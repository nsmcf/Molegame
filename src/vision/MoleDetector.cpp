#include "MoleDetector.h"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace {
void validate(const MoleDetectorConfig& c) {
    if (c.hueMin < 0 || c.hueMax > 179 || c.hueMin > c.hueMax ||
        c.saturationMin < 0 || c.saturationMin > 255 || c.valueMin < 0 || c.valueMin > 255 ||
        !std::isfinite(c.absentRatio) || !std::isfinite(c.visibleRatio) ||
        c.absentRatio < 0 || c.absentRatio >= c.visibleRatio || c.visibleRatio > 1 ||
        !std::isfinite(c.minUnoccludedRatio) || c.minUnoccludedRatio <= 0 ||
        c.minUnoccludedRatio > 1 || c.handMargin < 0 || c.handMargin > 100)
        throw std::runtime_error("Invalid thresholds in config/mole.yml");
}
}

MoleDetectorConfig loadMoleDetectorConfig(const std::string& path) {
    MoleDetectorConfig c;
    cv::FileStorage file(path, cv::FileStorage::READ);
    if (!file.isOpened()) throw std::runtime_error("Cannot read " + path);
    const auto read = [&](const char* key, auto& value) { if (!file[key].empty()) file[key] >> value; };
    read("hue_min", c.hueMin); read("hue_max", c.hueMax);
    read("saturation_min", c.saturationMin); read("value_min", c.valueMin);
    read("visible_ratio", c.visibleRatio); read("absent_ratio", c.absentRatio);
    read("min_unoccluded_ratio", c.minUnoccludedRatio); read("hand_margin", c.handMargin);
    validate(c);
    return c;
}

const char* visibilityName(MoleVisibility state) {
    switch (state) {
    case MoleVisibility::Visible: return "VISIBLE";
    case MoleVisibility::Absent: return "ABSENT";
    default: return "UNKNOWN";
    }
}

MoleObservation observeMole(const cv::Mat& bgr, const std::vector<cv::Point2f>& region,
                            const std::vector<cv::Rect>& hands, const MoleDetectorConfig& c) {
    validate(c);
    const auto start = std::chrono::steady_clock::now();
    MoleObservation result;
    const auto finish = [&](const char* reason) {
        result.reason = reason;
        result.detectorMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        return result;
    };
    if (bgr.empty() || bgr.type() != CV_8UC3) return finish("invalid_frame");
    if (region.size() < 3) return finish("no_region");
    std::vector<cv::Point> polygon;
    for (const auto& p : region) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 ||
            p.x >= bgr.cols || p.y >= bgr.rows) return finish("region_outside_frame");
        polygon.emplace_back(cvRound(p.x), cvRound(p.y));
    }
    // 小投影区域取整后会出现共线点或轻微凹折，不能据此否定有效标定。
    // 先验证浮点区域，再用整数凸包消除栅格化造成的重复/共线边。
    if (!cv::isContourConvex(region)) return finish("invalid_region");
    std::vector<cv::Point> hull;
    cv::convexHull(polygon, hull);
    if (hull.size() < 3) return finish("region_too_small");
    cv::Mat roi(bgr.size(), CV_8U, cv::Scalar(0));
    cv::fillConvexPoly(roi, hull, cv::Scalar(255));
    const int total = cv::countNonZero(roi);
    if (total < 50) return finish("region_too_small");
    cv::Mat available = roi.clone();
    for (const auto& hand : hands) {
        if (hand.width <= 0 || hand.height <= 0) continue;
        const cv::Rect expanded(hand.x - c.handMargin, hand.y - c.handMargin,
                                hand.width + 2 * c.handMargin, hand.height + 2 * c.handMargin);
        const auto clipped = expanded & cv::Rect(0, 0, bgr.cols, bgr.rows);
        if (!clipped.empty()) available(clipped).setTo(0);
    }
    const int remaining = cv::countNonZero(available);
    result.unoccludedRatio = static_cast<double>(remaining) / total;
    if (result.unoccludedRatio < c.minUnoccludedRatio) return finish("occluded");
    cv::Mat hsv, color;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(c.hueMin, c.saturationMin, c.valueMin),
                cv::Scalar(c.hueMax, 255, 255), color);
    cv::bitwise_and(color, available, color);
    result.colorRatio = static_cast<double>(cv::countNonZero(color)) / remaining;
    if (result.colorRatio >= c.visibleRatio) {
        result.state = MoleVisibility::Visible;
        return finish("color_present");
    }
    // 手框遮住区域时，缺少颜色不能证明地鼠已消失。
    if (remaining != total) return finish("occluded_without_color");
    if (result.colorRatio <= c.absentRatio) {
        result.state = MoleVisibility::Absent;
        return finish("color_absent");
    }
    return finish("color_ambiguous");
}

MoleObservation observeMoleRegions(const cv::Mat& bgr, const std::vector<std::vector<cv::Point2f>>& regions,
                                   const std::vector<cv::Rect>& hands, const MoleDetectorConfig& config) {
    if (regions.size() == 1) return observeMole(bgr, regions.front(), hands, config);
    MoleObservation result;
    bool visible = false, absent = false, allOccluded = !regions.empty();
    double totalMs = 0;
    for (const auto& region : regions) {
        const auto evidence = observeMole(bgr, region, hands, config);
        totalMs += evidence.detectorMs;
        visible |= evidence.state == MoleVisibility::Visible;
        absent |= evidence.state == MoleVisibility::Absent;
        allOccluded &= evidence.reason == "occluded" || evidence.reason == "occluded_without_color";
        if (evidence.state != MoleVisibility::Unknown &&
            (result.state == MoleVisibility::Unknown || evidence.unoccludedRatio > result.unoccludedRatio)) result = evidence;
        else if (result.state == MoleVisibility::Unknown && evidence.unoccludedRatio > result.unoccludedRatio) result = evidence;
    }
    result.detectorMs = totalMs;
    if (visible && absent) {
        result.state = MoleVisibility::Unknown;
        result.reason = "state_marker_conflict";
    } else if (visible) result.reason = "state_marker_visible";
    else if (absent) result.reason = "state_marker_absent";
    else result.reason = regions.empty() ? "no_region" : allOccluded ? "state_markers_occluded" : "state_markers_unknown";
    return result;
}
