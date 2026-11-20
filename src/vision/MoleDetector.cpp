#include "MoleDetector.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
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
        c.minUnoccludedRatio > 1 || c.handMargin < 0 || c.handMargin > 100 ||
        c.handYellowContrast < 1 || c.handYellowContrast > 255)
        throw std::runtime_error("Invalid thresholds in config/mole.yml");
}

int largestPatch(const cv::Mat& mask) {
    cv::Mat labels, stats, centers;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centers, 8);
    int largest = 0;
    for (int label = 1; label < count; ++label)
        largest = std::max(largest, stats.at<int>(label, cv::CC_STAT_AREA));
    return largest;
}

// 黄色光往往同时抬高红、绿通道，相对于蓝通道的差值可用于比较局部投影。
int yellowStrength(const cv::Vec3b& bgr) {
    return std::max(0, std::min(bgr[1], bgr[2]) - bgr[0]);
}

cv::Rect padded(cv::Rect rect, int dx, int dy) {
    return {rect.x - dx, rect.y - dy, rect.width + 2 * dx, rect.height + 2 * dy};
}

// 在同一个手框内取样。外围参照跳过暗背景和灰色像素；样本不足时不给手上颜色投票。
void addHandEvidence(const cv::Mat& frame, const cv::Mat& yellow, cv::Rect roi,
                     cv::Rect hand, int contrast, cv::Mat& accepted) {
    const cv::Rect inside = hand & roi;
    if (inside.empty()) return;
    const cv::Rect guard = padded(roi, (roi.width + 7) / 8, (roi.height + 7) / 8);
    const cv::Rect nearby = padded(guard, std::max(4, roi.width / 3), std::max(4, roi.height / 3));
    const cv::Rect reference = nearby & hand & cv::Rect(0, 0, frame.cols, frame.rows);
    std::array<int, 256> histogram{};
    int samples = 0;
    for (int y = reference.y; y < reference.br().y; ++y) {
        const auto* row = frame.ptr<cv::Vec3b>(y);
        for (int x = reference.x; x < reference.br().x; ++x) {
            if (guard.contains(cv::Point(x, y))) continue;
            const auto& pixel = row[x];
            if (pixel[2] < 40 || pixel[2] <= pixel[0] + 10 || pixel[1] <= pixel[0]) continue;
            ++histogram[yellowStrength(pixel)];
            ++samples;
        }
    }
    if (samples < std::max(16, inside.area() / 50)) return;
    int level = 0, cumulative = histogram[0];
    while (cumulative * 4 < samples * 3)
        cumulative += histogram[++level];
    for (int y = inside.y; y < inside.br().y; ++y) {
        const auto* row = frame.ptr<cv::Vec3b>(y);
        const auto* candidate = yellow.ptr<uchar>(y - roi.y);
        auto* output = accepted.ptr<uchar>(y - roi.y);
        for (int x = inside.x; x < inside.br().x; ++x) {
            if (candidate[x - roi.x] && yellowStrength(row[x]) >= level + contrast)
                output[x - roi.x] = 255;
        }
    }
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
    read("hand_yellow_contrast", c.handYellowContrast);
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
    const cv::Rect bounds = cv::boundingRect(hull) & cv::Rect(0, 0, bgr.cols, bgr.rows);
    for (auto& point : hull) point -= bounds.tl();
    cv::Mat roi(bounds.size(), CV_8U, cv::Scalar(0));
    cv::fillConvexPoly(roi, hull, cv::Scalar(255));
    const int total = cv::countNonZero(roi);
    if (total < 50) return finish("region_too_small");

    cv::Mat hsv, yellow;
    cv::cvtColor(bgr(bounds), hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(c.hueMin, c.saturationMin, c.valueMin),
                cv::Scalar(c.hueMax, 255, 255), yellow);
    cv::bitwise_and(yellow, roi, yellow);
    cv::Mat available = roi.clone();
    for (const auto& hand : hands) {
        if (hand.empty()) continue;
        const auto clipped = padded(hand, c.handMargin, c.handMargin) & bounds;
        if (!clipped.empty()) available(clipped - bounds.tl()).setTo(0);
    }
    const int remaining = cv::countNonZero(available);
    result.unoccludedRatio = static_cast<double>(remaining) / total;
    cv::Mat accepted;
    cv::bitwise_and(yellow, available, accepted);
    const int clearYellow = cv::countNonZero(accepted);
    result.colorRatio = static_cast<double>(clearYellow) / total;
    if (result.unoccludedRatio >= c.minUnoccludedRatio &&
        clearYellow >= c.visibleRatio * remaining && largestPatch(accepted) >= c.visibleRatio * remaining) {
        result.state = MoleVisibility::Visible;
        return finish("color_present");
    }
    if (remaining < total) {
        for (const auto& hand : hands)
            addHandEvidence(bgr, yellow, bounds, hand, c.handYellowContrast, accepted);
        const int combined = cv::countNonZero(accepted);
        result.colorRatio = static_cast<double>(combined) / total;
        if (combined > clearYellow && combined >= c.visibleRatio * total &&
            largestPatch(accepted) >= c.visibleRatio * total) {
            result.state = MoleVisibility::Visible;
            return finish("projected_color_on_hand");
        }
        return finish(remaining == 0 ? "occluded" : "occluded_without_color");
    }
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
