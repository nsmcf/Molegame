#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

enum class MoleVisibility { Unknown, Visible, Absent };

struct MoleDetectorConfig {
    int hueMin = 10, hueMax = 45;
    int saturationMin = 90, valueMin = 70;
    double visibleRatio = 0.20, absentRatio = 0.02;
    double minUnoccludedRatio = 0.30;
    int handMargin = 6;
};

struct MoleObservation {
    MoleVisibility state = MoleVisibility::Unknown;
    double colorRatio = 0;
    double unoccludedRatio = 0;
    double detectorMs = 0;
    std::string reason = "no_region";
};

MoleDetectorConfig loadMoleDetectorConfig(const std::string& path = "config/mole.yml");
const char* visibilityName(MoleVisibility state);
// 仅使用本帧原图、标定区域和同帧手框；不使用程序显示状态或跨帧保持。
MoleObservation observeMole(const cv::Mat& bgr, const std::vector<cv::Point2f>& region,
                            const std::vector<cv::Rect>& hands, const MoleDetectorConfig& config);
// 同帧同步色块：一块清晰即可提供证据；相互矛盾或全部不明时保留 UNKNOWN。
MoleObservation observeMoleRegions(const cv::Mat& bgr, const std::vector<std::vector<cv::Point2f>>& regions,
                                   const std::vector<cv::Rect>& hands, const MoleDetectorConfig& config);
