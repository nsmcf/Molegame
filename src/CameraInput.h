#pragma once
#include <opencv2/core.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct HandObservation {
    using Clock = std::chrono::steady_clock;
    std::uint64_t sequence = 0;
    Clock::time_point capturedAt{};  // 转换并复制后的主机时间，不是曝光时间。
    Clock::time_point completedAt{};
    std::vector<cv::Point2f> centers;
    double detectorMs = 0;
};

struct CameraUpdate {
    cv::Mat preview;
    std::vector<HandObservation> observations;
    std::size_t dropped = 0;
};

// 采集和 YOLO 在后台顺序执行，主线程负责窗口和游戏计时。
class CameraInput {
public:
    CameraInput();
    ~CameraInput();
    CameraInput(const CameraInput&) = delete;
    CameraInput& operator=(const CameraInput&) = delete;
    cv::Size frameSize() const;
    const std::string& deviceId() const;
    void poll(CameraUpdate& update);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::vector<cv::Point2f> calibrationTargets(cv::Size gameSize);
// 点击顺序对应投影中的 1、2、3、4；允许相机旋转，不按图像上下位置重新排序。
cv::Mat cameraToGameTransform(const std::vector<cv::Point2f>& corners,
                              cv::Size cameraSize, cv::Size gameSize);

int runCameraPreview();
int checkHandModel(const std::string& imagePath);
