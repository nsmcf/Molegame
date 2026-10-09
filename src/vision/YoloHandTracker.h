#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

enum class YoloBackend
{
    OpenCvDnn,
    TensorRt
};

struct YoloHandBox
{
    int classId = -1;
    std::string label;
    float confidence = 0.0f;
    cv::Rect box;
    cv::Point center;
};

struct YoloHandDetection
{
    bool found = false;
    YoloHandBox hand;
    cv::Point gameCenter;
    std::vector<YoloHandBox> boxes;
};

struct YoloHandPerformance
{
    double preprocessMs = 0.0;
    double inferenceMs = 0.0;
    double postprocessMs = 0.0;
    double totalMs = 0.0;
};

struct YoloHandTrackerConfig
{
    YoloBackend backend = YoloBackend::TensorRt;

    std::string onnxModelPath = "models/hand_detector.onnx";
    std::string tensorRtEnginePath = "models/hand_detector.engine";
    std::string tensorRtFp16IoEnginePath = "models/hand_detector_fp16_io.engine";
    std::string tensorRtFastEnginePath = "models/hand_detector_416_fp16_io.engine";
    std::string classNamesPath = "models/classes.names";

    int handClassId = 0;
    float confidenceThreshold = 0.65f;
    float nmsThreshold = 0.45f;
    int holdFramesAfterMiss = 0;
    float smoothingAlpha = 0.80f;
    float maxCenterJumpRatio = 0.40f;
    float maxBoxAreaChangeRatio = 4.0f;

    cv::Size networkInputSize = cv::Size(640, 640);
    cv::Size cameraFrameSize = cv::Size(720, 540);
    cv::Size gameOutputSize = cv::Size(1920, 1080);

    bool useCudaForOpenCvDnn = false;
    bool useCudaGraphForTensorRt = true;
    bool useGpuPipelineForTensorRt = true;
    bool useSpinWaitForTensorRt = false;
    bool preferFastTensorRtEngine = true;
    bool preferFp16IoTensorRtEngine = true;
    bool mirrorX = false;
    // Return up to two NMS-filtered per-frame hand boxes; temporal association
    // remains the responsibility of ShortHandTracker.
    bool multiHandTracking = false;

    // Optional 3x3 homography from camera coordinates to game/projector coordinates.
    cv::Mat cameraToGameHomography;
};

class YoloHandTracker
{
public:
    explicit YoloHandTracker(const YoloHandTrackerConfig& config = YoloHandTrackerConfig());
    ~YoloHandTracker();

    YoloHandTracker(const YoloHandTracker&) = delete;
    YoloHandTracker& operator=(const YoloHandTracker&) = delete;

    bool initialize();
    bool isInitialized() const;

    bool detect(const cv::Mat& frame, YoloHandDetection& detection);
    cv::Point mapCameraPointToGame(const cv::Point& cameraPoint, const cv::Size& cameraFrameSize) const;
    cv::Point mapCameraPointToGame(const cv::Point& cameraPoint) const;

    void drawDebug(cv::Mat& frame, const YoloHandDetection& detection) const;

    const YoloHandTrackerConfig& config() const;
    const std::string& lastError() const;
    const YoloHandPerformance& lastPerformance() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
