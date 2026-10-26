#include "CameraInput.h"
#include <iostream>
#include <utility>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>

std::vector<cv::Point2f> calibrationTargets(cv::Size size) {
    // 围绕击打中心取中间一半的区域，减少相机画面边缘畸变的影响。
    const float cx = size.width * 0.5f, cy = size.height * 0.5f + 30;
    const float dx = size.width * 0.25f, dy = size.height * 0.25f;
    return {{cx - dx, cy - dy}, {cx + dx, cy - dy}, {cx + dx, cy + dy}, {cx - dx, cy + dy}};
}

cv::Mat cameraToGameTransform(const std::vector<cv::Point2f>& corners,
                              cv::Size cameraSize, cv::Size gameSize) {
    if (corners.size() != 4 || cameraSize.width <= 0 || cameraSize.height <= 0 ||
        gameSize.width <= 0 || gameSize.height <= 0)
        throw std::runtime_error("Calibration requires four points and positive image sizes");
    for (const auto& p : corners) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 ||
            p.x >= cameraSize.width || p.y >= cameraSize.height)
            throw std::runtime_error("Calibration points must be inside the camera image");
    }
    if (!cv::isContourConvex(corners) ||
        std::abs(cv::contourArea(corners)) < cameraSize.area() * 0.01)
        throw std::runtime_error("Invalid point order or area. Press H and click projected labels 1,2,3,4");
    const auto targets = calibrationTargets(gameSize);
    cv::Mat transform = cv::getPerspectiveTransform(corners, targets);
    if (!cv::checkRange(transform) || std::abs(cv::determinant(transform)) < 1e-10)
        throw std::runtime_error("Degenerate calibration");
    return transform;
}

std::vector<cv::Point2f> moleRegionInCamera(const cv::Mat& transform, cv::Size gameSize, int hitRadius) {
    if (transform.empty()) return {};
    std::vector<cv::Point2f> disk, region;
    for (int i = 0; i < 24; ++i) {
        const double angle = i * 2 * CV_PI / 24;
        disk.emplace_back(gameSize.width / 2.f + hitRadius * 0.70f * static_cast<float>(std::cos(angle)),
            gameSize.height / 2.f + 30 + hitRadius * 0.70f * static_cast<float>(std::sin(angle)));
    }
    cv::perspectiveTransform(disk, region, transform.inv());
    return region;
}

std::vector<cv::Rect> moleStateMarkers(cv::Size size, int hitRadius) {
    const int side = std::clamp(hitRadius, 40, std::min(size.width, size.height) / 8);
    const cv::Point center(size.width / 2, size.height / 2 + 30);
    const int offset = std::min(3 * hitRadius, size.width / 2 - side / 2 - 20);
    return {{center.x - offset - side / 2, center.y - side / 2, side, side},
            {center.x + offset - side / 2, center.y - side / 2, side, side}};
}

std::vector<std::vector<cv::Point2f>> moleObservationRegionsInCamera(
    const cv::Mat& transform, cv::Size gameSize, int hitRadius, bool stateMarkers) {
    if (transform.empty()) return {};
    if (!stateMarkers) return {moleRegionInCamera(transform, gameSize, hitRadius)};
    std::vector<std::vector<cv::Point2f>> regions;
    for (const auto& rect : moleStateMarkers(gameSize, hitRadius)) {
        std::vector<cv::Point2f> corners{{static_cast<float>(rect.x), static_cast<float>(rect.y)},
            {static_cast<float>(rect.x + rect.width - 1), static_cast<float>(rect.y)},
            {static_cast<float>(rect.x + rect.width - 1), static_cast<float>(rect.y + rect.height - 1)},
            {static_cast<float>(rect.x), static_cast<float>(rect.y + rect.height - 1)}};
        std::vector<cv::Point2f> mapped;
        cv::perspectiveTransform(corners, mapped, transform.inv());
        regions.push_back(std::move(mapped));
    }
    return regions;
}

#ifdef MOLE_WITH_CAMERA
#include "vision/YoloHandTracker.h"
#include <pylon/PylonIncludes.h>
#include <pylon/BaslerUniversalInstantCamera.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <exception>
#include <future>
#include <mutex>
#include <thread>

namespace {
struct CameraSettings {
    int width = 720;
    int height = 540;
    double exposureUs = 1500.0;
};

// 读取相机尺寸和曝光，避免无效参数进入采集。
CameraSettings readCameraSettings() {
    CameraSettings settings;
    cv::FileStorage file("config/camera.yml", cv::FileStorage::READ);
    if (!file.isOpened()) throw std::runtime_error("Cannot read config/camera.yml");
    if (!file["width"].empty()) file["width"] >> settings.width;
    if (!file["height"].empty()) file["height"] >> settings.height;
    if (!file["exposure_us"].empty()) file["exposure_us"] >> settings.exposureUs;
    if (settings.width <= 0 || settings.height <= 0 ||
        !std::isfinite(settings.exposureUs) || settings.exposureUs <= 0)
        throw std::runtime_error("Camera size and exposure must be positive");
    return settings;
}

// 使用旧模型的原始检测框，预览阶段不做平滑或坐标标定。
YoloHandTrackerConfig readYoloSettings(cv::Size frameSize) {
    YoloHandTrackerConfig config;
    config.cameraFrameSize = frameSize;
    config.multiHandTracking = true;
    cv::FileStorage file("config/yolo.yml", cv::FileStorage::READ);
    if (!file.isOpened()) throw std::runtime_error("Cannot read config/yolo.yml");
    std::string backend = "tensorrt";
    if (!file["backend"].empty()) file["backend"] >> backend;
    if (backend == "tensorrt") config.backend = YoloBackend::TensorRt;
    else if (backend == "opencv") config.backend = YoloBackend::OpenCvDnn;
    else throw std::runtime_error("YOLO backend must be tensorrt or opencv");
    auto path = [&](const char* key, std::string& value) {
        if (!file[key].empty()) file[key] >> value;
        if (value.empty()) throw std::runtime_error(std::string("Empty path: ") + key);
    };
    path("onnx_model", config.onnxModelPath);
    path("tensorrt_engine", config.tensorRtEnginePath);
    path("tensorrt_fp16_engine", config.tensorRtFp16IoEnginePath);
    path("tensorrt_fast_engine", config.tensorRtFastEnginePath);
    path("class_names", config.classNamesPath);
    if (!file["confidence_threshold"].empty()) file["confidence_threshold"] >> config.confidenceThreshold;
    if (!file["nms_threshold"].empty()) file["nms_threshold"] >> config.nmsThreshold;
    auto flag = [&](const char* key, bool& value) {
        if (!file[key].empty()) { int number = 0; file[key] >> number; value = number != 0; }
    };
    flag("use_cuda_graph", config.useCudaGraphForTensorRt);
    flag("use_gpu_pipeline", config.useGpuPipelineForTensorRt);
    flag("use_spin_wait", config.useSpinWaitForTensorRt);
    if (!std::isfinite(config.confidenceThreshold) || config.confidenceThreshold < 0 || config.confidenceThreshold > 1 ||
        !std::isfinite(config.nmsThreshold) || config.nmsThreshold < 0 || config.nmsThreshold > 1)
        throw std::runtime_error("YOLO thresholds must be between zero and one");
    return config;
}

void requireDetector(YoloHandTracker& detector) {
    if (!detector.initialize()) throw std::runtime_error("YOLO initialization failed: " + detector.lastError());
}
}

struct CameraInput::Impl {
    cv::Size size;
    std::string serial;
    std::mutex mutex;
    cv::Mat preview;
    cv::Mat rawPreview;
    HandObservation previewSample;
    std::vector<HandObservation> pending;
    std::size_t dropped = 0;
    std::exception_ptr failure;
    std::vector<std::vector<cv::Point2f>> moleRegions;
    std::uint64_t regionRevision = 0;
    // 最后声明，确保销毁时先停止并等待线程，再释放共享数据。
    std::jthread worker;

    explicit Impl(bool enableMole) {
        std::promise<void> ready;
        auto initialized = ready.get_future();
        worker = std::jthread([this, enableMole, signal = std::move(ready)](std::stop_token stop) mutable {
            bool started = false;
            const auto fail = [&](std::exception_ptr error) {
                { std::lock_guard lock(mutex); failure = error; }
                if (!started) signal.set_exception(error);
            };
            try {
                Pylon::PylonAutoInitTerm pylon;
                Pylon::CBaslerUniversalInstantCamera camera(Pylon::CTlFactory::GetInstance().CreateFirstDevice());
                camera.Open();
                const auto settings = readCameraSettings();
                camera.Width.SetValue(settings.width);
                camera.Height.SetValue(settings.height);
                camera.ExposureAuto.SetValue(Basler_UniversalCameraParams::ExposureAuto_Off);
                camera.ExposureTime.SetValue(settings.exposureUs);
                if (camera.AcquisitionFrameRateEnable.IsWritable()) camera.AcquisitionFrameRateEnable.SetValue(false);
                size = {static_cast<int>(camera.Width.GetValue()), static_cast<int>(camera.Height.GetValue())};
                serial = camera.GetDeviceInfo().GetSerialNumber().c_str();
                YoloHandTracker detector(readYoloSettings(size));
                requireDetector(detector);
                const auto moleSettings = enableMole ? loadMoleDetectorConfig() : MoleDetectorConfig{};
                Pylon::CImageFormatConverter converter;
                converter.OutputPixelFormat = Pylon::PixelType_BGR8packed;
                converter.OutputBitAlignment = Pylon::OutputBitAlignment_MsbAligned;
                Pylon::CPylonImage image;
                camera.StartGrabbing(Pylon::GrabStrategy_LatestImageOnly);
                signal.set_value();
                started = true;
                std::uint64_t sequence = 0;
                auto lastFrameAt = HandObservation::Clock::now();
                while (!stop.stop_requested()) {
                    Pylon::CGrabResultPtr grab;
                    if (!camera.RetrieveResult(100, grab, Pylon::TimeoutHandling_Return)) {
                        if (HandObservation::Clock::now() - lastFrameAt > std::chrono::seconds(2))
                            throw std::runtime_error("No camera frames for 2 seconds; check connection and camera trigger settings");
                        continue;
                    }
                    if (!grab->GrabSucceeded()) throw std::runtime_error("Camera grab failed");
                    converter.Convert(image, grab);
                    cv::Mat frame = cv::Mat(size.height, size.width, CV_8UC3, image.GetBuffer()).clone();
                    HandObservation observation;
                    observation.sequence = ++sequence;
                    observation.capturedAt = HandObservation::Clock::now();
                    lastFrameAt = observation.capturedAt;
                    std::vector<std::vector<cv::Point2f>> regions;
                    {
                        std::lock_guard lock(mutex);
                        regions = moleRegions;
                        observation.regionRevision = regionRevision;
                    }
                    YoloHandDetection detection;
                    detector.detect(frame, detection);
                    if (!detector.lastError().empty()) throw std::runtime_error(detector.lastError());
                    observation.detectorMs = detector.lastPerformance().totalMs;
                    for (const auto& hand : detection.boxes) {
                        observation.centers.emplace_back(hand.center);
                        observation.boxes.push_back(hand.box);
                    }
                    // 两种检测使用同一张未绘制调试标记的图像。
                    if (enableMole) observation.mole = observeMoleRegions(frame, regions, observation.boxes, moleSettings);
                    observation.completedAt = HandObservation::Clock::now();
                    cv::Mat raw = frame.clone();
                    for (const auto& hand : detection.boxes) {
                        cv::rectangle(frame, hand.box, cv::Scalar(0, 220, 0), 2);
                        cv::putText(frame, "hand " + std::to_string(hand.confidence).substr(0, 4),
                            {hand.box.x, std::max(12, hand.box.y - 4)}, cv::FONT_HERSHEY_SIMPLEX,
                            0.45, cv::Scalar(0, 220, 0), 1);
                        cv::circle(frame, hand.center, 5, cv::Scalar(0, 220, 255), -1);
                    }
                    std::lock_guard lock(mutex);
                    if (pending.size() >= 512) {
                        dropped += pending.size();
                        pending.clear();
                    }
                    previewSample = observation;
                    pending.push_back(std::move(observation));
                    rawPreview = std::move(raw);
                    preview = std::move(frame);
                }
                camera.StopGrabbing();
                camera.Close();
            } catch (const GenICam::GenericException& error) {
                fail(std::make_exception_ptr(std::runtime_error(error.GetDescription())));
            } catch (...) { fail(std::current_exception()); }
        });
        initialized.get();
    }
};

CameraInput::CameraInput(bool observeMole) : impl_(std::make_unique<Impl>(observeMole)) {}
CameraInput::~CameraInput() = default;
cv::Size CameraInput::frameSize() const { return impl_->size; }
const std::string& CameraInput::deviceId() const { return impl_->serial; }
std::uint64_t CameraInput::setMoleRegions(const std::vector<std::vector<cv::Point2f>>& regions) {
    std::lock_guard lock(impl_->mutex);
    impl_->moleRegions = regions;
    return ++impl_->regionRevision;
}
void CameraInput::poll(CameraUpdate& update) {
    std::lock_guard lock(impl_->mutex);
    if (impl_->failure) std::rethrow_exception(impl_->failure);
    update.preview = impl_->preview;
    update.rawPreview = impl_->rawPreview;
    update.previewSample = impl_->previewSample;
    update.observations.clear();
    update.observations.swap(impl_->pending);
    update.dropped = std::exchange(impl_->dropped, 0);
}

// 离线诊断原图、灰度和局部对比度；仅本次检查降低候选门槛，不改变游戏设置。
int checkHandModel(const std::string& imagePath) {
    try {
        const cv::Mat frame = cv::imread(imagePath);
        if (frame.empty()) throw std::runtime_error("Cannot read model-check image: " + imagePath);
        auto settings = readYoloSettings(frame.size());
        const float configuredThreshold = settings.confidenceThreshold;
        settings.confidenceThreshold = std::min(configuredThreshold, 0.05f);
        YoloHandTracker detector(settings);
        requireDetector(detector);
        cv::Mat gray, grayBgr, contrast, contrastBgr;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(gray, grayBgr, cv::COLOR_GRAY2BGR);
        cv::createCLAHE(2.0)->apply(gray, contrast);
        cv::cvtColor(contrast, contrastBgr, cv::COLOR_GRAY2BGR);
        for (const auto& [name, input] : std::vector<std::pair<const char*, cv::Mat>>{
                {"raw", frame}, {"gray", grayBgr}, {"gray_clahe", contrastBgr}}) {
            YoloHandDetection detection;
            detector.detect(input, detection);
            if (!detector.lastError().empty()) throw std::runtime_error(detector.lastError());
            const auto accepted = std::count_if(detection.boxes.begin(), detection.boxes.end(),
                [&](const auto& hand) { return hand.confidence >= configuredThreshold; });
            std::cout << "Model check OK: variant=" << name << ", frame=" << frame.cols << 'x' << frame.rows
                      << ", hands=" << accepted << ", configured_threshold=" << configuredThreshold
                      << ", candidate_threshold=" << settings.confidenceThreshold
                      << ", candidates=" << detection.boxes.size()
                      << ", detector_total_ms=" << detector.lastPerformance().totalMs << '\n';
            for (const auto& hand : detection.boxes)
                std::cout << "  candidate confidence=" << hand.confidence << " box=" << hand.box
                          << " accepted=" << (hand.confidence >= configuredThreshold) << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

// 独立预览只验证采集和手模型，不用于正式地鼠判定。
int runCameraPreview() {
    try {
        CameraInput camera;
        const std::string window = "MoleGame camera preview";
        cv::namedWindow(window, cv::WINDOW_NORMAL);
        cv::resizeWindow(window, camera.frameSize().width, camera.frameSize().height);
        auto nextLog = HandObservation::Clock::now();
        std::cout << "Camera serial: " << camera.deviceId() << "; raw hand preview. ESC exits.\n";
        for (;;) {
            CameraUpdate update;
            camera.poll(update);
            if (!update.preview.empty()) cv::imshow(window, update.preview);
            const auto now = HandObservation::Clock::now();
            if (!update.observations.empty() && now >= nextLog) {
                const auto& sample = update.observations.back();
                std::cout << "[Preview] frame=" << sample.sequence << " hands=" << sample.centers.size()
                          << " detector_total_ms=" << sample.detectorMs
                          << " capture_to_preview_submit_ms="
                          << std::chrono::duration<double, std::milli>(now - sample.capturedAt).count()
                          << " dropped=" << update.dropped << '\n';
                nextLog = now + std::chrono::seconds(1);
            }
            if (cv::waitKeyEx(1) == 27 || cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1) break;
        }
        cv::destroyAllWindows();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Camera preview failed: " << error.what() << '\n';
        cv::destroyAllWindows();
        return 1;
    }
}
#else
struct CameraInput::Impl {};
CameraInput::CameraInput(bool) { throw std::runtime_error("Camera input unavailable: build with MOLE_WITH_CAMERA=ON"); }
CameraInput::~CameraInput() = default;
cv::Size CameraInput::frameSize() const { return {}; }
const std::string& CameraInput::deviceId() const { static const std::string empty; return empty; }
void CameraInput::poll(CameraUpdate&) { throw std::runtime_error("Camera input unavailable"); }
std::uint64_t CameraInput::setMoleRegions(const std::vector<std::vector<cv::Point2f>>&) { return 0; }
int runCameraPreview() {
    std::cerr << "Camera preview is unavailable: build with MOLE_WITH_CAMERA=ON.\n";
    return 1;
}
int checkHandModel(const std::string&) {
    std::cerr << "Model check is unavailable: build with MOLE_WITH_CAMERA=ON.\n";
    return 1;
}
#endif
