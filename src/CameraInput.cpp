#include "CameraInput.h"
#include <iostream>

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

// 仅检查迁移模型，不连接相机或打开窗口。
int checkHandModel(const std::string& imagePath) {
    try {
        const cv::Mat frame = cv::imread(imagePath);
        if (frame.empty()) throw std::runtime_error("Cannot read model-check image: " + imagePath);
        YoloHandTracker detector(readYoloSettings(frame.size()));
        requireDetector(detector);
        YoloHandDetection detection;
        detector.detect(frame, detection);
        if (!detector.lastError().empty()) throw std::runtime_error(detector.lastError());
        std::cout << "Model check OK: frame=" << frame.cols << 'x' << frame.rows
                  << ", hands=" << detection.boxes.size()
                  << ", detector_total_ms=" << detector.lastPerformance().totalMs << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

// 独立预览只验证采集和手模型，不用于正式地鼠判定。
int runCameraPreview() {
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
        const cv::Size frameSize(static_cast<int>(camera.Width.GetValue()), static_cast<int>(camera.Height.GetValue()));
        YoloHandTracker detector(readYoloSettings(frameSize));
        requireDetector(detector);
        Pylon::CImageFormatConverter converter;
        converter.OutputPixelFormat = Pylon::PixelType_BGR8packed;
        converter.OutputBitAlignment = Pylon::OutputBitAlignment_MsbAligned;
        Pylon::CPylonImage image;
        camera.StartGrabbing(Pylon::GrabStrategy_LatestImageOnly);
        cv::namedWindow("MoleGame camera preview", cv::WINDOW_NORMAL);
        cv::resizeWindow("MoleGame camera preview", frameSize.width, frameSize.height);
        std::uint64_t sequence = 0;
        auto nextLog = std::chrono::steady_clock::now();
        std::cout << "Camera: " << camera.GetDeviceInfo().GetModelName()
                  << "; raw hand preview, no mole judgment or calibration. ESC exits.\n";
        while (camera.IsGrabbing()) {
            if (cv::getWindowProperty("MoleGame camera preview", cv::WND_PROP_VISIBLE) < 1) break;
            Pylon::CGrabResultPtr grab;
            if (camera.RetrieveResult(100, grab, Pylon::TimeoutHandling_Return) && grab->GrabSucceeded()) {
                converter.Convert(image, grab);
                cv::Mat frame(static_cast<int>(grab->GetHeight()), static_cast<int>(grab->GetWidth()),
                              CV_8UC3, image.GetBuffer());
                frame = frame.clone();
                const auto capturedAt = std::chrono::steady_clock::now();
                ++sequence;
                YoloHandDetection detection;
                detector.detect(frame, detection);
                if (!detector.lastError().empty()) throw std::runtime_error(detector.lastError());
                for (const auto& hand : detection.boxes) {
                    cv::rectangle(frame, hand.box, cv::Scalar(0, 220, 0), 2);
                    cv::circle(frame, hand.center, 5, cv::Scalar(0, 220, 255), -1);
                }
                cv::putText(frame, "Hand preview only | ESC: exit", {12, 28},
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 220, 255), 2);
                cv::imshow("MoleGame camera preview", frame);
                const auto now = std::chrono::steady_clock::now();
                if (now >= nextLog) {
                    std::cout << "[Preview] frame=" << sequence << " hands=" << detection.boxes.size()
                              << " detector_total_ms=" << detector.lastPerformance().totalMs
                              << " capture_to_preview_submit_ms="
                              << std::chrono::duration<double, std::milli>(now - capturedAt).count() << '\n';
                    nextLog = now + std::chrono::seconds(1);
                }
            }
            if (cv::waitKeyEx(1) == 27) break;
        }
        camera.StopGrabbing();
        camera.Close();
        cv::destroyAllWindows();
        return 0;
    } catch (const GenICam::GenericException& error) {
        std::cerr << "Camera preview failed: " << error.GetDescription() << '\n';
        cv::destroyAllWindows();
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "Camera preview failed: " << error.what() << '\n';
        cv::destroyAllWindows();
        return 1;
    }
}
#else
int runCameraPreview() {
    std::cerr << "Camera preview is unavailable: build with MOLE_WITH_CAMERA=ON.\n";
    return 1;
}
int checkHandModel(const std::string&) {
    std::cerr << "Model check is unavailable: build with MOLE_WITH_CAMERA=ON.\n";
    return 1;
}
#endif
