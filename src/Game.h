#pragma once

#include <opencv2/core.hpp>
#include "vision/MoleDetector.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <vector>

struct GameConfig {
    int width = 960;
    int height = 720;
    int outputRotationCcw = 0;
    int hitRadius = 85;
    std::string targetStyle = "square";
    int targetBrightness = 100;
    int stateMarkers = 1;
    int displayMs = 500;
    int waitMinMs = 800;
    int waitMaxMs = 1800;
    int feedbackMs = 600;
    int inputTimeoutMs = 200;
    int observationTimeoutMs = 1000;
};

enum class GameMode { Mouse, InternalTiming, CameraObservation, ObservationProbe };

// 读取并检查游戏参数。
GameConfig loadGameConfig(const std::string& path = "config/game.yml");

class Game {
public:
    explicit Game(GameConfig config, GameMode mode = GameMode::Mouse);
    // 两种评分共用手部进入规则；探测模式用于检查颜色与遮挡。
    void run();

private:
    friend struct GameTestAccess;
    using Clock = std::chrono::steady_clock;
    enum class Phase { AwaitExit, Waiting, Active, Resolving, Feedback };
    enum class Outcome { Hit, Miss, Unjudged };

    GameConfig config_;
    Phase phase_ = Phase::AwaitExit;
    std::uint32_t randomSeed_ = std::random_device{}();
    std::mt19937 random_{randomSeed_};
    bool useCamera_ = false;
    bool observationProbe_ = false;
    bool cameraJudgment_ = false;
    MoleVisibility currentMole_ = MoleVisibility::Unknown;
    bool observedOn_ = false;
    std::ofstream observations_;
    cv::Point2f pointer_{-1, -1};
    bool inputKnown_ = false;
    bool inputInside_ = false;
    bool hasSample_ = false;
    bool trialUncertain_ = false;
    std::string uncertaintyReason_;
    bool fullscreen_ = false;
    cv::Rect displayedContent_; // 等比缩放后游戏画面在窗口中的位置，黑边不属于游戏坐标。
    bool activeSubmitted_ = false;
    int activeDurationMs_ = 500;
    int activeTargetBrightness_ = 100;
    int trials_ = 0;
    std::uint64_t trialId_ = 0;
    int hits_ = 0;
    int unjudged_ = 0;
    std::string lastResult_ = "No result yet";
    Clock::time_point deadline_{};
    Clock::time_point submittedAt_{};
    Clock::time_point observationDeadline_{};
    Clock::time_point lastSampleAt_{};
    Clock::time_point ignoreBefore_{};
    const std::string windowName_;
    bool calibrating_ = false;
    cv::Size cameraSize_;
    std::string cameraId_;
    std::vector<cv::Point2f> calibrationCorners_;
    cv::Mat cameraToGame_;
    std::string calibrationMessage_ = "Click projected labels 1,2,3,4 in the camera window";
    Clock::time_point sessionStart_ = Clock::now();
    std::ofstream events_;
    std::string sessionDirectory_;
    std::uint64_t currentFrame_ = 0;
    Clock::time_point currentSampleAt_{};

    // 接收鼠标位置，并检测首次进入。
    static void mouseCallback(int event, int x, int y, int flags, void* userdata);
    void onMouse(int x, int y);
    void onInput(std::optional<cv::Point2f> point, Clock::time_point sampledAt,
                 Clock::time_point receivedAt, std::uint64_t frame = 0,
                 MoleVisibility mole = MoleVisibility::Unknown, const std::string& evidenceReason = "");
    bool inputFresh(Clock::time_point now) const;
    // 离开区域后安排随机等待。
    void scheduleTrial(Clock::time_point now);
    // 更新状态，并结束超时试次。
    void update(Clock::time_point now);
    void submitFrame(Clock::time_point now);
    void finishTrial(Outcome result, Clock::time_point now, Clock::time_point entryAt = {}, const std::string& reason = "");
    void markUncertain(const std::string& reason);
    // 清空成绩，保留当前鼠标位置。
    void reset(Clock::time_point now = Clock::now());
    static void calibrationClick(int event, int x, int y, int flags, void* userdata);
    void beginCalibration();
    void loadCalibration(const std::string& path = "config/camera-calibration.yml");
    void confirmCalibration();
    cv::Mat renderCalibration() const;
    void openLog();
    void logEvent(const std::string& event, Clock::time_point now,
                  const std::string& detail = "", int handCount = -1);
    // 绘制固定坑、地鼠与预览提示。
    cv::Mat render() const;
    // 只旋转最终输出；标定和评分仍使用原来的游戏坐标。
    cv::Mat rotateForOutput(const cv::Mat& frame) const;
    cv::Size outputSize() const;
    cv::Size gameSize() const;
    cv::Mat fitForWindow(const cv::Mat& frame, cv::Size viewport);
    void toggleOutputRotation();
    void adjustTargetBrightness(int delta);
};
