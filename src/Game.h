#pragma once

#include <opencv2/core.hpp>

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
    int hitRadius = 85;
    int displayMs = 500;
    int waitMinMs = 800;
    int waitMaxMs = 1800;
    int feedbackMs = 600;
    int inputTimeoutMs = 200;
};

// 读取并检查游戏参数。
GameConfig loadGameConfig(const std::string& path = "config/game.yml");

class Game {
public:
    explicit Game(GameConfig config, bool useCamera = false);
    // 鼠标预览或 YOLO 手部输入；两者共用内部计时规则。
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
    cv::Point2f pointer_{-1, -1};
    bool inputKnown_ = false;
    bool inputInside_ = false;
    bool hasSample_ = false;
    bool trialUncertain_ = false;
    bool fullscreen_ = false;
    bool activeSubmitted_ = false;
    int activeDurationMs_ = 500;
    int trials_ = 0;
    std::uint64_t trialId_ = 0;
    int hits_ = 0;
    int unjudged_ = 0;
    std::string lastResult_ = "No result yet";
    Clock::time_point deadline_{};
    Clock::time_point submittedAt_{};
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
                 Clock::time_point receivedAt, std::uint64_t frame = 0);
    bool inputFresh(Clock::time_point now) const;
    // 离开区域后安排随机等待。
    void scheduleTrial(Clock::time_point now);
    // 更新状态，并结束超时试次。
    void update(Clock::time_point now);
    void submitFrame(Clock::time_point now);
    void finishTrial(Outcome result, Clock::time_point now, Clock::time_point entryAt = {});
    // 清空成绩，保留当前鼠标位置。
    void reset(Clock::time_point now = Clock::now());
    static void calibrationClick(int event, int x, int y, int flags, void* userdata);
    void beginCalibration();
    void loadCalibration();
    void confirmCalibration();
    cv::Mat renderCalibration() const;
    void openLog();
    void logEvent(const std::string& event, Clock::time_point now,
                  const std::string& detail = "", int handCount = -1);
    // 绘制固定坑、地鼠与预览提示。
    cv::Mat render() const;
};
