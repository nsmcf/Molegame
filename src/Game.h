#pragma once

#include <opencv2/core.hpp>

#include <chrono>
#include <random>
#include <string>

struct GameConfig {
    int width = 960;
    int height = 720;
    int hitRadius = 85;
    int displayMs = 500;
    int waitMinMs = 800;
    int waitMaxMs = 1800;
    int feedbackMs = 600;
};

// 读取并检查游戏参数。
GameConfig loadGameConfig(const std::string& path = "config/game.yml");

class Game {
public:
    explicit Game(GameConfig config);
    // 运行鼠标预览，不连接相机判定。
    void run();

private:
    using Clock = std::chrono::steady_clock;
    enum class Phase { AwaitExit, Waiting, Active, Feedback };

    GameConfig config_;
    Phase phase_ = Phase::AwaitExit;
    std::mt19937 random_{std::random_device{}()};
    cv::Point mouse_{-1, -1};
    bool mouseKnown_ = false;
    bool mouseInside_ = false;
    bool fullscreen_ = false;
    bool activeSubmitted_ = false;
    int activeDurationMs_ = 500;
    int trials_ = 0;
    int hits_ = 0;
    std::string lastResult_ = "No result yet";
    Clock::time_point deadline_{};
    Clock::time_point submittedAt_{};
    const std::string windowName_ = "MoleGame - Mouse preview";

    // 接收鼠标位置，并检测首次进入。
    static void mouseCallback(int event, int x, int y, int flags, void* userdata);
    void onMouse(int x, int y);
    // 离开区域后安排随机等待。
    void scheduleTrial(Clock::time_point now);
    // 更新状态，并结束超时试次。
    void update(Clock::time_point now);
    void finishTrial(bool hit, Clock::time_point now);
    // 清空成绩，保留当前鼠标位置。
    void reset();
    // 绘制固定坑、地鼠与预览提示。
    cv::Mat render() const;
};
