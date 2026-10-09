#include "Game.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

GameConfig loadGameConfig(const std::string& path) {
    cv::FileStorage file(path, cv::FileStorage::READ);
    if (!file.isOpened()) {
        throw std::runtime_error("Cannot open " + path + ". Run from the MoleGame project folder.");
    }
    GameConfig config;
    const auto read = [&file](const char* key, int& value) {
        if (!file[key].empty()) file[key] >> value;
    };
    read("width", config.width);
    read("height", config.height);
    read("hit_radius", config.hitRadius);
    read("display_ms", config.displayMs);
    read("wait_min_ms", config.waitMinMs);
    read("wait_max_ms", config.waitMaxMs);
    read("feedback_ms", config.feedbackMs);
    if (config.width < 640 || config.width > 3840 || config.height < 480 || config.height > 2160 ||
        config.hitRadius < 20 || config.hitRadius > std::min(config.width, config.height) / 4 ||
        config.displayMs < 50 || config.displayMs > 10000 ||
        config.waitMinMs < 100 || config.waitMaxMs < config.waitMinMs || config.waitMaxMs > 60000 ||
        config.feedbackMs < 100 || config.feedbackMs > 10000) {
        throw std::runtime_error("Invalid values in " + path + ". Check the ranges in config/game.yml.");
    }
    return config;
}

Game::Game(GameConfig config) : config_(std::move(config)) {}

void Game::mouseCallback(int, int x, int y, int, void* userdata) {
    static_cast<Game*>(userdata)->onMouse(x, y);
}

void Game::onMouse(int x, int y) {
    const auto now = Clock::now();
    // 回调可能在下一次画面更新前到达，先检查内部截止时刻。
    if (phase_ == Phase::Active && activeSubmitted_ && now >= deadline_) {
        finishTrial(false, now);
    }
    const cv::Point center(config_.width / 2, config_.height / 2 + 30);
    const double dx = static_cast<double>(x) - center.x;
    const double dy = static_cast<double>(y) - center.y;
    const bool inside = dx * dx + dy * dy <= static_cast<double>(config_.hitRadius) * config_.hitRadius;
    const bool entered = mouseKnown_ && !mouseInside_ && inside;
    mouse_ = {x, y};
    mouseKnown_ = true;
    mouseInside_ = inside;

    if (!entered) return;
    if (phase_ == Phase::Active && activeSubmitted_) {
        finishTrial(true, now);
    } else if (phase_ == Phase::Waiting || (phase_ == Phase::Active && !activeSubmitted_)) {
        phase_ = Phase::AwaitExit;
        activeSubmitted_ = false;
        lastResult_ = "Too early - move outside and wait for the mole";
    }
}

void Game::scheduleTrial(Clock::time_point now) {
    const int delay = std::uniform_int_distribution<int>(config_.waitMinMs, config_.waitMaxMs)(random_);
    phase_ = Phase::Waiting;
    deadline_ = now + std::chrono::milliseconds(delay);
}

void Game::update(Clock::time_point now) {
    if (phase_ == Phase::AwaitExit) {
        if (mouseKnown_ && !mouseInside_) scheduleTrial(now);
    } else if (phase_ == Phase::Waiting && now >= deadline_) {
        if (mouseInside_) {
            phase_ = Phase::AwaitExit;
        } else {
            phase_ = Phase::Active;
            activeDurationMs_ = config_.displayMs;
            activeSubmitted_ = false;
        }
    } else if (phase_ == Phase::Active && activeSubmitted_ && now >= deadline_) {
        finishTrial(false, now);
    } else if (phase_ == Phase::Feedback && now >= deadline_) {
        phase_ = Phase::AwaitExit;
        if (mouseKnown_ && !mouseInside_) scheduleTrial(now);
    }
}

void Game::finishTrial(bool hit, Clock::time_point now) {
    if (phase_ != Phase::Active || !activeSubmitted_) return;
    if (hit) {
        ++hits_;
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - submittedAt_).count();
        lastResult_ = "HIT - mouse entry after " + std::to_string(elapsed) + " ms (internal time)";
    } else {
        lastResult_ = "MISS - internal window ended";
    }
    phase_ = Phase::Feedback;
    deadline_ = now + std::chrono::milliseconds(config_.feedbackMs);
}

void Game::reset() {
    trials_ = 0;
    hits_ = 0;
    phase_ = Phase::AwaitExit;
    activeSubmitted_ = false;
    lastResult_ = "Reset - no result yet";
}

cv::Mat Game::render() const {
    cv::Mat frame(config_.height, config_.width, CV_8UC3, cv::Scalar(29, 25, 24));
    const cv::Scalar white(235, 235, 235);
    const cv::Scalar muted(160, 160, 160);
    const cv::Point center(config_.width / 2, config_.height / 2 + 30);
    const int radius = config_.hitRadius;
    cv::putText(frame, "MoleGame | Mouse preview", {28, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.85, white, 2, cv::LINE_AA);
    cv::putText(frame, "Internal timing only - camera experiment not connected", {28, 73},
        cv::FONT_HERSHEY_SIMPLEX, 0.5, muted, 1, cv::LINE_AA);
    cv::putText(frame, "Display: " + std::to_string(config_.displayMs) + " ms   Hits: " + std::to_string(hits_) +
        "   Trials: " + std::to_string(trials_), {28, 108}, cv::FONT_HERSHEY_SIMPLEX, 0.65, white, 1, cv::LINE_AA);
    cv::putText(frame, lastResult_, {28, 142}, cv::FONT_HERSHEY_SIMPLEX, 0.5, muted, 1, cv::LINE_AA);

    cv::circle(frame, center, radius + 25, cv::Scalar(66, 60, 57), -1, cv::LINE_AA);
    cv::circle(frame, center, radius + 13, cv::Scalar(10, 10, 10), -1, cv::LINE_AA);
    if (phase_ == Phase::Active) {
        const cv::Scalar mole(0, 200, 255);
        cv::circle(frame, center + cv::Point(-radius / 2, -radius / 2), radius / 3, mole, -1, cv::LINE_AA);
        cv::circle(frame, center + cv::Point(radius / 2, -radius / 2), radius / 3, mole, -1, cv::LINE_AA);
        cv::circle(frame, center, radius * 4 / 5, mole, -1, cv::LINE_AA);
        cv::circle(frame, center + cv::Point(-radius / 4, -radius / 5), radius / 12, cv::Scalar(15, 15, 15), -1, cv::LINE_AA);
        cv::circle(frame, center + cv::Point(radius / 4, -radius / 5), radius / 12, cv::Scalar(15, 15, 15), -1, cv::LINE_AA);
        cv::ellipse(frame, center + cv::Point(0, radius / 4), {radius / 4, radius / 6}, 0, 0, 360,
            cv::Scalar(200, 235, 255), -1, cv::LINE_AA);
    }
    cv::circle(frame, center, radius, cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
    const std::string status = phase_ == Phase::Active ? "MOVE INTO THE CIRCLE" :
        phase_ == Phase::Waiting ? "Wait for the mole" : "Move outside the circle";
    const int textWidth = cv::getTextSize(status, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, nullptr).width;
    cv::putText(frame, status, {center.x - textWidth / 2, center.y + radius + 65},
        cv::FONT_HERSHEY_SIMPLEX, 0.7, white, 1, cv::LINE_AA);
    if (mouseKnown_) {
        cv::circle(frame, mouse_, 7, cv::Scalar(80, 80, 255), -1, cv::LINE_AA);
        cv::circle(frame, mouse_, 9, white, 1, cv::LINE_AA);
    }
    cv::putText(frame, "Move outside -> wait -> enter the circle. No click needed.", {28, config_.height - 58},
        cv::FONT_HERSHEY_SIMPLEX, 0.5, muted, 1, cv::LINE_AA);
    cv::putText(frame, "Esc: exit   Space: reset   F: fullscreen   Up/Down: display +/- 50 ms", {28, config_.height - 28},
        cv::FONT_HERSHEY_SIMPLEX, 0.5, muted, 1, cv::LINE_AA);
    return frame;
}

void Game::run() {
    cv::namedWindow(windowName_, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(windowName_, config_.width, config_.height);
    cv::setMouseCallback(windowName_, &Game::mouseCallback, this);
    for (;;) {
        update(Clock::now());
        if (phase_ == Phase::Active && !activeSubmitted_) ++trials_;
        const cv::Mat frame = render();
        if (phase_ == Phase::Active && !activeSubmitted_) {
            // 记录画面提交时间，不把它当成投影面实际可见时间。
            submittedAt_ = Clock::now();
            deadline_ = submittedAt_ + std::chrono::milliseconds(activeDurationMs_);
            activeSubmitted_ = true;
        }
        cv::imshow(windowName_, frame);
        const int key = cv::waitKeyEx(1);
        if (key == 27 || cv::getWindowProperty(windowName_, cv::WND_PROP_VISIBLE) < 1) break;
        if (key == ' ') reset();
        else if (key == 'f' || key == 'F') {
            fullscreen_ = !fullscreen_;
            cv::setWindowProperty(windowName_, cv::WND_PROP_FULLSCREEN,
                fullscreen_ ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
        } else if (key == 2490368 || key == 65362) {
            config_.displayMs = std::min(10000, config_.displayMs + 50);
        } else if (key == 2621440 || key == 65364) {
            config_.displayMs = std::max(50, config_.displayMs - 50);
        }
    }
    cv::destroyAllWindows();
}
