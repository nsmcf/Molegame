#include "Game.h"
#include "CameraInput.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
// 不同观看方向下，提示文字按可用宽度缩放，避免裁掉右端。
void putFittedText(cv::Mat& frame, const std::string& text, cv::Point origin, int font,
                   double scale, cv::Scalar color, int thickness = 1, int lineType = cv::LINE_8) {
    const int width = cv::getTextSize(text, font, scale, thickness, nullptr).width;
    const int available = std::max(1, frame.cols - origin.x - 20);
    if (width > available) scale *= static_cast<double>(available) / width;
    cv::putText(frame, text, origin, font, scale, color, thickness, lineType);
}
}

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
    read("output_rotation_ccw", config.outputRotationCcw);
    read("hit_radius", config.hitRadius);
    if (!file["target_style"].empty()) file["target_style"] >> config.targetStyle;
    read("target_brightness", config.targetBrightness);
    read("state_markers", config.stateMarkers);
    read("display_ms", config.displayMs);
    read("wait_min_ms", config.waitMinMs);
    read("wait_max_ms", config.waitMaxMs);
    read("feedback_ms", config.feedbackMs);
    read("input_timeout_ms", config.inputTimeoutMs);
    read("observation_timeout_ms", config.observationTimeoutMs);
    if (config.width < 640 || config.width > 3840 || config.height < 480 || config.height > 2160 ||
        (config.outputRotationCcw != 0 && config.outputRotationCcw != 90 && config.outputRotationCcw != 180) ||
        (config.targetStyle != "square" && config.targetStyle != "mole") ||
        config.targetBrightness < 10 || config.targetBrightness > 100 ||
        (config.stateMarkers != 0 && config.stateMarkers != 1) ||
        config.hitRadius < 20 || config.hitRadius > std::min(config.width, config.height) / 4 ||
        config.displayMs < 50 || config.displayMs > 10000 ||
        config.waitMinMs < 100 || config.waitMaxMs < config.waitMinMs || config.waitMaxMs > 60000 ||
        config.feedbackMs < 100 || config.feedbackMs > 10000 ||
        config.inputTimeoutMs < 50 || config.inputTimeoutMs > 2000 ||
        config.observationTimeoutMs < 50 || config.observationTimeoutMs > 5000) {
        throw std::runtime_error("Invalid values in " + path + ". Check the ranges in config/game.yml.");
    }
    return config;
}

Game::Game(GameConfig config, GameMode mode)
    : config_(std::move(config)), useCamera_(mode != GameMode::Mouse),
      observationProbe_(mode == GameMode::ObservationProbe), cameraJudgment_(mode == GameMode::CameraObservation),
      windowName_(mode == GameMode::ObservationProbe ? "MoleGame - Observation probe" :
                  mode == GameMode::CameraObservation ? "MoleGame - Camera observation" :
                  mode == GameMode::InternalTiming ? "MoleGame - Internal timing" : "MoleGame - Mouse preview"),
      calibrating_(mode != GameMode::Mouse) {}

void Game::mouseCallback(int, int x, int y, int, void* userdata) {
    static_cast<Game*>(userdata)->onMouse(x, y);
}

void Game::onMouse(int x, int y) {
    if (useCamera_) return;
    const auto now = Clock::now();
    cv::Point2f outputPoint(static_cast<float>(x), static_cast<float>(y));
    if (!displayedContent_.empty()) {
        if (!displayedContent_.contains(cv::Point(x, y))) {
            onInput(std::nullopt, now, now);
            return;
        }
        outputPoint.x = static_cast<float>(x - displayedContent_.x) * outputSize().width / displayedContent_.width;
        outputPoint.y = static_cast<float>(y - displayedContent_.y) * outputSize().height / displayedContent_.height;
    }
    // 将显示坐标反变换回游戏坐标，保持鼠标和相机使用同一命中区域。
    cv::Point2f point = outputPoint;
    if (config_.outputRotationCcw == 90)
        point = {gameSize().width - 1.f - outputPoint.y, outputPoint.x};
    else if (config_.outputRotationCcw == 180)
        point = {gameSize().width - 1.f - outputPoint.x, gameSize().height - 1.f - outputPoint.y};
    onInput(point, now, now);
}

cv::Size Game::outputSize() const {
    return {config_.width, config_.height};
}

cv::Size Game::gameSize() const {
    return config_.outputRotationCcw == 90 ? cv::Size(config_.height, config_.width) : outputSize();
}

cv::Mat Game::rotateForOutput(const cv::Mat& frame) const {
    if (config_.outputRotationCcw == 0) return frame;
    cv::Mat rotated;
    cv::rotate(frame, rotated, config_.outputRotationCcw == 180 ? cv::ROTATE_180 : cv::ROTATE_90_COUNTERCLOCKWISE);
    return rotated;
}

cv::Mat Game::fitForWindow(const cv::Mat& frame, cv::Size viewport) {
    if (viewport.width <= 0 || viewport.height <= 0) viewport = frame.size();
    const double scale = std::min(static_cast<double>(viewport.width) / frame.cols,
                                 static_cast<double>(viewport.height) / frame.rows);
    const int width = std::clamp(cvRound(frame.cols * scale), 1, viewport.width);
    const int height = std::clamp(cvRound(frame.rows * scale), 1, viewport.height);
    displayedContent_ = {(viewport.width - width) / 2, (viewport.height - height) / 2, width, height};
    cv::Mat displayed(viewport, frame.type(), cv::Scalar::all(0));
    cv::resize(frame, displayed(displayedContent_), displayedContent_.size(), 0, 0,
        scale < 1 ? cv::INTER_AREA : cv::INTER_LINEAR);
    return displayed;
}

void Game::toggleOutputRotation() {
    config_.outputRotationCcw = config_.outputRotationCcw == 180 ? 0 : 180;
    if (useCamera_) beginCalibration();
    else {
        reset();
        inputKnown_ = false;
        hasSample_ = false;
    }
    logEvent("OUTPUT_ROTATION_CHANGED", Clock::now(), std::to_string(config_.outputRotationCcw));
}

void Game::adjustTargetBrightness(int delta) {
    config_.targetBrightness = std::clamp(config_.targetBrightness + delta, 10, 100);
    logEvent("TARGET_BRIGHTNESS_CHANGED", Clock::now(), std::to_string(config_.targetBrightness));
}

bool Game::inputFresh(Clock::time_point now) const {
    return inputKnown_ && (!useCamera_ || (hasSample_ &&
        now - lastSampleAt_ <= std::chrono::milliseconds(config_.inputTimeoutMs)));
}

void Game::markUncertain(const std::string& reason) {
    trialUncertain_ = true;
    if (uncertaintyReason_.empty()) uncertaintyReason_ = reason;
}

void Game::onInput(std::optional<cv::Point2f> point, Clock::time_point sampledAt,
                   Clock::time_point receivedAt, std::uint64_t frame, MoleVisibility mole, const std::string& evidenceReason) {
    if (sampledAt < ignoreBefore_ || (hasSample_ && sampledAt <= lastSampleAt_)) return;
    currentFrame_ = frame;
    currentSampleAt_ = sampledAt;
    bool trialOpen = activeSubmitted_ && (phase_ == Phase::Active || phase_ == Phase::Resolving);
    const auto timeout = std::chrono::milliseconds(config_.inputTimeoutMs);
    const bool gap = useCamera_ && hasSample_ && sampledAt - lastSampleAt_ > timeout;
    const bool stale = useCamera_ && (receivedAt - sampledAt > timeout || sampledAt > receivedAt);
    if (gap || stale) {
        inputKnown_ = false;
        if (trialOpen) markUncertain(stale ? "sample_too_late" : "frame_gap");
    }
    hasSample_ = true;
    lastSampleAt_ = sampledAt;
    if (stale) point.reset();
    if (point && (!std::isfinite(point->x) || !std::isfinite(point->y))) point.reset();
    currentMole_ = stale ? MoleVisibility::Unknown : mole;
    if (trialOpen && sampledAt >= submittedAt_ &&
        (cameraJudgment_ || sampledAt < deadline_) && !point)
        markUncertain(evidenceReason.empty() ? "hand_input_missing" : evidenceReason);

    // 先处理队列中的样本，再按当前时间更新显示。结果晚到不改变样本所属的计时窗口。
    if (cameraJudgment_ && trialOpen && sampledAt >= submittedAt_) {
        // 内部截止时间只控制显示；命中取决于动作帧中的真实颜色证据。
        if (sampledAt >= observationDeadline_) {
            finishTrial(Outcome::Unjudged, receivedAt, {}, "mole_observation_timeout");
            trialOpen = false;
        } else if (currentMole_ == MoleVisibility::Unknown) {
            markUncertain(evidenceReason.empty() ? "mole_unknown" : evidenceReason);
        } else if (currentMole_ == MoleVisibility::Visible) {
            observedOn_ = true;
        } else if (observedOn_) {
            finishTrial(trialUncertain_ ? Outcome::Unjudged : Outcome::Miss, receivedAt);
            trialOpen = false;
        }
    } else if (!cameraJudgment_ && trialOpen && sampledAt >= deadline_) {
        finishTrial(trialUncertain_ ? Outcome::Unjudged : Outcome::Miss, receivedAt);
        trialOpen = false;
    }
    if (!point) {
        inputKnown_ = false;
        if (trialOpen && sampledAt >= submittedAt_ && sampledAt < deadline_)
            markUncertain(evidenceReason.empty() ? "hand_input_missing" : evidenceReason);
        if (phase_ == Phase::Waiting) phase_ = Phase::AwaitExit;
        logEvent("INPUT_UNKNOWN", receivedAt);
        return;
    }
    const cv::Point2f center(gameSize().width / 2.f, gameSize().height / 2.f + 30);
    const auto delta = *point - center;
    const bool inside = delta.dot(delta) <= static_cast<float>(config_.hitRadius * config_.hitRadius);
    const bool entered = inputKnown_ && !inputInside_ && inside;
    pointer_ = *point;
    inputKnown_ = true;
    inputInside_ = inside;
    logEvent("INPUT", receivedAt);
    if (!entered) return;
    if (cameraJudgment_ && trialOpen && sampledAt >= submittedAt_) {
        const auto result = currentMole_ == MoleVisibility::Visible ? Outcome::Hit :
            currentMole_ == MoleVisibility::Absent ? Outcome::Miss : Outcome::Unjudged;
        finishTrial(result, receivedAt, sampledAt, result == Outcome::Unjudged ?
            (evidenceReason.empty() ? "mole_unknown_on_entry" : evidenceReason) : "");
        return;
    }
    if (activeSubmitted_ && (phase_ == Phase::Active || phase_ == Phase::Resolving) &&
        sampledAt >= submittedAt_ && sampledAt < deadline_) {
        finishTrial(Outcome::Hit, receivedAt, sampledAt);
    } else if (phase_ == Phase::Waiting || (phase_ == Phase::Active && !activeSubmitted_)) {
        phase_ = Phase::AwaitExit;
        activeSubmitted_ = false;
        lastResult_ = "Too early - move outside and wait for the mole";
        logEvent("EARLY_ENTRY", receivedAt);
    } else if ((phase_ == Phase::Active || phase_ == Phase::Resolving) && activeSubmitted_ && sampledAt < submittedAt_) {
        // 在显示前进入，但推理结果在显示后才到达：取消这个未正确准备的试次。
        logEvent("EARLY_ENTRY", receivedAt);
        finishTrial(Outcome::Unjudged, receivedAt);
        lastResult_ = "Too early - move outside and wait for the mole";
    }
}

void Game::scheduleTrial(Clock::time_point now) {
    const int delay = std::uniform_int_distribution<int>(config_.waitMinMs, config_.waitMaxMs)(random_);
    phase_ = Phase::Waiting;
    deadline_ = now + std::chrono::milliseconds(delay);
    logEvent("WAIT", now);
}

void Game::update(Clock::time_point now) {
    if (useCamera_ && !inputFresh(now)) {
        if (phase_ == Phase::Waiting) phase_ = Phase::AwaitExit;
        if ((phase_ == Phase::Active || phase_ == Phase::Resolving) && activeSubmitted_)
            markUncertain("hand_input_timeout");
    }
    if (phase_ == Phase::AwaitExit) {
        if (inputFresh(now) && !inputInside_ &&
            (!cameraJudgment_ || currentMole_ == MoleVisibility::Absent)) scheduleTrial(now);
    } else if (phase_ == Phase::Waiting && now >= deadline_) {
        if (!inputFresh(now) || inputInside_ || (cameraJudgment_ && currentMole_ != MoleVisibility::Absent)) {
            phase_ = Phase::AwaitExit;
        } else {
            phase_ = Phase::Active;
            activeDurationMs_ = config_.displayMs;
            activeSubmitted_ = false;
        }
    } else if (phase_ == Phase::Active && activeSubmitted_ && now >= deadline_) {
        if (useCamera_) {
            phase_ = Phase::Resolving;
            logEvent("WINDOW_END", now);
        } else finishTrial(Outcome::Miss, now);
    } else if (phase_ == Phase::Resolving &&
               now >= (cameraJudgment_ ? observationDeadline_ : deadline_) + std::chrono::milliseconds(config_.inputTimeoutMs)) {
        finishTrial(Outcome::Unjudged, now, {}, cameraJudgment_ ? "mole_observation_timeout" : "inflight_result_timeout");
    } else if (phase_ == Phase::Feedback && now >= deadline_) {
        phase_ = Phase::AwaitExit;
        if (inputFresh(now) && !inputInside_ &&
            (!cameraJudgment_ || currentMole_ == MoleVisibility::Absent)) scheduleTrial(now);
    }
}

void Game::submitFrame(Clock::time_point now) {
    if (phase_ != Phase::Active || activeSubmitted_) return;
    ++trials_;
    ++trialId_;
    submittedAt_ = now;
    deadline_ = now + std::chrono::milliseconds(activeDurationMs_);
    observationDeadline_ = deadline_ + std::chrono::milliseconds(config_.observationTimeoutMs);
    observedOn_ = false;
    activeSubmitted_ = true;
    activeTargetBrightness_ = config_.targetBrightness;
    trialUncertain_ = !inputKnown_;
    uncertaintyReason_ = trialUncertain_ ? "hand_input_missing_at_start" : "";
    logEvent("WINDOW_START", now);
    logEvent("TRIAL_BRIGHTNESS", now, std::to_string(activeTargetBrightness_));
}

void Game::finishTrial(Outcome result, Clock::time_point now, Clock::time_point entryAt, const std::string& reason) {
    if ((phase_ != Phase::Active && phase_ != Phase::Resolving) || !activeSubmitted_) return;
    if (result == Outcome::Hit) {
        ++hits_;
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(entryAt - submittedAt_).count();
        lastResult_ = "HIT - entry after " + std::to_string(elapsed) +
            (cameraJudgment_ ? " ms from submission (same-frame mole visible)" : " ms (internal time)");
    } else if (result == Outcome::Unjudged) {
        ++unjudged_;
        const std::string cause = !reason.empty() ? reason : !uncertaintyReason_.empty() ? uncertaintyReason_ : "missing_evidence";
        lastResult_ = "UNJUDGED - " + cause;
        logEvent("UNJUDGED_REASON", now, cause);
    } else {
        lastResult_ = cameraJudgment_ ? "MISS - no entry while observed mole was visible" : "MISS - internal window ended";
    }
    logEvent("RESULT", now, result == Outcome::Hit ? "HIT" : result == Outcome::Miss ? "MISS" : "UNJUDGED");
    if (events_.is_open()) events_.flush();
    phase_ = Phase::Feedback;
    deadline_ = now + std::chrono::milliseconds(config_.feedbackMs);
}

void Game::reset(Clock::time_point now) {
    if (activeSubmitted_ && (phase_ == Phase::Active || phase_ == Phase::Resolving))
        finishTrial(Outcome::Unjudged, now, {}, "manual_reset");
    logEvent("RESET", now);
    trials_ = 0;
    hits_ = 0;
    unjudged_ = 0;
    phase_ = Phase::AwaitExit;
    activeSubmitted_ = false;
    observedOn_ = false;
    ignoreBefore_ = now;
    lastResult_ = "Reset - no result yet";
}

cv::Mat Game::render() const {
    const auto size = gameSize();
    cv::Mat frame(size, CV_8UC3, cv::Scalar(29, 25, 24));
    const cv::Scalar white(235, 235, 235);
    const cv::Scalar muted(160, 160, 160);
    const cv::Point center(size.width / 2, size.height / 2 + 30);
    const int radius = config_.hitRadius;
    const int brightness = !observationProbe_ && activeSubmitted_ &&
        (phase_ == Phase::Active || phase_ == Phase::Resolving) ? activeTargetBrightness_ : config_.targetBrightness;
    const double intensity = brightness / 100.0;
    putFittedText(frame, observationProbe_ ? "MoleGame | Same-frame observation probe" :
        cameraJudgment_ ? "MoleGame | YOLO + Camera observation" :
        useCamera_ ? "MoleGame | YOLO + Internal timing" : "MoleGame | Mouse preview",
        {28, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.85, white, 2, cv::LINE_AA);
    putFittedText(frame, observationProbe_ ? "No scoring | V: toggle mole | A: auto cycle | S: save frame" :
        cameraJudgment_ ? "Single hand | H: recalibrate | same-frame mole visibility" :
        useCamera_ ? "Single hand | H: recalibrate | internal display window" :
        "Mouse preview - internal timing only", {28, 73},
        cv::FONT_HERSHEY_SIMPLEX, 0.5, muted, 1, cv::LINE_AA);
    putFittedText(frame, "Display: " + std::to_string(config_.displayMs) + " ms   Brightness: " + std::to_string(brightness) + "%" +
        (brightness != config_.targetBrightness ? " (next " + std::to_string(config_.targetBrightness) + "%)" : "") +
        (observationProbe_ ? "   Observation only - no trial scores" : "   Hits: " + std::to_string(hits_) +
        "   Trials: " + std::to_string(trials_) + "   Unjudged: " + std::to_string(unjudged_)),
        {28, 108}, cv::FONT_HERSHEY_SIMPLEX, 0.65, white, 1, cv::LINE_AA);
    putFittedText(frame, lastResult_, {28, 142}, cv::FONT_HERSHEY_SIMPLEX, 0.5, muted, 1, cv::LINE_AA);

    cv::circle(frame, center, radius + 25, cv::Scalar(66, 60, 57), -1, cv::LINE_AA);
    cv::circle(frame, center, radius + 13, cv::Scalar(10, 10, 10), -1, cv::LINE_AA);
    if (phase_ == Phase::Active) {
        const cv::Scalar mole = cv::Scalar(0, 200, 255) * intensity;
        if (config_.stateMarkers)
            for (const auto& rect : moleStateMarkers(size, radius)) cv::rectangle(frame, rect, mole, -1);
        if (config_.targetStyle == "square") {
            // 先用无五官图案排查手模型把地鼠图案误检为手的问题。
            const int half = radius * 2 / 3;
            cv::rectangle(frame, center - cv::Point(half, half), center + cv::Point(half, half),
                mole, -1, cv::LINE_8);
        } else {
            cv::circle(frame, center + cv::Point(-radius / 2, -radius / 2), radius / 3, mole, -1, cv::LINE_AA);
            cv::circle(frame, center + cv::Point(radius / 2, -radius / 2), radius / 3, mole, -1, cv::LINE_AA);
            cv::circle(frame, center, radius * 4 / 5, mole, -1, cv::LINE_AA);
            cv::circle(frame, center + cv::Point(-radius / 4, -radius / 5), radius / 12, cv::Scalar(15, 15, 15) * intensity, -1, cv::LINE_AA);
            cv::circle(frame, center + cv::Point(radius / 4, -radius / 5), radius / 12, cv::Scalar(15, 15, 15) * intensity, -1, cv::LINE_AA);
            cv::ellipse(frame, center + cv::Point(0, radius / 4), {radius / 4, radius / 6}, 0, 0, 360,
                cv::Scalar(200, 235, 255) * intensity, -1, cv::LINE_AA);
        }
    }
    cv::circle(frame, center, radius, cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
    const std::string status = observationProbe_ ? "Check observed state in the CAMERA window" :
        phase_ == Phase::Active ? "MOVE INTO THE CIRCLE" :
        phase_ == Phase::Resolving ? (cameraJudgment_ ? "Waiting for camera evidence" : "Waiting for result") :
        !inputFresh(Clock::now()) ? (useCamera_ ? "Show one hand outside the circle" : "Move mouse outside the circle") :
        cameraJudgment_ && phase_ == Phase::AwaitExit && currentMole_ != MoleVisibility::Absent ? "Clear the hole for camera observation" :
        phase_ == Phase::Waiting ? "Wait for the mole" : "Move outside the circle";
    const int fullTextWidth = cv::getTextSize(status, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, nullptr).width;
    const double statusScale = 0.7 * std::min(1.0, static_cast<double>(size.width - 40) / std::max(1, fullTextWidth));
    const int textWidth = cv::getTextSize(status, cv::FONT_HERSHEY_SIMPLEX, statusScale, 1, nullptr).width;
    cv::putText(frame, status, {center.x - textWidth / 2, center.y + radius + 65},
        cv::FONT_HERSHEY_SIMPLEX, statusScale, white, 1, cv::LINE_AA);
    if (inputFresh(Clock::now())) {
        cv::circle(frame, pointer_, 7, cv::Scalar(80, 80, 255), -1, cv::LINE_AA);
        cv::circle(frame, pointer_, 9, white, 1, cv::LINE_AA);
    }
    putFittedText(frame, observationProbe_ ? "Test: absent / visible / partial cover / full cover" :
        "Move outside -> wait -> enter the circle. No click needed.", {28, size.height - 58},
        cv::FONT_HERSHEY_SIMPLEX, 0.5, muted, 1, cv::LINE_AA);
    putFittedText(frame, "Esc: exit   Space: reset   F: fullscreen   R: rotate   Up/Down: time   Left/Right: brightness", {28, size.height - 28},
        cv::FONT_HERSHEY_SIMPLEX, 0.5, muted, 1, cv::LINE_AA);
    return frame;
}

void Game::logEvent(const std::string& event, Clock::time_point now, const std::string& detail, int count) {
    if (!events_.is_open()) return;
    const auto ms = [&](Clock::time_point t) {
        return std::chrono::duration<double, std::milli>(t - sessionStart_).count();
    };
    events_ << ms(now) << ',' << event << ',' << trialId_ << ',' << currentFrame_ << ','
            << (currentFrame_ ? ms(currentSampleAt_) : -1) << ',' << count << ',';
    if (inputKnown_) events_ << pointer_.x << ',' << pointer_.y;
    else events_ << ',';
    events_ << ',' << (observationProbe_ ? config_.displayMs : activeDurationMs_) << ',' << (trials_ ? ms(submittedAt_) : -1) << ','
            << (trials_ ? ms(submittedAt_ + std::chrono::milliseconds(activeDurationMs_)) : -1)
            << ',' << detail << '\n';
    if (!events_) throw std::runtime_error("Cannot write session events");
}

void Game::openLog() {
    namespace fs = std::filesystem;
    const auto id = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    sessionDirectory_ = std::string("outputs/") + (observationProbe_ ? "observation-" :
        cameraJudgment_ ? "camera-" : "internal-") + std::to_string(id);
    fs::create_directories(sessionDirectory_);
    events_.open(sessionDirectory_ + "/events.csv");
    if (!events_) throw std::runtime_error("Cannot create event log");
    events_ << std::fixed << std::setprecision(3)
            << "received_ms,event,trial,frame,sample_ms,hand_count,x,y,display_ms,window_start_ms,window_end_ms,detail\n";
    for (const auto& name : {"game.yml", "camera.yml", "yolo.yml"})
        fs::copy_file(fs::path("config") / name, fs::path(sessionDirectory_) / name);
    if (fs::exists("config/mole.yml"))
        fs::copy_file("config/mole.yml", fs::path(sessionDirectory_) / "mole.yml");
    if (observationProbe_) {
        observations_.open(sessionDirectory_ + "/observations.csv");
        if (!observations_) throw std::runtime_error("Cannot create observation log");
        observations_ << std::fixed << std::setprecision(3)
            << "frame,sample_ms,received_ms,hand_count,mole_state,color_ratio,unoccluded_ratio,"
               "yolo_ms,mole_ms,host_sample_to_receive_ms,reason\n";
    }
    cv::FileStorage metadata(sessionDirectory_ + "/session.yml", cv::FileStorage::WRITE);
    if (!metadata.isOpened()) throw std::runtime_error("Cannot save session metadata");
    metadata << "mode" << (observationProbe_ ? "observation_probe" :
        cameraJudgment_ ? "camera_observation" : "internal_timing") << "camera_serial" << cameraId_
             << "camera_size" << cameraSize_ << "game_size" << gameSize() << "output_size" << outputSize()
             << "output_rotation_ccw" << config_.outputRotationCcw
             << "target_brightness" << config_.targetBrightness
             << "game_points" << calibrationTargets(gameSize())
             << "input_timeout_ms" << config_.inputTimeoutMs
             << "observation_timeout_ms" << config_.observationTimeoutMs
             << "mole_state_source" << (config_.stateMarkers ? "same_frame_synchronous_side_markers" : "same_frame_center_color")
             << "timestamp_source" << "host_after_conversion_and_clone"
             << "random_engine" << "mt19937" << "random_seed" << std::to_string(randomSeed_);
    if (observationProbe_)
        metadata << "sampling" << "every_processed_camera_frame"
                 << "scoring" << "disabled"
                 << "mole_region" << (config_.stateMarkers ? "two_side_markers" : "disk_at_hit_center_radius_0.70_times_hit_radius");
    std::cout << "Session: " << sessionDirectory_ << '\n';
}

void Game::beginCalibration() {
    if (activeSubmitted_ && (phase_ == Phase::Active || phase_ == Phase::Resolving))
        finishTrial(Outcome::Unjudged, Clock::now(), {}, "calibration_changed");
    calibrating_ = true;
    calibrationCorners_.clear();
    cameraToGame_.release();
    inputKnown_ = false;
    phase_ = Phase::AwaitExit;
    activeSubmitted_ = false;
    calibrationMessage_ = "Click projected labels 1,2,3,4 in the camera window";
    logEvent("CALIBRATION_START", Clock::now());
}

void Game::calibrationClick(int event, int x, int y, int, void* userdata) {
    auto& game = *static_cast<Game*>(userdata);
    if (event != cv::EVENT_LBUTTONDOWN || !game.calibrating_ || game.calibrationCorners_.size() >= 4) return;
    game.calibrationCorners_.emplace_back(static_cast<float>(x), static_cast<float>(y));
    if (game.calibrationCorners_.size() == 4) {
        try {
            game.cameraToGame_ = cameraToGameTransform(game.calibrationCorners_, game.cameraSize_,
                                                       game.gameSize());
            game.calibrationMessage_ = "Check CENTER ring + mapped hand; ENTER to start, H to retry";
        } catch (const std::exception&) {
            game.calibrationMessage_ = "Invalid points. Press H and click labels 1,2,3,4 again";
        }
    }
}

void Game::loadCalibration(const std::string& path) {
    if (!std::filesystem::exists(path)) return;
    try {
        cv::FileStorage file(path, cv::FileStorage::READ);
        cv::Size cameraSize, gameSize;
        std::string serial;
        file["camera_size"] >> cameraSize;
        file["game_size"] >> gameSize;
        file["camera_serial"] >> serial;
        int rotation = 0;
        if (!file["output_rotation_ccw"].empty()) file["output_rotation_ccw"] >> rotation;
        if (cameraSize != cameraSize_ || gameSize != this->gameSize() ||
            serial != cameraId_ || rotation != config_.outputRotationCcw)
            throw std::runtime_error("Calibration device, size or output rotation changed");
        const auto targets = calibrationTargets(this->gameSize());
        std::vector<cv::Point2f> savedTargets;
        if (!file["game_points"].empty()) file["game_points"] >> savedTargets;
        if (savedTargets.size() != targets.size()) throw std::runtime_error("Calibration target layout changed");
        for (std::size_t i = 0; i < targets.size(); ++i)
            if (cv::norm(savedTargets[i] - targets[i]) > 0.01) throw std::runtime_error("Calibration target layout changed");
        file["camera_points"] >> calibrationCorners_;
        cameraToGame_ = cameraToGameTransform(calibrationCorners_, cameraSize_, gameSize);
        calibrationMessage_ = "Saved mapping: check CENTER ring + hand; ENTER to start, H to redo";
    } catch (const std::exception& error) {
        std::cerr << error.what() << "; recalibration required.\n";
        beginCalibration();
    }
}

void Game::confirmCalibration() {
    if (cameraToGame_.empty()) return;
    namespace fs = std::filesystem;
    const fs::path path = "config/camera-calibration.yml";
    if (fs::exists(path)) fs::copy_file(path, path.string() + ".bak", fs::copy_options::overwrite_existing);
    cv::FileStorage file(path.string(), cv::FileStorage::WRITE);
    if (!file.isOpened()) throw std::runtime_error("Cannot save camera calibration");
    file << "camera_size" << cameraSize_ << "game_size" << gameSize() << "output_size" << outputSize()
         << "output_rotation_ccw" << config_.outputRotationCcw
         << "game_points" << calibrationTargets(gameSize())
         << "camera_serial" << cameraId_ << "camera_points" << calibrationCorners_;
    file.release();
    const auto now = Clock::now();
    if (!sessionDirectory_.empty()) {
        const auto revision = std::chrono::duration_cast<std::chrono::microseconds>(now - sessionStart_).count();
        fs::copy_file(path, fs::path(sessionDirectory_) / ("calibration-" + std::to_string(revision) + ".yml"));
    }
    calibrating_ = false;
    inputKnown_ = false;
    hasSample_ = false;
    ignoreBefore_ = now;
    phase_ = Phase::AwaitExit;
    activeSubmitted_ = false;
    logEvent("CALIBRATION_CONFIRMED", now);
}

cv::Mat Game::renderCalibration() const {
    const auto size = gameSize();
    cv::Mat frame(size, CV_8UC3, cv::Scalar(29, 25, 24));
    putFittedText(frame, "Calibration | F: fullscreen | R: rotate 0/180 | H: restart | Esc: exit", {20, 35},
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(235, 235, 235), 1);
    putFittedText(frame, calibrationMessage_, {20, 70},
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(235, 235, 235), 1);
    // 与实际命中区域相同的参考圆，便于确认标定中心是否偏移。
    const cv::Point center(size.width / 2, size.height / 2 + 30);
    cv::circle(frame, center, config_.hitRadius, cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
    cv::drawMarker(frame, center, cv::Scalar(235, 235, 235), cv::MARKER_CROSS, 18, 1);
    putFittedText(frame, "CENTER - check only", {center.x - config_.hitRadius, center.y + config_.hitRadius + 25},
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(235, 235, 235), 1);
    const auto targets = calibrationTargets(size);
    for (std::size_t i = 0; i < targets.size(); ++i) {
        cv::drawMarker(frame, targets[i], cv::Scalar(0, 220, 255), cv::MARKER_CROSS, 60, 4);
        cv::putText(frame, std::to_string(i + 1), targets[i] + cv::Point2f(36, -36),
            cv::FONT_HERSHEY_SIMPLEX, 1.4, cv::Scalar(235, 235, 235), 3);
    }
    if (inputFresh(Clock::now())) cv::circle(frame, pointer_, 8, cv::Scalar(80, 80, 255), -1);
    return frame;
}

void Game::run() {
    std::unique_ptr<CameraInput> camera;
    const std::string previewName = "MoleGame camera preview";
    if (useCamera_) {
        camera = std::make_unique<CameraInput>(true);
        cameraSize_ = camera->frameSize();
        cameraId_ = camera->deviceId();
        openLog();
        loadCalibration();
        cv::namedWindow(previewName, cv::WINDOW_NORMAL);
        cv::resizeWindow(previewName, cameraSize_.width, cameraSize_.height);
        cv::setMouseCallback(previewName, &Game::calibrationClick, this);
    }
    // Win32 HighGUI 会铺满客户区；先自行等比缩放并填黑边，再交给窗口显示。
    cv::namedWindow(windowName_, cv::WINDOW_NORMAL | cv::WINDOW_FREERATIO);
    cv::resizeWindow(windowName_, outputSize().width, outputSize().height);
    if (!useCamera_) cv::setMouseCallback(windowName_, &Game::mouseCallback, this);
    std::optional<cv::Rect> previousWindowRect;
    auto nextLog = Clock::now();
    cv::Mat preview;
    cv::Mat rawPreview, savedAnnotation;
    HandObservation previewSample;
    MoleObservation mole;
    std::vector<std::vector<cv::Point>> molePolygons;
    std::uint64_t lastProbeFrame = 0;
    bool autoCycle = false;
    auto nextProbeSwitch = Clock::now();
    std::optional<bool> lastProbeSubmitted;
    cv::Mat appliedTransform;
    std::uint64_t regionRevision = 0;
    for (;;) {
        const auto windowRect = cv::getWindowImageRect(windowName_);
        if (windowRect.width > 0 && windowRect.height > 0) {
            if (camera && previousWindowRect && windowRect != *previousWindowRect) beginCalibration();
            previousWindowRect = windowRect;
        }
        if (camera) {
            // 标定更新带版本号，旧标定下已在途的结果不能用于新一轮评分。
            if (!calibrating_ && !cameraToGame_.empty()) {
                if (appliedTransform.empty() || cv::norm(appliedTransform, cameraToGame_) > 1e-9) {
                    regionRevision = camera->setMoleRegions(moleObservationRegionsInCamera(cameraToGame_,
                        gameSize(), config_.hitRadius, config_.stateMarkers != 0));
                    appliedTransform = cameraToGame_.clone();
                }
            } else if (!appliedTransform.empty()) {
                regionRevision = camera->setMoleRegions({});
                appliedTransform.release();
            }
            CameraUpdate batch;
            try { camera->poll(batch); }
            catch (const std::exception&) {
                if (activeSubmitted_ && (phase_ == Phase::Active || phase_ == Phase::Resolving))
                    finishTrial(Outcome::Unjudged, Clock::now(), {}, "camera_error");
                logEvent("CAMERA_ERROR", Clock::now());
                cv::destroyAllWindows();
                throw;
            }
            if (batch.dropped) {
                inputKnown_ = false;
                markUncertain("sample_queue_overflow");
                logEvent("QUEUE_OVERFLOW", Clock::now(), std::to_string(batch.dropped));
            }
            for (const auto& sample : batch.observations) {
                currentFrame_ = sample.sequence;
                currentSampleAt_ = sample.capturedAt;
                const auto receivedAt = Clock::now();
                std::optional<cv::Point2f> point;
                if (sample.centers.size() == 1 && !cameraToGame_.empty()) {
                    std::vector<cv::Point2f> mapped;
                    cv::perspectiveTransform(sample.centers, mapped, cameraToGame_);
                    if (std::isfinite(mapped[0].x) && std::isfinite(mapped[0].y) &&
                        std::abs(mapped[0].x) < 1e6 && std::abs(mapped[0].y) < 1e6) point = mapped[0];
                }
                const bool regionMatches = sample.regionRevision == regionRevision;
                if (!calibrating_ && !regionMatches) point.reset();
                const std::string evidenceReason = !regionMatches ? "calibration_changed" :
                    sample.centers.empty() ? "no_hand" : sample.centers.size() > 1 ? "multiple_hands" :
                    !point ? "invalid_hand_mapping" : sample.mole.reason;
                if (!calibrating_ && !observationProbe_)
                    onInput(point, sample.capturedAt, receivedAt, sample.sequence,
                        regionMatches ? sample.mole.state : MoleVisibility::Unknown, evidenceReason);
                else {
                    pointer_ = point.value_or(cv::Point2f(-1, -1));
                    inputKnown_ = point.has_value();
                    hasSample_ = true;
                    lastSampleAt_ = sample.capturedAt;
                }
                logEvent("DETECTION", receivedAt, regionMatches ? visibilityName(sample.mole.state) : "UNKNOWN",
                    static_cast<int>(sample.centers.size()));
                if (sample.mole.state == MoleVisibility::Unknown && !calibrating_)
                    logEvent("DETECTION_REASON", receivedAt, evidenceReason, static_cast<int>(sample.centers.size()));
                if (observationProbe_ && !calibrating_ && regionMatches && sample.capturedAt >= ignoreBefore_) {
                    const auto ms = [&](Clock::time_point t) {
                        return std::chrono::duration<double, std::milli>(t - sessionStart_).count();
                    };
                    observations_ << sample.sequence << ',' << ms(sample.capturedAt) << ',' << ms(receivedAt)
                        << ',' << sample.centers.size() << ',' << visibilityName(sample.mole.state)
                        << ',' << sample.mole.colorRatio << ',' << sample.mole.unoccludedRatio << ',' << sample.detectorMs
                        << ',' << sample.mole.detectorMs << ',' << ms(receivedAt) - ms(sample.capturedAt)
                        << ',' << sample.mole.reason << '\n';
                    if (!observations_) throw std::runtime_error("Cannot write observation log");
                }
                if (receivedAt >= nextLog) {
                    std::cout << "[YOLO] frame=" << sample.sequence << " hands=" << sample.centers.size()
                              << " detector_total_ms=" << sample.detectorMs
                              << " mole=" << visibilityName(sample.mole.state) << " mole_ms=" << sample.mole.detectorMs
                              << " capture_to_game_receive_ms="
                              << std::chrono::duration<double, std::milli>(receivedAt - sample.capturedAt).count() << '\n';
                    nextLog = receivedAt + std::chrono::seconds(1);
                }
            }
            if (!batch.preview.empty()) {
                preview = batch.preview;
                rawPreview = batch.rawPreview;
                previewSample = batch.previewSample;
                mole = previewSample.mole;
                molePolygons.clear();
                if (!calibrating_ && previewSample.regionRevision == regionRevision) {
                    for (const auto& region : moleObservationRegionsInCamera(cameraToGame_, gameSize(), config_.hitRadius, config_.stateMarkers != 0)) {
                        std::vector<cv::Point> polygon;
                        for (const auto& p : region) {
                            if (std::isfinite(p.x) && std::isfinite(p.y) && std::abs(p.x) < 1e6 && std::abs(p.y) < 1e6)
                                polygon.emplace_back(p);
                        }
                        if (polygon.size() == region.size()) molePolygons.push_back(std::move(polygon));
                    }
                    lastProbeFrame = previewSample.sequence;
                }
            }
            if (!preview.empty()) {
                cv::Mat annotated = preview.clone();
                for (std::size_t i = 0; i < calibrationCorners_.size(); ++i)
                    cv::putText(annotated, std::to_string(i + 1), calibrationCorners_[i],
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(80, 80, 255), 2);
                if (!cameraToGame_.empty()) {
                    std::vector<cv::Point2f> ring, cameraRing;
                    for (int i = 0; i < 64; ++i) {
                        const double angle = i * 2 * CV_PI / 64;
                        ring.emplace_back(gameSize().width / 2.f + config_.hitRadius * static_cast<float>(std::cos(angle)),
                            gameSize().height / 2.f + 30 + config_.hitRadius * static_cast<float>(std::sin(angle)));
                    }
                    cv::perspectiveTransform(ring, cameraRing, cameraToGame_.inv());
                    std::vector<cv::Point> polygon;
                    for (const auto& p : cameraRing) polygon.emplace_back(p);
                    cv::polylines(annotated, polygon, true, cv::Scalar(255, 150, 0), 2);
                }
                cv::putText(annotated, "One hand only | H: calibrate | Esc: exit", {10, 25},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 220, 255), 1);
                if (observationProbe_ || cameraJudgment_) {
                    if (!calibrating_ && !molePolygons.empty())
                        cv::polylines(annotated, molePolygons, true, cv::Scalar(255, 0, 255), 2);
                    std::ostringstream label;
                    const bool fresh = !calibrating_ && previewSample.capturedAt >= ignoreBefore_ &&
                        previewSample.regionRevision == regionRevision &&
                        Clock::now() - previewSample.capturedAt <= std::chrono::milliseconds(config_.inputTimeoutMs);
                    label << "frame=" << previewSample.sequence << " mole="
                          << (calibrating_ ? "NOT_READY" : fresh ? visibilityName(mole.state) : "UNKNOWN")
                          << " color=" << std::fixed << std::setprecision(2) << mole.colorRatio
                          << " clear=" << mole.unoccludedRatio;
                    cv::putText(annotated, label.str(), {10, 50}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                    const std::string hint = calibrating_ ? (cameraToGame_.empty() ?
                        "Click cross centers 1,2,3,4" : "Check CENTER then press ENTER") : fresh ? mole.reason : "sample_stale_or_region_changed";
                    cv::putText(annotated, hint, {10, 75},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                    savedAnnotation = annotated;
                }
                cv::putText(annotated, "Brightness setting: " + std::to_string(config_.targetBrightness) +
                    "% | Left/Right: -/+ 10%", {10, annotated.rows - 12},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(235, 235, 235), 1);
                cv::imshow(previewName, annotated);
            }
        }
        if (!calibrating_ && !observationProbe_) update(Clock::now());
        if (observationProbe_ && !calibrating_ && autoCycle && Clock::now() >= nextProbeSwitch) {
            phase_ = phase_ == Phase::Active ? Phase::Waiting : Phase::Active;
            const int duration = phase_ == Phase::Active ? config_.displayMs :
                std::uniform_int_distribution<int>(config_.waitMinMs, config_.waitMaxMs)(random_);
            nextProbeSwitch = Clock::now() + std::chrono::milliseconds(duration);
        }
        const cv::Mat frame = fitForWindow(rotateForOutput(calibrating_ ? renderCalibration() : render()), windowRect.size());
        // 这是内部窗口起点，不声称是投影面实际可见时刻。
        if (!calibrating_ && !observationProbe_) submitFrame(Clock::now());
        if (observationProbe_) {
            const bool visible = !calibrating_ && phase_ == Phase::Active;
            if (!lastProbeSubmitted || visible != *lastProbeSubmitted) {
                logEvent(visible ? "STIMULUS_SUBMIT_ON" : "STIMULUS_SUBMIT_OFF", Clock::now());
                lastProbeSubmitted = visible;
            }
        }
        cv::imshow(windowName_, frame);
        const int key = cv::waitKeyEx(1);
        if (key == 27 || cv::getWindowProperty(windowName_, cv::WND_PROP_VISIBLE) < 1 ||
            (camera && cv::getWindowProperty(previewName, cv::WND_PROP_VISIBLE) < 1)) break;
        if (camera && (key == 'h' || key == 'H')) { beginCalibration(); continue; }
        if (camera && calibrating_ && (key == 13 || key == 10)) { confirmCalibration(); continue; }
        if (key == 'r' || key == 'R') {
            toggleOutputRotation();
            autoCycle = false;
            if (!fullscreen_) cv::resizeWindow(windowName_, outputSize().width, outputSize().height);
            previousWindowRect.reset();
            continue;
        }
        if (observationProbe_ && !calibrating_) {
            if (key == 'v' || key == 'V') {
                autoCycle = false;
                phase_ = phase_ == Phase::Active ? Phase::Waiting : Phase::Active;
                logEvent("MANUAL_TOGGLE", Clock::now(), phase_ == Phase::Active ? "ON" : "OFF");
            } else if (key == 'a' || key == 'A') {
                autoCycle = !autoCycle;
                nextProbeSwitch = Clock::now();
                logEvent("AUTO_CYCLE", Clock::now(), autoCycle ? "ON" : "OFF");
            } else if ((key == 's' || key == 'S') && !rawPreview.empty() &&
                       lastProbeFrame == previewSample.sequence && previewSample.capturedAt >= ignoreBefore_) {
                const auto prefix = sessionDirectory_ + "/frame-" + std::to_string(lastProbeFrame);
                if (!cv::imwrite(prefix + "-raw.png", rawPreview) || !cv::imwrite(prefix + "-preview.png", savedAnnotation))
                    throw std::runtime_error("Cannot save observation snapshot");
                observations_.flush();
                logEvent("SNAPSHOT", Clock::now(), std::to_string(lastProbeFrame));
                events_.flush();
                std::cout << "Saved " << prefix << " (raw + annotated)\n";
            }
        }
        if (key == ' ') reset();
        else if (key == 2424832 || key == 65361 || key == '[') adjustTargetBrightness(-10);
        else if (key == 2555904 || key == 65363 || key == ']') adjustTargetBrightness(10);
        else if (key == 'f' || key == 'F') {
            fullscreen_ = !fullscreen_;
            cv::setWindowProperty(windowName_, cv::WND_PROP_FULLSCREEN,
                fullscreen_ ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
            if (camera) beginCalibration();
        } else if (key == 2490368 || key == 65362) {
            config_.displayMs = std::min(10000, config_.displayMs + 50);
            if (observationProbe_) logEvent("DISPLAY_MS_CHANGED", Clock::now(), std::to_string(config_.displayMs));
        } else if (key == 2621440 || key == 65364) {
            config_.displayMs = std::max(50, config_.displayMs - 50);
            if (observationProbe_) logEvent("DISPLAY_MS_CHANGED", Clock::now(), std::to_string(config_.displayMs));
        }
    }
    if (useCamera_ && activeSubmitted_ && (phase_ == Phase::Active || phase_ == Phase::Resolving))
        finishTrial(Outcome::Unjudged, Clock::now(), {}, "session_end");
    logEvent("SESSION_END", Clock::now());
    if (observations_.is_open()) observations_.flush();
    cv::destroyAllWindows();
}
