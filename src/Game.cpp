#include "Game.h"
#include "CameraInput.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
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
    read("input_timeout_ms", config.inputTimeoutMs);
    if (config.width < 640 || config.width > 3840 || config.height < 480 || config.height > 2160 ||
        config.hitRadius < 20 || config.hitRadius > std::min(config.width, config.height) / 4 ||
        config.displayMs < 50 || config.displayMs > 10000 ||
        config.waitMinMs < 100 || config.waitMaxMs < config.waitMinMs || config.waitMaxMs > 60000 ||
        config.feedbackMs < 100 || config.feedbackMs > 10000 ||
        config.inputTimeoutMs < 50 || config.inputTimeoutMs > 2000) {
        throw std::runtime_error("Invalid values in " + path + ". Check the ranges in config/game.yml.");
    }
    return config;
}

Game::Game(GameConfig config, bool useCamera)
    : config_(std::move(config)), useCamera_(useCamera),
      windowName_(useCamera ? "MoleGame - Internal timing" : "MoleGame - Mouse preview"),
      calibrating_(useCamera) {}

void Game::mouseCallback(int, int x, int y, int, void* userdata) {
    static_cast<Game*>(userdata)->onMouse(x, y);
}

void Game::onMouse(int x, int y) {
    if (useCamera_) return;
    const auto now = Clock::now();
    onInput(cv::Point2f(static_cast<float>(x), static_cast<float>(y)), now, now);
}

bool Game::inputFresh(Clock::time_point now) const {
    return inputKnown_ && (!useCamera_ || (hasSample_ &&
        now - lastSampleAt_ <= std::chrono::milliseconds(config_.inputTimeoutMs)));
}

void Game::onInput(std::optional<cv::Point2f> point, Clock::time_point sampledAt,
                   Clock::time_point receivedAt, std::uint64_t frame) {
    if (sampledAt < ignoreBefore_ || (hasSample_ && sampledAt <= lastSampleAt_)) return;
    currentFrame_ = frame;
    currentSampleAt_ = sampledAt;
    const bool trialOpen = activeSubmitted_ && (phase_ == Phase::Active || phase_ == Phase::Resolving);
    const auto timeout = std::chrono::milliseconds(config_.inputTimeoutMs);
    const bool gap = useCamera_ && hasSample_ && sampledAt - lastSampleAt_ > timeout;
    const bool stale = useCamera_ && (receivedAt - sampledAt > timeout || sampledAt > receivedAt);
    if (gap || stale) {
        inputKnown_ = false;
        if (trialOpen) trialUncertain_ = true;
    }
    hasSample_ = true;
    lastSampleAt_ = sampledAt;
    if (stale) point.reset();
    if (point && (!std::isfinite(point->x) || !std::isfinite(point->y))) point.reset();

    // 先处理队列中的样本，再按当前时间更新显示。结果晚到不改变样本所属的计时窗口。
    if (trialOpen && sampledAt >= deadline_) {
        finishTrial(trialUncertain_ ? Outcome::Unjudged : Outcome::Miss, receivedAt);
    }
    if (!point) {
        inputKnown_ = false;
        if (trialOpen && sampledAt >= submittedAt_ && sampledAt < deadline_) trialUncertain_ = true;
        if (phase_ == Phase::Waiting) phase_ = Phase::AwaitExit;
        logEvent("INPUT_UNKNOWN", receivedAt);
        return;
    }
    const cv::Point2f center(config_.width / 2.f, config_.height / 2.f + 30);
    const auto delta = *point - center;
    const bool inside = delta.dot(delta) <= static_cast<float>(config_.hitRadius * config_.hitRadius);
    const bool entered = inputKnown_ && !inputInside_ && inside;
    pointer_ = *point;
    inputKnown_ = true;
    inputInside_ = inside;
    logEvent("INPUT", receivedAt);
    if (!entered) return;
    if (activeSubmitted_ && (phase_ == Phase::Active || phase_ == Phase::Resolving) &&
        sampledAt >= submittedAt_ && sampledAt < deadline_) {
        finishTrial(Outcome::Hit, receivedAt, sampledAt);
    } else if (phase_ == Phase::Waiting || (phase_ == Phase::Active && !activeSubmitted_)) {
        phase_ = Phase::AwaitExit;
        activeSubmitted_ = false;
        lastResult_ = "Too early - move outside and wait for the mole";
        logEvent("EARLY_ENTRY", receivedAt);
    } else if (phase_ == Phase::Active && activeSubmitted_ && sampledAt < submittedAt_) {
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
            trialUncertain_ = true;
    }
    if (phase_ == Phase::AwaitExit) {
        if (inputFresh(now) && !inputInside_) scheduleTrial(now);
    } else if (phase_ == Phase::Waiting && now >= deadline_) {
        if (!inputFresh(now) || inputInside_) {
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
               now >= deadline_ + std::chrono::milliseconds(config_.inputTimeoutMs)) {
        finishTrial(Outcome::Unjudged, now);
    } else if (phase_ == Phase::Feedback && now >= deadline_) {
        phase_ = Phase::AwaitExit;
        if (inputFresh(now) && !inputInside_) scheduleTrial(now);
    }
}

void Game::submitFrame(Clock::time_point now) {
    if (phase_ != Phase::Active || activeSubmitted_) return;
    ++trials_;
    ++trialId_;
    submittedAt_ = now;
    deadline_ = now + std::chrono::milliseconds(activeDurationMs_);
    activeSubmitted_ = true;
    trialUncertain_ = !inputKnown_;
    logEvent("WINDOW_START", now);
}

void Game::finishTrial(Outcome result, Clock::time_point now, Clock::time_point entryAt) {
    if ((phase_ != Phase::Active && phase_ != Phase::Resolving) || !activeSubmitted_) return;
    if (result == Outcome::Hit) {
        ++hits_;
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(entryAt - submittedAt_).count();
        lastResult_ = "HIT - entry after " + std::to_string(elapsed) + " ms (internal time)";
    } else if (result == Outcome::Unjudged) {
        ++unjudged_;
        lastResult_ = "UNJUDGED - missing, ambiguous or delayed hand input";
    } else {
        lastResult_ = "MISS - internal window ended";
    }
    logEvent("RESULT", now, result == Outcome::Hit ? "HIT" : result == Outcome::Miss ? "MISS" : "UNJUDGED");
    if (events_.is_open()) events_.flush();
    phase_ = Phase::Feedback;
    deadline_ = now + std::chrono::milliseconds(config_.feedbackMs);
}

void Game::reset(Clock::time_point now) {
    if (activeSubmitted_ && (phase_ == Phase::Active || phase_ == Phase::Resolving))
        finishTrial(Outcome::Unjudged, now);
    logEvent("RESET", now);
    trials_ = 0;
    hits_ = 0;
    unjudged_ = 0;
    phase_ = Phase::AwaitExit;
    activeSubmitted_ = false;
    ignoreBefore_ = now;
    lastResult_ = "Reset - no result yet";
}

cv::Mat Game::render() const {
    cv::Mat frame(config_.height, config_.width, CV_8UC3, cv::Scalar(29, 25, 24));
    const cv::Scalar white(235, 235, 235);
    const cv::Scalar muted(160, 160, 160);
    const cv::Point center(config_.width / 2, config_.height / 2 + 30);
    const int radius = config_.hitRadius;
    cv::putText(frame, useCamera_ ? "MoleGame | YOLO + Internal timing" : "MoleGame | Mouse preview",
        {28, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.85, white, 2, cv::LINE_AA);
    cv::putText(frame, useCamera_ ? "Single hand | H: recalibrate | internal display window" :
        "Mouse preview - internal timing only", {28, 73},
        cv::FONT_HERSHEY_SIMPLEX, 0.5, muted, 1, cv::LINE_AA);
    cv::putText(frame, "Display: " + std::to_string(config_.displayMs) + " ms   Hits: " + std::to_string(hits_) +
        "   Trials: " + std::to_string(trials_) + "   Unjudged: " + std::to_string(unjudged_),
        {28, 108}, cv::FONT_HERSHEY_SIMPLEX, 0.65, white, 1, cv::LINE_AA);
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
        phase_ == Phase::Resolving ? "Waiting for result" :
        !inputFresh(Clock::now()) ? (useCamera_ ? "Show one hand outside the circle" : "Move mouse outside the circle") :
        phase_ == Phase::Waiting ? "Wait for the mole" : "Move outside the circle";
    const int textWidth = cv::getTextSize(status, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, nullptr).width;
    cv::putText(frame, status, {center.x - textWidth / 2, center.y + radius + 65},
        cv::FONT_HERSHEY_SIMPLEX, 0.7, white, 1, cv::LINE_AA);
    if (inputFresh(Clock::now())) {
        cv::circle(frame, pointer_, 7, cv::Scalar(80, 80, 255), -1, cv::LINE_AA);
        cv::circle(frame, pointer_, 9, white, 1, cv::LINE_AA);
    }
    cv::putText(frame, "Move outside -> wait -> enter the circle. No click needed.", {28, config_.height - 58},
        cv::FONT_HERSHEY_SIMPLEX, 0.5, muted, 1, cv::LINE_AA);
    cv::putText(frame, "Esc: exit   Space: reset   F: fullscreen   Up/Down: display +/- 50 ms", {28, config_.height - 28},
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
    events_ << ',' << activeDurationMs_ << ',' << (trials_ ? ms(submittedAt_) : -1) << ','
            << (trials_ ? ms(submittedAt_ + std::chrono::milliseconds(activeDurationMs_)) : -1)
            << ',' << detail << '\n';
    if (!events_) throw std::runtime_error("Cannot write session events");
}

void Game::openLog() {
    namespace fs = std::filesystem;
    const auto id = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    sessionDirectory_ = "outputs/internal-" + std::to_string(id);
    fs::create_directories(sessionDirectory_);
    events_.open(sessionDirectory_ + "/events.csv");
    if (!events_) throw std::runtime_error("Cannot create event log");
    events_ << std::fixed << std::setprecision(3)
            << "received_ms,event,trial,frame,sample_ms,hand_count,x,y,display_ms,window_start_ms,window_end_ms,detail\n";
    for (const auto& name : {"game.yml", "camera.yml", "yolo.yml"})
        fs::copy_file(fs::path("config") / name, fs::path(sessionDirectory_) / name);
    cv::FileStorage metadata(sessionDirectory_ + "/session.yml", cv::FileStorage::WRITE);
    if (!metadata.isOpened()) throw std::runtime_error("Cannot save session metadata");
    metadata << "mode" << "internal_timing" << "camera_serial" << cameraId_
             << "camera_size" << cameraSize_ << "game_size" << cv::Size(config_.width, config_.height)
             << "input_timeout_ms" << config_.inputTimeoutMs
             << "timestamp_source" << "host_after_conversion_and_clone"
             << "random_engine" << "mt19937" << "random_seed" << std::to_string(randomSeed_);
    std::cout << "Session: " << sessionDirectory_ << '\n';
}

void Game::beginCalibration() {
    if (activeSubmitted_ && (phase_ == Phase::Active || phase_ == Phase::Resolving))
        finishTrial(Outcome::Unjudged, Clock::now());
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
                                                       {game.config_.width, game.config_.height});
            game.calibrationMessage_ = "Check mapped hand point, then ENTER to start; H to retry";
        } catch (const std::exception&) {
            game.calibrationMessage_ = "Invalid points. Press H and click labels 1,2,3,4 again";
        }
    }
}

void Game::loadCalibration() {
    const std::string path = "config/camera-calibration.yml";
    if (!std::filesystem::exists(path)) return;
    try {
        cv::FileStorage file(path, cv::FileStorage::READ);
        cv::Size cameraSize, gameSize;
        std::string serial;
        file["camera_size"] >> cameraSize;
        file["game_size"] >> gameSize;
        file["camera_serial"] >> serial;
        if (cameraSize != cameraSize_ || gameSize != cv::Size(config_.width, config_.height) || serial != cameraId_)
            throw std::runtime_error("Calibration device or size changed");
        file["camera_points"] >> calibrationCorners_;
        cameraToGame_ = cameraToGameTransform(calibrationCorners_, cameraSize_, gameSize);
        calibrationMessage_ = "Saved mapping: verify hand point; ENTER to start or H to redo";
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
    file << "camera_size" << cameraSize_ << "game_size" << cv::Size(config_.width, config_.height)
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
    cv::Mat frame(config_.height, config_.width, CV_8UC3, cv::Scalar(29, 25, 24));
    cv::putText(frame, "Calibration | F: fullscreen | H: restart | Esc: exit", {20, 35},
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(235, 235, 235), 1);
    cv::putText(frame, calibrationMessage_, {20, config_.height / 2},
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(235, 235, 235), 1);
    const auto targets = calibrationTargets({config_.width, config_.height});
    for (std::size_t i = 0; i < targets.size(); ++i) {
        cv::drawMarker(frame, targets[i], cv::Scalar(0, 220, 255), cv::MARKER_CROSS, 30, 2);
        cv::putText(frame, std::to_string(i + 1), targets[i] + cv::Point2f(15, -15),
            cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(235, 235, 235), 2);
    }
    if (inputFresh(Clock::now())) cv::circle(frame, pointer_, 8, cv::Scalar(80, 80, 255), -1);
    return frame;
}

void Game::run() {
    std::unique_ptr<CameraInput> camera;
    const std::string previewName = "MoleGame camera preview";
    if (useCamera_) {
        camera = std::make_unique<CameraInput>();
        cameraSize_ = camera->frameSize();
        cameraId_ = camera->deviceId();
        openLog();
        loadCalibration();
        cv::namedWindow(previewName, cv::WINDOW_NORMAL);
        cv::resizeWindow(previewName, cameraSize_.width, cameraSize_.height);
        cv::setMouseCallback(previewName, &Game::calibrationClick, this);
    }
    cv::namedWindow(windowName_, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(windowName_, config_.width, config_.height);
    if (!useCamera_) cv::setMouseCallback(windowName_, &Game::mouseCallback, this);
    std::optional<cv::Rect> previousWindowRect;
    auto nextLog = Clock::now();
    cv::Mat preview;
    for (;;) {
        if (camera) {
            CameraUpdate batch;
            try { camera->poll(batch); }
            catch (const std::exception&) {
                if (activeSubmitted_ && (phase_ == Phase::Active || phase_ == Phase::Resolving))
                    finishTrial(Outcome::Unjudged, Clock::now());
                logEvent("CAMERA_ERROR", Clock::now());
                cv::destroyAllWindows();
                throw;
            }
            if (batch.dropped) {
                inputKnown_ = false;
                trialUncertain_ = true;
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
                if (!calibrating_) onInput(point, sample.capturedAt, receivedAt, sample.sequence);
                else {
                    pointer_ = point.value_or(cv::Point2f(-1, -1));
                    inputKnown_ = point.has_value();
                    hasSample_ = true;
                    lastSampleAt_ = sample.capturedAt;
                }
                logEvent("DETECTION", receivedAt, "", static_cast<int>(sample.centers.size()));
                if (receivedAt >= nextLog) {
                    std::cout << "[YOLO] frame=" << sample.sequence << " hands=" << sample.centers.size()
                              << " detector_total_ms=" << sample.detectorMs
                              << " capture_to_game_receive_ms="
                              << std::chrono::duration<double, std::milli>(receivedAt - sample.capturedAt).count() << '\n';
                    nextLog = receivedAt + std::chrono::seconds(1);
                }
            }
            if (!batch.preview.empty()) preview = batch.preview;
            if (!preview.empty()) {
                cv::Mat annotated = preview.clone();
                for (std::size_t i = 0; i < calibrationCorners_.size(); ++i)
                    cv::putText(annotated, std::to_string(i + 1), calibrationCorners_[i],
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(80, 80, 255), 2);
                if (!cameraToGame_.empty()) {
                    std::vector<cv::Point2f> ring, cameraRing;
                    for (int i = 0; i < 64; ++i) {
                        const double angle = i * 2 * CV_PI / 64;
                        ring.emplace_back(config_.width / 2.f + config_.hitRadius * static_cast<float>(std::cos(angle)),
                            config_.height / 2.f + 30 + config_.hitRadius * static_cast<float>(std::sin(angle)));
                    }
                    cv::perspectiveTransform(ring, cameraRing, cameraToGame_.inv());
                    std::vector<cv::Point> polygon;
                    for (const auto& p : cameraRing) polygon.emplace_back(p);
                    cv::polylines(annotated, polygon, true, cv::Scalar(255, 150, 0), 2);
                }
                cv::putText(annotated, "One hand only | H: calibrate | Esc: exit", {10, 25},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 220, 255), 1);
                cv::imshow(previewName, annotated);
            }
        }
        if (!calibrating_) update(Clock::now());
        const cv::Mat frame = calibrating_ ? renderCalibration() : render();
        // 这是内部窗口起点，不声称是投影面实际可见时刻。
        if (!calibrating_) submitFrame(Clock::now());
        cv::imshow(windowName_, frame);
        if (camera) {
            const auto rect = cv::getWindowImageRect(windowName_);
            if (previousWindowRect && rect != *previousWindowRect) beginCalibration();
            previousWindowRect = rect;
        }
        const int key = cv::waitKeyEx(1);
        if (key == 27 || cv::getWindowProperty(windowName_, cv::WND_PROP_VISIBLE) < 1 ||
            (camera && cv::getWindowProperty(previewName, cv::WND_PROP_VISIBLE) < 1)) break;
        if (camera && (key == 'h' || key == 'H')) { beginCalibration(); continue; }
        if (camera && calibrating_ && (key == 13 || key == 10)) { confirmCalibration(); continue; }
        if (key == ' ') reset();
        else if (key == 'f' || key == 'F') {
            fullscreen_ = !fullscreen_;
            cv::setWindowProperty(windowName_, cv::WND_PROP_FULLSCREEN,
                fullscreen_ ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
            if (camera) beginCalibration();
        } else if (key == 2490368 || key == 65362) {
            config_.displayMs = std::min(10000, config_.displayMs + 50);
        } else if (key == 2621440 || key == 65364) {
            config_.displayMs = std::max(50, config_.displayMs - 50);
        }
    }
    if (useCamera_ && activeSubmitted_ && (phase_ == Phase::Active || phase_ == Phase::Resolving))
        finishTrial(Outcome::Unjudged, Clock::now());
    logEvent("SESSION_END", Clock::now());
    cv::destroyAllWindows();
}
