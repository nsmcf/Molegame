#include "Game.h"
#include "CameraInput.h"
#include <opencv2/imgproc.hpp>

#include <functional>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

// 用人工时间戳驱动实际游戏逻辑；不打开窗口，不依赖相机速度。
struct GameTestAccess {
    static auto time(int ms) { return Game::Clock::time_point{} + std::chrono::milliseconds(ms); }
    static Game make(bool camera = true, bool observed = false) {
        GameConfig config;
        config.waitMinMs = config.waitMaxMs = 100;
        config.displayMs = 100;
        config.feedbackMs = 100;
        config.inputTimeoutMs = 200;
        config.observationTimeoutMs = 200;
        return Game(config, observed ? GameMode::CameraObservation : camera ? GameMode::InternalTiming : GameMode::Mouse);
    }
    static void input(Game& game, std::optional<cv::Point2f> point, int sample, int received,
                      MoleVisibility mole = MoleVisibility::Unknown) {
        game.onInput(point, time(sample), time(received), static_cast<std::uint64_t>(sample), mole);
    }
    static cv::Point2f outside() { return {100, 100}; }
    static cv::Point2f inside() { return {480, 390}; }
    static void start(Game& game) {
        input(game, outside(), 1000, 1000);
        game.update(time(1000));
        game.update(time(1100));
        game.submitFrame(time(1100));
        input(game, outside(), 1120, 1121);
        require(game.trials_ == 1, "trial did not start");
    }
    static void startObserved(Game& game, bool visible = true) {
        input(game, outside(), 1000, 1000, MoleVisibility::Absent);
        game.update(time(1000));
        game.update(time(1100));
        game.submitFrame(time(1100));
        input(game, outside(), 1120, 1121, visible ? MoleVisibility::Visible : MoleVisibility::Absent);
        require(game.trials_ == 1, "observed trial did not start");
    }
    static void run(const std::string& filter = "") {
        const std::vector<std::pair<const char*, std::function<void()>>> cases{
            {"single entry scores once", [] {
                auto game = make(); start(game);
                input(game, inside(), 1150, 1155);
                input(game, inside(), 1160, 1165);
                require(game.hits_ == 1 && game.phase_ == Game::Phase::Feedback, "duplicate or missing hit");
            }},
            {"late inference uses sample timestamp", [] {
                auto game = make(); start(game);
                game.update(time(1205));
                require(game.phase_ == Game::Phase::Resolving, "mole should be hidden while resolving");
                input(game, inside(), 1150, 1220);
                require(game.hits_ == 1, "in-window sample rejected because inference finished later");
            }},
            {"entry exactly at deadline is outside window", [] {
                auto game = make(); start(game);
                input(game, inside(), 1200, 1205);
                require(game.hits_ == 0 && game.unjudged_ == 0 && game.lastResult_.starts_with("MISS"),
                        "deadline must be exclusive");
            }},
            {"clean observations produce miss", [] {
                auto game = make(); start(game);
                input(game, outside(), 1180, 1181);
                game.update(time(1205));
                input(game, outside(), 1210, 1211);
                require(game.lastResult_.starts_with("MISS"), "clean timeout must be a miss");
            }},
            {"lost hand reappearing inside is not an entry", [] {
                auto game = make(); start(game);
                input(game, std::nullopt, 1140, 1141);
                input(game, inside(), 1150, 1151);
                require(game.hits_ == 0, "detection gap created a false entry");
                input(game, outside(), 1210, 1211);
                require(game.unjudged_ == 1, "uncertain timeout counted as a miss");
            }},
            {"new outside-inside pair recovers after unknown input", [] {
                auto game = make(); start(game);
                input(game, std::nullopt, 1140, 1141);
                input(game, inside(), 1150, 1151);
                input(game, outside(), 1160, 1161);
                input(game, inside(), 1170, 1171);
                require(game.hits_ == 1, "valid new entry should recover");
            }},
            {"missing watermark has bounded wait and unjudged result", [] {
                auto game = make(); start(game);
                game.update(time(1201));
                game.update(time(1401));
                require(game.unjudged_ == 1 && game.phase_ == Game::Phase::Feedback, "unbounded result wait");
            }},
            {"stale results cannot score", [] {
                auto game = make(); start(game);
                input(game, inside(), 1150, 1360);
                input(game, outside(), 1370, 1371);
                require(game.hits_ == 0 && game.unjudged_ == 1, "stale result created a hit");
            }},
            {"large capture gap breaks continuity", [] {
                auto game = make();
                game.config_.displayMs = 500;
                start(game);
                input(game, inside(), 1350, 1351);
                require(game.hits_ == 0, "capture gap created false entry");
                input(game, outside(), 1620, 1621);
                require(game.unjudged_ == 1, "large gap not recorded");
            }},
            {"early entry cancels waiting", [] {
                auto game = make();
                input(game, outside(), 1000, 1000);
                game.update(time(1000));
                input(game, inside(), 1050, 1051);
                game.update(time(1100));
                require(game.phase_ == Game::Phase::AwaitExit && game.trials_ == 0, "early entry started a trial");
            }},
            {"pre-window entry received after submission cancels trial", [] {
                auto game = make();
                input(game, outside(), 1000, 1000);
                game.update(time(1000));
                game.update(time(1100));
                game.submitFrame(time(1100));
                input(game, inside(), 1090, 1120);
                require(game.hits_ == 0 && game.unjudged_ == 1 && game.phase_ == Game::Phase::Feedback,
                        "pre-window entry was not cancelled");
            }},
            {"first observation inside does not arm a trial", [] {
                auto game = make();
                input(game, inside(), 1000, 1000);
                game.update(time(1000));
                require(game.phase_ == Game::Phase::AwaitExit, "first inside observation armed trial");
            }},
            {"stale outside sample cannot keep waiting armed", [] {
                auto game = make();
                input(game, outside(), 1000, 1000);
                game.update(time(1000));
                game.update(time(1300));
                require(game.phase_ == Game::Phase::AwaitExit, "stale input started trial");
            }},
            {"reset rejects queued pre-reset samples", [] {
                auto game = make(); start(game);
                game.reset(time(1160));
                input(game, inside(), 1150, 1170);
                require(game.trials_ == 0 && game.hits_ == 0 && !game.inputInside_, "reset replayed queued sample");
            }},
            {"mouse timeout still completes without camera watermark", [] {
                auto game = make(false); start(game);
                game.update(time(1201));
                require(game.phase_ == Game::Phase::Feedback && game.lastResult_.starts_with("MISS"),
                        "mouse preview now depends on camera");
            }},
            {"CSV records sample time and unique trial ids across reset", [] {
                auto game = make();
                game.sessionStart_ = time(1000);
                game.events_.open("timing-test-events.csv");
                require(game.events_.is_open(), "cannot open test event file");
                start(game);
                input(game, inside(), 1150, 1220);
                game.reset(time(1300));
                input(game, outside(), 1310, 1311);
                game.update(time(1311));
                game.update(time(1411));
                game.submitFrame(time(1411));
                game.events_.close();
                require(game.trialId_ == 2 && game.trials_ == 1, "trial ids repeated after reset");
                std::ifstream file("timing-test-events.csv");
                std::string line;
                bool hitFound = false;
                while (std::getline(file, line)) {
                    std::vector<std::string> fields;
                    std::istringstream row(line);
                    std::string field;
                    while (std::getline(row, field, ',')) fields.push_back(field);
                    // Empty detail is a trailing CSV field, omitted by getline.
                    require(fields.size() == 11 || fields.size() == 12, "malformed CSV row");
                    if (fields[1] == "RESULT") {
                        require(std::stod(fields[0]) == 220 && std::stod(fields[4]) == 150 && fields[11] == "HIT",
                                "log confused received time with sample time");
                        hitFound = true;
                    }
                }
                require(hitFound, "hit event missing from CSV");
            }},
            {"session snapshot and event file can be created without camera", [] {
                namespace fs = std::filesystem;
                fs::create_directories("config");
                for (const auto& name : {"game.yml", "camera.yml", "yolo.yml"}) {
                    const auto path = fs::path("config") / name;
                    if (!fs::exists(path)) { std::ofstream file(path); file << "%YAML:1.0\n---\n"; }
                }
                auto game = make();
                game.cameraId_ = "test-device";
                game.cameraSize_ = {720, 540};
                game.openLog();
                start(game);
                game.events_.close();
                cv::FileStorage metadata(game.sessionDirectory_ + "/session.yml", cv::FileStorage::READ);
                std::string seed;
                metadata["random_seed"] >> seed;
                require(seed == std::to_string(game.randomSeed_), "session random seed missing");
                require(fs::file_size(game.sessionDirectory_ + "/events.csv") > 100, "empty event file");
                require(fs::exists(game.sessionDirectory_ + "/yolo.yml"), "configuration snapshot missing");
            }},
            {"calibration preserves projected label order even when rotated", [] {
                const std::vector<cv::Point2f> corners{{600, 450}, {100, 450}, {100, 80}, {600, 80}};
                const auto transform = cameraToGameTransform(corners, {720, 540}, {960, 720});
                std::vector<cv::Point2f> mapped;
                cv::perspectiveTransform(corners, mapped, transform);
                const auto expected = calibrationTargets({960, 720});
                for (int i = 0; i < 4; ++i)
                    require(cv::norm(mapped[i] - expected[i]) < 0.01, "calibration reordered labels");
            }},
            {"invalid calibration cannot be accepted", [] {
                const std::vector<std::vector<cv::Point2f>> bad{
                    {{100, 100}, {600, 450}, {600, 100}, {100, 450}},
                    {{100, 100}, {100, 100}, {600, 450}, {100, 450}},
                    {{-1, 100}, {600, 100}, {600, 450}, {100, 450}},
                    {{100, 100}, {101, 100}, {101, 101}, {100, 101}},
                    {{std::numeric_limits<float>::quiet_NaN(), 100}, {600, 100}, {600, 450}, {100, 450}}
                };
                for (const auto& corners : bad) {
                    bool rejected = false;
                    try { cameraToGameTransform(corners, {720, 540}, {960, 720}); }
                    catch (const std::exception&) { rejected = true; }
                    require(rejected, "bad calibration accepted");
                }
            }},
            {"rendered mole is visible and empty hole is absent", [] {
                auto game = make(false);
                const std::vector<cv::Point2f> region{{420, 330}, {540, 330}, {540, 450}, {420, 450}};
                game.phase_ = Game::Phase::Active;
                require(observeMole(game.render(), region, {}, {}).state == MoleVisibility::Visible,
                        "actual rendered mole color not recognized");
                game.phase_ = Game::Phase::Waiting;
                require(observeMole(game.render(), region, {}, {}).state == MoleVisibility::Absent,
                        "empty hole recognized as visible mole");
            }},
            {"both target styles provide the same on-off color evidence", [] {
                auto game = make(false);
                const auto region = moleRegionInCamera(cv::Mat::eye(3, 3, CV_64F), {960, 720}, 85);
                for (const auto& style : {"square", "mole"}) {
                    game.config_.targetStyle = style;
                    game.phase_ = Game::Phase::Active;
                    require(observeMole(game.render(), region, {}, {}).state == MoleVisibility::Visible,
                            "target style has no positive color evidence");
                    game.phase_ = Game::Phase::Waiting;
                    require(observeMole(game.render(), region, {}, {}).state == MoleVisibility::Absent,
                            "target style left positive evidence after hiding");
                }
            }},
            {"calibrated disk detects mole in a rotated perspective camera image", [] {
                auto game = make(false);
                game.phase_ = Game::Phase::Active;
                const cv::Size cameraSize(720, 540);
                const std::vector<cv::Point2f> corners{{600, 450}, {100, 440}, {110, 80}, {620, 90}};
                const auto transform = cameraToGameTransform(corners, cameraSize, {960, 720});
                const auto region = moleRegionInCamera(transform, {960, 720}, 85);
                cv::Mat captured;
                cv::warpPerspective(game.render(), captured, transform.inv(), cameraSize);
                require(observeMole(captured, region, {}, {}).state == MoleVisibility::Visible,
                        "rotated calibrated disk did not recognize projected mole");
            }},
            {"yellow outside calibrated region cannot create a visible mole", [] {
                cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(0, 200, 255));
                frame(cv::Rect(50, 50, 101, 101)).setTo(cv::Scalar(0, 0, 0));
                const std::vector<cv::Point2f> region{{50, 50}, {150, 50}, {150, 150}, {50, 150}};
                require(observeMole(frame, region, {}, {}).state == MoleVisibility::Absent,
                        "color from outside the mole region leaked into detection");
            }},
            {"small projected mole region survives pixel rounding at live calibration scale", [] {
                const cv::Size cameraSize(720, 540), gameSize(720, 960);
                const std::vector<std::vector<cv::Point2f>> calibrations{
                    {{188, 263}, {217, 65}, {528, 77}, {526, 284}},
                    {{189, 263}, {215, 65}, {527, 78}, {524, 284}}
                };
                for (const auto& points : calibrations) {
                    const auto transform = cameraToGameTransform(points, cameraSize, gameSize);
                    const auto region = moleRegionInCamera(transform, gameSize, 85);
                    cv::Mat frame(cameraSize, CV_8UC3, cv::Scalar(0, 200, 255));
                    const auto on = observeMole(frame, region, {}, {});
                    require(on.state == MoleVisibility::Visible, "valid live-scale region rejected after rounding");
                    frame.setTo(cv::Scalar(0, 0, 0));
                    require(observeMole(frame, region, {}, {}).state == MoleVisibility::Absent,
                            "live-scale empty region rejected after rounding");
                }
            }},
            {"self-crossed region remains invalid instead of being repaired by a hull", [] {
                const cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(0, 200, 255));
                const std::vector<cv::Point2f> crossed{{50, 50}, {150, 150}, {150, 50}, {50, 150}};
                const auto result = observeMole(frame, crossed, {}, {});
                require(result.state == MoleVisibility::Unknown && result.reason == "invalid_region",
                        "invalid crossed region was accepted");
            }},
            {"partial occlusion can preserve positive color evidence", [] {
                cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(0, 200, 255));
                const std::vector<cv::Point2f> region{{50, 50}, {150, 50}, {150, 150}, {50, 150}};
                const cv::Rect hand(50, 50, 45, 101);
                frame(hand).setTo(cv::Scalar(0, 0, 0));
                const auto result = observeMole(frame, region, {hand}, {});
                require(result.state == MoleVisibility::Visible && result.unoccludedRatio < 1,
                        "remaining visible mole was lost under partial occlusion");
            }},
            {"full occlusion is unknown even when hand color matches mole", [] {
                const cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(0, 200, 255));
                const std::vector<cv::Point2f> region{{50, 50}, {150, 50}, {150, 150}, {50, 150}};
                const auto result = observeMole(frame, region, {{40, 40, 120, 120}}, {});
                require(result.state == MoleVisibility::Unknown && result.reason == "occluded",
                        "fully covering hand created a visible or absent claim");
            }},
            {"absence under partial occlusion remains unknown", [] {
                const cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));
                const std::vector<cv::Point2f> region{{50, 50}, {150, 50}, {150, 150}, {50, 150}};
                require(observeMole(frame, region, {{50, 50, 30, 101}}, {}).state == MoleVisibility::Unknown,
                        "missing color under occlusion was called definite absence");
            }},
            {"ambiguous color amount is unknown and no previous state is held", [] {
                cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(0, 200, 255));
                const std::vector<cv::Point2f> region{{50, 50}, {150, 50}, {150, 150}, {50, 150}};
                require(observeMole(frame, region, {}, {}).state == MoleVisibility::Visible, "fixture not visible");
                frame.setTo(cv::Scalar(0, 0, 0));
                frame(cv::Rect(50, 50, 10, 101)).setTo(cv::Scalar(0, 200, 255));
                require(observeMole(frame, region, {}, {}).state == MoleVisibility::Unknown,
                        "weak color evidence was accepted or previous visible state was held");
                frame.setTo(cv::Scalar(0, 0, 0));
                require(observeMole(frame, region, {}, {}).state == MoleVisibility::Absent,
                        "previous frame prevented disappearance");
            }},
            {"missing or off-image detection regions cannot claim absence", [] {
                const cv::Mat frame(200, 200, CV_8UC3, cv::Scalar(0, 0, 0));
                require(observeMole(frame, {}, {}, {}).state == MoleVisibility::Unknown, "uncalibrated region accepted");
                const std::vector<cv::Point2f> region{{-5, 50}, {150, 50}, {150, 150}, {-5, 150}};
                require(observeMole(frame, region, {}, {}).state == MoleVisibility::Unknown, "clipped region called absent");
                const std::vector<cv::Point2f> tiny{{50, 50}, {51, 50}, {51, 51}, {50, 51}};
                require(observeMole(frame, tiny, {}, {}).state == MoleVisibility::Unknown, "tiny region accepted");
            }},
            {"invalid mole thresholds fail before detection", [] {
                MoleDetectorConfig config;
                config.absentRatio = config.visibleRatio;
                bool rejected = false;
                try { observeMole({}, {}, {}, config); }
                catch (const std::exception&) { rejected = true; }
                require(rejected, "ambiguous threshold configuration accepted");
            }},
            {"observation session is labeled as a probe and snapshots detector settings", [] {
                namespace fs = std::filesystem;
                fs::create_directories("config");
                for (const auto& name : {"game.yml", "camera.yml", "yolo.yml", "mole.yml"}) {
                    const auto path = fs::path("config") / name;
                    if (!fs::exists(path)) { std::ofstream file(path); file << "%YAML:1.0\n---\n"; }
                }
                Game probe(GameConfig{}, GameMode::ObservationProbe);
                probe.cameraId_ = "test-device";
                probe.cameraSize_ = {720, 540};
                probe.openLog();
                probe.config_.displayMs = 250;
                probe.logEvent("DISPLAY_MS_CHANGED", time(1000), "250");
                probe.events_.close();
                probe.observations_.close();
                cv::FileStorage metadata(probe.sessionDirectory_ + "/session.yml", cv::FileStorage::READ);
                std::string mode, scoring;
                metadata["mode"] >> mode; metadata["scoring"] >> scoring;
                require(mode == "observation_probe" && scoring == "disabled", "probe confused with scoring session");
                require(fs::exists(probe.sessionDirectory_ + "/mole.yml"), "mole settings snapshot missing");
                std::ifstream data(probe.sessionDirectory_ + "/observations.csv");
                std::string header;
                std::getline(data, header);
                require(header.find("yolo_ms,mole_ms") != std::string::npos, "detector timings not separated");
                std::ifstream events(probe.sessionDirectory_ + "/events.csv");
                std::getline(events, header); std::getline(events, header);
                std::istringstream row(header);
                std::vector<std::string> fields;
                std::string field;
                while (std::getline(row, field, ',')) fields.push_back(field);
                require(fields.size() == 12 && std::stoi(fields[8]) == 250 &&
                        std::stod(fields[9]) == -1 && std::stod(fields[10]) == -1 && fields[11] == "250",
                        "probe log lost adjusted display duration");
            }},
            {"camera scores visible entry after internal deadline while internal mode rejects it", [] {
                auto internal = make(); start(internal);
                auto observed = make(true, true); startObserved(observed);
                internal.update(time(1205)); observed.update(time(1205));
                input(internal, inside(), 1250, 1255, MoleVisibility::Visible);
                input(observed, inside(), 1250, 1255, MoleVisibility::Visible);
                require(internal.hits_ == 0 && observed.hits_ == 1,
                        "camera mode reused internal cutoff or internal mode used image visibility");
            }},
            {"visible entry exactly at nominal cutoff is valid only for camera mode", [] {
                auto game = make(true, true); startObserved(game);
                input(game, inside(), 1200, 1205, MoleVisibility::Visible);
                require(game.hits_ == 1, "nominal deadline incorrectly closed observed window");
            }},
            {"absent mole rejects an entry before the internal deadline", [] {
                auto game = make(true, true); startObserved(game);
                input(game, inside(), 1150, 1155, MoleVisibility::Absent);
                require(game.hits_ == 0 && game.unjudged_ == 0 && game.lastResult_.starts_with("MISS"),
                        "camera mode accepted entry because internal window was open");
            }},
            {"entry before observed onset is a miss", [] {
                auto game = make(true, true); startObserved(game, false);
                input(game, inside(), 1150, 1155, MoleVisibility::Absent);
                require(game.hits_ == 0 && game.lastResult_.starts_with("MISS"), "entry before actual appearance scored");
            }},
            {"occluded entry is unjudged without holding previous visible state", [] {
                auto game = make(true, true); startObserved(game);
                input(game, inside(), 1150, 1155, MoleVisibility::Unknown);
                require(game.hits_ == 0 && game.unjudged_ == 1, "occlusion extended previous visible state");
            }},
            {"clean observed disappearance ends a missed trial", [] {
                auto game = make(true, true); startObserved(game);
                game.update(time(1205));
                input(game, outside(), 1230, 1235, MoleVisibility::Absent);
                require(game.unjudged_ == 0 && game.lastResult_.starts_with("MISS"), "clear disappearance did not end trial");
            }},
            {"unobserved onset times out as unjudged rather than a definite miss", [] {
                auto game = make(true, true); startObserved(game, false);
                game.update(time(1205));
                input(game, outside(), 1300, 1305, MoleVisibility::Absent);
                game.update(time(1600));
                require(game.unjudged_ == 1 && game.phase_ == Game::Phase::Feedback, "missing onset was called a miss or waited forever");
            }},
            {"visible state cannot bypass the observation safety timeout", [] {
                auto game = make(true, true); startObserved(game);
                input(game, outside(), 1300, 1305, MoleVisibility::Visible);
                input(game, inside(), 1400, 1405, MoleVisibility::Visible);
                require(game.hits_ == 0 && game.unjudged_ == 1, "persistent color scored after safety timeout");
            }},
            {"missing observed disappearance has a bounded wait", [] {
                auto game = make(true, true); startObserved(game);
                game.update(time(1205));
                game.update(time(1600));
                require(game.unjudged_ == 1 && game.phase_ == Game::Phase::Feedback, "observed trial waited indefinitely");
            }},
            {"in-flight visible entry uses sample time within observation guard", [] {
                auto game = make(true, true); startObserved(game);
                game.update(time(1405));
                input(game, inside(), 1250, 1420, MoleVisibility::Visible);
                require(game.hits_ == 1, "fresh in-flight entry was rejected by receive time");
            }},
            {"stale observed entry cannot create a hit", [] {
                auto game = make(true, true); startObserved(game);
                game.update(time(1205));
                input(game, inside(), 1250, 1460, MoleVisibility::Visible);
                game.update(time(1600));
                require(game.hits_ == 0 && game.unjudged_ == 1, "stale observed entry scored");
            }},
            {"lost hand reappearing inside does not score in observed mode", [] {
                auto game = make(true, true); startObserved(game);
                input(game, std::nullopt, 1140, 1145, MoleVisibility::Visible);
                input(game, inside(), 1150, 1155, MoleVisibility::Visible);
                input(game, outside(), 1230, 1235, MoleVisibility::Absent);
                require(game.hits_ == 0 && game.unjudged_ == 1, "detection gap created an observed hit");
            }},
            {"unknown interval prevents a definite missed-trial claim", [] {
                auto game = make(true, true); startObserved(game);
                input(game, outside(), 1150, 1155, MoleVisibility::Unknown);
                input(game, outside(), 1230, 1235, MoleVisibility::Absent);
                require(game.unjudged_ == 1, "unknown mole interval was silently called a clean miss");
            }},
            {"next observed trial waits for an empty hole after feedback", [] {
                auto game = make(true, true); startObserved(game);
                input(game, inside(), 1150, 1155, MoleVisibility::Visible);
                input(game, outside(), 1260, 1265, MoleVisibility::Visible);
                game.update(time(1270));
                require(game.phase_ == Game::Phase::AwaitExit, "next trial started while previous projection remained visible");
                input(game, outside(), 1280, 1285, MoleVisibility::Absent);
                game.update(time(1290));
                require(game.phase_ == Game::Phase::Waiting, "empty hole did not rearm next trial");
            }},
            {"unknown mole cannot arm an observed trial", [] {
                auto game = make(true, true);
                input(game, outside(), 1000, 1000, MoleVisibility::Unknown);
                game.update(time(1000));
                require(game.phase_ == Game::Phase::AwaitExit, "unknown initial state armed a trial");
            }},
            {"pre-submission observed entry cannot score despite a visible label", [] {
                auto game = make(true, true);
                input(game, outside(), 1000, 1000, MoleVisibility::Absent);
                game.update(time(1000)); game.update(time(1100)); game.submitFrame(time(1100));
                input(game, inside(), 1090, 1120, MoleVisibility::Visible);
                require(game.hits_ == 0 && game.unjudged_ == 1, "queued early entry was assigned to the new observed trial");
            }},
            {"output turns counterclockwise without cropping or changing source pixels", [] {
                auto game = make(false);
                const cv::Mat original = (cv::Mat_<unsigned char>(2, 3) << 1, 2, 3, 4, 5, 6);
                game.config_.outputRotationCcw = 90;
                const auto rotated = game.rotateForOutput(original);
                require(rotated.rows == 3 && rotated.cols == 2 &&
                        rotated.at<unsigned char>(0, 0) == 3 && rotated.at<unsigned char>(0, 1) == 6 &&
                        rotated.at<unsigned char>(2, 0) == 1 && rotated.at<unsigned char>(2, 1) == 4,
                        "output rotated clockwise or lost pixels");
                require(original.at<unsigned char>(0, 0) == 1, "rotation modified source frame");
                require(game.outputSize() == cv::Size(960, 720) && game.gameSize() == cv::Size(720, 960),
                        "right-side view changed the landscape output size");
                require(game.rotateForOutput(game.render()).size() == game.outputSize(),
                        "rotated game does not fill the configured output frame");
                game.config_.outputRotationCcw = 0;
                require(cv::norm(game.rotateForOutput(original), original) == 0, "zero rotation changed image");
            }},
            {"mouse positions map back from counterclockwise output", [] {
                auto game = make(false);
                game.config_.outputRotationCcw = 90;
                game.onMouse(100, 619);
                require(game.pointer_ == outside() && !game.inputInside_, "rotated outside mouse position is incorrect");
                game.onMouse(510, 359);
                require(game.pointer_ == cv::Point2f(360, 510) && game.inputInside_, "rotated mouse center missed the hit region");
            }},
            {"rotated calibration maps hand and mole into the same logical region", [] {
                auto game = make(false);
                game.config_.outputRotationCcw = 90;
                game.phase_ = Game::Phase::Active;
                std::vector<cv::Point2f> corners;
                for (const auto& target : calibrationTargets(game.gameSize()))
                    corners.emplace_back(target.y, 719.f - target.x);
                const auto transform = cameraToGameTransform(corners, game.outputSize(), game.gameSize());
                std::vector<cv::Point2f> mapped;
                cv::perspectiveTransform(std::vector<cv::Point2f>{{510, 359}}, mapped, transform);
                require(cv::norm(mapped[0] - cv::Point2f(360, 510)) < 0.01, "rotated camera hand was mirrored or offset");
                const auto region = moleRegionInCamera(transform, game.gameSize(), 85);
                require(observeMole(game.rotateForOutput(game.render()), region, {}, {}).state == MoleVisibility::Visible,
                        "mole region did not follow rotated projection");
            }},
            {"changing output rotation cancels trial and clears camera calibration", [] {
                auto game = make(true, true); startObserved(game);
                game.calibrating_ = false;
                game.cameraToGame_ = cv::Mat::eye(3, 3, CV_64F);
                game.calibrationCorners_ = {{100, 100}, {600, 100}, {600, 450}, {100, 450}};
                game.toggleOutputRotation();
                require(game.config_.outputRotationCcw == 180 && game.calibrating_ &&
                        game.cameraToGame_.empty() && game.calibrationCorners_.empty() &&
                        !game.activeSubmitted_ && game.unjudged_ == 1,
                        "rotation reused old calibration or left an active trial");
                game.toggleOutputRotation();
                require(game.config_.outputRotationCcw == 0 && game.unjudged_ == 1, "rotation toggle did not restore original direction");
            }},
            {"inset calibration cache checks target layout and output rotation", [] {
                const std::string path = "rotation-calibration-test.yml";
                const std::vector<cv::Point2f> corners{{100, 80}, {600, 80}, {600, 450}, {100, 450}};
                const auto write = [&](int rotation) {
                    cv::FileStorage file(path, cv::FileStorage::WRITE);
                    file << "camera_size" << cv::Size(720, 540) << "game_size" <<
                        (rotation == 90 ? cv::Size(720, 960) : cv::Size(960, 720))
                         << "game_points" << calibrationTargets(rotation == 90 ? cv::Size(720, 960) : cv::Size(960, 720))
                         << "camera_serial" << "rotation-test" << "camera_points" << corners;
                    if (rotation >= 0) file << "output_rotation_ccw" << rotation;
                };
                auto game = make();
                game.cameraSize_ = {720, 540}; game.cameraId_ = "rotation-test";
                write(-1);
                game.loadCalibration(path);
                require(!game.cameraToGame_.empty(), "legacy calibration should remain valid for unrotated output");
                game.config_.outputRotationCcw = 90;
                game.loadCalibration(path);
                require(game.cameraToGame_.empty(), "legacy unrotated calibration accepted for rotated output");
                write(90);
                game.loadCalibration(path);
                require(!game.cameraToGame_.empty(), "matching rotated calibration could not be loaded");
                game.config_.outputRotationCcw = 180;
                game.loadCalibration(path);
                require(game.cameraToGame_.empty(), "side-view calibration was reused for opposite view");
                write(180);
                game.loadCalibration(path);
                require(!game.cameraToGame_.empty(), "opposite-view calibration could not be loaded");
                game.config_.outputRotationCcw = 0;
                game.loadCalibration(path);
                require(game.cameraToGame_.empty(), "180 degree calibration was accepted for unrotated output");
            }},
            {"180 degree output keeps a wide layout without changing source pixels", [] {
                auto game = make(false);
                game.config_.outputRotationCcw = 180;
                const cv::Mat original = (cv::Mat_<unsigned char>(2, 3) << 1, 2, 3, 4, 5, 6);
                const auto rotated = game.rotateForOutput(original);
                require(rotated.size() == original.size() && rotated.at<unsigned char>(0, 0) == 6 &&
                        rotated.at<unsigned char>(0, 2) == 4 && rotated.at<unsigned char>(1, 0) == 3 &&
                        rotated.at<unsigned char>(1, 2) == 1, "180 degree output is mirrored or changes aspect ratio");
                require(original.at<unsigned char>(0, 0) == 1, "180 degree output modified source pixels");
                require(game.gameSize() == cv::Size(960, 720) && game.outputSize() == cv::Size(960, 720) &&
                        game.rotateForOutput(game.render()).size() == game.outputSize(), "opposite-view layout is not landscape");
            }},
            {"mouse coordinates map back from 180 degree output", [] {
                auto game = make(false);
                game.config_.outputRotationCcw = 180;
                game.onMouse(859, 619);
                require(game.pointer_ == outside() && !game.inputInside_, "opposite-view outside point is incorrect");
                game.onMouse(479, 329);
                require(game.pointer_ == inside() && game.inputInside_, "opposite-view mouse center misses hit region");
            }},
            {"opposite-view calibration aligns hand and mole in landscape layout", [] {
                auto game = make(false);
                game.config_.outputRotationCcw = 180;
                game.phase_ = Game::Phase::Active;
                std::vector<cv::Point2f> corners;
                for (const auto& p : calibrationTargets(game.gameSize())) corners.emplace_back(959.f - p.x, 719.f - p.y);
                const auto transform = cameraToGameTransform(corners, game.outputSize(), game.gameSize());
                std::vector<cv::Point2f> mapped;
                cv::perspectiveTransform(std::vector<cv::Point2f>{{479, 329}}, mapped, transform);
                require(cv::norm(mapped[0] - inside()) < 0.01, "opposite-view hand mapping is mirrored or offset");
                const auto region = moleRegionInCamera(transform, game.gameSize(), 85);
                require(observeMole(game.rotateForOutput(game.render()), region, {}, {}).state == MoleVisibility::Visible,
                        "mole region did not follow 180 degree output");
            }},
            {"window fitting preserves aspect ratio at fullscreen and letterboxed sizes", [] {
                auto game = make(false);
                game.config_.width = 1280;
                cv::Mat source(720, 1280, CV_8UC3, cv::Scalar::all(255));
                auto displayed = game.fitForWindow(source, {2560, 1440});
                require(displayed.size() == cv::Size(2560, 1440) &&
                        game.displayedContent_ == cv::Rect(0, 0, 2560, 1440), "16:9 fullscreen did not fill uniformly");
                displayed = game.fitForWindow(source, {1920, 1200});
                require(game.displayedContent_ == cv::Rect(0, 60, 1920, 1080) &&
                        displayed.at<cv::Vec3b>(0, 0) == cv::Vec3b(0, 0, 0) &&
                        displayed.at<cv::Vec3b>(60, 0) == cv::Vec3b(255, 255, 255),
                        "different-aspect window stretched content instead of adding bars");
                const cv::Mat oldCanvas(720, 960, CV_8UC3, cv::Scalar::all(255));
                game.fitForWindow(oldCanvas, {2560, 1440});
                require(game.displayedContent_ == cv::Rect(320, 0, 1920, 1440), "legacy 4:3 canvas was stretched");
            }},
            {"window fitting maps scaled mouse input and excludes black bars", [] {
                auto game = make(false);
                game.config_.width = 1280;
                game.config_.outputRotationCcw = 180;
                game.fitForWindow(game.rotateForOutput(game.render()), {1920, 1200});
                game.onMouse(960, 555);
                require(cv::norm(game.pointer_ - cv::Point2f(639, 389)) < 0.01 && game.inputInside_,
                        "scaled mouse input did not map back through padding and 180 degree rotation");
                game.onMouse(960, 10);
                require(!game.inputKnown_, "black bar was accepted as a game coordinate");
                game.config_.outputRotationCcw = 0;
                game.onMouse(960, 555);
                require(cv::norm(game.pointer_ - cv::Point2f(640, 330)) < 0.01, "scaled unrotated mouse input is incorrect");
            }},
            {"state marker geometry stays outside the fixed hit region", [] {
                auto game = make(false);
                const auto markers = moleStateMarkers(game.gameSize(), 85);
                require(markers.size() == 2, "two state markers were not generated");
                const cv::Rect hitBounds(395, 305, 171, 171);
                for (const auto& rect : markers) require((rect & hitBounds).empty(), "state marker overlaps hit region");
                const auto regions = moleObservationRegionsInCamera(cv::Mat::eye(3, 3, CV_64F), game.gameSize(), 85, true);
                game.phase_ = Game::Phase::Active;
                require(observeMoleRegions(game.render(), regions, {}, {}).state == MoleVisibility::Visible,
                        "rendered markers did not turn on with target");
                game.phase_ = Game::Phase::Waiting;
                require(observeMoleRegions(game.render(), regions, {}, {}).state == MoleVisibility::Absent,
                        "markers remained visible after target hid");
            }},
            {"state marker aggregation handles partial occlusion and conflicts", [] {
                const std::vector<std::vector<cv::Point2f>> regions{
                    {{20, 40}, {60, 40}, {60, 80}, {20, 80}},
                    {{140, 40}, {180, 40}, {180, 80}, {140, 80}}};
                cv::Mat frame(120, 200, CV_8UC3, cv::Scalar(0, 200, 255));
                require(observeMoleRegions(frame, regions, {{10, 30, 60, 60}}, {}).state == MoleVisibility::Visible,
                        "one blocked marker discarded clear positive evidence from other marker");
                require(observeMoleRegions(frame, regions, {{0, 0, 200, 120}}, {}).state == MoleVisibility::Unknown,
                        "both blocked markers created a known state");
                frame(cv::Rect(130, 30, 60, 60)).setTo(cv::Scalar(0, 0, 0));
                require(observeMoleRegions(frame, regions, {}, {}).state == MoleVisibility::Unknown,
                        "conflicting markers were silently accepted");
                frame.setTo(cv::Scalar(0, 0, 0));
                require(observeMoleRegions(frame, regions, {{10, 30, 60, 60}}, {}).state == MoleVisibility::Absent,
                        "one clear absent marker could not close observation");
            }},
            {"state marker evidence permits central hand entry without holding stale state", [] {
                auto game = make(true, true); startObserved(game);
                const auto transform = cv::Mat::eye(3, 3, CV_64F);
                const auto regions = moleObservationRegionsInCamera(transform, game.gameSize(), 85, true);
                const std::vector<cv::Rect> hand{{310, 220, 340, 340}};
                const auto frame = game.render();
                require(observeMole(frame, moleRegionInCamera(transform, game.gameSize(), 85), hand, {}).state == MoleVisibility::Unknown,
                        "fixture should hide original center region");
                const auto evidence = observeMoleRegions(frame, regions, hand, {});
                game.onInput(inside(), time(1150), time(1155), 42, evidence.state, evidence.reason);
                require(game.hits_ == 1, "clear same-frame state markers did not support an entry");
                auto blocked = make(true, true); startObserved(blocked);
                const auto unknown = observeMoleRegions(frame, regions, {{0, 0, 960, 720}}, {});
                blocked.onInput(inside(), time(1150), time(1155), 42, unknown.state, unknown.reason);
                require(blocked.hits_ == 0 && blocked.unjudged_ == 1 &&
                        blocked.lastResult_.find("state_markers_occluded") != std::string::npos,
                        "hidden markers reused previous visible state or lost diagnostic reason");
            }},
            {"inset calibration targets surround the hit center with correct label mapping", [] {
                const auto targets = calibrationTargets({1280, 720});
                const std::vector<cv::Point2f> expected{{320, 210}, {960, 210}, {960, 570}, {320, 570}};
                require(targets == expected, "inset targets are not centered around the hit region");
                const std::vector<cv::Point2f> clicked{{250, 170}, {450, 170}, {450, 350}, {250, 350}};
                const auto transform = cameraToGameTransform(clicked, {720, 540}, {1280, 720});
                std::vector<cv::Point2f> mapped;
                cv::perspectiveTransform(clicked, mapped, transform);
                for (int i = 0; i < 4; ++i)
                    require(cv::norm(mapped[i] - targets[i]) < 0.01, "inset labels mapped to old edge targets");
                cv::perspectiveTransform(std::vector<cv::Point2f>{{350, 260}}, mapped, transform);
                require(cv::norm(mapped[0] - cv::Point2f(640, 390)) < 0.01, "inset mapping shifted hit center");
            }},
            {"inset calibration rejects old or missing target layout", [] {
                const std::string path = "inset-calibration-test.yml";
                const std::vector<cv::Point2f> points{{250, 170}, {450, 170}, {450, 350}, {250, 350}};
                const auto write = [&](const std::vector<cv::Point2f>& targets) {
                    cv::FileStorage file(path, cv::FileStorage::WRITE);
                    file << "camera_size" << cv::Size(720, 540) << "game_size" << cv::Size(1280, 720)
                         << "output_rotation_ccw" << 0 << "camera_serial" << "inset-test" << "camera_points" << points;
                    if (!targets.empty()) file << "game_points" << targets;
                };
                auto game = make(); game.config_.width = 1280;
                game.cameraSize_ = {720, 540}; game.cameraId_ = "inset-test";
                write({{128, 108}, {1152, 108}, {1152, 612}, {128, 612}});
                game.loadCalibration(path);
                require(game.cameraToGame_.empty(), "old edge-based calibration was accepted for inset targets");
                write({}); game.loadCalibration(path);
                require(game.cameraToGame_.empty(), "unversioned calibration was accepted for new target layout");
                write(calibrationTargets({1280, 720})); game.loadCalibration(path);
                require(!game.cameraToGame_.empty(), "new inset calibration could not be loaded");
            }},
            {"target brightness scales target pixels and clamps adjustment", [] {
                auto game = make(false);
                game.phase_ = Game::Phase::Active;
                for (const auto& style : {"mole", "square"}) {
                    game.config_.targetStyle = style;
                    game.config_.targetBrightness = 100;
                    const auto bright = game.render();
                    game.config_.targetBrightness = 50;
                    const auto dim = game.render();
                    require(bright.at<cv::Vec3b>(390, 480) == cv::Vec3b(0, 200, 255) &&
                            dim.at<cv::Vec3b>(390, 480) == cv::Vec3b(0, 100, 128), "target color did not dim uniformly");
                    require(bright.size() == dim.size(), "brightness changed target geometry");
                }
                game.config_.targetStyle = "mole";
                require(game.render().at<cv::Vec3b>(411, 480) == cv::Vec3b(100, 118, 128), "mole face highlight stayed at full brightness");
                game.adjustTargetBrightness(-1000);
                require(game.config_.targetBrightness == 10, "brightness fell below supported range");
                game.adjustTargetBrightness(1000);
                require(game.config_.targetBrightness == 100, "brightness exceeded supported range");
            }},
            {"target brightness is immediate in probe and stable within a scored trial", [] {
                Game probe(GameConfig{}, GameMode::ObservationProbe);
                probe.phase_ = Game::Phase::Active;
                probe.config_.targetBrightness = 40;
                require(probe.render().at<cv::Vec3b>(390, 480) == cv::Vec3b(0, 80, 102), "probe brightness did not change immediately");
                auto game = make(); start(game);
                game.adjustTargetBrightness(-50);
                require(game.render().at<cv::Vec3b>(390, 480) == cv::Vec3b(0, 200, 255), "brightness changed during a scored trial");
                input(game, inside(), 1150, 1155);
                input(game, outside(), 1260, 1261);
                game.update(time(1262)); game.update(time(1362)); game.submitFrame(time(1362));
                require(game.render().at<cv::Vec3b>(390, 480) == cv::Vec3b(0, 100, 128), "next trial did not use requested brightness");
            }}
        };
        std::size_t passed = 0;
        for (const auto& [name, check] : cases) {
            if (!filter.empty() && std::string(name).find(filter) == std::string::npos) continue;
            check();
            ++passed;
            std::cout << "PASS: " << name << '\n';
        }
        require(passed != 0, "no test matched MOLE_TEST_FILTER");
        std::cout << passed << " cases passed\n";
    }
};

int main() {
    try {
        const char* filter = std::getenv("MOLE_TEST_FILTER");
        GameTestAccess::run(filter ? filter : "");
        return 0;
    }
    catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
