#include "Game.h"
#include "CameraInput.h"

#include <functional>
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
    static Game make(bool camera = true) {
        GameConfig config;
        config.waitMinMs = config.waitMaxMs = 100;
        config.displayMs = 100;
        config.feedbackMs = 100;
        config.inputTimeoutMs = 200;
        return Game(config, camera);
    }
    static void input(Game& game, std::optional<cv::Point2f> point, int sample, int received) {
        game.onInput(point, time(sample), time(received), static_cast<std::uint64_t>(sample));
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
    static void run() {
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
            }}
        };
        for (const auto& [name, check] : cases) {
            check();
            std::cout << "PASS: " << name << '\n';
        }
        std::cout << cases.size() << " cases passed\n";
    }
};

int main() {
    try { GameTestAccess::run(); return 0; }
    catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
