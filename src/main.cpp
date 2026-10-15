#include "CameraInput.h"
#include "Game.h"

#include <opencv2/core/utils/logger.hpp>

#include <exception>
#include <iostream>
#include <string>

// 选择游戏预览或相机预览入口。
int main(int argc, char* argv[]) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "MoleGame\n"
                "  No arguments: one-hole mouse preview with internal timing\n"
                "  --camera-preview: camera and hand-detector preview\n"
                "  --camera-game: calibrated YOLO hand input with internal timing\n"
                "  --check-model <image-path>: check the hand model without a camera or GUI\n"
                "  --help: show this help\n\n"
                "Run from the project folder; settings are in config/game.yml.\n"
                "Mouse preview: enter the circle after the mole appears; no click needed.\n"
                "Esc: exit, Space: reset, F: fullscreen, Up/Down: display +/- 50 ms.\n"
                "Camera game: click projected labels 1,2,3,4 in camera preview, then Enter.\n"
                "H: recalibrate. Use one hand; missing/ambiguous input may be UNJUDGED.\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--camera-preview") return runCameraPreview();
        if (argc == 2 && std::string(argv[1]) == "--camera-game") {
            Game game(loadGameConfig(), true);
            game.run();
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--check-model") return checkHandModel(argv[2]);
        if (argc != 1) {
            std::cerr << "Unknown arguments. Use --help.\n";
            return 2;
        }
        Game game(loadGameConfig());
        game.run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MoleGame error: " << error.what() << '\n';
        return 1;
    }
}
