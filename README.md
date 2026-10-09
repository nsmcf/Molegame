# MoleGame

一个坑的投影打地鼠实验原型，使用 C++、OpenCV、Basler pylon 和已有 YOLO 手模型。

当前提供单坑鼠标预览和独立相机手检测预览。相机标定、地鼠显示状态识别、两种正式判定和实验日志待接入。
鼠标成绩与鼠标进入时间仅用于检查原型，不是人的手部反应实验结果。

## 开始运行

在 VS Code 中打开 **整个 `D:\project\MoleGame` 文件夹**。按 F5，选择
`MoleGame mouse preview`；相机测试选择 `MoleGame camera preview`。

也可以从项目根目录执行：

```powershell
# 构建并打开单坑鼠标预览
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1 -Run

# 构建并打开 Basler + YOLO 手部预览
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1 -CameraPreview
```

构建输出为 `build/camera-release/MoleGame.exe`。脚本只设置当前进程的 SDK 环境，
不修改系统 PATH 或 PowerShell 执行策略。
首次环境配置参考 `config/local.example.json`，本机配置位于忽略提交的 `config/local.json`。
VS Code 的 `launch.json` 也包含本机 DLL 路径；换机器时与本机配置一起调整。

只有 OpenCV 时，可以运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1 -NoCamera -Run
```

这会构建 `build/mouse-release/MoleGame.exe`，不查找相机和 GPU SDK。

## 鼠标预览

将鼠标移到圆圈外，等待地鼠出现，再移动进圆圈即可命中，不需要点击。
过早进入会取消等待；停在圈内不会自动命中或连续启动新试次。

| 按键 | 操作 |
| --- | --- |
| Esc | 退出 |
| Space | 重置成绩 |
| F | 全屏切换 |
| ↑ / ↓ | 下一个地鼠的显示时间增加 / 减少 50 ms |

默认显示时长是 **500 ms**，可在 `config/game.yml` 修改；200 ms 暂未定为实验参数。
显示计时从程序提交首帧开始，不等于投影面实际可见时刻。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `src/main.cpp` | 选择鼠标预览、相机预览或模型检查 |
| `src/Game.h/.cpp` | 单坑显示、随机等待、鼠标进入与内部计时 |
| `src/CameraInput.h/.cpp` | 独立 Basler 原始帧采集和 YOLO 手框预览 |
| `src/vision/YoloHandTracker.h/.cpp` | 从旧项目复制的 TensorRT / OpenCV DNN 推理实现 |
| `config/game.yml` | 窗口、区域和出没时间 |
| `config/camera.yml`、`config/yolo.yml` | 相机和手模型参数 |
| `scripts/build.ps1` | 本机依赖环境、CMake / Ninja 构建和运行 |

默认流程：`main.cpp` → `Game::run()`。
相机流程：`main.cpp` → `runCameraPreview()` → Basler 原始 BGR 帧 → YOLO → 同帧手框。
此阶段相机预览没有游戏坐标标定，不能直接把预览手点当作正式击打输入。

## 复用模型

`models/` 已复制旧项目的 ONNX、三个 TensorRT engine 和类别文件。
优先加载 `hand_detector_416_fp16_io.engine`，使用 GPU 预处理和筛选。
engine 文件与硬件、TensorRT 环境相关；保留 ONNX 作为重新构建来源。
当前复用引擎会打印跨设备构建警告；本机图片推理检查已成功，正式测试前仍需核对实际手部识别。
模型二进制未纳入 Git；原始模型来源是 `D:\project\RhythmProcamPrototype\RhythmProcamPrototype\models`。

不连接相机、不打开窗口的模型检查：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1 -CheckModel "实际相机截图.png"
```

相机预览关闭平滑和漏检保持，使用原始检测框；输入转换后复制图像，识别后才画框。
预览的 `capture_to_preview_submit_ms` 从图像转换完成计时，不包含相机曝光、读出和传输，
也不是投影面可见反馈延迟。基础预览采用同步循环，正式实验前再评估采集与处理线程。

## 下一步

1. 实机确认相机预览和手检测正常。
2. 标定相机到投影区域的坐标映射。
3. 加入固定坑的实际地鼠显示状态识别，并验证手遮挡。
4. 两种判定共用同一手部动作与采集帧，再接入事件日志和实际投影录像。

研究问题、命中定义及评价方式见 [研究准备](docs/whack_a_mole_plan.md)。

2026-10-07 基础验证：独立目录的 Release 和仅 OpenCV 构建通过；GPU 模型图片检查通过。
鼠标 GUI 已验证随机出现、命中、超时、停留不刷分、调时长、重置和正常退出。
真实 Basler 采集与投影实验尚未在新项目中验证。
