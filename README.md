# MoleGame

一个坑的投影打地鼠实验原型，使用 C++、OpenCV、Basler pylon 和已有 YOLO 手模型。

当前提供单坑鼠标预览、独立 YOLO 相机预览，以及四点标定后的手部内部计时原型和事件日志。
相机观测实际地鼠显示状态的方法尚未实现；实机识别、标定和投影判定仍待验证。
鼠标成绩与鼠标进入时间仅用于检查原型，不是人的手部反应实验结果。

## 开始运行

在 VS Code 中打开 **整个 `D:\project\MoleGame` 文件夹**。按 F5，选择
`MoleGame mouse preview`；相机测试选择 `MoleGame camera preview`；
手部内部计时选择 `MoleGame YOLO internal timing`。

也可以从项目根目录执行：

```powershell
# 构建并打开单坑鼠标预览
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1 -Run

# 构建并打开 Basler + YOLO 手部预览
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1 -CameraPreview

# 构建并运行 YOLO 手部输入 + 内部计时
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1 -CameraGame
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

## YOLO 手部输入与内部计时

设备接好后，先用 `-CameraPreview` 检查真实手框和控制台中的 `hands`、
`detector_total_ms`、`capture_to_preview_submit_ms`，再运行 `-CameraGame`。
仅有模型加载成功或空画面零手框，不能证明实际手部检测正确。

1. 把游戏窗口放到投影输出，先用 F 确定全屏状态；相机预览留在电脑屏幕上。
2. 在**相机预览**里依次点击投影标签 **1、2、3、4** 的十字中心。
   点击顺序对应标签编号，不按照相机图像中的上下左右重新排序。
3. 标定后，预览中蓝色轮廓表示固定击打区域，游戏画面的红点表示映射后的手中心。
   用一只手检查两者位置是否一致，再按 Enter 确认并开始。
4. 手中心先在圈外，等待地鼠出现，再进入圈内。停留不重复计分，提前进入取消等待。

H 重新标定。移动/缩放游戏窗口或切换全屏会清除标定并要求重新确认。
相机、镜头、投影仪或投影面发生移动时也应按 H；程序无法自动检测这些物理变化。
本机标定保存到忽略提交的 `config/camera-calibration.yml`，覆盖前保存 `.bak`。
再次启动会读取相同设备和分辨率的标定，但仍要求检查红点后按 Enter 确认。

该模式只使用一只手。零手、两只手、过期输入或过大取帧间隔会中断进入判断，
不通过插值或漏检保持生成动作；重新检测到圈内的手不会直接算命中。
没有可靠进入动作且试次中存在输入缺口时记为 **UNJUDGED**，与明确的 **MISS** 分开。
这是单手原型，没有加入跨帧手部身份跟踪。

采集和 YOLO 在后台线程依次执行，游戏显示在主线程更新。
内部接受窗口为 `[首帧提交时间, 首帧提交时间 + display_ms)`，
用手点所在图像转换并复制后的主机时间判断，而不是推理完成或主线程接收时间。
窗口结束立即隐藏地鼠；最多等待 `input_timeout_ms` 接收在途结果，**不延长接受窗口**。
该参数也限制输入间隔和样本接收延迟，默认 200 ms；它是原型可靠性阈值，不是显示延迟补偿。
超过时限无法取得结果、队列溢出或相机错误时保留无法判定结果。

每次运行保存 `outputs/internal-<会话编号>/events.csv`、配置快照、随机种子及确认的标定。
CSV 分开记录 `sample_ms` 和 `received_ms`、帧号、手数、映射手点、内部窗口与结果；
事件中的试次编号不随 Space 重置成绩而重复。时间均相对会话起点。
`sample_ms` **不是相机曝光时刻**，也没有测量实际投影出现/消失时间；
结果晚到时，显示反馈的延迟仍然存在。正式实验前仍需实际投影录像和时序验证。

不连接设备的自动测试：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1 -Test
```

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `src/main.cpp` | 选择鼠标预览、相机预览、手部内部计时或模型检查 |
| `src/Game.h/.cpp` | 共用进入规则、显示与内部计时，标定交互和事件记录 |
| `src/CameraInput.h/.cpp` | 后台 Basler 采集、YOLO 原始手框和样本时间戳，四点映射 |
| `src/vision/YoloHandTracker.h/.cpp` | 从旧项目复制的 TensorRT / OpenCV DNN 推理实现 |
| `config/game.yml` | 窗口、区域和出没时间 |
| `config/camera.yml`、`config/yolo.yml` | 相机和手模型参数 |
| `scripts/build.ps1` | 本机依赖环境、CMake / Ninja 构建和运行 |
| `tests/GameTests.cpp` | 无设备的计时、输入缺口、CSV 和标定边界测试 |

默认流程：`main.cpp` → `Game::run()`。
相机流程：`main.cpp` → `runCameraPreview()` → Basler 原始 BGR 帧 → YOLO → 同帧手框。
独立相机预览不做标定；相机游戏必须先标定和确认，再使用映射手点。

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
预览的 `capture_to_preview_submit_ms` 从图像转换并复制后计时，不包含相机曝光、读出和传输，
也不是投影面可见反馈延迟。采集/推理与窗口分线程，取帧和推理仍在同一后台线程顺序执行。

## 下一步

1. 实机确认相机预览和手检测正常。
2. 实机检查四点标定、手部进入和内部窗口日志。
3. 加入固定坑的实际地鼠显示状态识别，并验证手遮挡。
4. 两种判定共用同一手部动作与采集帧，并用实际投影录像核对。

研究问题、命中定义及评价方式见 [研究准备](docs/whack_a_mole_plan.md)。

2026-10-07 基础验证：独立目录的 Release 和仅 OpenCV 构建通过；GPU 模型图片检查通过。
鼠标 GUI 已验证随机出现、命中、超时、停留不刷分、调时长、重置和正常退出。
真实 Basler 采集与投影实验尚未在新项目中验证。

2026-10-15 验证：相机版和仅 OpenCV 版 Release 构建通过；19 个确定性场景通过，
覆盖延迟结果、截止边界、输入缺口、CSV、会话快照及四点映射。
TensorRT 合成空图推理通路检查通过；未接相机时能报告设备不可用并正常退出。
当天没有连接设备，未验证真实手部识别、标定交互或投影判定效果。
