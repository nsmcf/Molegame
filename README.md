# MoleGame

开发约定见 [AGENTS.md](AGENTS.md)：仅在用户明确要求时编写或运行测试，普通开发不预设测试待办。

一个坑的投影打地鼠实验原型，使用 C++、OpenCV、Basler pylon 和已有 YOLO 手模型。

当前提供单坑鼠标预览、独立 YOLO 相机预览，以及四点标定后的内部计时和同帧相机观测两种评分。
另有不计分的地鼠颜色观测入口，供调整真实投影和遮挡条件下的识别。
两种评分的代码流程已接通。
鼠标成绩与鼠标进入时间仅用于检查原型，不是人的手部反应实验结果。

## 开始运行

在 VS Code 中打开 **整个 `D:\project\MoleGame` 文件夹**，在“运行和调试”中选择入口后按 F5：

| 启动项 | 用途 |
| --- | --- |
| `MoleGame practice` | 运行独立的 `practice.cpp` 练习 |
| `MoleGame YOLO internal timing` | 手部内部计时评分 |
| `MoleGame YOLO camera observation` | 同帧手部与地鼠观测评分 |
| `MoleGame same-frame observation probe` | 不计分的观测调试，可保存原图 |

两种评分模式目前在启动时选择；切换时先按 Esc 退出，再选择另一启动项并按 F5。
启动前会自动检查构建，代码没有变化时无需重新编译。

也可以从项目根目录执行：

```powershell
# 构建并打开单坑鼠标预览
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1 -Run

# 构建并打开 Basler + YOLO 手部预览
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1 -CameraPreview

# 构建并运行 YOLO 手部输入 + 内部计时
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1 -CameraGame

# 构建并运行同帧手部 + 地鼠观测评分
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1 -ObservedGame

# 构建并运行同帧手部 + 地鼠观测（不计分）
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1 -ObserveMole
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
| R | 原方向 / 180° 对面观看切换；相机模式会要求重新标定 |
| ↑ / ↓ | 下一个地鼠的显示时间增加 / 减少 50 ms |
| ← / → 或 [ / ] | 地鼠亮度减少 / 增加 10%，范围 10～100% |

默认显示时长是 **500 ms**，可在 `config/game.yml` 修改；200 ms 暂未定为实验参数。
显示计时从程序提交首帧开始，不等于投影面实际可见时刻。

`target_brightness` 设置启动时的地鼠亮度百分比，当前配置为 50；按左右箭头（或方括号）实时调整。
观测模式立即生效；按 V 可保持地鼠常亮。
计分模式当前回合保持原亮度，调整从下一回合生效，画面显示当前值及待应用值。
调整不写回配置文件；调节事件保存在现有日志中，合适的启动值可手动写入 config/game.yml。
该调节只缩放目标图案的颜色强度，不修改相机曝光或 YOLO 阈值；若启用同步标记，标记与目标一起调暗。
两种评分对比时使用相同亮度。

当前按参与者站在相机对面操作：游戏旋转 180°，参与者看到的布局仍为横向宽幅。
`config/game.yml` 中 `output_rotation_ccw: 180` 选择该方向，`0` 为原方向；
`90` 保留为之前从右侧观看的布局，仅通过配置选择。
运行时按 R 可临时切换，本次切换不写回配置文件；鼠标预览切换时清空成绩。
当前游戏画布为 1280 × 720（16:9），在 2560 × 1440 屏幕上可等比放大两倍铺满。
程序先按观看方向布局、旋转，再按实际窗口尺寸等比缩放；其他宽高比会居中补黑边，
保持地鼠和命中圆的比例。Windows 进程使用每显示器 DPI 感知，以实际像素处理全屏和鼠标位置。
此设置只影响本程序，不修改系统分辨率或显示缩放。改变画布尺寸、窗口或全屏状态后重新标定。

相机预览按 `config/camera.yml` 的 `undistort` 设置显示原图或去畸变图。
四点映射负责相机与游戏坐标对应；投影仪梯形校正和相机斜拍仍会影响画面外观。

## YOLO 手部输入与内部计时

`-CameraPreview` 打开独立相机预览，`-CameraGame` 启动手部内部计时评分。

1. 把游戏窗口放到投影输出，用 R 确定方向、F 确定全屏状态；相机预览留在电脑屏幕上。
2. 在**相机预览**里依次点击投影标签 **1、2、3、4** 的十字中心。
   点击顺序对应标签编号，不按照相机图像中的上下左右重新排序。
   四个十字已向击打中心收拢，覆盖画布中间约一半宽高；十字放大到 60 像素并加粗，编号也已放大。
3. 点完四个十字后，相机预览显示蓝色轮廓，游戏显示映射手点。
   CENTER 是参考圆，不是第五个标定点；按 Enter 开始，按 H 重新选择四点。
4. 手中心先在圈外，等待地鼠出现，再进入圈内。停留不重复计分，提前进入取消等待。

H 重新标定。旋转、移动/缩放游戏窗口或切换全屏会清除标定并要求重新确认。
旋转后仍按投影标签 1、2、3、4 点击，不按相机画面中的上下左右排序。
投影输出在最后一步旋转，手部映射、命中圆和地鼠检测区域共用原来的游戏坐标。
相机、镜头、投影仪或投影面发生移动时也应按 H；程序无法自动检测这些物理变化。
本机标定保存到忽略提交的 `config/camera-calibration.yml`，覆盖前保存 `.bak`。
再次启动会读取相同设备、分辨率、旋转方向和标定点布局的标定，按 Enter 开始。
旧的靠边标定不会复用到新的内收十字位置，首次运行需重新点选四个十字。

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
结果晚到时，显示反馈的延迟仍然存在。

## 相机观测评分

运行 `-ObservedGame`，对应程序参数 `--camera-observed-game`。
与内部计时共用四点标定、单手检测、固定击打区域、随机等待和显示时长；
两种评分均在后台对同帧原图执行 YOLO 和地鼠颜色检测，再按顺序处理所有已产生的样本。

| 判定 | 内部计时 `-CameraGame` | 相机观测 `-ObservedGame` |
| --- | --- | --- |
| 首次从圈外进入 | 采样时间在内部窗口内才命中 | 进入帧中的颜色证据为 VISIBLE 才命中，当前直接检测地鼠中心 |
| 内部窗口结束、投影仍可见 | 不再接受窗口外动作 | 等待实际消失，期间仍可接受同帧 VISIBLE 的进入 |
| 进入时地鼠已不可见 | 由内部窗口判定 | 明确 ABSENT 为 MISS，UNKNOWN 为 UNJUDGED |
| 零手、多手、过期或间隔过大 | 中断进入连续性 | 相同，不用插值补出进入动作 |

相机模式要求手先在圈外，并且地鼠区域明确为 ABSENT，才安排下一次随机等待。
程序仍按 `display_ms` 控制投影 ON/OFF；相机评分不会把内部隐藏时刻当成实际投影消失时刻。
观察到 VISIBLE 后再观察到 ABSENT，则结束无命中的试次；中途证据不完整时记 UNJUDGED。
若始终没观察到出现、没有可靠消失、或进入时地鼠中心被遮挡而无法确认，仍保留无法判定结果。

当前设置 `state_markers: 1`：在地鼠左右两侧显示两个黄色方块，与目标在同一张输出帧中同时显示/隐藏，
两种评分使用完全相同的画面。相机从动作所在帧的标记读取投影显示状态，避免中心地鼠被手框遮住就失去证据。
至少一个标记清楚、且所有清楚标记的状态一致时才采用该状态；矛盾或都不明确时为 UNKNOWN。
这属于通过同步标记观察显示状态，不声称看清了手下面的地鼠，也不沿用上一帧的 VISIBLE。
设为 `state_markers: 0` 并重启可恢复中心颜色区域检测。

UNJUDGED 现在会显示具体原因，并写入现有调试事件：
`no_hand` / `multiple_hands` 表示没有唯一手点，`sample_too_late` / `frame_gap` 表示时效或连续性不足，
`state_markers_occluded` 表示标记被遮住，`state_marker_conflict` 表示标记状态冲突，
`mole_observation_timeout` 表示没有在上限内确认投影出现/消失。

`observation_timeout_ms` 限制内部隐藏后等待图像证据的时长，默认 1000 ms，
另允许最多 `input_timeout_ms` 接收在途结果；超过观察边界的采样不能得分。
这是异常保护上限，不是显示延迟补偿，也不保持或延长地鼠的 VISIBLE 状态。
设备正常时，仍由动作所在帧的实际颜色证据决定命中。

相机评分会话的现有调试日志保存在 `outputs/camera-<会话编号>/`，与内部计时目录区分。
目前暂缓实验成绩自动汇总与分析功能。

## 同帧地鼠观测

运行 `-ObserveMole`，或在 F5 中选择 `MoleGame same-frame observation probe`。
把游戏窗口放到投影输出，按前述步骤完成四点标定和 Enter 确认。
此入口不计分，可显示颜色观测状态并保存图片。

| 按键 | 操作 |
| --- | --- |
| V | 手动切换地鼠 ON/OFF，并停止自动循环 |
| A | 开启/停止自动循环；ON 时长为 `display_ms`，OFF 时长为随机等待时间 |
| S | 保存同帧原图和检测预览；开启去畸变时另存校正图 |
| C | 开启/停止自动采图，每 0.5 秒保存一组原图和检测预览，默认关闭 |
| H / F / Esc | 重新标定 / 全屏切换并重新标定 / 退出 |
| R | 切换原方向 / 180° 对面观看，停止自动循环并重新标定 |
| ↑ / ↓ | 调整后续 ON 时长 50 ms |
| ← / → 或 [ / ] | 立即调整地鼠亮度，每次 10%，画面显示 Brightness 百分比 |

采集训练图片时，先按 V 保持地鼠显示，再按 C 开启自动采图，此时可以持续移动手；
再次按 C 停止。S 仍可随时手动保存，V 只控制地鼠显示，不负责截图。
自动和手动采图均写入项目下同一个 `outputs/observation-<会话编号>/` 目录，
每组包含 `*-raw.png` 原图和 `*-preview.png` 检测预览；开启去畸变时另存 `*-undistorted.png`。
相机预览显示自动采图状态和本次已保存组数；重新标定期间暂停采集，确认后继续。
没有新帧、帧过期或标定版本不符时跳过，不重复保存同一帧，也不补拍积压时段。
YOLO 漏检或没有手时仍可保存，便于收集失败样本和负样本；PNG 写入耗时和相机帧率可能使实际间隔超过 0.5 秒。

紫色轮廓是颜色观测区域：当前对应两侧同步标记；偏移时可按 H 重新标定。
关闭标记后，轮廓对应固定击打中心、半径为击打半径 0.70 倍的区域；此时中心被手框完全覆盖仍会 UNKNOWN。
识别使用画框前的 BGR 原图，以 OpenCV HSV 颜色分割统计区域内的黄色比例；
排除同帧 YOLO 手框（带边距）覆盖的像素。每个区域颜色足够时为 `VISIBLE`，
无遮挡且颜色很少时为 `ABSENT`，中间比例、区域越界、可用面积不足或遮挡下缺少颜色时为 `UNKNOWN`。
标记模式再合并两个区域的本帧证据；没有历史状态保持。
不使用内部 ON/OFF 状态帮助识别，也不把上帧状态延长到本帧。
漏检的手、阴影、同色物体和实际投影颜色变化仍可能导致错误；这三种标签是检测输出，不能作为真值。
HSV 和比例阈值在 `config/mole.yml`，修改后重启，每次会话保存参数快照。

实机已出现 YOLO 将带五官地鼠误检为手、继而整块排除地鼠区域的情况。
曾使用无五官黄色方块排查此问题，目前按使用要求恢复 `config/game.yml` 的 `target_style: mole`；
`square` 仍可用于诊断。两种评分使用相同图案。
绿色框标注为 `hand <置信度>`，用于辨认手模型输出；不能仅凭位于地鼠区域就丢弃手框。

标定时显示 NOT_READY，点完十字并确认 CENTER 后按 Enter 才进入检测/评分；过期样本另行显示原因。
连续出现 UNJUDGED 时，先检查预览的 mole 状态和下一行原因：
`occluded` / `occluded_without_color` 表示手框遮挡后证据不足，`color_ambiguous` 表示颜色比例不明确，
`invalid_region` / `region_outside_frame` 表示区域几何或标定有问题。
当前标记模式下，手进入中心时至少一侧标记应保持清楚；两侧都遮住或出现 `state_marker_conflict` 时保留 UNKNOWN。
保持一只手在相机视野内的圈外位置；零手、多手和检测间断也会影响判定。
投影上只显示游戏窗口，避免把相机预览本身投影出去形成重复画面。

每次运行保存到 `outputs/observation-<会话编号>/`：

- `observations.csv`：所处理预览帧的编号、采样/接收时间、手数、状态、颜色/可用面积比例、原因，
  YOLO 耗时、地鼠检测耗时和主机采样至接收耗时分别记录。
- `events.csv`：标定、内部显示提交 ON/OFF、手动/自动操作、截图及退出事件。
- `frame-<帧号>-raw.png`、`frame-<帧号>-preview.png`：按 S 手动或按 C 开启自动采集保存的同帧原图与检测预览。
- `session.yml`、配置及标定快照：标记本次为 `observation_probe`，评分关闭。

手框、原图和观测结果按同一帧绑定；颜色检测在后台对每个已处理的采样帧执行，
主线程顺序接收结果，预览只显示最新图像。队列溢出会记录 QUEUE_OVERFLOW。
相机使用最新帧采集策略，不能将此 CSV 当成全部曝光帧序列或显示延迟的精确测量。
采样时间仍为转换复制后的主机时间。截图用于排查和汇报。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `src/main.cpp` | 通过 GameMode 选择两种评分、预览或模型检查 |
| `src/Game.h/.cpp` | 共用进入规则与显示，两种判定、标定交互和调试事件 |
| `src/CameraInput.h/.cpp` | 后台 Basler 采集、同帧 YOLO/地鼠检测、标定区域版本及样本时间戳 |
| `src/vision/YoloHandTracker.h/.cpp` | 从旧项目复制的 TensorRT / OpenCV DNN 推理实现 |
| `src/vision/MoleDetector.h/.cpp` | 同帧原图的地鼠颜色证据、手框排除和三状态输出 |
| `config/game.yml` | 窗口、区域和出没时间 |
| `config/camera.yml`、`config/yolo.yml` | 相机和手模型参数 |
| `config/mole.yml` | 地鼠 HSV、颜色比例及遮挡阈值候选值 |
| `scripts/build.ps1` | 本机依赖环境、CMake / Ninja 构建和运行 |

默认流程：`main.cpp` → `Game::run()`。
相机流程：`main.cpp` → `runCameraPreview()` → Basler 采集 → 按配置去畸变 → YOLO → 同帧手框。
独立相机预览不做四点标定；相机游戏标定后使用映射手点。

## 复用模型

模型路径在 `config/yolo.yml`。当前默认使用补充训练后的 YOLOv10n，
加载 `models/MoleGame-finetune-20261108-deploy/hand_detector_finetuned_416_fp16_io.engine`，
使用 GPU 预处理、筛选及 CUDA graph。该引擎由转入的 ONNX 在本机 GPU 上构建。
当前根据已有数据集结果选择 416。

旧项目的 ONNX、三个 engine 和类别文件仍保留在 `models/` 根目录，
旧配置备份在 `outputs/model-deploy-20261110/yolo-original.yml`，可用于回退。
已有指标、逐图记录位置和回退步骤见 [模型接入记录](docs/model_deployment.md)。
engine 与硬件、TensorRT 环境相关；保留 ONNX 作为重新构建来源。
模型二进制及部署 ZIP 不纳入 Git。

相机预览关闭平滑和漏检保持，使用原始检测框；输入转换后复制图像，识别后才画框。
预览的 `capture_to_preview_submit_ms` 从图像转换并复制后计时，不包含相机曝光、读出和传输，
也不是投影面可见反馈延迟。采集/推理与窗口分线程，取帧和推理仍在同一后台线程顺序执行。

研究问题、命中定义及评价方式见 [研究准备](docs/whack_a_mole_plan.md)。

## 试用镜头去畸变（2026-11-17）

`config/camera.yml` 的 `undistort: 1` 启用本次 Basler_Pic 棋盘格试算参数，
`intrinsics_file` 指向 `config/camera-intrinsics-trial.yml`。
采用收紧视野的 720×540 输出，和 `before-after-valid-view.png` 的显示方式一致。
这份参数为当前项目使用的棋盘格试算配置。

相机帧先去畸变，再交给 YOLO 和地鼠识别；预览、手框、中心圈和四点映射使用同一坐标系。
启动时预计算映射表，逐帧只执行重映射。画面尺寸或参数无效会报错，不会默默换回原图。
相机窗口标题会显示 `(undistorted)`，控制台会显示 `[Camera] Undistortion ON`。

首次启用会拒绝复用原图上的旧四点坐标。请在新的相机预览里重新点击投影十字
**1→2→3→4**，按 **Enter** 开始。
之后修改镜头参数或切换开关，也会要求重新进行四点标定。
如需关闭，设置 `undistort: 0` 并重启。

观测模式的 S／C 采集仍将原始传感器图像保存为 `*-raw.png`；
启用校正时额外保存无 UI 的 `*-undistorted.png`，检测预览保存为 `*-preview.png`。
手框、圈和观测日志坐标对应去畸变画面；`*-raw.png` 保留供镜头标定使用。
每次会话记录坐标系标识，并备份实际使用的镜头参数。
