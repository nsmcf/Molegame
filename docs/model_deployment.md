# 补充训练模型接入记录（2026-11-10）

当前配置已切换到 `models/MoleGame-finetune-20261108-deploy/hand_detector_finetuned_416_fp16_io.engine`，即补充训练后的 YOLOv10n，输入 416×416、输出 1×300×6，FP16 输入输出。
CUDA graph 和 GPU 预处理/筛选均成功启用。旧模型文件仍在 models/ 根目录，启动配置备份在 outputs/model-deploy-20261110/yolo-original.yml。

## 文件与环境

- 新 ONNX、PT、类别文件位于 models/MoleGame-finetune-20261108-deploy/；转入包内 7 个文件的 SHA-256 均核验通过。
- 416 与 640 engine 都在本机 RTX 3060 Laptop GPU 上用 TensorRT 8.6.1.6 构建；没有复制另一台电脑的 engine。
- 构建启用 FP16、线性 CHW 输入输出、2 个辅助流、1024 MiB workspace、builderOptimizationLevel=3。
- 构建进程需要 TensorRT lib、CUDA bin、cuDNN bin 在当前进程 PATH 中；没有修改系统 PATH。
- 构建日志、部署信息、逐图 CSV、对照预览、评测源码均保存于 outputs/model-deploy-20261110/。

## 对比方法

使用项目现有 YoloHandTracker 推理路径，multiHandTracking=true，每帧最多保留两只手。
新旧模型均固定置信度 0.65、NMS 0.45；人工框与预测框按 IoU≥0.5 一对一匹配。
TP 为匹配框，FP 为未匹配预测框，FN 为未匹配人工框。定位偏差也可能同时产生 FP 与 FN，因此 FN 不完全等于整只手未检出。
下列 Precision/Recall 是单一阈值下的框匹配指标，不是 mAP；耗时为 detector_total_ms，不是采集到显示的端到端延迟。
使用新增数据的 156 张验证图选择尺寸，最后使用此前留出的 89 张测试图比较所选新模型与旧模型。没有用测试集选择尺寸或调整阈值。

## 验证集结果（156 张、196 个人工手框）

| 模型 | TP | FP | FN | Precision | Recall | F1 | 整图全部框匹配 | 平均检测 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old416-val | 74 | 8 | 122 | 90.2% | 37.8% | 53.2% | 59/156 | 1.315 |
| new416-val | 163 | 1 | 33 | 99.4% | 83.2% | 90.6% | 123/156 | 1.271 |
| new640-val | 157 | 3 | 39 | 98.1% | 80.1% | 88.2% | 114/156 | 2.231 |

选择 `new416`：验证集框匹配 F1 为 90.6%，高于 640 版的 88.2%；整图匹配 123/156，高于 114/156，平均检测耗时也更低。

## 留出测试结果（89 张、106 个人工手框）

| 模型 | TP | FP | FN | Precision | Recall | 整图全部框匹配 | 有手却零框的图片 | 平均检测 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old416-test | 70 | 3 | 36 | 95.9% | 66.0% | 55/89 | 22 | 1.683 |
| new416-test | 103 | 0 | 3 | 100.0% | 97.2% | 86/89 | 1 | 1.701 |

这些图片来自同一场采集；没有重新量化第一批数据上的保持情况。
11 月 10 日的模型部署只更新手部模型，当时仍使用中心颜色检测，手遮挡中心时可能 UNKNOWN。
11 月 11 日起已启用 `config/game.yml` 的 `state_markers: 1`，从两侧同步标记观察显示状态，避免依赖被手遮住的中心。
手模型仍为 416 FP16；HSV 阈值及同帧手框排除规则不变。操作见 [同帧地鼠观测](../README.md#同帧地鼠观测)。

## 运行与回退

下次从 VS Code 启动相机模式会加载新配置。控制台 engine 名称带 finetuned，可用于确认加载的是新引擎。
回到原模型时，退出程序后执行：

```powershell
Copy-Item -LiteralPath outputs/model-deploy-20261110/yolo-original.yml -Destination config/yolo.yml
```

配置在启动时读取，切换后需要重新启动。模型二进制、部署 ZIP 和运行输出均不纳入 Git。

TensorRT 工具参考：[NVIDIA trtexec](https://github.com/NVIDIA/TensorRT/blob/main/samples/trtexec/README.md)。
