# EfficientPhys-C 心率测量复现（UBFC-rPPG Dataset 2）

本仓库记录基于 [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) 的 EfficientPhys-C 复现实验：完整模型、POS 人脸 ROI 基线、去空间注意力和去输入 BatchNorm 两组消融。实验使用受试者级固定划分（训练 29 人、验证 4 人、测试 9 人；测试者为 subject41–49），随机种子 100。

## 仓库内容

| 路径 | 用途 |
| --- | --- |
| `splits/fixed_subject_split.csv` | 本次使用的受试者划分 |
| `configs/*.yaml` | 完整模型与两组消融的训练、测试配置 |
| `toolbox_overlay/` | 相对 rPPG-Toolbox 提交 `b7500b848f84ad7f86e277b4612563b69f4f88f9` 修改的 5 个文件；按同名路径覆盖原文件 |
| `scripts/preprocess_raw_subjects.py` | 原始视频与 BVP 预处理，生成 181 帧输入、180 点 DiffNormalized 标签 |
| `scripts/prepare_raw_paper_experiment.py`、`scripts/check_raw_paper_setup.py`、`scripts/run_training_clean.py` | 核查缓存、准备正式配置、冒烟检查及启动完整模型训练/测试 |
| `scripts/prepare_raw_paper_ablations.py`、`scripts/run_raw_paper_ablation.py` | 生成并运行两组消融；使用相同划分和评价协议 |
| `scripts/run_pos_face_roi_baseline.py`、`scripts/run_pos_raw_paper_baseline.py` | 正式人脸 ROI POS 基线及其共用函数；**正式 POS 结果来自前者** |
| `scripts/analyze_raw_paper_subjects.py`、`scripts/plot_raw_paper_waveforms.py`、`scripts/diagnose_subject43_44.py` | 逐受试者结果、波形图和高误差受试者频谱检查 |
| `scripts/benchmark_raw_paper_inference.py` | 最终模型的单片段 GPU 前向耗时测量 |
| `scripts/make_corrected_bland_altman.py`、`scripts/make_comparison_table.py`、`scripts/make_inference_timing_card.py` | 报告可视化的辅助脚本 |
| `results/metrics_summary.csv` | 四种方法的五项心率指标与最佳验证轮次 |

原始 UBFC 视频、预处理缓存、模型权重及完整训练日志体积较大，未上传；请保留本地实验文件。本仓库的 `toolbox_overlay/` 包含 rPPG-Toolbox 派生代码，原项目许可证见 `LICENSE.rPPG-Toolbox.txt`。

## 关键实现

- 输入是完整 RGB 视频帧，缩放到 72×72，标称采样率设为 30 Hz。每段 181 帧产生 180 组相邻帧差，对齐 180 点标签。没有对视频进行外部标准化或人脸裁剪；模型内部执行帧差与输入 BatchNorm。
- 训练采用负 Pearson 损失、AdamW、学习率 0.001、权重衰减 0.01、batch 4、10 轮。每组实验独立选取验证损失最低的轮次，测试集仅作最终评价。权重衰减 0.01 是本次实现值。
- 测试阶段对预测和真实的差分波形分别重建、去趋势和 0.75–2.5 Hz 带通滤波，然后在同一频段用整位受试者有效波形的 FFT 主峰估计心率。SNR 沿用 Toolbox 算法，背景频率统计范围也设为 0.75–2.5 Hz。
- POS 使用相同 9 名测试受试者和有效时间范围，但在原视频上构建人脸 ROI；`run_pos_raw_paper_baseline.py` 仅为共用函数模块，其早期直接用整帧的结果不计入正式对照。
- 消融通过环境变量 `EFFICIENTPHYS_ABLATION` 选择 `full`、`no_attention` 或 `no_bn`。`no_attention` 跳过两处空间注意力加权，`no_bn` 跳过模型输入 BatchNorm。

## 复现步骤（Windows PowerShell）

以下脚本保留了本次实验的绝对路径：Toolbox `D:\rppg\rPPG-Toolbox`、原始视频 `D:\Microsoft  edge`、缓存与结果 `G:`。在其他机器运行前，先修改脚本顶部的路径常量和三个 YAML 中的路径。软件环境记录为 Python 3.12.10、PyTorch 2.7.1+cu128、CUDA 12.8、NumPy 1.26.4、SciPy 1.11.4、OpenCV 4.8.1。

1. 取得 rPPG-Toolbox 并固定到上面的提交。将 `toolbox_overlay/` 中 5 个文件按相同相对路径覆盖到 Toolbox。`main.py` 中 DataLoader 的 `num_workers=0` 及 `trainer/__init__.py` 中关闭 PhysMamba 导入，是这台 Windows 环境的兼容性改动。将 `splits/fixed_subject_split.csv` 复制到 `D:\rppg\EfficientPhysC_UBFC_Dataset2_subject_split.csv`。
2. 从仓库根目录运行下面命令。已有本次缓存和模型时无需重跑训练；预处理脚本会保护现有配置，消融脚本会避免覆盖已有模型。

```powershell
$py = 'D:\rppg\.venv\Scripts\python.exe'
& $py -X utf8 .\scripts\preprocess_raw_subjects.py --source 'D:\Microsoft  edge' --run
& $py -X utf8 .\scripts\prepare_raw_paper_experiment.py --apply
& $py -X utf8 .\scripts\check_raw_paper_setup.py
& $py -X utf8 .\scripts\run_training_clean.py --config_file 'G:\efficientphys_raw_paper_seed100.yaml'
& $py -X utf8 .\scripts\analyze_raw_paper_subjects.py
& $py -X utf8 .\scripts\plot_raw_paper_waveforms.py
& $py -X utf8 .\scripts\run_pos_face_roi_baseline.py
& $py -X utf8 .\scripts\prepare_raw_paper_ablations.py
& $py -X utf8 .\scripts\run_raw_paper_ablation.py --variant no_attention
& $py -X utf8 .\scripts\run_raw_paper_ablation.py --variant no_bn
& $py -X utf8 .\scripts\benchmark_raw_paper_inference.py
```

`configs/` 中的 YAML 是脚本生成配置的归档副本。完整模型的正式结果位于 `G:\第三次训练结果_Raw_Paper`，POS 位于 `G:\POS基线_FaceROI_Paper`，两组消融位于 `G:\Ablation_Raw_Paper`。

## 主要结果

| 方法 | MAE ↓ (bpm) | RMSE ↓ (bpm) | MAPE ↓ (%) | 心率 Pearson ↑ | SNR ↑ (dB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| EfficientPhys-C 完整模型 | 1.465 | 3.275 | 1.612 | 0.955 | 0.119 |
| 去空间注意力 | 2.148 | 4.576 | 2.472 | 0.960 | −0.469 |
| 去输入 BatchNorm | 1.660 | 3.327 | 1.808 | 0.952 | 0.112 |
| POS（人脸 ROI） | 4.297 | 8.151 | 4.532 | 0.751 | −1.076 |

完整模型最低验证损失为 0.3508，对应 epoch 7（从 0 计数）。去注意力和去 BatchNorm 分别选中 epoch 4 与 epoch 7。去注意力的心率 Pearson 略高，但 MAE、RMSE、MAPE 和 SNR 均较差；相关性不能单独代表绝对心率误差。逐项原始数值及 Toolbox 输出的标准误见 `results/metrics_summary.csv`。

最终模型在 RTX 5060 Laptop GPU 上以 FP32、batch=1、181 帧输入、180 点输出进行测速：热身 30 次、计时 200 次，平均 **19.0564 ± 0.3425 ms/片段**（± 为重复测量标准差），P95 为 **19.6592 ms/片段**。仅计输入已经在 GPU 上的模型前向调用，不含读取、传输、滤波和心率计算。

这些数值对应本次固定受试者划分和实现；与论文报告值不构成严格同条件比较。实验以论文 EfficientPhys-C 的方法为目标，同时记录了 Toolbox 版本、固定划分和具体评价协议。
