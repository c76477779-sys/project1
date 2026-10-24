"""Measure final EfficientPhys-C Raw181 checkpoint forward time on a real test clip.

CUDA events measure only the forward call. Data loading, CPU-to-GPU transfer,
filtering, FFT and file writing happen outside the timed region.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch


TOOLBOX_ROOT = Path("D:/rppg/rPPG-Toolbox")
RESULT_ROOT = Path("G:/第三次训练结果_Raw_Paper")
MODEL_FILE_NAME = "UBFC_Dataset2_EfficientPhysC_Raw_Paper_seed100"
CHECKPOINT = (
    RESULT_ROOT / "train/PreTrainedModels" / f"{MODEL_FILE_NAME}_Epoch7.pth"
)
SAMPLE = Path("G:/UBFC-processed-raw/test/subject41/subject41_input0.npy")
OUTPUT_DIR = RESULT_ROOT / "inference_timing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=30, help="GPU warmup forwards")
    parser.add_argument("--repeats", type=int, default=200, help="timed forwards")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 10:
        parser.error("--warmup 至少1次，--repeats 至少10次")
    return args


def main() -> None:
    args = parse_args()
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"找不到完整模型最佳权重：{CHECKPOINT}")
    if not SAMPLE.is_file():
        raise FileNotFoundError(f"找不到 Raw181 测试片段：{SAMPLE}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 未检测到 CUDA GPU，无法进行这次 GPU 测速")

    # Explicitly select the complete model, even if a parent shell previously
    # set an ablation environment variable.
    os.environ["EFFICIENTPHYS_ABLATION"] = "full"
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(TOOLBOX_ROOT))
    from neural_methods.model.EfficientPhys import EfficientPhys

    frames = np.load(SAMPLE, allow_pickle=False)
    if frames.shape != (181, 72, 72, 3) or frames.dtype != np.float32:
        raise ValueError(
            f"测试片段与最终 Raw181 配置不符：shape={frames.shape}, dtype={frames.dtype}"
        )
    # Match the Toolbox's NDCHW loader. Transfer occurs before timing.
    ndchw = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
    input_tensor = torch.from_numpy(ndchw).to(device="cuda:0", dtype=torch.float32)

    core_model = EfficientPhys(frame_depth=10, img_size=72).to("cuda:0")
    model = torch.nn.DataParallel(core_model, device_ids=[0])
    state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"权重内容不是 state_dict：{type(state_dict)}")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"模型权重：{CHECKPOINT}", flush=True)
    print(f"测试片段：{SAMPLE}", flush=True)
    print(f"设备：{torch.cuda.get_device_name(0)}", flush=True)
    print(f"模式：{core_model.ablation}；FP32；单GPU；batch=1个片段", flush=True)
    print(f"输入：{tuple(input_tensor.shape)}；热身{args.warmup}次；重复{args.repeats}次", flush=True)

    with torch.inference_mode():
        for _ in range(args.warmup):
            prediction = model(input_tensor)
        torch.cuda.synchronize()

        if tuple(prediction.shape) != (180, 1):
            raise ValueError(f"输出形状应为 (180, 1)，实际是 {tuple(prediction.shape)}")

        times_ms = []
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            prediction = model(input_tensor)
            end.record()
            end.synchronize()
            times_ms.append(start.elapsed_time(end))

    times = np.asarray(times_ms, dtype=np.float64)
    mean_ms = float(times.mean())
    effective_frames = 180
    metrics = {
        "experiment": "EfficientPhys-C Raw/Paper final full model",
        "checkpoint": str(CHECKPOINT),
        "sample": str(SAMPLE),
        "model_mode": core_model.ablation,
        "gpu": torch.cuda.get_device_name(0),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "precision": "FP32, TF32 disabled",
        "execution": "eval + torch.inference_mode + DataParallel(single GPU)",
        "input_shape": list(input_tensor.shape),
        "output_shape": list(prediction.shape),
        "input_frames": 181,
        "effective_output_frames": effective_frames,
        "sample_rate_hz": 30,
        "batch_size_clips": 1,
        "warmup_repetitions": args.warmup,
        "timed_repetitions": args.repeats,
        "mean_ms_per_clip": mean_ms,
        "std_ms_per_clip": float(times.std(ddof=1)),
        "median_ms_per_clip": float(np.median(times)),
        "p95_ms_per_clip": float(np.percentile(times, 95)),
        "mean_ms_per_effective_frame": mean_ms / effective_frames,
        "effective_frame_throughput_fps": effective_frames * 1000.0 / mean_ms,
        "timed_scope": "CUDA events around single model forward only; input already on GPU",
        "excluded": [
            "disk and data loading",
            "CPU to GPU transfer",
            "video resizing and face handling",
            "signal reconstruction and filtering",
            "FFT and heart-rate calculation",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "raw_paper_epoch7_forward_timing.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (OUTPUT_DIR / "raw_paper_epoch7_forward_times.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["repetition", "forward_ms"])
        writer.writerows((index, value) for index, value in enumerate(times_ms, start=1))

    report = "\n".join(
        [
            "EfficientPhys-C Raw/Paper 最终模型前向测速",
            f"GPU：{metrics['gpu']}",
            f"权重：{CHECKPOINT}",
            "FP32，batch=1，单GPU，输入181帧、输出180个有效点",
            f"热身：{args.warmup} 次；重复计时：{args.repeats} 次",
            f"平均片段前向耗时：{mean_ms:.4f} ms",
            f"重复测量标准差：{metrics['std_ms_per_clip']:.4f} ms",
            f"中位数：{metrics['median_ms_per_clip']:.4f} ms",
            f"P95：{metrics['p95_ms_per_clip']:.4f} ms",
            f"每有效帧摊销耗时：{metrics['mean_ms_per_effective_frame']:.4f} ms/frame",
            f"有效帧吞吐率：{metrics['effective_frame_throughput_fps']:.2f} FPS",
            "范围：仅计输入已在GPU上的单片段模型前向调用。",
            "不含视频读取、数据传输、滤波及心率计算；每帧摊销值不等于在线逐帧延迟。",
        ]
    )
    (OUTPUT_DIR / "raw_paper_epoch7_forward_timing.txt").write_text(
        report + "\n", encoding="utf-8"
    )
    print("\n" + report, flush=True)
    print(f"\n结果目录：{OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
