"""Run a controlled POS baseline on the final Raw/Paper UBFC test split.

This uses the same subject41-subject49 cached frames, effective duration, ground
truth reconstruction and 0.75-2.5 Hz evaluation range as the final
EfficientPhys-C experiment. POS has no training stage or learned parameters.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


TOOLBOX_ROOT = Path("D:/rppg/rPPG-Toolbox")
CACHE_ROOT = Path("G:/UBFC-processed-raw/test")
OUTPUT_DIR = Path("G:/POS基线_Raw_Paper")
EXPECTED_SUBJECTS = [f"subject{number}" for number in range(41, 50)]
FS = 30
LOW_HZ = 0.75
HIGH_HZ = 2.5
TARGETS_PER_CHUNK = 180


if str(TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_ROOT))

from evaluation.post_process import calculate_metric_per_video  # noqa: E402
from unsupervised_methods.utils import detrend  # noqa: E402


def chunk_number(path: Path) -> int:
    stem = path.stem
    return int(stem.rsplit("input", 1)[1])


def load_subject(subject: str) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame mean RGB and reconstructed GT used by the model test."""
    folder = CACHE_ROOT / subject
    if not folder.is_dir():
        raise FileNotFoundError(f"缺少测试缓存目录：{folder}")

    input_paths = sorted(folder.glob(f"{subject}_input*.npy"), key=chunk_number)
    if not input_paths:
        raise FileNotFoundError(f"没有找到 {subject} 的输入片段：{folder}")

    rgb_parts = []
    diff_label_parts = []
    for input_path in input_paths:
        index = chunk_number(input_path)
        label_path = folder / f"{subject}_label{index}.npy"
        frames = np.load(input_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(label_path, allow_pickle=False)
        if frames.shape != (181, 72, 72, 3):
            raise ValueError(f"输入形状异常：{input_path} -> {frames.shape}")
        if labels.shape != (TARGETS_PER_CHUNK,):
            raise ValueError(f"标签形状异常：{label_path} -> {labels.shape}")

        # The 181st frame is only for EfficientPhys temporal differencing. The
        # first 180 frames align with this chunk's 180 target points.
        rgb_parts.append(np.asarray(frames[:TARGETS_PER_CHUNK]).mean(axis=(1, 2)))
        diff_label_parts.append(np.asarray(labels, dtype=np.float64))

    rgb = np.concatenate(rgb_parts, axis=0).astype(np.float64, copy=False)
    diff_label = np.concatenate(diff_label_parts)
    ground_truth = np.cumsum(diff_label)
    if rgb.shape != (len(ground_truth), 3):
        raise ValueError(f"{subject} RGB/标签未对齐：{rgb.shape}, {ground_truth.shape}")
    if not np.isfinite(rgb).all() or not np.isfinite(ground_truth).all():
        raise ValueError(f"{subject} 包含非有限值")
    return rgb, ground_truth


def pos_from_mean_rgb(rgb: np.ndarray) -> np.ndarray:
    """Toolbox POS_WANG logic with its internal upper cutoff set to 2.5 Hz."""
    window_seconds = 1.6
    sample_count = rgb.shape[0]
    output = np.zeros((1, sample_count), dtype=np.float64)
    window_length = math.ceil(window_seconds * FS)

    for n in range(sample_count):
        start = n - window_length
        if start < 0:
            continue
        window = rgb[start:n, :]
        channel_mean = np.mean(window, axis=0)
        if np.any(np.abs(channel_mean) < np.finfo(np.float64).eps):
            raise ValueError("POS窗口出现零通道均值")
        normalized = (window / channel_mean).T
        projected = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64) @ normalized
        denominator = np.std(projected[1])
        if denominator <= np.finfo(np.float64).eps:
            continue
        pulse = projected[0] + (np.std(projected[0]) / denominator) * projected[1]
        pulse -= np.mean(pulse)
        output[0, start:n] += pulse

    # These two operations match Toolbox POS_WANG. Only the high cutoff is
    # changed from the Toolbox default 3.0 Hz to the paper evaluation 2.5 Hz.
    bvp = np.asarray(detrend(output.T, 100).T)[0]
    b, a = signal.butter(
        1,
        [LOW_HZ / FS * 2, HIGH_HZ / FS * 2],
        btype="bandpass",
    )
    return signal.filtfilt(b, a, bvp.astype(np.float64))


def plot_results(rows: list[dict]) -> None:
    subjects = [row["subject"] for row in rows]
    gt = np.array([row["gt_hr_bpm"] for row in rows])
    pred = np.array([row["pred_hr_bpm"] for row in rows])
    error = pred - gt
    x = np.arange(len(subjects))

    fig, axis = plt.subplots(figsize=(11, 5))
    width = 0.36
    axis.bar(x - width / 2, gt, width, label="Ground-truth HR")
    axis.bar(x + width / 2, pred, width, label="POS HR")
    axis.set_xticks(x, subjects)
    axis.set_ylabel("Heart rate (bpm)")
    axis.set_title("POS baseline: per-subject FFT heart rate")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pos_per_subject_hr.png", dpi=200)
    fig.savefig(OUTPUT_DIR / "pos_per_subject_hr.pdf")
    plt.close(fig)

    mean_hr = (gt + pred) / 2
    bias = float(np.mean(error))
    sd = float(np.std(error, ddof=1))
    upper = bias + 1.96 * sd
    lower = bias - 1.96 * sd
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(mean_hr, error, s=45)
    axis.axhline(bias, color="black", label=f"Bias={bias:.2f}")
    axis.axhline(upper, color="red", linestyle="--", label=f"+1.96 SD={upper:.2f}")
    axis.axhline(lower, color="red", linestyle="--", label=f"-1.96 SD={lower:.2f}")
    for i, subject in enumerate(subjects):
        axis.annotate(subject.replace("subject", "S"), (mean_hr[i], error[i]), fontsize=8)
    axis.set_xlabel("Mean of GT and POS HR (bpm)")
    axis.set_ylabel("POS HR - GT HR (bpm)")
    axis.set_title("POS Bland-Altman difference plot")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pos_bland_altman_difference.png", dpi=200)
    fig.savefig(OUTPUT_DIR / "pos_bland_altman_difference.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(gt, pred, s=50)
    minimum = min(float(gt.min()), float(pred.min())) - 3
    maximum = max(float(gt.max()), float(pred.max())) + 3
    axis.plot([minimum, maximum], [minimum, maximum], "k--", label="Identity")
    for i, subject in enumerate(subjects):
        axis.annotate(subject.replace("subject", "S"), (gt[i], pred[i]), fontsize=8)
    axis.set_xlim(minimum, maximum)
    axis.set_ylim(minimum, maximum)
    axis.set_xlabel("Ground-truth HR (bpm)")
    axis.set_ylabel("POS HR (bpm)")
    axis.set_title("POS versus ground-truth heart rate")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pos_hr_scatter.png", dpi=200)
    fig.savefig(OUTPUT_DIR / "pos_hr_scatter.pdf")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    present = sorted(path.name for path in CACHE_ROOT.glob("subject*") if path.is_dir())
    if present != EXPECTED_SUBJECTS:
        raise ValueError(f"测试缓存受试者不是 subject41～49：{present}")

    rows = []
    start_time = time.perf_counter()
    for subject in EXPECTED_SUBJECTS:
        subject_start = time.perf_counter()
        rgb, ground_truth = load_subject(subject)
        prediction = pos_from_mean_rgb(rgb)
        gt_hr, pred_hr, snr, macc = calculate_metric_per_video(
            prediction,
            ground_truth,
            fs=FS,
            diff_flag=False,
            use_bandpass=True,
            hr_method="FFT",
        )
        signed_error = float(pred_hr - gt_hr)
        absolute_error = abs(signed_error)
        rows.append(
            {
                "subject": subject,
                "points": len(prediction),
                "duration_seconds": len(prediction) / FS,
                "gt_hr_bpm": float(gt_hr),
                "pred_hr_bpm": float(pred_hr),
                "signed_error_bpm": signed_error,
                "absolute_error_bpm": absolute_error,
                "absolute_percentage_error_percent": absolute_error / float(gt_hr) * 100,
                "snr_db": float(np.asarray(snr).squeeze()),
                "macc": float(macc),
            }
        )
        print(
            f"{subject}: GT={gt_hr:.2f}, POS={pred_hr:.2f}, "
            f"误差={signed_error:+.2f}, SNR={float(np.asarray(snr).squeeze()):.2f} dB, "
            f"耗时={time.perf_counter() - subject_start:.1f}s",
            flush=True,
        )

    csv_path = OUTPUT_DIR / "pos_per_subject_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    gt = np.array([row["gt_hr_bpm"] for row in rows])
    pred = np.array([row["pred_hr_bpm"] for row in rows])
    snr = np.array([row["snr_db"] for row in rows])
    error = pred - gt
    count = len(rows)
    absolute_error = np.abs(error)
    squared_error = error ** 2
    mae = float(np.mean(absolute_error))
    rmse = float(np.sqrt(np.mean(squared_error)))
    mape = float(np.mean(absolute_error / gt) * 100)
    pearson = float(np.corrcoef(pred, gt)[0, 1])
    mean_snr = float(np.mean(snr))
    values = {
        "MAE_bpm": mae,
        "MAE_standard_error": float(np.std(absolute_error) / np.sqrt(count)),
        "RMSE_bpm": rmse,
        "RMSE_standard_error": float(np.sqrt(np.std(squared_error) / np.sqrt(count))),
        "MAPE_percent": mape,
        "MAPE_standard_error": float(np.std(absolute_error / gt) / np.sqrt(count) * 100),
        "Pearson": pearson,
        "Pearson_standard_error": float(np.sqrt((1 - pearson ** 2) / (count - 2))),
        "SNR_dB": mean_snr,
        "SNR_standard_error": float(np.std(snr) / np.sqrt(count)),
    }

    methodology = {
        "method": "POS_WANG from rPPG-Toolbox, adapted internal high cutoff 3.0->2.5 Hz",
        "subjects": EXPECTED_SUBJECTS,
        "input": "same Raw 72x72 cached frames as final EfficientPhys-C test",
        "fs": FS,
        "duration": "same effective whole-subject duration as final model test",
        "pos_internal_bandpass_hz": [LOW_HZ, HIGH_HZ],
        "evaluation_bandpass_fft_snr_hz": [LOW_HZ, HIGH_HZ],
        "evaluation_method": "whole-subject FFT",
        "training": False,
    }
    (OUTPUT_DIR / "methodology.json").write_text(
        json.dumps(methodology, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_lines = [
        f"FFT MAE (FFT Label): {mae} +/- {values['MAE_standard_error']}",
        f"FFT RMSE (FFT Label): {rmse} +/- {values['RMSE_standard_error']}",
        f"FFT MAPE (FFT Label): {mape} +/- {values['MAPE_standard_error']}",
        f"FFT Pearson (FFT Label): {pearson} +/- {values['Pearson_standard_error']}",
        f"FFT SNR (FFT Label): {mean_snr} +/- {values['SNR_standard_error']} (dB)",
    ]
    (OUTPUT_DIR / "pos_metrics.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    plot_results(rows)

    print("\n" + "\n".join(summary_lines), flush=True)
    print(f"\n总耗时：{time.perf_counter() - start_time:.1f}s", flush=True)
    print(f"结果目录：{OUTPUT_DIR}", flush=True)
    print("已生成五项指标、逐受试者CSV、方法记录和对比/Bland-Altman图。", flush=True)


if __name__ == "__main__":
    main()
