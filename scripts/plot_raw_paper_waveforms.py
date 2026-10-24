"""Plot filtered predicted/ground-truth waveforms for the 9 UBFC test subjects."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


RESULT_ROOT = Path("G:/第三次训练结果_Raw_Paper")
MODEL_NAME = "UBFC_Dataset2_EfficientPhysC_Raw_Paper_seed100"
PICKLE_PATH = RESULT_ROOT / "test/saved_test_outputs" / f"{MODEL_NAME}_outputs.pickle"
OUTPUT_DIR = RESULT_ROOT / "test/waveform_plots"

FS = 30
LOW_HZ = 0.75
HIGH_HZ = 2.5
FIRST_SECONDS = 6
EXPECTED_SUBJECTS = [f"subject{number}" for number in range(41, 50)]


def natural_subject_key(subject: str) -> int:
    return int("".join(character for character in subject if character.isdigit()))


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def concatenate_chunks(chunks: dict) -> np.ndarray:
    def chunk_key(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    return np.concatenate([to_numpy(chunks[key]) for key in sorted(chunks, key=chunk_key)])


def reconstruct_and_filter(values: np.ndarray, diff_normalized: bool) -> np.ndarray:
    if diff_normalized:
        values = np.cumsum(values)
    values = signal.detrend(values)
    b, a = signal.butter(1, [LOW_HZ / FS * 2, HIGH_HZ / FS * 2], btype="bandpass")
    return signal.filtfilt(b, a, values)


def standardize(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Waveform has zero or invalid standard deviation")
    return (values - np.mean(values)) / scale


def plot_subject(subject: str, prediction: np.ndarray, label: np.ndarray) -> None:
    points = min(FIRST_SECONDS * FS, len(prediction), len(label))
    time_axis = np.arange(points) / FS
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(time_axis, standardize(label[:points]), linewidth=2, label="Ground truth")
    axis.plot(time_axis, standardize(prediction[:points]), linewidth=2, label="Predicted rPPG")
    axis.set_title(f"EfficientPhys-C (Raw, best epoch 7): {subject}\nBandpass: {LOW_HZ}-{HIGH_HZ} Hz")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Standardized amplitude")
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right")
    fig.tight_layout()
    stem = OUTPUT_DIR / f"{subject}_first6s_filtered"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    if not PICKLE_PATH.is_file():
        raise FileNotFoundError(f"找不到测试输出：{PICKLE_PATH}")
    with PICKLE_PATH.open("rb") as file:
        saved = pickle.load(file)

    predictions = saved["predictions"]
    labels = saved["labels"]
    label_type = saved.get("label_type", "")
    saved_fs = saved.get("fs", FS)
    if saved_fs != FS:
        raise ValueError(f"测试输出采样率是 {saved_fs}，期望 {FS}")

    subjects = sorted(predictions, key=natural_subject_key)
    if subjects != EXPECTED_SUBJECTS or set(labels) != set(EXPECTED_SUBJECTS):
        raise ValueError(f"测试受试者不是 subject41～49：{subjects}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    prepared = {}
    for subject in subjects:
        prediction = concatenate_chunks(predictions[subject])
        label = concatenate_chunks(labels[subject])
        if prediction.shape != label.shape:
            raise ValueError(f"{subject} 预测和标签长度不同：{prediction.shape}, {label.shape}")
        prediction = reconstruct_and_filter(prediction, label_type == "DiffNormalized")
        label = reconstruct_and_filter(label, label_type == "DiffNormalized")
        correlation = float(np.corrcoef(prediction, label)[0, 1])
        if not np.isfinite(correlation):
            raise ValueError(f"{subject} 波形相关性无效")
        rows.append({
            "subject": subject,
            "points": len(prediction),
            "duration_seconds": len(prediction) / FS,
            "filtered_waveform_pearson": correlation,
        })
        prepared[subject] = (prediction, label)
        plot_subject(subject, prediction, label)
        print(f"完成 {subject}：{len(prediction)} 点，波形 Pearson={correlation:.4f}")

    csv_path = OUTPUT_DIR / "filtered_waveform_correlations.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    points = FIRST_SECONDS * FS
    time_axis = np.arange(points) / FS
    for axis, subject in zip(axes.flat, subjects):
        prediction, label = prepared[subject]
        actual_points = min(points, len(prediction), len(label))
        axis.plot(time_axis[:actual_points], standardize(label[:actual_points]), linewidth=1.4, label="Ground truth")
        axis.plot(time_axis[:actual_points], standardize(prediction[:actual_points]), linewidth=1.4, label="Predicted")
        axis.set_title(subject)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("Time (s)")
    fig.supylabel("Standardized amplitude")
    fig.suptitle(f"EfficientPhys-C test subjects: first {FIRST_SECONDS} s after {LOW_HZ}-{HIGH_HZ} Hz bandpass")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    summary = OUTPUT_DIR / "all_test_subjects_first6s_filtered"
    fig.savefig(summary.with_suffix(".png"), dpi=200)
    fig.savefig(summary.with_suffix(".pdf"))
    plt.close(fig)

    print(f"\n输出目录：{OUTPUT_DIR}")
    print("已生成9名受试者的PNG/PDF、9人汇总图和 filtered_waveform_correlations.csv")
    print("注意：表中的波形Pearson是辅助分析，不是五项指标中的心率Pearson。")


if __name__ == "__main__":
    main()
