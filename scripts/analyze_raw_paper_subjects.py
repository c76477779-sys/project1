"""Generate per-subject FFT heart-rate results for the final Raw/Paper run.

The calculations call rPPG-Toolbox's calculate_metric_per_video directly, so
the reconstruction, detrending, bandpass, FFT HR and SNR definitions match the
five headline test metrics.
"""

from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOOLBOX_ROOT = Path("D:/rppg/rPPG-Toolbox")
RESULT_ROOT = Path("G:/第三次训练结果_Raw_Paper")
MODEL_NAME = "UBFC_Dataset2_EfficientPhysC_Raw_Paper_seed100"
PICKLE_PATH = (
    RESULT_ROOT
    / "test/saved_test_outputs"
    / f"{MODEL_NAME}_outputs.pickle"
)
OUTPUT_DIR = RESULT_ROOT / "test/subject_analysis"
EXPECTED_SUBJECTS = [f"subject{number}" for number in range(41, 50)]
FS = 30


if str(TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_ROOT))

from evaluation.post_process import calculate_metric_per_video  # noqa: E402


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def concatenate_chunks(chunks: dict) -> np.ndarray:
    def chunk_key(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)

    return np.concatenate(
        [to_numpy(chunks[key]) for key in sorted(chunks, key=chunk_key)]
    )


def subject_number(subject: str) -> int:
    return int(subject.removeprefix("subject"))


def save_plots(rows: list[dict]) -> None:
    subjects = [row["subject"] for row in rows]
    x = np.arange(len(subjects))
    gt = np.array([row["gt_hr_bpm"] for row in rows])
    pred = np.array([row["pred_hr_bpm"] for row in rows])
    absolute_error = np.array([row["absolute_error_bpm"] for row in rows])

    fig, axis = plt.subplots(figsize=(11, 5))
    width = 0.36
    axis.bar(x - width / 2, gt, width, label="Ground-truth HR")
    axis.bar(x + width / 2, pred, width, label="Predicted HR")
    axis.set_xticks(x, subjects)
    axis.set_ylabel("Heart rate (bpm)")
    axis.set_title("Per-subject FFT heart rate (0.75-2.5 Hz)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    stem = OUTPUT_DIR / "per_subject_hr_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)

    colors = ["#d95f02" if value >= 5 else "#1b9e77" for value in absolute_error]
    fig, axis = plt.subplots(figsize=(11, 5))
    bars = axis.bar(subjects, absolute_error, color=colors)
    axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.axhline(np.mean(absolute_error), color="black", linestyle="--",
                 label=f"Mean = {np.mean(absolute_error):.2f} bpm")
    axis.set_ylabel("Absolute HR error (bpm)")
    axis.set_title("Per-subject absolute FFT heart-rate error")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    stem = OUTPUT_DIR / "per_subject_absolute_hr_error"
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
    label_type = saved.get("label_type", "DiffNormalized")
    saved_fs = int(saved.get("fs", FS))
    subjects = sorted(predictions, key=subject_number)

    if subjects != EXPECTED_SUBJECTS or set(labels) != set(EXPECTED_SUBJECTS):
        raise ValueError(f"测试受试者不是 subject41～49：{subjects}")
    if saved_fs != FS:
        raise ValueError(f"测试输出 FS={saved_fs}，期望 FS={FS}")
    if label_type != "DiffNormalized":
        raise ValueError(f"标签类型为 {label_type}，期望 DiffNormalized")

    rows = []
    for subject in subjects:
        prediction = concatenate_chunks(predictions[subject])
        label = concatenate_chunks(labels[subject])
        if prediction.shape != label.shape:
            raise ValueError(
                f"{subject} 预测与标签长度不同：{prediction.shape}, {label.shape}"
            )

        gt_hr, pred_hr, snr, macc = calculate_metric_per_video(
            prediction,
            label,
            fs=FS,
            diff_flag=True,
            use_bandpass=True,
            hr_method="FFT",
        )
        signed_error = float(pred_hr - gt_hr)
        absolute_error = abs(signed_error)
        ape = absolute_error / float(gt_hr) * 100
        row = {
            "subject": subject,
            "points": len(prediction),
            "duration_seconds": len(prediction) / FS,
            "gt_hr_bpm": float(gt_hr),
            "pred_hr_bpm": float(pred_hr),
            "signed_error_bpm": signed_error,
            "absolute_error_bpm": absolute_error,
            "absolute_percentage_error_percent": ape,
            "snr_db": float(np.asarray(snr).squeeze()),
            "macc": float(macc),
        }
        rows.append(row)
        print(
            f"{subject}: GT={gt_hr:.2f}, Pred={pred_hr:.2f}, "
            f"误差={signed_error:+.2f}, 绝对误差={absolute_error:.2f} bpm, "
            f"SNR={row['snr_db']:.2f} dB"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "per_subject_fft_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    gt = np.array([row["gt_hr_bpm"] for row in rows])
    pred = np.array([row["pred_hr_bpm"] for row in rows])
    snr = np.array([row["snr_db"] for row in rows])
    error = pred - gt
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mape = float(np.mean(np.abs(error / gt)) * 100)
    pearson = float(np.corrcoef(pred, gt)[0, 1])
    mean_snr = float(np.mean(snr))

    summary_path = OUTPUT_DIR / "summary.txt"
    lines = [
        f"subjects={len(rows)}",
        f"MAE={mae:.12f} bpm",
        f"RMSE={rmse:.12f} bpm",
        f"MAPE={mape:.12f} %",
        f"Pearson={pearson:.12f}",
        f"SNR={mean_snr:.12f} dB",
        "evaluation=whole subject, FFT, DiffNormalized reconstruction, "
        "0.75-2.5 Hz bandpass/search/SNR range",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_plots(rows)

    print("\n五项指标复核：")
    print(f"MAE={mae:.8f} bpm")
    print(f"RMSE={rmse:.8f} bpm")
    print(f"MAPE={mape:.8f} %")
    print(f"Pearson={pearson:.8f}")
    print(f"SNR={mean_snr:.8f} dB")
    print(f"\n输出目录：{OUTPUT_DIR}")
    print("已生成逐受试者 CSV、两组 PNG/PDF 图和 summary.txt。")


if __name__ == "__main__":
    main()
