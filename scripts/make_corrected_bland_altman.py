"""Regenerate the final heart-rate Bland–Altman plot with the correct sign label.

Reads the already evaluated, per-subject FFT heart rates. Does not retrain,
retest, or modify the original Toolbox PDF.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("G:/第三次训练结果_Raw_Paper/test")
SOURCE = ROOT / "subject_analysis/per_subject_fft_metrics.csv"
OUTPUT = ROOT / "bland_altman_plots"
STEM = "UBFC_Dataset2_EfficientPhysC_Raw_Paper_seed100_FFT_BlandAltman_DifferencePlot_corrected"


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"找不到逐受试者指标：{SOURCE}")
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    expected = [f"subject{index}" for index in range(41, 50)]
    if [row["subject"] for row in rows] != expected:
        raise ValueError("CSV 中的受试者不是按 subject41～49 排列")

    gt = np.array([float(row["gt_hr_bpm"]) for row in rows])
    pred = np.array([float(row["pred_hr_bpm"]) for row in rows])
    if not np.isfinite(gt).all() or not np.isfinite(pred).all():
        raise ValueError("CSV 中存在无效心率")

    average = (gt + pred) / 2
    difference = gt - pred  # Exactly the sign used by Toolbox's BlandAltmanPy.
    bias = float(np.mean(difference))
    sample_sd = float(np.std(difference, ddof=1))
    # Keep the original Toolbox's averaged=True limit calculation unchanged.
    half_width = 1.96 * np.sqrt(2) * sample_sd

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.scatter(average, difference, s=65, color="#1676a2", zorder=3)
    for row, x, y in zip(rows, average, difference):
        ax.annotate(row["subject"], (x, y), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    ax.axhline(bias, color="black", label=f"Mean difference: {bias:.2f} bpm")
    ax.axhline(bias + half_width, color="#b5543d", linestyle="--",
               label=f"Toolbox upper limit: {bias + half_width:.2f} bpm")
    ax.axhline(bias - half_width, color="#b5543d", linestyle="--",
               label=f"Toolbox lower limit: {bias - half_width:.2f} bpm")
    ax.set_xlabel("Average of ground-truth and predicted HR (bpm)")
    ax.set_ylabel("Ground-truth HR minus predicted HR (bpm)")
    ax.set_title("EfficientPhys-C: heart-rate difference plot")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT / f"{STEM}.pdf"
    png = OUTPUT / f"{STEM}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=220)
    plt.close(fig)

    print(f"真实心率 - 预测心率：平均差值 {bias:.4f} bpm")
    print(f"采用原 Toolbox averaged=True 公式的上下界："
          f"{bias - half_width:.4f}, {bias + half_width:.4f} bpm")
    print(f"已保存：{pdf}")
    print(f"已保存：{png}")
    print("原始 Toolbox PDF 未修改。")


if __name__ == "__main__":
    main()
