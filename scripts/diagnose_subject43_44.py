"""Check two Raw/Paper test subjects without training or changing saved outputs."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


ROOT = Path("G:/第三次训练结果_Raw_Paper")
CACHE = Path("G:/UBFC-processed-raw/test")
PICKLE = ROOT / (
    "test/saved_test_outputs/"
    "UBFC_Dataset2_EfficientPhysC_Raw_Paper_seed100_outputs.pickle"
)
OUT = ROOT / "test/subject_diagnostics"
SUBJECTS = ("subject43", "subject44")
FS = 30
TOOLBOX = Path("D:/rppg/rPPG-Toolbox")
sys.path.insert(0, str(TOOLBOX))

from evaluation.post_process import _detrend, calculate_metric_per_video  # noqa: E402


def as_vector(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def join_chunks(chunks):
    return np.concatenate([as_vector(chunks[key]) for key in sorted(chunks, key=int)])


def evaluated_waveform(differences):
    """Mirror the reconstruction, detrending and bandpass in Toolbox metrics."""
    restored = _detrend(np.cumsum(differences), 100)
    b, a = signal.butter(1, [0.75 / FS * 2, 2.5 / FS * 2], btype="bandpass")
    return signal.filtfilt(b, a, restored)


def check_cache(subject, saved_labels):
    folder = CACHE / subject
    metadata = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    count = int(metadata["clips"])
    assert metadata["subject"] == subject and metadata["split"] == "test"
    assert metadata["settings"]["label_type"] == "DiffNormalized"
    assert metadata["settings"]["input_frames"] == 181
    assert metadata["settings"]["target_frames"] == 180
    assert count * 180 == len(saved_labels), "Saved test labels do not match cache length"

    cached = []
    previous_last = None
    for index in range(count):
        x = np.load(folder / f"{subject}_input{index}.npy", mmap_mode="r")
        y = np.load(folder / f"{subject}_label{index}.npy")
        assert x.shape == (181, 72, 72, 3) and y.shape == (180,)
        if previous_last is not None:
            assert np.array_equal(previous_last, x[0]), f"Frame boundary mismatch at clip {index}"
        previous_last = np.array(x[-1], copy=True)
        cached.append(y)
    cached = np.concatenate(cached)
    max_saved_diff = float(np.max(np.abs(cached - saved_labels)))
    assert np.allclose(cached, saved_labels, rtol=0, atol=1e-6), (
        f"Saved test labels differ from cache: {max_saved_diff}"
    )

    bvp = np.load(folder / "bvp_raw.npy")
    expected = np.concatenate((np.diff(bvp) / np.std(np.diff(bvp)), [0.0]))
    max_raw_diff = float(np.max(np.abs(expected[: len(cached)] - cached)))
    assert np.allclose(expected[: len(cached)], cached, rtol=0, atol=1e-6), (
        f"Cached labels differ from raw BVP: {max_raw_diff}"
    )

    source = metadata["source"]
    source_video = Path(source["video_path"])
    source_label = source_video.with_name("ground_truth.txt")
    source_status = "source files unavailable; cache checked"
    if source_video.is_file() and source_label.is_file():
        digest = hashlib.sha256(source_label.read_bytes()).hexdigest()
        assert digest == source["label_sha256"], "Source ground_truth.txt changed"
        assert source_video.stat().st_size == source["video_bytes"], "Source video size changed"
        source_status = "source label hash and video size match cache metadata"

    timestamps_path = folder / "timestamps.npy"
    timestamp_note = "not saved"
    if timestamps_path.is_file():
        timestamps = np.load(timestamps_path)
        assert len(timestamps) == len(bvp) and np.all(np.diff(timestamps) >= 0)
        timestamp_note = f"saved, median step={np.median(np.diff(timestamps)):.6g} (source units)"

    return {
        "clips": count,
        "original_bvp_points": len(bvp),
        "discarded_tail_points": metadata["discarded_tail_points"],
        "video_reported_fps": metadata["video_reported_fps"],
        "max_saved_label_difference": max_saved_diff,
        "max_raw_bvp_label_difference": max_raw_diff,
        "timestamp_note": timestamp_note,
        "source_video": str(source_video),
        "source_status": source_status,
    }


def draw_spectrum(subject, prediction, label, gt_hr, pred_hr):
    pred_signal = evaluated_waveform(prediction)
    gt_signal = evaluated_waveform(label)
    nfft = 1 << (len(pred_signal) - 1).bit_length()
    freq, pred_power = signal.periodogram(pred_signal, fs=FS, nfft=nfft, detrend=False)
    _, gt_power = signal.periodogram(gt_signal, fs=FS, nfft=nfft, detrend=False)
    selected = (freq >= 0.75) & (freq <= 2.5)
    bpm = freq[selected] * 60
    pred = pred_power.reshape(-1)[selected]
    gt = gt_power.reshape(-1)[selected]
    assert abs(bpm[np.argmax(pred)] - pred_hr) < 1e-8
    assert abs(bpm[np.argmax(gt)] - gt_hr) < 1e-8

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(bpm, gt / gt.max(), label=f"Ground truth peak: {gt_hr:.2f} bpm")
    ax.plot(bpm, pred / pred.max(), label=f"Predicted peak: {pred_hr:.2f} bpm")
    ax.axvline(gt_hr, color="C0", linestyle="--", alpha=0.7)
    ax.axvline(pred_hr, color="C1", linestyle="--", alpha=0.7)
    ax.set(xlabel="Heart rate (bpm)", ylabel="Power / each spectrum maximum",
           title=f"{subject}: whole-subject FFT spectrum (0.75–2.5 Hz)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = OUT / f"{subject}_fft_spectrum.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    if not PICKLE.is_file():
        raise FileNotFoundError(f"Missing saved test output: {PICKLE}")
    with PICKLE.open("rb") as handle:
        saved = pickle.load(handle)
    assert saved.get("label_type") == "DiffNormalized"
    assert int(saved.get("fs")) == FS
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []
    for subject in SUBJECTS:
        pred_chunks = saved["predictions"][subject]
        label_chunks = saved["labels"][subject]
        assert set(pred_chunks) == set(label_chunks)
        prediction = join_chunks(pred_chunks)
        label = join_chunks(label_chunks)
        assert len(prediction) == len(label)
        info = check_cache(subject, label)
        gt_hr, pred_hr, snr, _ = calculate_metric_per_video(
            prediction, label, fs=FS, diff_flag=True,
            use_bandpass=True, hr_method="FFT"
        )
        image_path = draw_spectrum(subject, prediction, label, gt_hr, pred_hr)
        line = (f"{subject}: GT={gt_hr:.2f}, Pred={pred_hr:.2f}, "
                f"error={pred_hr-gt_hr:+.2f} bpm, SNR={float(np.asarray(snr).squeeze()):.2f} dB\n"
                f"  cache={info}\n  spectrum={image_path}")
        lines.append(line)
        print(line)
    report = OUT / "subject43_44_checks.txt"
    report.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nCheck report: {report}")
    print("These checks cannot prove timestamp synchronization or identify motion/lighting causes.")


if __name__ == "__main__":
    main()
