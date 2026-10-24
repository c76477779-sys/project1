"""Run the report-ready POS baseline with the Toolbox-style face ROI."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np


WORK_DIR = Path(__file__).resolve().parent
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

import run_pos_raw_paper_baseline as common  # noqa: E402


OUTPUT_DIR = Path("G:/POS基线_FaceROI_Paper")
PROCESSED_TEST_DIR = OUTPUT_DIR / "processed_test"
CASCADE_PATH = (
    common.TOOLBOX_ROOT / "dataset/haarcascade_frontalface_default.xml"
)
LARGE_BOX_COEF = 1.5


def load_ground_truth(subject: str) -> tuple[np.ndarray, Path]:
    folder = common.CACHE_ROOT / subject
    metadata_path = folder / "complete.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"缺少缓存元数据：{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_path = Path(metadata["source"]["video_path"])
    if not source_path.is_file():
        raise FileNotFoundError(
            f"{subject} 原视频已不在预处理时的位置：{source_path}\n"
            "请把视频放回原位置后重试。"
        )

    label_parts = []
    input_paths = sorted(
        folder.glob(f"{subject}_input*.npy"), key=common.chunk_number
    )
    for input_path in input_paths:
        index = common.chunk_number(input_path)
        label_path = folder / f"{subject}_label{index}.npy"
        label = np.load(label_path, allow_pickle=False)
        if label.shape != (common.TARGETS_PER_CHUNK,):
            raise ValueError(f"标签形状异常：{label_path} -> {label.shape}")
        label_parts.append(np.asarray(label, dtype=np.float64))
    if not label_parts:
        raise FileNotFoundError(f"没有找到 {subject} 的标签")
    ground_truth = np.cumsum(np.concatenate(label_parts))
    return ground_truth, source_path


def detect_face_box(frame_bgr: np.ndarray, detector: cv2.CascadeClassifier) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray)
    if len(faces) == 0:
        raise RuntimeError("第一帧未检测到人脸，停止以避免退化成整帧POS")
    face = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    x, y, width, height = [float(value) for value in face]
    x = max(0.0, x - (LARGE_BOX_COEF - 1.0) / 2.0 * width)
    y = max(0.0, y - (LARGE_BOX_COEF - 1.0) / 2.0 * height)
    width *= LARGE_BOX_COEF
    height *= LARGE_BOX_COEF
    frame_height, frame_width = frame_bgr.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2 = min(frame_width, int(x + width))
    y2 = min(frame_height, int(y + height))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"无效人脸框：{(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def video_mean_rgb(video_path: Path, points: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        raise FileNotFoundError(f"无法加载Haar检测器：{CASCADE_PATH}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    values = np.empty((points, 3), dtype=np.float64)
    face_box = None
    try:
        for index in range(points):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"视频在第 {index} 帧提前结束：{video_path}")
            if face_box is None:
                face_box = detect_face_box(frame, detector)
            x1, y1, x2, y2 = face_box
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (72, 72), interpolation=cv2.INTER_AREA)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            values[index] = crop_rgb.mean(axis=(0, 1))
    finally:
        capture.release()
    assert face_box is not None
    return values, face_box


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_TEST_DIR.mkdir(parents=True, exist_ok=True)
    common.OUTPUT_DIR = OUTPUT_DIR
    rows = []
    boxes = {}
    total_start = time.perf_counter()

    for subject in common.EXPECTED_SUBJECTS:
        subject_start = time.perf_counter()
        ground_truth, video_path = load_ground_truth(subject)
        rgb, face_box = video_mean_rgb(video_path, len(ground_truth))
        # POS only consumes the spatial mean RGB sequence. Saving this compact
        # representation makes the preprocessing reproducible without writing
        # several gigabytes of cropped video frames.
        np.save(PROCESSED_TEST_DIR / f"{subject}_mean_rgb.npy", rgb, allow_pickle=False)
        np.save(
            PROCESSED_TEST_DIR / f"{subject}_ground_truth.npy",
            ground_truth,
            allow_pickle=False,
        )
        (PROCESSED_TEST_DIR / f"{subject}_metadata.json").write_text(
            json.dumps(
                {
                    "subject": subject,
                    "source_video": str(video_path),
                    "points": len(ground_truth),
                    "fs": common.FS,
                    "face_detector": "OpenCV Haar cascade",
                    "fixed_face_box_xyxy": list(face_box),
                    "large_box_coef": LARGE_BOX_COEF,
                    "resize_before_spatial_mean": [72, 72],
                    "color_order": "RGB",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        prediction = common.pos_from_mean_rgb(rgb)
        gt_hr, pred_hr, snr, macc = common.calculate_metric_per_video(
            prediction,
            ground_truth,
            fs=common.FS,
            diff_flag=False,
            use_bandpass=True,
            hr_method="FFT",
        )
        error = float(pred_hr - gt_hr)
        absolute_error = abs(error)
        snr_value = float(np.asarray(snr).squeeze())
        rows.append(
            {
                "subject": subject,
                "points": len(prediction),
                "duration_seconds": len(prediction) / common.FS,
                "gt_hr_bpm": float(gt_hr),
                "pred_hr_bpm": float(pred_hr),
                "signed_error_bpm": error,
                "absolute_error_bpm": absolute_error,
                "absolute_percentage_error_percent": absolute_error / float(gt_hr) * 100,
                "snr_db": snr_value,
                "macc": float(macc),
            }
        )
        boxes[subject] = {
            "video_path": str(video_path),
            "fixed_face_box_xyxy": list(face_box),
        }
        print(
            f"{subject}: GT={gt_hr:.2f}, POS={pred_hr:.2f}, "
            f"误差={error:+.2f}, SNR={snr_value:.2f} dB, "
            f"人脸框={face_box}, 耗时={time.perf_counter() - subject_start:.1f}s",
            flush=True,
        )

    csv_path = OUTPUT_DIR / "pos_face_roi_per_subject_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    gt = np.array([row["gt_hr_bpm"] for row in rows])
    pred = np.array([row["pred_hr_bpm"] for row in rows])
    snr_values = np.array([row["snr_db"] for row in rows])
    error = pred - gt
    absolute_error = np.abs(error)
    squared_error = error ** 2
    count = len(rows)
    mae = float(np.mean(absolute_error))
    rmse = float(np.sqrt(np.mean(squared_error)))
    mape = float(np.mean(absolute_error / gt) * 100)
    pearson = float(np.corrcoef(pred, gt)[0, 1])
    mean_snr = float(np.mean(snr_values))
    mae_se = float(np.std(absolute_error) / np.sqrt(count))
    rmse_se = float(np.sqrt(np.std(squared_error) / np.sqrt(count)))
    mape_se = float(np.std(absolute_error / gt) / np.sqrt(count) * 100)
    pearson_se = float(np.sqrt((1 - pearson ** 2) / (count - 2)))
    snr_se = float(np.std(snr_values) / np.sqrt(count))

    lines = [
        f"FFT MAE (FFT Label): {mae} +/- {mae_se}",
        f"FFT RMSE (FFT Label): {rmse} +/- {rmse_se}",
        f"FFT MAPE (FFT Label): {mape} +/- {mape_se}",
        f"FFT Pearson (FFT Label): {pearson} +/- {pearson_se}",
        f"FFT SNR (FFT Label): {mean_snr} +/- {snr_se} (dB)",
    ]
    (OUTPUT_DIR / "pos_face_roi_metrics.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    methodology = {
        "method": "rPPG-Toolbox POS_WANG logic",
        "subjects": common.EXPECTED_SUBJECTS,
        "face_roi": "Haar first-frame fixed box, enlarged by 1.5, resized to 72x72",
        "source_paths_and_boxes": boxes,
        "fs": common.FS,
        "pos_internal_bandpass_hz": [common.LOW_HZ, common.HIGH_HZ],
        "evaluation_bandpass_fft_snr_hz": [common.LOW_HZ, common.HIGH_HZ],
        "evaluation": "same effective whole-subject durations as EfficientPhys-C",
        "saved_processed_test": str(PROCESSED_TEST_DIR),
        "training": False,
    }
    (OUTPUT_DIR / "methodology.json").write_text(
        json.dumps(methodology, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    common.plot_results(rows)

    print("\n" + "\n".join(lines), flush=True)
    print(f"\n总耗时：{time.perf_counter() - total_start:.1f}s", flush=True)
    print(f"正式POS结果目录：{OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
