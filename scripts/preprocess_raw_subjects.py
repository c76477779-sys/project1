"""UBFC Dataset2: full-frame RGB, 72x72, 181 input frames / 180 targets.

No video standardization, face crop, video differencing or bandpass filtering.
Requires the matching 181-to-180 training entry, not the old Toolbox trainer.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SOURCE = Path(r"D:\Microsoft  edge")
OUTPUT_ROOT = Path("G:/UBFC-processed-raw")
SPLIT_CSV = Path("D:/rppg/EfficientPhysC_UBFC_Dataset2_subject_split.csv")
SEED = 100
FS = 30
LENGTH = 180
SIZE = 72
SETTINGS = {
    "schema": "efficientphys_raw181_v1",
    "data_type": "Raw",
    "color": "RGB",
    "pixel_scale": "0..255; no division or standardization",
    "crop_face": False,
    "resize": [SIZE, SIZE],
    "interpolation": "cv2.INTER_AREA",
    "label_type": "DiffNormalized",
    "label_normalization": "whole-video np.diff / np.std(ddof=0), append zero",
    "input_frames": LENGTH + 1,
    "target_frames": LENGTH,
    "stride": LENGTH,
    "fs": FS,
    "seed": SEED,
    "last_frame": "repeat only if the final input frame is unavailable",
    "tail": "discard fewer than 180 target points",
    "training_entry": "requires per-clip difference before flattening",
}


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, text):
        self.console.write(text)
        self.file.write(text)
        return len(text)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def isatty(self):
        return False


def save_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_split():
    with SPLIT_CSV.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    splits = {}
    for row in rows:
        subject, split = row["subject_id"].strip(), row["split"].strip()
        if subject in splits or split not in ("train", "valid", "test"):
            raise ValueError(f"Invalid or duplicate split row: {row}")
        if int(row["seed"]) != SEED:
            raise ValueError(f"Unexpected seed in split CSV: {row}")
        splits[subject] = split
    counts = {name: list(splits.values()).count(name) for name in ("train", "valid", "test")}
    if counts != {"train": 29, "valid": 4, "test": 9}:
        raise ValueError(f"Unexpected split counts: {counts}")
    return splits


def read_labels(path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    bvp = np.array([float(value) for value in lines[0].split()], dtype=np.float64)
    if len(bvp) < LENGTH or not np.isfinite(bvp).all():
        raise ValueError(f"Empty, short or non-finite BVP: {path}")
    difference = np.diff(bvp)
    scale = float(np.std(difference))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"BVP difference has invalid standard deviation: {path}")
    labels = np.concatenate((difference / scale, [0.0])).astype(np.float32)
    timestamps = None
    if len(lines) >= 3 and lines[2].strip():
        timestamps = np.array([float(value) for value in lines[2].split()], dtype=np.float64)
        if len(timestamps) != len(bvp) or not np.isfinite(timestamps).all():
            raise ValueError(f"Timestamp count or values invalid: {path}")
        if np.any(np.diff(timestamps) < 0):
            raise ValueError(f"Timestamps are not ordered: {path}")
    return bvp, labels, timestamps


def scan_source(root, splits):
    if not root.is_dir():
        raise FileNotFoundError(f"Source folder unavailable: {root}")
    jobs = []
    problems = []
    for outer in root.iterdir():
        subject = outer.name.lower()
        if not outer.is_dir() or subject not in splits:
            continue
        candidates = (outer, outer / outer.name)
        folder = next((p for p in candidates if (p / "vid.avi").is_file()
                       and (p / "ground_truth.txt").is_file()), None)
        if folder is None:
            problems.append(f"{outer}: missing vid.avi / ground_truth.txt")
            continue
        video = folder / "vid.avi"
        try:
            bvp, labels, timestamps = read_labels(folder / "ground_truth.txt")
            capture = cv2.VideoCapture(str(video))
            try:
                if not capture.isOpened():
                    raise ValueError("Cannot open video")
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                capture.release()
            if reported > 0 and reported != len(bvp):
                raise ValueError(f"Reported frames {reported} != labels {len(bvp)}; check alignment")
            stat = video.stat()
            signature = {"video_path": str(video.resolve()), "video_bytes": stat.st_size,
                         "video_mtime_ns": stat.st_mtime_ns,
                         "label_sha256": hashlib.sha256((folder / "ground_truth.txt").read_bytes()).hexdigest()}
            jobs.append({"subject": subject, "split": splits[subject], "folder": folder,
                         "bvp": bvp, "labels": labels, "timestamps": timestamps,
                         "fps": fps, "signature": signature})
        except Exception as error:
            problems.append(f"{subject}: {error}")
    if problems:
        raise RuntimeError("Source check failed:\n" + "\n".join(problems))
    if not jobs:
        raise RuntimeError(f"No subjects from the split CSV found directly under {root}")
    return sorted(jobs, key=lambda job: int(job["subject"][7:]))


def resize_video(path, expected_frames):
    frames = np.empty((expected_frames, SIZE, SIZE, 3), dtype=np.uint8)
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if count >= expected_frames:
                raise ValueError(f"Video contains more frames than labels: {path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames[count] = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            count += 1
    finally:
        capture.release()
    if count != expected_frames:
        raise ValueError(f"Decoded {count} frames, expected {expected_frames}: {path}")
    return frames


def make_clip(frames, labels, index):
    start = index * LENGTH
    clip = frames[start:start + LENGTH + 1]
    padded = len(clip) == LENGTH
    if padded:
        clip = np.concatenate((clip, clip[-1:]), axis=0)
    target = labels[start:start + LENGTH]
    if clip.shape != (LENGTH + 1, SIZE, SIZE, 3) or target.shape != (LENGTH,):
        raise ValueError(f"Invalid clip shape: {clip.shape}, {target.shape}")
    return clip.astype(np.float32), target, padded


def verify_subject(folder, subject, split):
    metadata = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if (metadata["settings"] != SETTINGS or metadata["subject"] != subject
            or metadata["split"] != split or metadata["clips"] <= 0):
        raise ValueError(f"Cache settings or subject mismatch: {folder}")
    count = metadata["clips"]
    if (len(list(folder.glob(f"{subject}_input*.npy"))) != count
            or len(list(folder.glob(f"{subject}_label*.npy"))) != count):
        raise ValueError(f"Cache file count mismatch: {folder}")
    inputs = []
    previous_last = None
    for index in range(count):
        path = folder / f"{subject}_input{index}.npy"
        x = np.load(path, mmap_mode="r", allow_pickle=False)
        y = np.load(folder / f"{subject}_label{index}.npy", allow_pickle=False)
        if x.shape != (LENGTH + 1, SIZE, SIZE, 3) or x.dtype != np.float32:
            raise ValueError(f"Invalid input shape or dtype: {path}")
        if y.shape != (LENGTH,) or y.dtype != np.float32 or not np.isfinite(y).all():
            raise ValueError(f"Invalid labels: {folder}, clip {index}")
        if previous_last is not None and not np.array_equal(previous_last, x[0]):
            raise ValueError(f"Clip boundary mismatch: {path}")
        previous_last = np.array(x[-1], copy=True)
        del x
        inputs.append(path)
    return metadata, inputs


def process_subject(job):
    subject, split = job["subject"], job["split"]
    final = OUTPUT_ROOT / split / subject
    if final.exists():
        metadata, _ = verify_subject(final, subject, split)
        if metadata["source"] != job["signature"]:
            raise ValueError(f"Source files changed since this cache was made: {subject}")
        print(f"跳过已完成且检查通过：{subject}", flush=True)
        return
    temporary = OUTPUT_ROOT / "_incomplete" / subject
    if temporary.exists():
        raise RuntimeError(f"发现未完成的目录，请保留报错信息再处理：{temporary}")
    started = time.perf_counter()
    print(f"\n处理 {subject} -> {split}", flush=True)
    frames = resize_video(job["folder"] / "vid.avi", len(job["bvp"]))
    count = len(frames) // LENGTH
    temporary.mkdir(parents=True)
    padded_count = 0
    for index in range(count):
        clip, target, padded = make_clip(frames, job["labels"], index)
        np.save(temporary / f"{subject}_input{index}.npy", clip, allow_pickle=False)
        np.save(temporary / f"{subject}_label{index}.npy", target, allow_pickle=False)
        padded_count += int(padded)
    del frames
    np.save(temporary / "bvp_raw.npy", job["bvp"], allow_pickle=False)
    if job["timestamps"] is not None:
        np.save(temporary / "timestamps.npy", job["timestamps"], allow_pickle=False)
    save_json(temporary / "complete.json", {
        "subject": subject, "split": split, "settings": SETTINGS,
        "source": job["signature"], "clips": count,
        "original_frames": len(job["bvp"]), "effective_target_points": count * LENGTH,
        "discarded_tail_points": len(job["bvp"]) % LENGTH,
        "padded_final_inputs": padded_count, "video_reported_fps": job["fps"],
        "numpy": np.__version__, "opencv": cv2.__version__,
    })
    verify_subject(temporary, subject, split)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary.rename(final)
    print(f"完成 {subject}：{count} 段，末尾补帧 {padded_count}，"
          f"耗时 {(time.perf_counter() - started) / 60:.2f} 分钟", flush=True)


def write_lists(splits):
    complete = set()
    for split in ("train", "valid", "test"):
        inputs = []
        for subject in sorted(splits, key=lambda value: int(value[7:])):
            if splits[subject] != split:
                continue
            folder = OUTPUT_ROOT / split / subject
            if not folder.exists():
                continue
            _, paths = verify_subject(folder, subject, split)
            inputs.extend(paths)
            complete.add(subject)
        path = OUTPUT_ROOT / f"{split}_file_list.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["input_files"])
            writer.writerows([[str(path)] for path in inputs])
        temporary.replace(path)
        people = sum(splits[s] == split for s in complete)
        print(f"{split}: {people} 人，{len(inputs)} 段 -> {path}", flush=True)
    missing = sorted(set(splits) - complete, key=lambda value: int(value[7:]))
    save_json(OUTPUT_ROOT / "progress.json", {"complete": len(complete), "total": len(splits), "missing": missing})
    print(f"\n累计完成：{len(complete)}/{len(splits)} 人", flush=True)
    if missing:
        print("尚未完成：" + ", ".join(missing), flush=True)
    else:
        print("全部完成：训练 29 人、验证 4 人、测试 9 人。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run", action="store_true", help="实际处理；默认仅检查源文件")
    parser.add_argument("--verify-only", action="store_true", help="核验已完成缓存并重建累计清单")
    args = parser.parse_args()
    if args.run and args.verify_only:
        parser.error("--run 与 --verify-only 不能同时使用")
    splits = read_split()
    jobs = [] if args.verify_only else scan_source(args.source, splits)
    print("视频：完整 RGB 画面 -> 72x72 INTER_AREA；Raw，保留 0..255 像素尺度")
    print("标签：全视频 DiffNormalized；181 张输入对应 180 个标签；配置采样率 30 Hz")
    print(f"输出：{OUTPUT_ROOT}")
    for job in jobs:
        print(f"{job['subject']} -> {job['split']}，{len(job['bvp'])} 帧，"
              f"{len(job['bvp']) // LENGTH} 段，文件标注帧率 {job['fps']:.4f}")
    if not args.run and not args.verify_only:
        print("\n源文件及元数据检查通过。未解码全部视频；实际运行还会检查完整解码帧数。")
        print("加 --run 开始处理；本批无需找齐全部 42 人。")
        return
    if not Path("G:/").is_dir():
        raise FileNotFoundError("G 盘未连接或无法访问")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT_ROOT / "preprocessing_config.json"
    expected = {"settings": SETTINGS, "subject_split": splits}
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"已有缓存使用其他设置，不能混用：{manifest}")
    elif args.verify_only or any(OUTPUT_ROOT.iterdir()):
        raise ValueError(f"缺少配置记录，不能认定目录为本次缓存：{OUTPUT_ROOT}")
    else:
        save_json(manifest, expected)
    lock = OUTPUT_ROOT / ".preprocess.lock"
    with lock.open("x", encoding="utf-8") as file:
        file.write(f"pid={os.getpid()}\n")
    try:
        random.seed(SEED)
        np.random.seed(SEED)
        log_path = OUTPUT_ROOT / f"preprocess_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                print(f"日志：{log_path}", flush=True)
                print(json.dumps(expected, ensure_ascii=False, indent=2), flush=True)
                with (OUTPUT_ROOT / "fixed_subject_split.csv").open("w", encoding="utf-8", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["subject_id", "split", "seed"])
                    writer.writerows([[subject, split, SEED] for subject, split in splits.items()])
                try:
                    for job in jobs:
                        process_subject(job)
                    write_lists(splits)
                except Exception:
                    import traceback
                    traceback.print_exc(file=sys.stdout)
                    raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
