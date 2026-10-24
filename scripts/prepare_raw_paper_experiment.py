"""Prepare the paper-aligned EfficientPhys-C experiment after preprocessing.

Default mode is read-only and verifies the 42-subject cache.  Pass --apply to:
1. back up and patch EfficientPhysTrainer for 181 RGB frames -> 180 targets;
2. use paper settings for EfficientPhys-C training;
3. use 0.75-2.5 Hz for evaluation bandpass, FFT search and the rPPG-Toolbox
   SNR calculation;
4. write the experiment YAML to G:\\efficientphys_raw_paper_seed100.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import py_compile
import shutil
from pathlib import Path

import numpy as np


CACHE_ROOT = Path("G:/UBFC-processed-raw")
TOOLBOX_ROOT = Path("D:/rppg/rPPG-Toolbox")
TRAINER_PATH = TOOLBOX_ROOT / "neural_methods/trainer/EfficientPhysTrainer.py"
MODEL_PATH = TOOLBOX_ROOT / "neural_methods/model/EfficientPhys.py"
POST_PROCESS_PATH = TOOLBOX_ROOT / "evaluation/post_process.py"
YAML_PATH = Path("G:/efficientphys_raw_paper_seed100.yaml")

EXPECTED_SUBJECTS = {"train": 29, "valid": 4, "test": 9}
EXPECTED_SHAPES = {
    "input": ((181, 72, 72, 3), np.dtype("float32")),
    "label": ((180,), np.dtype("float32")),
}

TRAINER_MARKER = "# RAW181_PAPER_PATCH_V1"

YAML_TEXT = """BASE: ['']
TOOLBOX_MODE: train_and_test

TRAIN:
  BATCH_SIZE: 4
  EPOCHS: 10
  LR: 1e-3
  MODEL_FILE_NAME: UBFC_Dataset2_EfficientPhysC_Raw_Paper_seed100
  PLOT_LOSSES_AND_LR: True
  DATA:
    FS: 30
    DATASET: UBFC-rPPG
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: G:/UBFC-processed-raw
    CACHED_PATH: G:/UBFC-processed-raw
    FILE_LIST_PATH: G:/UBFC-processed-raw/train_file_list.csv
    EXP_DATA_NAME: train
    BEGIN: 0.0
    END: 1.0
    PREPROCESS:
      DATA_TYPE: ['Raw']
      DATA_AUG: ['None']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: False
        BACKEND: HC
        USE_LARGE_FACE_BOX: False
        LARGE_BOX_COEF: 1.0
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72

VALID:
  DATA:
    FS: 30
    DATASET: UBFC-rPPG
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: G:/UBFC-processed-raw
    CACHED_PATH: G:/UBFC-processed-raw
    FILE_LIST_PATH: G:/UBFC-processed-raw/valid_file_list.csv
    EXP_DATA_NAME: valid
    BEGIN: 0.0
    END: 1.0
    PREPROCESS:
      DATA_TYPE: ['Raw']
      DATA_AUG: ['None']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: False
        BACKEND: HC
        USE_LARGE_FACE_BOX: False
        LARGE_BOX_COEF: 1.0
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72

TEST:
  METRICS: ['MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'BA']
  USE_LAST_EPOCH: False
  DATA:
    FS: 30
    DATASET: UBFC-rPPG
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: G:/UBFC-processed-raw
    CACHED_PATH: G:/UBFC-processed-raw
    FILE_LIST_PATH: G:/UBFC-processed-raw/test_file_list.csv
    EXP_DATA_NAME: test
    BEGIN: 0.0
    END: 1.0
    PREPROCESS:
      DATA_TYPE: ['Raw']
      DATA_AUG: ['None']
      LABEL_TYPE: DiffNormalized
      DO_CHUNK: True
      CHUNK_LENGTH: 180
      CROP_FACE:
        DO_CROP_FACE: False
        BACKEND: HC
        USE_LARGE_FACE_BOX: False
        LARGE_BOX_COEF: 1.0
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 72
        W: 72

DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1
LOG:
  PATH: G:/第三次训练结果_Raw_Paper
MODEL:
  NAME: EfficientPhys
  MODEL_DIR: PreTrainedModels
  EFFICIENTPHYS:
    FRAME_DEPTH: 10
INFERENCE:
  BATCH_SIZE: 1
  EVALUATION_METHOD: FFT
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: False
    WINDOW_SIZE: 10
  MODEL_PATH: ''
"""


def load_csv(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少清单：{path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or "input_files" not in rows[0]:
        raise ValueError(f"清单为空或缺少 input_files 列：{path}")
    return [Path(row["input_files"]) for row in rows]


def verify_cache() -> dict[str, int]:
    manifest = CACHE_ROOT / "preprocessing_config.json"
    progress = CACHE_ROOT / "progress.json"
    if not manifest.is_file() or not progress.is_file():
        raise FileNotFoundError("缺少 preprocessing_config.json 或 progress.json，请先运行 --verify-only")

    settings = json.loads(manifest.read_text(encoding="utf-8"))["settings"]
    required = {
        "data_type": "Raw",
        "label_type": "DiffNormalized",
        "input_frames": 181,
        "target_frames": 180,
        "fs": 30,
        "seed": 100,
    }
    for key, value in required.items():
        if settings.get(key) != value:
            raise ValueError(f"预处理配置不匹配：{key}={settings.get(key)!r}，期望 {value!r}")

    state = json.loads(progress.read_text(encoding="utf-8"))
    if state.get("complete") != 42 or state.get("missing"):
        raise ValueError(f"预处理尚未完成 42/42：{state}")

    result = {}
    all_paths: list[Path] = []
    for split, expected_people in EXPECTED_SUBJECTS.items():
        subjects = [p for p in (CACHE_ROOT / split).iterdir() if p.is_dir() and p.name.startswith("subject")]
        if len(subjects) != expected_people:
            raise ValueError(f"{split} 受试者数为 {len(subjects)}，期望 {expected_people}")
        paths = load_csv(CACHE_ROOT / f"{split}_file_list.csv")
        if not paths:
            raise ValueError(f"{split} 清单没有片段")
        result[split] = len(paths)
        all_paths.extend(paths)

    for input_path in all_paths:
        label_path = Path(str(input_path).replace("input", "label"))
        if not input_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"片段或标签不存在：{input_path} / {label_path}")
        x = np.load(input_path, mmap_mode="r", allow_pickle=False)
        y = np.load(label_path, mmap_mode="r", allow_pickle=False)
        if (x.shape, x.dtype) != EXPECTED_SHAPES["input"]:
            raise ValueError(f"输入格式错误：{input_path}，{x.shape}，{x.dtype}")
        if (y.shape, y.dtype) != EXPECTED_SHAPES["label"] or not np.isfinite(y).all():
            raise ValueError(f"标签格式错误：{label_path}，{y.shape}，{y.dtype}")
    return result


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"无法唯一定位 {description}：找到 {count} 处；未修改文件")
    return text.replace(old, new, 1)


def patch_trainer() -> None:
    original = TRAINER_PATH.read_text(encoding="utf-8")
    if TRAINER_MARKER in original:
        print(f"训练器已经是 Raw181 版本：{TRAINER_PATH}")
        return

    text = original
    text = replace_once(
        text,
        '"""Trainer for EfficientPhys."""',
        '"""Trainer for EfficientPhys."""\n\n' + TRAINER_MARKER,
        "训练器文件头",
    )
    text = replace_once(
        text,
        "            self.criterion = torch.nn.MSELoss()\n"
        "            self.optimizer = optim.AdamW(\n"
        "                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)\n"
        "            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html\n"
        "            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(\n"
        "                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)",
        "            # EfficientPhys-C paper: negative Pearson, AdamW, lr=0.001.\n"
        "            # Paper does not specify weight decay; this run uses 0.01.\n"
        "            self.criterion = Neg_Pearson()\n"
        "            self.optimizer = optim.AdamW(\n"
        "                self.model.parameters(), lr=config.TRAIN.LR,\n"
        "                weight_decay=0.01)",
        "损失函数和优化器",
    )

    insertion_point = '            raise ValueError("EfficientPhys trainer initialized in incorrect toolbox mode!")\n\n'
    helper = '''            raise ValueError("EfficientPhys trainer initialized in incorrect toolbox mode!")

    def _forward_raw181(self, data, labels):
        """Run each 181-frame clip independently and return 180 predictions."""
        if data.ndim != 5 or labels.ndim != 2:
            raise ValueError(f"Expected data [N,181,C,H,W] and labels [N,180], got {data.shape}, {labels.shape}")
        if data.shape[1] != labels.shape[1] + 1 or labels.shape[1] != self.chunk_len:
            raise ValueError(f"Raw181 alignment error: data {data.shape}, labels {labels.shape}")
        predictions = [self.model(data[index]) for index in range(data.shape[0])]
        predictions = torch.cat(predictions, dim=0)
        targets = labels.reshape(-1, 1)
        if predictions.shape != targets.shape:
            raise ValueError(f"Prediction/label mismatch: {predictions.shape}, {targets.shape}")
        return predictions, targets

'''
    text = replace_once(text, insertion_point, helper, "Raw181 前向函数插入点")

    old_train = '''                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                data = torch.cat((data, last_frame), 0)
                labels = labels[:(N * D) // self.base_len * self.base_len]
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
'''
    new_train = '''                self.optimizer.zero_grad()
                pred_ppg, labels = self._forward_raw181(data, labels)
'''
    text = replace_once(text, old_train, new_train, "训练阶段 181→180 前向")
    text = replace_once(
        text,
        "                lrs.append(self.scheduler.get_last_lr())",
        "                lrs.append(self.optimizer.param_groups[0]['lr'])",
        "学习率记录",
    )
    text = replace_once(text, "                self.scheduler.step()\n", "", "学习率调度器步进")

    old_valid = '''                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data_valid[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                data_valid = torch.cat((data_valid, last_frame), 0)
                labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
                pred_ppg_valid = self.model(data_valid)
'''
    text = replace_once(
        text,
        old_valid,
        "                pred_ppg_valid, labels_valid = self._forward_raw181(data_valid, labels_valid)\n",
        "验证阶段 181→180 前向",
    )

    old_test = '''                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data_test[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                data_test = torch.cat((data_test, last_frame), 0)
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                pred_ppg_test = self.model(data_test)
'''
    text = replace_once(
        text,
        old_test,
        "                pred_ppg_test, labels_test = self._forward_raw181(data_test, labels_test)\n",
        "测试阶段 181→180 前向",
    )

    backup = TRAINER_PATH.with_suffix(".py.before_raw181.bak")
    if not backup.exists():
        shutil.copy2(TRAINER_PATH, backup)
    TRAINER_PATH.write_text(text, encoding="utf-8")
    py_compile.compile(str(TRAINER_PATH), doraise=True)
    print(f"训练器已备份并更新：{TRAINER_PATH}")


def patch_model_reshape() -> None:
    """Allow flattening a non-contiguous tensor produced by the TSM path."""
    original = MODEL_PATH.read_text(encoding="utf-8")
    old = "        d9 = d8.view(d8.size(0), -1)"
    new = "        d9 = d8.reshape(d8.size(0), -1)"
    if old in original:
        text = replace_once(original, old, new, "EfficientPhys 非连续张量展平")
        backup = MODEL_PATH.with_suffix(".py.before_raw181_reshape.bak")
        if not backup.exists():
            shutil.copy2(MODEL_PATH, backup)
        MODEL_PATH.write_text(text, encoding="utf-8")
        py_compile.compile(str(MODEL_PATH), doraise=True)
        print(f"模型展平操作已备份并更新：{MODEL_PATH}")
    elif new in original:
        print(f"模型展平操作已经使用 reshape：{MODEL_PATH}")
    else:
        raise RuntimeError("无法识别 EfficientPhys 特征展平代码，未修改模型")


def patch_evaluation() -> None:
    original = POST_PROCESS_PATH.read_text(encoding="utf-8")
    text = original
    fft_old = "def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.6, high_pass=3.3):"
    fft_new = "def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):"
    if fft_old in text:
        text = replace_once(text, fft_old, fft_new, "FFT 搜索范围")
    elif fft_new not in text:
        raise RuntimeError("无法识别 FFT 搜索范围，未修改评价代码")

    filter_old = "[b, a] = butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')"
    filter_new = "[b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')"
    if filter_old in text:
        text = replace_once(text, filter_old, filter_new, "带通滤波范围")
    elif filter_new not in text:
        raise RuntimeError("无法识别带通滤波范围，未修改评价代码")

    snr_old = "def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.6, high_pass=3.3):"
    snr_new = "def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.75, high_pass=2.5):"
    if snr_old in text:
        text = replace_once(text, snr_old, snr_new, "SNR 统计范围")
    elif snr_new not in text:
        raise RuntimeError("无法识别 SNR 统计范围，未修改评价代码")

    if text != original:
        backup = POST_PROCESS_PATH.with_suffix(".py.before_paper_band.bak")
        if not backup.exists():
            shutil.copy2(POST_PROCESS_PATH, backup)
        POST_PROCESS_PATH.write_text(text, encoding="utf-8")
        py_compile.compile(str(POST_PROCESS_PATH), doraise=True)
        print(f"评价代码已备份并更新：{POST_PROCESS_PATH}")
    else:
        print(f"评价代码已经是论文频段：{POST_PROCESS_PATH}")


def write_yaml() -> None:
    if YAML_PATH.exists() and YAML_PATH.read_text(encoding="utf-8") != YAML_TEXT:
        raise FileExistsError(f"YAML 已存在但内容不同，请先保留并检查：{YAML_PATH}")
    YAML_PATH.write_text(YAML_TEXT, encoding="utf-8")
    print(f"YAML 已生成：{YAML_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="备份并更新 Toolbox，生成 YAML；默认只核验")
    args = parser.parse_args()

    counts = verify_cache()
    print("预处理缓存检查通过：" + "，".join(f"{key}={value} 段" for key, value in counts.items()))
    print("受试者划分：train=29，valid=4，test=9；输入181帧，标签180点。")
    if not args.apply:
        print("当前为只读检查；加 --apply 准备训练代码和 YAML，不会启动训练。")
        return

    patch_trainer()
    patch_model_reshape()
    patch_evaluation()
    write_yaml()
    print("准备完成。下一步先检查 YAML 和数据加载，再启动正式训练。")


if __name__ == "__main__":
    main()
