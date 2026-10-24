"""Read-only smoke test for the Raw181 EfficientPhys-C experiment."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


TOOLBOX_ROOT = Path("D:/rppg/rPPG-Toolbox")
CONFIG_PATH = Path("G:/efficientphys_raw_paper_seed100.yaml")
CACHE_ROOT = Path("G:/UBFC-processed-raw")

sys.path.insert(0, str(TOOLBOX_ROOT))

from config import get_config  # noqa: E402
from dataset.data_loader.UBFCrPPGLoader import UBFCrPPGLoader  # noqa: E402
from neural_methods.trainer.EfficientPhysTrainer import EfficientPhysTrainer  # noqa: E402


def csv_count(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def make_dataset(name, config_data, device):
    dataset = UBFCrPPGLoader(
        name=name,
        data_path=config_data.DATA_PATH,
        config_data=config_data,
        device=device,
    )
    data, label, subject, chunk = dataset[0]
    if data.shape != (181, 3, 72, 72):
        raise ValueError(f"{name} 输入形状错误：{data.shape}")
    if label.shape != (180,):
        raise ValueError(f"{name} 标签形状错误：{label.shape}")
    if not np.isfinite(data).all() or not np.isfinite(label).all():
        raise ValueError(f"{name} 包含 NaN 或 Inf")
    print(f"{name}: {len(dataset)} 段；样本={subject} chunk={chunk}；输入={data.shape}；标签={label.shape}")
    return dataset, data, label


def main():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"YAML 不存在：{CONFIG_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 当前不可用，停止模型前向检查")

    config = get_config(SimpleNamespace(config_file=str(CONFIG_PATH)))
    checks = {
        "epoch": (config.TRAIN.EPOCHS, 10),
        "lr": (float(config.TRAIN.LR), 0.001),
        "batch": (config.TRAIN.BATCH_SIZE, 4),
        "fs": (config.TEST.DATA.FS, 30),
        "frame_depth": (config.MODEL.EFFICIENTPHYS.FRAME_DEPTH, 10),
        "train_list": (Path(config.TRAIN.DATA.FILE_LIST_PATH), CACHE_ROOT / "train_file_list.csv"),
        "valid_list": (Path(config.VALID.DATA.FILE_LIST_PATH), CACHE_ROOT / "valid_file_list.csv"),
        "test_list": (Path(config.TEST.DATA.FILE_LIST_PATH), CACHE_ROOT / "test_file_list.csv"),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"YAML 检查失败：{name}={actual!r}，期望 {expected!r}")

    print("YAML 参数检查通过：10 epochs，lr=0.001，batch=4，FS=30，frame_depth=10")
    print("清单行数："
          f"train={csv_count(CACHE_ROOT / 'train_file_list.csv')}，"
          f"valid={csv_count(CACHE_ROOT / 'valid_file_list.csv')}，"
          f"test={csv_count(CACHE_ROOT / 'test_file_list.csv')}")

    device = torch.device(config.DEVICE)
    train_set, train_x, train_y = make_dataset("train", config.TRAIN.DATA, device)
    make_dataset("valid", config.VALID.DATA, device)
    make_dataset("test", config.TEST.DATA, device)

    trainer = EfficientPhysTrainer(config, {"train": [None]})
    trainer.model.eval()
    x = torch.from_numpy(train_x).unsqueeze(0).to(device)
    y = torch.from_numpy(train_y).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction, target = trainer._forward_raw181(x, y)
        loss = trainer.criterion(prediction, target)

    if prediction.shape != (180, 1) or target.shape != (180, 1):
        raise ValueError(f"前向形状错误：prediction={prediction.shape}, target={target.shape}")
    if not math.isfinite(float(loss)):
        raise ValueError(f"损失不是有限数：{loss}")

    print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"单片段前向通过：输入={tuple(x.shape)} -> 预测={tuple(prediction.shape)}")
    print(f"负 Pearson 损失（仅冒烟检查，无性能含义）：{float(loss):.6f}")
    print("冒烟检查全部通过。没有训练，也没有保存模型权重。")


if __name__ == "__main__":
    main()
