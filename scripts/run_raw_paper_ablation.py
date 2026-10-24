"""Launch one controlled Raw/Paper EfficientPhys-C ablation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parent
CONFIG_ROOT = Path("G:/Ablation_Raw_Paper")
VARIANTS = ("no_attention", "no_bn")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    args = parser.parse_args()

    config_path = CONFIG_ROOT / f"{args.variant}_seed100.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"找不到配置：{config_path}\n请先运行 prepare_raw_paper_ablations.py"
        )

    result_root = CONFIG_ROOT / f"{args.variant}_seed100"
    if (result_root / "train/PreTrainedModels").exists():
        raise RuntimeError(
            f"结果目录已经包含模型，停止以避免覆盖：{result_root}\n"
            "如果是一次完整的旧实验，请先保留；不要直接重复运行。"
        )

    os.environ["EFFICIENTPHYS_ABLATION"] = args.variant
    print(f"消融模式：{args.variant}", flush=True)
    print(f"配置：{config_path}", flush=True)
    print(f"结果：{result_root}", flush=True)
    print("即将调用修改后的 rPPG-Toolbox 进行10轮训练、最佳验证轮选择和测试。\n", flush=True)

    sys.path.insert(0, str(WORK_DIR))
    import run_training_clean

    sys.argv = [
        str(WORK_DIR / "run_training_clean.py"),
        "--config_file",
        str(config_path),
    ]
    run_training_clean.main()


if __name__ == "__main__":
    main()
