"""Prepare controlled no-attention and no-BN EfficientPhys-C ablations."""

from __future__ import annotations

import shutil
from pathlib import Path


TOOLBOX_ROOT = Path("D:/rppg/rPPG-Toolbox")
MODEL_PATH = TOOLBOX_ROOT / "neural_methods/model/EfficientPhys.py"
BACKUP_PATH = MODEL_PATH.with_suffix(".py.before_raw_paper_ablations.bak")
BASE_YAML = Path("G:/efficientphys_raw_paper_seed100.yaml")
OUTPUT_ROOT = Path("G:/Ablation_Raw_Paper")
MARKER = "# RAW_PAPER_ABLATION_SWITCH_V1"


def patch_model() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到模型代码：{MODEL_PATH}")
    text = MODEL_PATH.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"模型消融开关已经存在：{MODEL_PATH}")
        return

    required = [
        "import torch\nimport torch.nn as nn\n",
        "        self.channel = channel\n",
        "        inputs = torch.diff(inputs, dim=0)\n        inputs = self.batch_norm(inputs)\n",
        """        g1 = torch.sigmoid(self.apperance_att_conv1(d2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1
""",
        """        g2 = torch.sigmoid(self.apperance_att_conv2(d6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2
""",
    ]
    missing = [snippet.splitlines()[0] for snippet in required if snippet not in text]
    if missing:
        raise RuntimeError(
            "EfficientPhys.py 与预期版本不一致，未做任何修改。缺少：\n"
            + "\n".join(missing)
        )

    if not BACKUP_PATH.exists():
        shutil.copy2(MODEL_PATH, BACKUP_PATH)

    text = text.replace(
        "import torch\nimport torch.nn as nn\n",
        f"import os\n\nimport torch\nimport torch.nn as nn\n\n{MARKER}\n",
        1,
    )
    text = text.replace(
        "        self.channel = channel\n",
        """        self.channel = channel
        self.ablation = os.environ.get("EFFICIENTPHYS_ABLATION", "full")
        if self.ablation not in {"full", "no_attention", "no_bn"}:
            raise ValueError(f"Unsupported EFFICIENTPHYS_ABLATION: {self.ablation}")
        print(f"EfficientPhys ablation mode: {self.ablation}")
""",
        1,
    )
    text = text.replace(
        "        inputs = torch.diff(inputs, dim=0)\n        inputs = self.batch_norm(inputs)\n",
        """        inputs = torch.diff(inputs, dim=0)
        if self.ablation != "no_bn":
            inputs = self.batch_norm(inputs)
""",
        1,
    )
    text = text.replace(
        """        g1 = torch.sigmoid(self.apperance_att_conv1(d2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1
""",
        """        if self.ablation == "no_attention":
            gated1 = d2
        else:
            g1 = torch.sigmoid(self.apperance_att_conv1(d2))
            g1 = self.attn_mask_1(g1)
            gated1 = d2 * g1
""",
        1,
    )
    text = text.replace(
        """        g2 = torch.sigmoid(self.apperance_att_conv2(d6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2
""",
        """        if self.ablation == "no_attention":
            gated2 = d6
        else:
            g2 = torch.sigmoid(self.apperance_att_conv2(d6))
            g2 = self.attn_mask_2(g2)
            gated2 = d6 * g2
""",
        1,
    )
    compile(text, str(MODEL_PATH), "exec")
    MODEL_PATH.write_text(text, encoding="utf-8")
    print(f"已加入消融开关：{MODEL_PATH}")
    print(f"修改前备份：{BACKUP_PATH}")


def make_configs() -> None:
    if not BASE_YAML.is_file():
        raise FileNotFoundError(f"找不到最终完整模型配置：{BASE_YAML}")
    base = BASE_YAML.read_text(encoding="utf-8-sig")
    expected_model = "MODEL_FILE_NAME: UBFC_Dataset2_EfficientPhysC_Raw_Paper_seed100"
    expected_log = "PATH: G:/第三次训练结果_Raw_Paper"
    if expected_model not in base or expected_log not in base:
        raise RuntimeError("基础YAML内容与最终实验不一致，未生成消融配置。")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    variants = {
        "no_attention": "UBFC_Dataset2_EfficientPhysC_Raw_Paper_no_attention_seed100",
        "no_bn": "UBFC_Dataset2_EfficientPhysC_Raw_Paper_no_bn_seed100",
    }
    for variant, model_name in variants.items():
        result_root = f"G:/Ablation_Raw_Paper/{variant}_seed100"
        result_path = Path(result_root)
        result_path.mkdir(parents=True, exist_ok=True)
        content = base.replace(expected_model, f"MODEL_FILE_NAME: {model_name}", 1)
        content = content.replace(expected_log, f"PATH: {result_root}", 1)
        config_path = OUTPUT_ROOT / f"{variant}_seed100.yaml"
        config_path.write_text(content, encoding="utf-8")
        (result_path / "ablation_definition.txt").write_text(
            (
                "variant=" + variant + "\n"
                "base_config=G:/efficientphys_raw_paper_seed100.yaml\n"
                "seed=100\n"
                "epochs=10\n"
                "loss=negative Pearson\n"
                "learning_rate=0.001\n"
                "optimizer=AdamW, weight_decay=0.01\n"
                "subjects=train29, valid4, test9 (subject41-subject49)\n"
                "evaluation=whole-subject FFT, 0.75-2.5 Hz\n"
                + (
                    "change=skip both spatial attention gates; keep input BatchNorm\n"
                    if variant == "no_attention"
                    else "change=skip input BatchNorm; keep both spatial attention gates\n"
                )
            ),
            encoding="utf-8",
        )
        print(f"已生成 {variant} 配置：{config_path}")


def main() -> None:
    patch_model()
    make_configs()
    print("\n准备完成。完整模型默认仍使用 full 模式。")
    print("下一步单独启动 no_attention 训练。")


if __name__ == "__main__":
    main()
