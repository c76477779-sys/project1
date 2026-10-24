from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:\project\2026-10-17\wei")
FONT = r"C:\Windows\Fonts\msyh.ttc"
BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

W, H = 1680, 570
im = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(im)
title = ImageFont.truetype(BOLD, 43)
label = ImageFont.truetype(FONT, 32)
number = ImageFont.truetype(BOLD, 41)
small = ImageFont.truetype(FONT, 26)

d.rectangle((0, 0, W, 105), fill="#946400")
d.text((55, 25), "EfficientPhys-C 最终模型前向耗时", font=title, fill="white")

items = [
    ("设备与精度", "RTX 5060 Laptop GPU · FP32"),
    ("输入 / 输出", "181 帧 / 180 个有效点"),
    ("测量设置", "batch=1 · 热身 30 次 · 计时 200 次"),
    ("平均耗时", "19.0564 ± 0.3425 ms / 片段"),
    ("P95 耗时", "19.6592 ms / 片段"),
    ("每有效帧摊销", "0.1059 ms / 帧"),
]

left, right, top, row_height = 55, 1625, 123, 65
for i, (key, value) in enumerate(items):
    y0 = top + i * row_height
    d.rectangle((left, y0, right, y0 + row_height),
                fill="#FFF3CF" if i % 2 == 0 else "#FFFAEB")
    d.text((75, y0 + 12), key, font=label, fill="#705000")
    d.text((525, y0 + 7 if i == 3 else y0 + 12), value,
           font=number if i == 3 else label, fill="#3E310C")

d.text((60, 530), "± 为重复测量标准差；仅模型前向，不含数据读取、传输及心率后处理。", font=small,
       fill="#756447")

im.save(ROOT / "最终模型运行效率.png")
im.save(ROOT / "最终模型运行效率.jpg", quality=95, subsampling=0)
