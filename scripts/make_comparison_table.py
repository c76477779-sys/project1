from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(r"D:\project\2026-10-17\wei")
FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

columns = [
    ("方法", 560),
    ("MAE↓ (bpm)", 260),
    ("RMSE↓ (bpm)", 280),
    ("MAPE↓ (%)", 255),
    ("Pearson↑", 235),
    ("SNR↑ (dB)", 250),
]
rows = [
    ("EfficientPhys-C 完整模型", "1.465", "3.275", "1.612", "0.955", "0.119"),
    ("去注意力", "2.148", "4.576", "2.472", "0.960", "−0.469"),
    ("去 BatchNorm", "1.660", "3.327", "1.808", "0.952", "0.112"),
    ("POS（人脸 ROI）", "4.297", "8.151", "4.532", "0.751", "−1.076"),
]

margin = 34
header_height = 80
row_height = 78
width = sum(item[1] for item in columns) + margin * 2
height = header_height + row_height * len(rows) + margin * 2
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)
header_font = ImageFont.truetype(FONT_BOLD, 31)
body_font = ImageFont.truetype(FONT, 32)
body_bold = ImageFont.truetype(FONT_BOLD, 32)

x = margin
for title, col_width in columns:
    draw.rectangle((x, margin, x + col_width, margin + header_height), fill="#234E70")
    bounds = draw.textbbox((0, 0), title, font=header_font)
    text_width = bounds[2] - bounds[0]
    text_x = x + 18 if x == margin else x + (col_width - text_width) / 2
    draw.text((text_x, margin + 18), title, font=header_font, fill="white")
    x += col_width

for row_index, row in enumerate(rows):
    y0 = margin + header_height + row_index * row_height
    y1 = y0 + row_height
    fill = "#EAF2F9" if row_index == 0 else ("#F7F9FB" if row_index % 2 == 0 else "white")
    draw.rectangle((margin, y0, width - margin, y1), fill=fill)
    draw.line((margin, y1, width - margin, y1), fill="#D8E1E9", width=2)
    x = margin
    for col_index, (value, (_, col_width)) in enumerate(zip(row, columns)):
        is_best = (row_index == 0 and col_index in (1, 2, 3, 5)) or (row_index == 1 and col_index == 4)
        font = body_bold if is_best else body_font
        bounds = draw.textbbox((0, 0), value, font=font)
        text_width = bounds[2] - bounds[0]
        text_x = x + 18 if col_index == 0 else x + (col_width - text_width) / 2
        draw.text((text_x, y0 + 18), value, font=font, fill="#182B3B")
        x += col_width

png = OUT / "完整模型_POS_消融对比表.png"
jpg = OUT / "完整模型_POS_消融对比表.jpg"
image.save(png)
image.save(jpg, quality=95, subsampling=0)
print(png)
print(jpg)
