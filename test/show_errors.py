# -*- coding: gb18030 -*-
import os
import sys
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).parent.parent
misclassified_dir = PROJECT_ROOT / "test" / "预览数据" / "fack"
real_search_dirs = [
    PROJECT_ROOT / "data" / "新数据" / "RealImage_Img" / "RealImage_Img" / "COCO",
    PROJECT_ROOT / "data" / "新数据" / "RealImage_Img" / "RealImage_Img" / "ImageNet",
    PROJECT_ROOT / "data" / "新数据" / "RealImage_Img" / "RealImage_Img" / "Places"
]

out_dir = PROJECT_ROOT / "test" / "score"
out_dir.mkdir(exist_ok=True)

fakes = list(misclassified_dir.glob("*_stuff.png")) + list(misclassified_dir.glob("*.jpg"))
fakes = [f for f in fakes if "_stuff" in f.name or f.stem.isdigit()]

random.shuffle(fakes)
selected_fakes = fakes[:10]

def create_comparison(real_path, fake_path, out_path):
    img_real = Image.open(real_path).convert('RGB')
    img_fake = Image.open(fake_path).convert('RGB')
    if img_real.size != img_fake.size:
        img_fake = img_fake.resize(img_real.size)
    w, h = img_real.size
    new_w = w * 2
    new_img = Image.new('RGB', (new_w, h))
    new_img.paste(img_real, (0, 0))
    new_img.paste(img_fake, (w, 0))
    draw = ImageDraw.Draw(new_img)
    try: font = ImageFont.truetype("arial.ttf", size=max(20, h//20))
    except: font = ImageFont.load_default()
    draw.rectangle([10, 10, 200, 50], fill=(0,0,0,128))
    draw.text((20, 15), "Real Image", fill="white", font=font)
    draw.rectangle([w+10, 10, w+400, 50], fill=(0,0,0,128))
    draw.text((w+20, 15), "Fake Image (Misclassified)", fill="red", font=font)
    new_img.save(out_path)

found_count = 0
for fake in selected_fakes:
    stem = fake.stem.replace('_stuff', '')
    real_img_path = None
    for d in real_search_dirs:
        for ext in ['.jpg', '.png', '.jpeg']:
            cand = d / (stem + ext)
            if cand.exists():
                real_img_path = cand
                break
        if real_img_path: break
    if real_img_path:
        out_path = out_dir / ('compare_' + stem + '.jpg')
        create_comparison(real_img_path, fake, out_path)
        found_count += 1
print("Generated", found_count, "comparison images.")
