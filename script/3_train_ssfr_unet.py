"""
Step 3: 训练 MicroForensicUNet (SSFR 去噪先验模型)
在全流程中位于 1_gen_annotations.py 之后、2_extract_features.py 之前
仅用真实图像训练，输出: outputs/ssfr_unet.pth
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from service.ssfr_extractor_module import train_forensic_unet


def collect_real_paths(data_root, max_images=2000):
    """收集 data/Real/ 下所有真实图像路径"""
    real_root = Path(data_root) / "Real"
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        paths.extend(real_root.rglob(ext))
    paths = [str(p) for p in paths]

    np.random.seed(42)
    np.random.shuffle(paths)
    paths = paths[:max_images]
    print(f"[SSFR UNet] 找到 {len(paths)} 张真实图像 (上限 {max_images})")
    return paths


def main():
    parser = argparse.ArgumentParser(description="训练 SSFR MicroForensicUNet")
    parser.add_argument("--data_root", type=str, default="data", help="数据根目录")
    parser.add_argument("--output", type=str, default="outputs/ssfr_unet.pth", help="模型保存路径")
    parser.add_argument("--epochs", type=int, default=40, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_images", type=int, default=2000, help="最大训练图像数")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("  训练 MicroForensicUNet (SSFR 去噪先验)")
    print("=" * 60)

    real_paths = collect_real_paths(args.data_root, args.max_images)
    if not real_paths:
        print("错误: 未找到真实图像，请检查 data/Real/ 目录")
        return

    train_forensic_unet(
        real_image_paths=real_paths,
        save_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    print("\n完成！模型保存在:", args.output)


if __name__ == "__main__":
    main()
