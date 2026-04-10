"""
生成独立测试集：从训练/验证未使用的盈余数据中抽取。
确保与 train_v2.txt / val_v2.txt / eval_change.txt 零重叠。

Usage:
    python script/gen_test_set.py
    python script/gen_test_set.py --sdxl 300 --flux 300 --ffhq 400 --forlab 200 --doubao 50 --change 200
"""
import argparse
import random
from pathlib import Path
from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / '.env')

DATA_ROOT = PROJECT_ROOT / os.getenv('DATA_ROOT', 'data')
SEED = 99  # 用不同 seed，避免和训练集采样顺序相关


def list_images(folder, exts=('.png', '.jpg', '.jpeg', '.webp')):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in exts])


def load_used_stems(*ann_files):
    """从多个注释文件中收集已使用的文件 stem，用于排除。"""
    stems = set()
    for fpath in ann_files:
        fpath = Path(fpath)
        if not fpath.exists():
            continue
        for enc in ['utf-8', 'gbk']:
            try:
                with open(fpath, encoding=enc) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        sep = '\t' if '\t' in line else ' '
                        path = line.rsplit(sep, 1)[0]
                        stems.add(Path(path).stem)
                break
            except UnicodeDecodeError:
                continue
    return stems


def main():
    parser = argparse.ArgumentParser(description="生成独立测试集")
    parser.add_argument('--sdxl',   type=int, default=300, help='每个SDXL子类抽取数量')
    parser.add_argument('--flux',   type=int, default=300, help='每个Flux子类抽取数量')
    parser.add_argument('--ffhq',   type=int, default=400, help='FFHQ抽取数量')
    parser.add_argument('--forlab', type=int, default=200, help='FORLAB抽取数量')
    parser.add_argument('--doubao', type=int, default=50,  help='doubao(fack+real各)抽取数量')
    parser.add_argument('--change', type=int, default=0,   help='change额外抽取（默认0，eval_change已有235张）')
    parser.add_argument('--out',    type=str, default='annotation/test_v2.txt')
    args = parser.parse_args()

    rng = random.Random(SEED)

    # 收集所有已用于训练/验证/评估的文件
    used_stems = load_used_stems(
        PROJECT_ROOT / 'annotation' / 'train_v2.txt',
        PROJECT_ROOT / 'annotation' / 'val_v2.txt',
        PROJECT_ROOT / 'annotation' / 'eval_change.txt',
    )
    print(f"已排除的训练/验证/评估文件: {len(used_stems)} 个 stem")

    entries = []

    # ── SDXL ──
    for cat in ['animals', 'faces', 'general', 'landscapes']:
        imgs = list_images(DATA_ROOT / 'sdxl' / cat)
        available = [f for f in imgs if f.stem not in used_stems]
        sampled = rng.sample(available, min(args.sdxl, len(available)))
        entries.extend(f"{f}\t1" for f in sampled)
        print(f"sdxl/{cat:12s} 可用 {len(available):6d} → 抽取 {len(sampled)}")

    # ── Flux ──
    for cat in ['animals', 'faces', 'general', 'landscapes']:
        imgs = list_images(DATA_ROOT / 'flux' / cat)
        available = [f for f in imgs if f.stem not in used_stems]
        sampled = rng.sample(available, min(args.flux, len(available)))
        entries.extend(f"{f}\t1" for f in sampled)
        print(f"flux/{cat:12s} 可用 {len(available):6d} → 抽取 {len(sampled)}")

    # ── Real/FFHQ ──
    imgs = list_images(DATA_ROOT / 'Real' / 'FFHQ')
    available = [f for f in imgs if f.stem not in used_stems]
    sampled = rng.sample(available, min(args.ffhq, len(available)))
    entries.extend(f"{f}\t0" for f in sampled)
    print(f"Real/FFHQ      可用 {len(available):6d} → 抽取 {len(sampled)}")

    # ── Real/FORLAB ──
    imgs = list_images(DATA_ROOT / 'Real' / 'FORLAB')
    available = [f for f in imgs if f.stem not in used_stems]
    sampled = rng.sample(available, min(args.forlab, len(available)))
    entries.extend(f"{f}\t0" for f in sampled)
    print(f"Real/FORLAB    可用 {len(available):6d} → 抽取 {len(sampled)}")

    # ── doubao ──
    for sub, label in [('fack', 1), ('real', 0)]:
        imgs = list_images(DATA_ROOT / 'doubao' / sub)
        available = [f for f in imgs if f.stem not in used_stems]
        sampled = rng.sample(available, min(args.doubao, len(available)))
        entries.extend(f"{f}\t{label}" for f in sampled)
        print(f"doubao/{sub:8s} 可用 {len(available):6d} → 抽取 {len(sampled)}")

    # ── change（可选额外补充）──
    if args.change > 0:
        imgs = list_images(DATA_ROOT / 'change' / 'images')
        available = [f for f in imgs if f.stem not in used_stems]
        sampled = rng.sample(available, min(args.change, len(available)))
        entries.extend(f"{f}\t1" for f in sampled)
        print(f"change         可用 {len(available):6d} → 抽取 {len(sampled)}")

    # ── 打乱 & 写出 ──
    rng.shuffle(entries)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(entries) + '\n')

    real_n = sum(1 for e in entries if e.endswith('\t0'))
    fake_n = sum(1 for e in entries if e.endswith('\t1'))
    print(f"\n{'='*50}")
    print(f"测试集: {len(entries)} 条  (Real={real_n}, Fake={fake_n})")
    print(f"写出: {out_path}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
