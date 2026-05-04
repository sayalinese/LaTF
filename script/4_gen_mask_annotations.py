"""
生成 SegFormer 定位模型专用的 BR 标注文件。
仅使用 BR Stuff 伪造图（有 mask）作为正样本，RealImage_Img 作为负样本。

输出文件:
  - annotation/train_seg.txt
  - annotation/val_seg.txt

Usage:
    python script/gen_mask_annotations.py
"""
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / '.env')

DATA_ROOT = PROJECT_ROOT / os.getenv('DATA_ROOT', 'data')
SEED = int(os.getenv('SEED', '42'))
VAL_RATIO = float(os.getenv('VAL_RATIO', '0.12'))

# 每个 forged group(method/source)最多抽样数量。
BR_SEG_POS_SAMPLE_PER_GROUP = int(os.getenv('BR_SEG_POS_SAMPLE_PER_GROUP', '1000'))
# 负样本目标比例: target_neg = pos * ratio
BR_SEG_NEG_RATIO = float(os.getenv('BR_SEG_NEG_RATIO', '1.0'))
EXCLUDE_EVAL_IDS_FROM_REAL = os.getenv('EXCLUDE_EVAL_IDS_FROM_REAL', '0') == '1'

FORGED_METHODS = ('BrushNet', 'LaMa', 'MAT', 'PowerPaint', 'SDXL')
SOURCE_FOLDERS = ('COCO', 'ImageNet', 'Places')
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
MASK_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
NUM_PATTERN = re.compile(r'(\d{5,})')

BR_FORGED_STUFF_ROOT = DATA_ROOT / '新数据' / 'BR-Gen' / 'Forged'
BR_MASK_STUFF_ROOT = DATA_ROOT / '新数据' / 'BR-Gen' / 'Mask' / 'Stuff'
BR_REAL_ROOT = DATA_ROOT / '新数据' / 'RealImage_Img' / 'RealImage_Img'
EVAL_FILE = PROJECT_ROOT / 'annotation' / 'eval_brgen_stuff_val.txt'


def canonical_id(stem):
    stem = stem.lower()
    if stem.endswith('_mask'):
        stem = stem[:-5]
    if stem.endswith('_stuff'):
        stem = stem[:-6]
    matches = NUM_PATTERN.findall(stem)
    if matches:
        return matches[-1]
    return stem


def list_images(folder, exts=IMG_EXTS):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in exts)


def sample_files(files, n, rng):
    if n <= 0 or n >= len(files):
        return list(files)
    return rng.sample(files, n)


def load_eval_ids(eval_file):
    ids = set()
    eval_file = Path(eval_file)
    if not eval_file.exists():
        return ids

    for enc in ('utf-8', 'gbk', 'latin-1'):
        try:
            with open(eval_file, 'r', encoding=enc) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sep = '\t' if '\t' in line else ' '
                    path = line.rsplit(sep, 1)[0]
                    ids.add(canonical_id(Path(path).stem))
            break
        except UnicodeDecodeError:
            continue
    return ids


def has_br_mask(stem, source):
    source_mask_dir = BR_MASK_STUFF_ROOT / source
    for ext in MASK_EXTS:
        if (source_mask_dir / f'{stem}{ext}').exists():
            return True
    for ext in MASK_EXTS:
        if (source_mask_dir / f'{stem}_mask{ext}').exists():
            return True
    return False


def split_train_val(entries, val_ratio, rng):
    entries = list(entries)
    rng.shuffle(entries)
    total = len(entries)
    if total == 0:
        return [], []

    val_n = int(total * val_ratio)
    if total >= 2:
        val_n = max(1, val_n)
    else:
        val_n = 0

    if val_n >= total:
        val_n = total - 1

    val_entries = entries[:val_n]
    train_entries = entries[val_n:]
    return train_entries, val_entries


def main():
    rng = random.Random(SEED)
    eval_ids = load_eval_ids(EVAL_FILE)
    print(f'已排除评估ID: {len(eval_ids)}')

    # 1) 正样本: BR Forged Stuff (仅 mask 可用样本)
    positive = []
    for method in FORGED_METHODS:
        for source in SOURCE_FOLDERS:
            folder = BR_FORGED_STUFF_ROOT / method / 'Stuff' / source
            imgs = list_images(folder)
            filtered = [p for p in imgs if canonical_id(p.stem) not in eval_ids]
            with_mask = [p for p in filtered if has_br_mask(p.stem, source)]
            sampled = sample_files(with_mask, BR_SEG_POS_SAMPLE_PER_GROUP, rng)
            positive.extend((str(p), 1) for p in sampled)
            print(
                f'pos/{method:10s}/Stuff/{source:8s} '
                f'可用 {len(filtered):5d} | 有mask {len(with_mask):5d} -> 抽取 {len(sampled):5d}'
            )

    print(f'正样本总计: {len(positive)}')

    # 2) 负样本: BR RealImage_Img
    target_neg = int(len(positive) * BR_SEG_NEG_RATIO)
    all_real = []
    for source in SOURCE_FOLDERS:
        folder = BR_REAL_ROOT / source
        imgs = list_images(folder)
        if EXCLUDE_EVAL_IDS_FROM_REAL:
            filtered = [p for p in imgs if canonical_id(p.stem) not in eval_ids]
        else:
            filtered = imgs
        all_real.extend(filtered)
        print(f'neg_pool/{source:8s} 可用 {len(filtered):5d}')

    if target_neg <= 0:
        target_neg = len(positive)
    negative_files = sample_files(all_real, min(target_neg, len(all_real)), rng)
    negative = [(str(p), 0) for p in negative_files]
    if len(negative) < target_neg:
        print(f'[WARN] 负样本不足: 目标 {target_neg}, 实际 {len(negative)}')

    print(f'负样本总计: {len(negative)}')

    # 3) 切分并写出
    all_entries = positive + negative
    train_entries, val_entries = split_train_val(all_entries, VAL_RATIO, rng)

    out_dir = PROJECT_ROOT / 'annotation'
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / 'train_seg.txt'
    val_path = out_dir / 'val_seg.txt'
    with open(train_path, 'w', encoding='utf-8') as f:
        if train_entries:
            f.write('\n'.join(f'{p}\t{l}' for p, l in train_entries) + '\n')
    with open(val_path, 'w', encoding='utf-8') as f:
        if val_entries:
            f.write('\n'.join(f'{p}\t{l}' for p, l in val_entries) + '\n')

    tr_pos = sum(1 for _, label in train_entries if label == 1)
    tr_neg = sum(1 for _, label in train_entries if label == 0)
    va_pos = sum(1 for _, label in val_entries if label == 1)
    va_neg = sum(1 for _, label in val_entries if label == 0)

    print(f"\n{'=' * 60}")
    print(
        f'训练集: {len(train_entries)} 条 '
        f'(正={tr_pos}, 负={tr_neg}, 比例 1:{tr_neg / max(1, tr_pos):.2f})'
    )
    print(f'验证集: {len(val_entries)} 条 (正={va_pos}, 负={va_neg})')
    print(f'写出: {train_path}')
    print(f'写出: {val_path}')
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
