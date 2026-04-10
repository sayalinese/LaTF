"""
生成 SegFormer 定位模型专用的训练/验证标注文件。
只保留有 mask 的正样本 (change + doubao/fack) 和等量负样本 (real)。

Usage:
    python script/gen_mask_annotations.py
"""
import os
import random
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / '.env')

DATA_ROOT = PROJECT_ROOT / os.getenv('DATA_ROOT', 'data')
SEED = 42
VAL_RATIO = 0.12

# 已有的定位评估集（不能混入训练/验证）
EVAL_CHANGE_FILE = PROJECT_ROOT / 'annotation' / 'eval_change.txt'

MASK_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}


def load_eval_stems():
    """读取 eval_change.txt 中已占用的文件名，避免泄漏到训练集。"""
    stems = set()
    if EVAL_CHANGE_FILE.exists():
        for enc in ['utf-8', 'gbk']:
            try:
                with open(EVAL_CHANGE_FILE, encoding=enc) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        path = line.rsplit('\t', 1)[0] if '\t' in line else line.rsplit(' ', 1)[0]
                        stems.add(Path(path).stem)
                break
            except UnicodeDecodeError:
                continue
    return stems


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)


def has_mask(stem, mask_dirs):
    for d in mask_dirs:
        for ext in MASK_EXTS:
            if (d / f"{stem}{ext}").exists():
                return True
    return False


def main():
    rng = random.Random(SEED)
    eval_stems = load_eval_stems()
    print(f"已排除的 eval_change 文件: {len(eval_stems)} 张")

    # ── 收集有 mask 的正样本 ──
    change_mask_dir = DATA_ROOT / 'change' / 'masks'
    doubao_mask_dir = DATA_ROOT / 'doubao' / 'masks'
    mask_dirs = [d for d in [change_mask_dir, doubao_mask_dir] if d.exists()]

    positive = []

    # change/images — 有 mask 的才要，排除 eval 集
    change_imgs = list_images(DATA_ROOT / 'change' / 'images')
    for f in change_imgs:
        if f.stem in eval_stems:
            continue
        if has_mask(f.stem, mask_dirs):
            positive.append((str(f), 1))
    print(f"change (有mask, 排除eval后): {len(positive)}")

    # doubao/fack — 有 mask 的才要
    doubao_fack = list_images(DATA_ROOT / 'doubao' / 'fack')
    n_before = len(positive)
    for f in doubao_fack:
        if has_mask(f.stem, mask_dirs):
            positive.append((str(f), 1))
    print(f"doubao/fack (有mask): {len(positive) - n_before}")

    print(f"正样本总计: {len(positive)}")

    # ── 收集负样本 (real)，数量 ≈ 正样本 ──
    target_neg = len(positive)
    negative = []

    # doubao/real — 全部用 (配对真图，质量高)
    doubao_real = list_images(DATA_ROOT / 'doubao' / 'real')
    for f in doubao_real:
        negative.append((str(f), 0))
    print(f"doubao/real: {len(doubao_real)}")

    # FFHQ — 补足剩余
    remain = target_neg - len(negative)
    if remain > 0:
        ffhq = list_images(DATA_ROOT / 'Real' / 'FFHQ')
        sampled = rng.sample(ffhq, min(remain, len(ffhq)))
        for f in sampled:
            negative.append((str(f), 0))
        print(f"Real/FFHQ: {len(sampled)}")

    # FORLAB — 如果还不够
    remain = target_neg - len(negative)
    if remain > 0:
        forlab = list_images(DATA_ROOT / 'Real' / 'FORLAB')
        sampled = rng.sample(forlab, min(remain, len(forlab)))
        for f in sampled:
            negative.append((str(f), 0))
        print(f"Real/FORLAB: {len(sampled)}")

    print(f"负样本总计: {len(negative)}")

    # ── 合并、切分、写出 ──
    all_entries = positive + negative
    rng.shuffle(all_entries)

    val_n = max(1, int(len(all_entries) * VAL_RATIO))
    val_entries = all_entries[:val_n]
    train_entries = all_entries[val_n:]

    out_dir = PROJECT_ROOT / 'annotation'
    out_dir.mkdir(exist_ok=True)

    for name, entries in [('train_seg.txt', train_entries), ('val_seg.txt', val_entries)]:
        path = out_dir / name
        with open(path, 'w', encoding='utf-8') as f:
            for filepath, label in entries:
                f.write(f"{filepath}\t{label}\n")

    # 统计
    tr_pos = sum(1 for _, l in train_entries if l == 1)
    tr_neg = sum(1 for _, l in train_entries if l == 0)
    va_pos = sum(1 for _, l in val_entries if l == 1)
    va_neg = sum(1 for _, l in val_entries if l == 0)

    print(f"\n{'='*60}")
    print(f"训练集: {len(train_entries)} 条 (正={tr_pos}, 负={tr_neg}, 比例 1:{tr_neg/max(1,tr_pos):.2f})")
    print(f"验证集: {len(val_entries)} 条 (正={va_pos}, 负={va_neg})")
    print(f"写出: {out_dir / 'train_seg.txt'}")
    print(f"写出: {out_dir / 'val_seg.txt'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
