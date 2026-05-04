"""
生成 BR 数据集下的分类训练/验证/测试注释文件。

输出文件:
  - annotation/train_v2.txt
  - annotation/val_v2.txt
  - annotation/test_v2.txt
  - annotation/eval_brgen_stuff_val.txt

Usage:
    python script/gen_annotations_v2.py
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
TEST_RATIO = float(os.getenv('TEST_RATIO', '0.12'))

# 每个 forged group(method/source)默认抽样数量。
BR_FAKE_SAMPLE_PER_GROUP = int(os.getenv('BR_FAKE_SAMPLE_PER_GROUP', '1000'))
# 每个 real source 默认抽样数量。
BR_REAL_SAMPLE_PER_SOURCE = int(os.getenv('BR_REAL_SAMPLE_PER_SOURCE', '5000'))
EXCLUDE_EVAL_IDS_FROM_REAL = os.getenv('EXCLUDE_EVAL_IDS_FROM_REAL', '0') == '1'

# 分类数据构成开关: BR 主体 + legacy 叠加
INCLUDE_BR_DATA = os.getenv('INCLUDE_BR_DATA', '0') == '1'
INCLUDE_LEGACY_SDXL_FLUX = os.getenv('INCLUDE_LEGACY_SDXL_FLUX', '1') == '1'
INCLUDE_DOUBAO = os.getenv('INCLUDE_DOUBAO', '1') == '1'
INCLUDE_LEGACY_REAL = os.getenv('INCLUDE_LEGACY_REAL', '1') == '1'
INCLUDE_CHANGE_FACK = os.getenv('INCLUDE_CHANGE_FACK', '1') == '1'

SDXL_SAMPLE_PER_CAT = int(os.getenv('SDXL_SAMPLE_PER_CAT', '50'))
FLUX_SAMPLE_PER_CAT = int(os.getenv('FLUX_SAMPLE_PER_CAT', '50'))
DOUBAO_FACK_SAMPLE = int(os.getenv('DOUBAO_FACK_SAMPLE', '281'))
DOUBAO_REAL_SAMPLE = int(os.getenv('DOUBAO_REAL_SAMPLE', '281'))
DOUBAO_OVERSAMPLE = int(os.getenv('DOUBAO_OVERSAMPLE', '3'))
CHANGE_FACK_SAMPLE = int(os.getenv('CHANGE_FACK_SAMPLE', '1000'))
CHANGE_OVERSAMPLE = int(os.getenv('CHANGE_OVERSAMPLE', '1'))
REAL_FFHQ_SAMPLE = int(os.getenv('REAL_FFHQ_SAMPLE', '400'))
REAL_FORLAB_SAMPLE = int(os.getenv('REAL_FORLAB_SAMPLE', '150'))

# [Data Update] Removing GAN-based/old inpainting methods (LaMa, MAT) to focus purely on Diffusion models.
# Keeping Diffusion-based ones: BrushNet, PowerPaint, SDXL.
FORGED_METHODS = ('BrushNet', 'PowerPaint', 'SDXL')
SOURCE_FOLDERS = ('COCO', 'ImageNet', 'Places')
LEGACY_CATEGORIES = ('animals', 'faces', 'general', 'landscapes')
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
MASK_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
NUM_PATTERN = re.compile(r'(\d{5,})')

BR_FORGED_STUFF_ROOT = DATA_ROOT / '新数据' / 'BR-Gen' / 'Forged'
BR_REAL_ROOT = DATA_ROOT / '新数据' / 'RealImage_Img' / 'RealImage_Img'
BR_VAL_TP_DIR = DATA_ROOT / '新数据' / 'BRGen_Stuff_Val' / 'Tp'
BR_VAL_GT_DIR = DATA_ROOT / '新数据' / 'BRGen_Stuff_Val' / 'Gt'

ANN_DIR = PROJECT_ROOT / 'annotation'
TRAIN_OUT = ANN_DIR / 'train_v2.txt'
VAL_OUT = ANN_DIR / 'val_v2.txt'
TEST_OUT = ANN_DIR / 'test_v2.txt'
EVAL_OUT = ANN_DIR / 'eval_brgen_stuff_val.txt'


def canonical_id(stem):
    """将不同命名风格归一到同一个 ID，避免评估泄漏。"""
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
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def sample_files(files, n, rng):
    if n <= 0 or n >= len(files):
        return list(files)
    return rng.sample(files, n)


def filter_eval_ids(paths, eval_ids, for_real=False):
    if not eval_ids:
        return list(paths)
    if for_real and not EXCLUDE_EVAL_IDS_FROM_REAL:
        return list(paths)
    return [p for p in paths if canonical_id(p.stem) not in eval_ids]


def split_train_val_test(entries, val_ratio, test_ratio, rng):
    entries = list(entries)
    rng.shuffle(entries)
    total = len(entries)
    if total == 0:
        return [], [], []

    test_n = int(total * test_ratio)
    val_n = int(total * val_ratio)

    if total >= 3:
        test_n = max(1, test_n)
        val_n = max(1, val_n)
    else:
        test_n = 0
        val_n = 0

    if test_n + val_n >= total:
        overflow = test_n + val_n - (total - 1)
        if overflow > 0:
            val_n = max(0, val_n - overflow)

    test_entries = entries[:test_n]
    val_entries = entries[test_n:test_n + val_n]
    train_entries = entries[test_n + val_n:]
    return train_entries, val_entries, test_entries


def build_eval_brgen_file():
    """从 BRGen_Stuff_Val/Tp + Gt 自动生成定位独立评估文件。"""
    entries = []
    eval_ids = set()
    missing_gt = 0

    for tp_path in list_images(BR_VAL_TP_DIR):
        stem = tp_path.stem
        gt_ok = False
        for ext in MASK_EXTS:
            gt_path = BR_VAL_GT_DIR / f'{stem}_mask{ext}'
            if gt_path.exists():
                gt_ok = True
                break
        if not gt_ok:
            for ext in MASK_EXTS:
                gt_path = BR_VAL_GT_DIR / f'{stem}{ext}'
                if gt_path.exists():
                    gt_ok = True
                    break
        if not gt_ok:
            missing_gt += 1
            continue

        entries.append(f'{tp_path}\t1')
        eval_ids.add(canonical_id(stem))

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUT, 'w', encoding='utf-8') as f:
        if entries:
            f.write('\n'.join(entries) + '\n')

    print(f'eval_brgen_stuff_val: {len(entries)} 条, 缺失GT: {missing_gt}')
    print(f'写出: {EVAL_OUT}')
    return eval_ids


def collect_fake_entries(eval_ids, rng):
    fake_entries = []
    for method in FORGED_METHODS:
        for source in SOURCE_FOLDERS:
            folder = BR_FORGED_STUFF_ROOT / method / 'Stuff' / source
            imgs = list_images(folder)
            filtered = filter_eval_ids(imgs, eval_ids, for_real=False)
            sampled = sample_files(filtered, BR_FAKE_SAMPLE_PER_GROUP, rng)
            fake_entries.extend((str(p), 1) for p in sampled)
            print(
                f'forged/{method:10s}/Stuff/{source:8s} '
                f'可用 {len(filtered):5d} -> 抽取 {len(sampled):5d}'
            )
    return fake_entries


def collect_real_entries(eval_ids, rng):
    real_entries = []
    for source in SOURCE_FOLDERS:
        folder = BR_REAL_ROOT / source
        imgs = list_images(folder)
        filtered = filter_eval_ids(imgs, eval_ids, for_real=True)
        sampled = sample_files(filtered, BR_REAL_SAMPLE_PER_SOURCE, rng)
        real_entries.extend((str(p), 0) for p in sampled)
        print(
            f'real/{source:8s} '
            f'可用 {len(filtered):5d} -> 抽取 {len(sampled):5d}'
        )
    return real_entries


def collect_legacy_sdxl_flux(eval_ids, rng):
    fake_entries = []

    for cat in LEGACY_CATEGORIES:
        sdxl_imgs = list_images(DATA_ROOT / 'sdxl' / cat)
        sdxl_filtered = filter_eval_ids(sdxl_imgs, eval_ids, for_real=False)
        sdxl_sampled = sample_files(sdxl_filtered, SDXL_SAMPLE_PER_CAT, rng)
        fake_entries.extend((str(p), 1) for p in sdxl_sampled)
        print(
            f'legacy/sdxl/{cat:12s} '
            f'可用 {len(sdxl_filtered):5d} -> 抽取 {len(sdxl_sampled):5d}'
        )

    for cat in LEGACY_CATEGORIES:
        flux_imgs = list_images(DATA_ROOT / 'flux' / cat)
        flux_filtered = filter_eval_ids(flux_imgs, eval_ids, for_real=False)
        flux_sampled = sample_files(flux_filtered, FLUX_SAMPLE_PER_CAT, rng)
        fake_entries.extend((str(p), 1) for p in flux_sampled)
        print(
            f'legacy/flux/{cat:12s} '
            f'可用 {len(flux_filtered):5d} -> 抽取 {len(flux_sampled):5d}'
        )

    return fake_entries


def collect_legacy_doubao(eval_ids, rng):
    fake_entries = []
    real_entries = []

    fack_imgs = list_images(DATA_ROOT / 'doubao' / 'fack')
    fack_filtered = filter_eval_ids(fack_imgs, eval_ids, for_real=False)
    fack_sampled = sample_files(fack_filtered, DOUBAO_FACK_SAMPLE, rng)
    fake_entries.extend((str(p), 1) for p in fack_sampled)
    if DOUBAO_OVERSAMPLE > 1:
        fake_entries = fake_entries * DOUBAO_OVERSAMPLE
    print(
        f'legacy/doubao/fack '
        f'可用 {len(fack_filtered):5d} -> 抽取 {len(fack_sampled):5d} '
        f'(x{DOUBAO_OVERSAMPLE})'
    )

    real_imgs = list_images(DATA_ROOT / 'doubao' / 'real')
    real_filtered = filter_eval_ids(real_imgs, eval_ids, for_real=True)
    real_sampled = sample_files(real_filtered, DOUBAO_REAL_SAMPLE, rng)
    real_entries.extend((str(p), 0) for p in real_sampled)
    if DOUBAO_OVERSAMPLE > 1:
        real_entries = real_entries * DOUBAO_OVERSAMPLE
    print(
        f'legacy/doubao/real '
        f'可用 {len(real_filtered):5d} -> 抽取 {len(real_sampled):5d} '
        f'(x{DOUBAO_OVERSAMPLE})'
    )

    return fake_entries, real_entries


def collect_change_fack(eval_ids, rng):
    fake_entries = []

    fack_imgs = list_images(DATA_ROOT / 'change' / 'pic_fack')
    fack_filtered = filter_eval_ids(fack_imgs, eval_ids, for_real=False)
    fack_sampled = sample_files(fack_filtered, CHANGE_FACK_SAMPLE, rng)
    fake_entries.extend((str(p), 1) for p in fack_sampled)
    
    if CHANGE_OVERSAMPLE > 1:
        fake_entries = fake_entries * CHANGE_OVERSAMPLE
        
    print(
        f'legacy/change/pic_fack '
        f'可用 {len(fack_filtered):5d} -> 抽取 {len(fack_sampled):5d} '
        f'(x{CHANGE_OVERSAMPLE})'
    )

    return fake_entries


def collect_legacy_real(eval_ids, rng):
    real_entries = []

    ffhq_imgs = list_images(DATA_ROOT / 'Real' / 'FFHQ')
    ffhq_filtered = filter_eval_ids(ffhq_imgs, eval_ids, for_real=True)
    ffhq_sampled = sample_files(ffhq_filtered, REAL_FFHQ_SAMPLE, rng)
    real_entries.extend((str(p), 0) for p in ffhq_sampled)
    print(
        f'legacy/Real/FFHQ '
        f'可用 {len(ffhq_filtered):5d} -> 抽取 {len(ffhq_sampled):5d}'
    )

    forlab_imgs = list_images(DATA_ROOT / 'Real' / 'FORLAB')
    forlab_filtered = filter_eval_ids(forlab_imgs, eval_ids, for_real=True)
    forlab_sampled = sample_files(forlab_filtered, REAL_FORLAB_SAMPLE, rng)
    real_entries.extend((str(p), 0) for p in forlab_sampled)
    print(
        f'legacy/Real/FORLAB '
        f'可用 {len(forlab_filtered):5d} -> 抽取 {len(forlab_sampled):5d}'
    )

    return real_entries


def count_stats(entries):
    real_n = sum(1 for _, label in entries if label == 0)
    fake_n = sum(1 for _, label in entries if label == 1)
    return real_n, fake_n


def write_entries(path, entries):
    with open(path, 'w', encoding='utf-8') as f:
        if entries:
            f.write('\n'.join(f'{p}\t{label}' for p, label in entries) + '\n')


def main():
    rng = random.Random(SEED)
    eval_ids = build_eval_brgen_file()
    print(f'已排除评估ID: {len(eval_ids)}')

    fake_entries = []
    real_entries = []

    if INCLUDE_BR_DATA:
        print('\n[Data] Include BR base data: ON')
        fake_entries.extend(collect_fake_entries(eval_ids, rng))
        real_entries.extend(collect_real_entries(eval_ids, rng))
    else:
        print('\n[Data] Include BR base data: OFF')

    if INCLUDE_LEGACY_SDXL_FLUX:
        print('[Data] Include legacy SDXL/Flux: ON')
        fake_entries.extend(collect_legacy_sdxl_flux(eval_ids, rng))
    else:
        print('[Data] Include legacy SDXL/Flux: OFF')

    if INCLUDE_DOUBAO:
        print('[Data] Include Doubao (with oversample): ON')
        doubao_fake, doubao_real = collect_legacy_doubao(eval_ids, rng)
        fake_entries.extend(doubao_fake)
        real_entries.extend(doubao_real)
    else:
        print('[Data] Include Doubao (with oversample): OFF')

    if INCLUDE_CHANGE_FACK:
        print('[Data] Include Change fack: ON')
        fake_entries.extend(collect_change_fack(eval_ids, rng))
    else:
        print('[Data] Include Change fack: OFF')

    if INCLUDE_LEGACY_REAL:
        print('[Data] Include legacy Real FFHQ/FORLAB: ON')
        real_entries.extend(collect_legacy_real(eval_ids, rng))
    else:
        print('[Data] Include legacy Real FFHQ/FORLAB: OFF')

    all_entries = fake_entries + real_entries

    if not all_entries:
        raise RuntimeError('没有可用样本: 请检查数据路径或开启至少一种数据源开关。')

    train_entries, val_entries, test_entries = split_train_val_test(
        all_entries, VAL_RATIO, TEST_RATIO, rng
    )

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    write_entries(TRAIN_OUT, train_entries)
    write_entries(VAL_OUT, val_entries)
    write_entries(TEST_OUT, test_entries)

    train_real, train_fake = count_stats(train_entries)
    val_real, val_fake = count_stats(val_entries)
    test_real, test_fake = count_stats(test_entries)

    print(f"\n{'=' * 60}")
    print(
        f'训练集: {len(train_entries)} 条 '
        f'(Real={train_real}, Fake={train_fake}, 比例 1:{train_fake / max(1, train_real):.2f})'
    )
    print(f'验证集: {len(val_entries)} 条 (Real={val_real}, Fake={val_fake})')
    print(f'测试集: {len(test_entries)} 条 (Real={test_real}, Fake={test_fake})')
    print(f'写出: {TRAIN_OUT}')
    print(f'写出: {VAL_OUT}')
    print(f'写出: {TEST_OUT}')
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
