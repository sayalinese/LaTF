"""
生成 V2 版本的训练/验证注释文件。
读取 .env 中的采样配置，按比例从各数据源抽取，写入 annotation/train_v2.txt 和 annotation/val_v2.txt。

Usage:
    python script/gen_annotations_v2.py
"""
import os
import random
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / '.env')

DATA_ROOT = PROJECT_ROOT / os.getenv('DATA_ROOT', 'data')
SEED = 42

# ── 从 .env 读取配置 ──
CHANGE_SAMPLE        = int(os.getenv('CHANGE_SAMPLE', '1920'))
DOUBAO_FACK_SAMPLE   = int(os.getenv('DOUBAO_FACK_SAMPLE', '281'))
DOUBAO_REAL_SAMPLE   = int(os.getenv('DOUBAO_REAL_SAMPLE', '281'))
DOUBAO_OVERSAMPLE    = int(os.getenv('DOUBAO_OVERSAMPLE', '3'))
REAL_FFHQ_SAMPLE     = int(os.getenv('REAL_FFHQ_SAMPLE', '400'))
REAL_FORLAB_SAMPLE   = int(os.getenv('REAL_FORLAB_SAMPLE', '150'))
SDXL_SAMPLE_PER_CAT  = int(os.getenv('SDXL_SAMPLE_PER_CAT', '50'))
FLUX_SAMPLE_PER_CAT  = int(os.getenv('FLUX_SAMPLE_PER_CAT', '50'))

VAL_RATIO = 0.12  # 验证集占比
TEST_RATIO = float(os.getenv('TEST_RATIO', '0.12'))  # 测试集占比

# 已有的定位评估集（不能混入训练/验证）
EVAL_CHANGE_FILE = PROJECT_ROOT / 'annotation' / 'eval_change.txt'


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


def list_images(folder, exts=('.png', '.jpg', '.jpeg', '.webp')):
    """列出目录下所有图片文件路径。"""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in exts])


def sample_files(files, n, rng):
    """从 files 中随机抽 n 个（不足则全取）。"""
    if n >= len(files):
        return list(files)
    return rng.sample(files, n)


def make_entry(filepath, label):
    """生成一行注释: 绝对路径\t标签"""
    return f"{filepath}\t{label}"


def split_train_val(entries, val_ratio, rng):
    """打乱后按比例切分训练/验证。"""
    rng.shuffle(entries)
    val_n = max(1, int(len(entries) * val_ratio))
    return entries[val_n:], entries[:val_n]


def split_train_val_test(entries, val_ratio, test_ratio, rng):
    """打乱后按比例切分训练/验证/测试。"""
    rng.shuffle(entries)
    total = len(entries)
    test_n = max(1, int(total * test_ratio))
    val_n = max(1, int(total * val_ratio))
    test_entries = entries[:test_n]
    val_entries = entries[test_n:test_n + val_n]
    train_entries = entries[test_n + val_n:]
    return train_entries, val_entries, test_entries


def main():
    rng = random.Random(SEED)
    eval_stems = load_eval_stems()
    print(f"已排除的 eval_change 文件: {len(eval_stems)} 张")

    all_train = []
    all_val = []
    all_test = []

    # ── 1. change（局部篡改，核心定位数据）──
    change_imgs = list_images(DATA_ROOT / 'change' / 'images')
    # 排除 eval 集
    change_imgs = [f for f in change_imgs if f.stem not in eval_stems]
    change_sampled = sample_files(change_imgs, CHANGE_SAMPLE, rng)
    change_entries = [make_entry(f, 1) for f in change_sampled]
    tr, va, te = split_train_val_test(change_entries, VAL_RATIO, TEST_RATIO, rng)
    all_train.extend(tr)
    all_val.extend(va)
    all_test.extend(te)
    print(f"change:        {len(change_sampled):5d} → train {len(tr)}, val {len(va)}, test {len(te)}")

    # ── 2. doubao/fack（有 mask，最有价值，过采样）──
    doubao_fack = list_images(DATA_ROOT / 'doubao' / 'fack')
    doubao_fack_sampled = sample_files(doubao_fack, DOUBAO_FACK_SAMPLE, rng)
    # 过采样
    doubao_fack_entries = [make_entry(f, 1) for f in doubao_fack_sampled] * DOUBAO_OVERSAMPLE
    tr, va, te = split_train_val_test(doubao_fack_entries, VAL_RATIO, TEST_RATIO, rng)
    all_train.extend(tr)
    all_val.extend(va)
    all_test.extend(te)
    print(f"doubao/fack:   {len(doubao_fack_sampled):5d} x{DOUBAO_OVERSAMPLE} = {len(doubao_fack_entries):5d} → train {len(tr)}, val {len(va)}, test {len(te)}")

    # ── 3. doubao/real（配对真图，同比过采样）──
    doubao_real = list_images(DATA_ROOT / 'doubao' / 'real')
    doubao_real_sampled = sample_files(doubao_real, DOUBAO_REAL_SAMPLE, rng)
    doubao_real_entries = [make_entry(f, 0) for f in doubao_real_sampled] * DOUBAO_OVERSAMPLE
    tr, va, te = split_train_val_test(doubao_real_entries, VAL_RATIO, TEST_RATIO, rng)
    all_train.extend(tr)
    all_val.extend(va)
    all_test.extend(te)
    print(f"doubao/real:   {len(doubao_real_sampled):5d} x{DOUBAO_OVERSAMPLE} = {len(doubao_real_entries):5d} → train {len(tr)}, val {len(va)}, test {len(te)}")

    # ── 4. Real/FFHQ ──
    ffhq = list_images(DATA_ROOT / 'Real' / 'FFHQ')
    ffhq_sampled = sample_files(ffhq, REAL_FFHQ_SAMPLE, rng)
    ffhq_entries = [make_entry(f, 0) for f in ffhq_sampled]
    tr, va, te = split_train_val_test(ffhq_entries, VAL_RATIO, TEST_RATIO, rng)
    all_train.extend(tr)
    all_val.extend(va)
    all_test.extend(te)
    print(f"Real/FFHQ:     {len(ffhq_sampled):5d} → train {len(tr)}, val {len(va)}, test {len(te)}")

    # ── 5. Real/FORLAB ──
    forlab = list_images(DATA_ROOT / 'Real' / 'FORLAB')
    forlab_sampled = sample_files(forlab, REAL_FORLAB_SAMPLE, rng)
    forlab_entries = [make_entry(f, 0) for f in forlab_sampled]
    tr, va, te = split_train_val_test(forlab_entries, VAL_RATIO, TEST_RATIO, rng)
    all_train.extend(tr)
    all_val.extend(va)
    all_test.extend(te)
    print(f"Real/FORLAB:   {len(forlab_sampled):5d} → train {len(tr)}, val {len(va)}, test {len(te)}")

    # ── 6. SDXL (4 子类) ──
    for cat in ['animals', 'faces', 'general', 'landscapes']:
        imgs = list_images(DATA_ROOT / 'sdxl' / cat)
        sampled = sample_files(imgs, SDXL_SAMPLE_PER_CAT, rng)
        entries = [make_entry(f, 1) for f in sampled]
        tr, va, te = split_train_val_test(entries, VAL_RATIO, TEST_RATIO, rng)
        all_train.extend(tr)
        all_val.extend(va)
        all_test.extend(te)
        print(f"sdxl/{cat:12s} {len(sampled):5d} → train {len(tr)}, val {len(va)}, test {len(te)}")

    # ── 7. Flux (4 子类) ──
    for cat in ['animals', 'faces', 'general', 'landscapes']:
        imgs = list_images(DATA_ROOT / 'flux' / cat)
        sampled = sample_files(imgs, FLUX_SAMPLE_PER_CAT, rng)
        entries = [make_entry(f, 1) for f in sampled]
        tr, va, te = split_train_val_test(entries, VAL_RATIO, TEST_RATIO, rng)
        all_train.extend(tr)
        all_val.extend(va)
        all_test.extend(te)
        print(f"flux/{cat:12s} {len(sampled):5d} → train {len(tr)}, val {len(va)}, test {len(te)}")

    # ── 打乱 & 写出 ──
    rng.shuffle(all_train)
    rng.shuffle(all_val)
    rng.shuffle(all_test)

    out_dir = PROJECT_ROOT / 'annotation'
    out_dir.mkdir(exist_ok=True)

    train_path = out_dir / 'train_v2.txt'
    val_path = out_dir / 'val_v2.txt'
    test_path = out_dir / 'test_v2.txt'

    with open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_train) + '\n')
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_val) + '\n')
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_test) + '\n')

    # ── 统计汇总 ──
    def count_stats(entries):
        r = sum(1 for e in entries if e.endswith('\t0'))
        f = sum(1 for e in entries if e.endswith('\t1'))
        return r, f

    train_real, train_fake = count_stats(all_train)
    val_real, val_fake = count_stats(all_val)
    test_real, test_fake = count_stats(all_test)

    print(f"\n{'='*60}")
    print(f"训练集: {len(all_train)} 条  (Real={train_real}, Fake={train_fake}, 比例 1:{train_fake/max(1,train_real):.1f})")
    print(f"验证集: {len(all_val)} 条  (Real={val_real}, Fake={val_fake})")
    print(f"测试集: {len(all_test)} 条  (Real={test_real}, Fake={test_fake})")
    print(f"写出: {train_path}")
    print(f"写出: {val_path}")
    print(f"写出: {test_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
