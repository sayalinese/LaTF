#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaRE 全流程训练脚本 (分类 + 定位 双管道, 解耦架构)

阶段:
  0 - 环境检查
  1 - 生成标注 (分类 train/val/test_v2 + 定位 train/val_seg)
  2 - 构建提取列表 + SSFR 特征提取
  3 - 训练分类模型 (LaREDeepFakeV11)
  4 - 训练定位模型 (SegFormer-B2)
  5 - 模型评估

用法:
    python train_full_pipeline.py                  # 全部阶段
    python train_full_pipeline.py --phase 3        # 从 Phase 3 开始
    python train_full_pipeline.py --skip_extract   # 跳过特征提取
    python train_full_pipeline.py --phase 4        # 只跑定位训练+评估
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)


def run(cmd, desc, fatal=True):
    """执行子进程，实时输出。"""
    print(f"\n  >>> {desc}")
    print(f"      {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        msg = f"失败: {desc} (exit code {result.returncode})"
        if fatal:
            print(f"  [ERROR] {msg}")
            sys.exit(1)
        else:
            print(f"  [WARN] {msg} — 跳过继续")
    return result.returncode


def phase0():
    """环境检查"""
    print("\n" + "=" * 60)
    print("[Phase 0] 环境检查")
    print("=" * 60)
    run('python -c "'
        "import torch; "
        "print(f'PyTorch {torch.__version__}, "
        "CUDA: {torch.cuda.is_available()}, "
        "GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else chr(78)+chr(47)+chr(65)}')"
        '"',
        "检查 PyTorch + CUDA")


def phase1():
    """生成标注文件"""
    print("\n" + "=" * 60)
    print("[Phase 1] 生成标注文件")
    print("=" * 60)
    run("python script/gen_annotations_v2.py",
        "分类标注 → train_v2 / val_v2 / test_v2")
    run("python script/gen_mask_annotations.py",
        "定位标注 → train_seg / val_seg")


def phase2(skip_extract=False):
    """构建特征提取列表 + 提取 SSFR 特征"""
    print("\n" + "=" * 60)
    print("[Phase 2] 特征提取")
    print("=" * 60)

    # 合并所有标注文件到一个去重提取列表
    ann_files = [
        ("annotation/train_v2.txt", "annotation/val_v2.txt",   "annotation/extract_v2.txt"),
        ("annotation/extract_v2.txt", "annotation/test_v2.txt",  "annotation/extract_v2.txt"),
        ("annotation/extract_v2.txt", "annotation/train_seg.txt", "annotation/extract_v2.txt"),
        ("annotation/extract_v2.txt", "annotation/val_seg.txt",  "annotation/extract_v2.txt"),
    ]
    for train_f, val_f, out_f in ann_files:
        run(f"python script/3_build_extract_list.py"
            f" --train {train_f} --val {val_f} --out {out_f}",
            f"合并 {val_f} → extract_v2")

    if skip_extract:
        print("  [跳过] 特征提取 (--skip_extract)")
        return

    run("python script/2_extract_features.py"
        " --input_path  annotation/extract_v2.txt"
        " --output_path dift.pt"
        " --extractor_type ssfr"
        " --bf16",
        "提取 SSFR 特征 (7ch, 32x32)")


def phase3():
    """训练分类模型"""
    print("\n" + "=" * 60)
    print("[Phase 3] 训练分类模型 (LaREDeepFakeV11)")
    print("=" * 60)
    run("python script/5_train_model_v11.py",
        "分类训练 (参数读自 .env)")


def phase4():
    """训练定位模型"""
    print("\n" + "=" * 60)
    print("[Phase 4] 训练定位模型 (SegFormer-B2)")
    print("=" * 60)
    run("python script/train_segformer.py"
        " --train_file annotation/train_seg.txt"
        " --val_file   annotation/val_seg.txt"
        " --out_dir    outputs/segformer_rgb"
        " --batch_size 12"
        " --epochs     40"
        " --patience   10",
        "SegFormer RGB 3ch 训练")


def phase5():
    """模型评估"""
    print("\n" + "=" * 60)
    print("[Phase 5] 模型评估")
    print("=" * 60)

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
    cls_dir = os.getenv('OUT_DIR', 'outputs/v14_multiscale')
    cls_model = PROJECT_ROOT / cls_dir / 'best.pth'
    seg_model = PROJECT_ROOT / 'outputs' / 'segformer_rgb' / 'best.pth'

    if cls_model.exists():
        run(f"python script/4_test_model.py --dir data --model {cls_model}",
            "分类测试", fatal=False)
    else:
        print(f"  [跳过] 分类模型不存在: {cls_model}")

    if seg_model.exists():
        run(f"python test/evaluate_segformer.py"
            f" --model {seg_model}"
            f" --ann_file annotation/val_seg.txt"
            f" --batch_size 12",
            "定位验证集评估", fatal=False)

        eval_change = PROJECT_ROOT / 'annotation' / 'eval_change.txt'
        if eval_change.exists():
            run(f"python test/evaluate_segformer.py"
                f" --model {seg_model}"
                f" --ann_file annotation/eval_change.txt"
                f" --batch_size 12",
                "定位独立评估集 (eval_change)", fatal=False)
    else:
        print(f"  [跳过] 定位模型不存在: {seg_model}")


def main():
    parser = argparse.ArgumentParser(description="LaRE 全流程训练 (分类 + 定位)")
    parser.add_argument("--phase", type=int, default=0,
                        help="从哪个阶段开始 (0=全部, 1=标注, 2=特征, 3=分类, 4=定位, 5=评估)")
    parser.add_argument("--skip_extract", action="store_true",
                        help="跳过特征提取 (已有 dift.pt 时使用)")
    args = parser.parse_args()

    start = args.phase
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 60)
    print("  LaRE 全流程训练 (分类 + 定位 解耦架构)")
    print(f"  起始阶段: Phase {start}")
    print(f"  时间: {ts}")
    print("=" * 60)

    phases = [
        (0, phase0),
        (1, phase1),
        (2, lambda: phase2(args.skip_extract)),
        (3, phase3),
        (4, phase4),
        (5, phase5),
    ]

    for phase_id, func in phases:
        if phase_id >= start:
            func()

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "=" * 60)
    print("  全流程完成!")
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / '.env')
    print(f"  分类模型: {os.getenv('OUT_DIR', 'outputs/v14_multiscale')}/best.pth")
    print(f"  定位模型: outputs/segformer_rgb/best.pth")
    print(f"  时间: {ts}")
    print("=" * 60)


if __name__ == "__main__":
    main()