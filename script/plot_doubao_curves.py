"""
豆包(Doubao) 专属：迭代准确率曲线 + ROC 对比脚本
=====================================================
对比两个版本的模型：
  - V14-旧（无加权）: outputs/模型备份/第三代/v13_doubao_focused/
  - V14-新（加权×3）: outputs/v13_doubao_focused/

生成图表:
  1. 训练迭代 Loss + Accuracy 曲线对比
  2. 全量 ROC 曲线对比
  3. 豆包专属 ROC 曲线（仅 Doubao_Real vs Doubao_Fake）
  4. 各类别准确率条形图对比

输出: outputs/doubao_curves/
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix
)
import collections

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from service.dataset import ImageDataset
from service.model_v11_fusion import LaREDeepFakeV11

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 路径配置 ──────────────────────────────────────────────────────────────────
VERSIONS = {
    "V14-旧（无加权）": {
        "csv":    PROJECT_ROOT / "outputs/模型备份/第三代/v13_doubao_focused/training_history.csv",
        "ckpt":   PROJECT_ROOT / "outputs/模型备份/第三代/v13_doubao_focused/best.pth",
        "color":  "#5C6BC0",
        "linestyle": "--",
    },
    "V14-新（加权×3）": {
        "csv":    PROJECT_ROOT / "outputs/v13_doubao_focused/training_history.csv",
        "ckpt":   PROJECT_ROOT / "outputs/v13_doubao_focused/best.pth",
        "color":  "#E53935",
        "linestyle": "-",
    },
}

OUT_DIR = PROJECT_ROOT / "outputs" / "doubao_curves"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 评估集标注文件
EVAL_FILES = {
    "Doubao_Fake": PROJECT_ROOT / "annotation/temp_eval_Doubao_Fake.txt",
    "Doubao_Real": PROJECT_ROOT / "annotation/temp_eval_Doubao_Real.txt",
    "Real":        PROJECT_ROOT / "annotation/temp_eval_Real.txt",
    "SDXL":        PROJECT_ROOT / "annotation/temp_eval_SDXL.txt",
    "Flux":        PROJECT_ROOT / "annotation/temp_eval_Flux.txt",
}

# 模型/数据集超参（与训练保持一致）
CLIP_TYPE    = os.getenv("CLIP_TYPE",    "RN50x64")
TEXTURE_MODEL= os.getenv("TEXTURE_MODEL","convnext_tiny")
DATA_ROOT    = os.getenv("DATA_ROOT",    str(PROJECT_ROOT))
MAP_FILE     = os.getenv("MAP_FILE",     "annotation/val_sdxl.txt")
HIGHRES_SIZE = int(os.getenv("HIGHRES_SIZE", "512"))
BATCH_SIZE   = 8


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: 训练曲线对比
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves():
    print("[1/4] 绘制训练迭代曲线对比 ...")

    dfs = {}
    for name, cfg in VERSIONS.items():
        if cfg["csv"].exists():
            dfs[name] = pd.read_csv(cfg["csv"])
        else:
            print(f"   ! 找不到 CSV: {cfg['csv']}")

    if not dfs:
        print("   (无 CSV 数据，跳过)")
        return

    has_lr = all("lr" in df.columns for df in dfs.values())

    ncols = 3 if has_lr else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    fig.suptitle("训练迭代曲线对比（旧版 vs 加权新版）", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    for name, df in dfs.items():
        cfg = VERSIONS[name]
        ax.plot(df["epoch"], df["train_loss"], color=cfg["color"],
                linestyle=cfg["linestyle"], alpha=0.5, linewidth=1.5)
        ax.plot(df["epoch"], df["val_loss"],   color=cfg["color"],
                linestyle=cfg["linestyle"], linewidth=2,
                label=f"{name} (val)")
    ax.set_title("Loss 对比")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)

    # Accuracy
    ax = axes[1]
    for name, df in dfs.items():
        cfg = VERSIONS[name]
        ax.plot(df["epoch"], df["train_acc"] * 100, color=cfg["color"],
                linestyle=cfg["linestyle"], alpha=0.5, linewidth=1.5)
        ax.plot(df["epoch"], df["val_acc"] * 100,   color=cfg["color"],
                linestyle=cfg["linestyle"], linewidth=2,
                label=f"{name} (val)")
        # 标注最佳点
        best_row = df.loc[df["val_acc"].idxmax()]
        ax.scatter(best_row["epoch"], best_row["val_acc"] * 100,
                   color=cfg["color"], s=80, zorder=5,
                   marker="*")
        ax.annotate(f'{best_row["val_acc"]*100:.2f}%',
                    xy=(best_row["epoch"], best_row["val_acc"] * 100),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=cfg["color"])
    ax.set_title("Accuracy 对比")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("准确率 (%)")
    ax.set_ylim([85, 101])
    ax.legend(fontsize=8)

    # LR
    if has_lr:
        ax = axes[2]
        for name, df in dfs.items():
            cfg = VERSIONS[name]
            ax.plot(df["epoch"], df["lr"], color=cfg["color"],
                    linestyle=cfg["linestyle"], linewidth=2, label=name)
        ax.set_title("学习率调度")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        ax.set_yscale("log")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "1_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> 已保存: 1_training_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# Part 2 & 3: 加载模型并推理
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: Path, device):
    model = LaREDeepFakeV11(
        num_classes=2,
        clip_type=CLIP_TYPE,
        texture_model=TEXTURE_MODEL,
    ).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    model.eval()
    return model


def build_combined_ann(categories=None) -> Path:
    """将多个 temp_eval_*.txt 合并为一个临时标注文件，供 ImageDataset 加载"""
    tmp_path = OUT_DIR / "_combined_eval.txt"
    lines = []
    used = categories or list(EVAL_FILES.keys())
    for cat in used:
        fpath = EVAL_FILES.get(cat)
        if fpath and fpath.exists():
            with open(fpath, encoding="utf-8") as f:
                lines.extend(f.readlines())
        else:
            print(f"   ! 标注文件不存在，跳过: {fpath}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return tmp_path


def run_inference(model, ann_file: Path, device) -> tuple[list, list, list[str]]:
    """对 ann_file 中的样本推理，返回 (labels, probs, paths)"""
    dataset = ImageDataset(
        data_root=DATA_ROOT,
        train_file=str(ann_file),
        map_file=MAP_FILE,
        data_size=448,
        highres_size=HIGHRES_SIZE,
        enable_v11=True,
        enable_trufor=True,
        is_train=False,
        drop_no_map=False,
    )
    # is_train=False 时数据存入 test_list，需要打开 val 模式才能被 DataLoader 访问
    dataset.set_val_mode(True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=False)

    all_labels, all_probs, all_paths = [], [], []

    # 路径列表直接来自 dataset.test_list（与 DataLoader 顺序严格对齐）
    raw_paths = [p for p, _ in dataset.test_list]
    path_iter = iter(raw_paths)

    with torch.no_grad():
        for batch in loader:
            batch_len = len(batch)
            img_clip  = batch[0].to(device)
            labels    = batch[1].to(device).view(-1)
            loss_maps = batch[2].to(device)
            img_highres = batch[3].to(device) if batch_len >= 4 else None
            trufor      = batch[5].to(device) if batch_len >= 6 else None

            try:
                from torch.amp import autocast as amp_autocast
                amp_t = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                with amp_autocast("cuda" if device.type == "cuda" else "cpu",
                                  dtype=amp_t, enabled=device.type == "cuda"):
                    logits = model(img_clip, img_highres, loss_maps, trufor_map=trufor)
            except Exception:
                logits = model(img_clip, img_highres, loss_maps, trufor_map=trufor)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            lbs   = labels.cpu().tolist()

            all_probs  += probs
            all_labels += lbs
            for _ in probs:
                all_paths.append(next(path_iter, ""))

    return all_labels, all_probs, all_paths


def infer_cat(path_str: str) -> str:
    p = path_str.lower().replace("\\", "/")
    if "doubao" in p:
        return "Doubao"
    if "sdxl" in p:
        return "SDXL"
    if "flux" in p:
        return "Flux"
    return "Real"


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: 全量 ROC 曲线对比
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(results: dict):
    print("[2/4] 全量 ROC 曲线对比 ...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ROC 曲线对比（全评估集 vs 豆包专属）", fontsize=14, fontweight="bold")

    for ax_idx, (title, filter_fn) in enumerate([
        ("全量 ROC（所有类别）",   lambda cat: True),
        ("豆包专属 ROC（仅Doubao）", lambda cat: cat == "Doubao"),
    ]):
        ax = axes[ax_idx]
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="随机猜测")

        for name, (labels, probs, paths) in results.items():
            cfg = VERSIONS[name]
            # 过滤类别
            mask = [filter_fn(infer_cat(p)) for p in paths]
            f_labels = [l for l, m in zip(labels, mask) if m]
            f_probs  = [p for p, m in zip(probs,  mask) if m]

            if len(set(f_labels)) < 2:
                print(f"   ! {name} - {title}: 标签只有一种，跳过ROC")
                continue

            fpr, tpr, _ = roc_curve(f_labels, f_probs)
            roc_auc = auc(fpr, tpr)
            n = len(f_labels)
            ax.plot(fpr, tpr, color=cfg["color"], lw=2,
                    linestyle=cfg["linestyle"],
                    label=f"{name}\nAUC={roc_auc:.4f} (n={n})")

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel("假阳率 (FPR)", fontsize=10)
        ax.set_ylabel("真阳率 (TPR)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "2_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> 已保存: 2_roc_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: 豆包分类别ROC（Real vs Fake 各自的置信度分布）
# ══════════════════════════════════════════════════════════════════════════════

def plot_doubao_score_dist(results: dict):
    print("[3/4] 豆包置信度分布对比 ...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("豆包样本 AI置信度 分布对比", fontsize=14, fontweight="bold")

    for ax_idx, (name, (labels, probs, paths)) in enumerate(results.items()):
        ax = axes[ax_idx]
        doubao_mask = [infer_cat(p) == "Doubao" for p in paths]

        real_scores = [s for s, l, m in zip(probs, labels, doubao_mask) if m and l == 0]
        fake_scores = [s for s, l, m in zip(probs, labels, doubao_mask) if m and l == 1]

        bins = np.linspace(0, 1, 31)
        ax.hist(real_scores, bins=bins, alpha=0.65, color="#4CAF50",
                label=f"Doubao_Real (n={len(real_scores)})", density=True, edgecolor="white")
        ax.hist(fake_scores, bins=bins, alpha=0.65, color="#F44336",
                label=f"Doubao_Fake (n={len(fake_scores)})", density=True, edgecolor="white")
        ax.axvline(x=0.5, color="black", linestyle="--", lw=1.5, label="阈值=0.5")

        # 准确率标注
        real_acc = sum(1 for s in real_scores if s < 0.5) / max(len(real_scores), 1) * 100
        fake_acc = sum(1 for s in fake_scores if s >= 0.5) / max(len(fake_scores), 1) * 100
        ax.text(0.02, 0.95, f"Real准确率: {real_acc:.1f}%\nFake准确率: {fake_acc:.1f}%",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        cfg = VERSIONS[name]
        ax.set_title(f"{name}", color=cfg["color"], fontsize=11)
        ax.set_xlabel("P(AI伪造)")
        ax.set_ylabel("密度")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "3_doubao_score_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> 已保存: 3_doubao_score_dist.png")


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: 各类别准确率条形图对比
# ══════════════════════════════════════════════════════════════════════════════

def plot_category_acc(results: dict):
    print("[4/4] 各类别准确率条形图对比 ...")

    CATS = ["Real", "SDXL", "Flux", "Doubao_Real", "Doubao_Fake"]
    CAT_COLORS = {
        "Real":        "#4CAF50",
        "SDXL":        "#2196F3",
        "Flux":        "#9C27B0",
        "Doubao_Real": "#FF9800",
        "Doubao_Fake": "#F44336",
    }

    def compute_cat_acc(labels, probs, paths):
        cat_data = collections.defaultdict(lambda: {"correct": 0, "total": 0})
        for label, prob, path in zip(labels, probs, paths):
            base_cat = infer_cat(path)
            if base_cat == "Doubao":
                cat = "Doubao_Real" if label == 0 else "Doubao_Fake"
            elif base_cat == "SDXL":
                cat = "SDXL"
            elif base_cat == "Flux":
                cat = "Flux"
            else:
                cat = "Real"
            pred = 0 if prob < 0.5 else 1
            cat_data[cat]["total"] += 1
            if pred == label:
                cat_data[cat]["correct"] += 1
        return {c: (d["correct"] / d["total"] * 100 if d["total"] else 0)
                for c, d in cat_data.items()}

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("各类别准确率对比", fontsize=14, fontweight="bold")

    x = np.arange(len(CATS))
    n_versions = len(results)
    width = 0.35

    for i, (name, (labels, probs, paths)) in enumerate(results.items()):
        cfg = VERSIONS[name]
        acc_dict = compute_cat_acc(labels, probs, paths)
        accs = [acc_dict.get(c, 0) for c in CATS]
        offset = (i - (n_versions - 1) / 2) * width
        bars = ax.bar(x + offset, accs, width * 0.9, label=name,
                      color=cfg["color"], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, accs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
                        color=cfg["color"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(CATS, fontsize=10)
    ax.set_ylabel("准确率 (%)")
    ax.set_ylim([0, 108])
    ax.axhline(y=85, color="gray", linestyle=":", lw=1, alpha=0.7, label="目标线 85%")
    ax.legend(fontsize=9)

    # 背景色区分豆包
    ax.axvspan(2.5, 4.5, alpha=0.06, color="orange", label="_nolegend_")
    ax.text(3.5, 105, "豆包区域", ha="center", fontsize=9, color="darkorange", alpha=0.8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "4_category_acc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> 已保存: 4_category_acc.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  豆包专属 迭代准确率 + ROC 曲线分析")
    print(f"  输出目录: {OUT_DIR}")
    print("=" * 60)

    # Part 1: 训练曲线（只需 CSV，不需要 GPU）
    plot_training_curves()

    # 检查是否有可用的模型
    available_versions = {
        name: cfg for name, cfg in VERSIONS.items() if cfg["ckpt"].exists()
    }
    if not available_versions:
        print("\n! 未找到任何模型 checkpoint，跳过推理相关图表")
        print("  请确认以下路径存在:")
        for name, cfg in VERSIONS.items():
            print(f"    {name}: {cfg['ckpt']}")
        return

    print(f"\n  找到 {len(available_versions)} 个模型，开始推理...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    # 合并评估集标注
    combined_ann = build_combined_ann()

    # 对每个版本运行推理
    results = {}
    for name, cfg in available_versions.items():
        print(f"\n  [{name}] 加载模型: {cfg['ckpt'].name}")
        try:
            model = load_model(cfg["ckpt"], device)
            print(f"  [{name}] 推理中...")
            labels, probs, paths = run_inference(model, combined_ann, device)
            results[name] = (labels, probs, paths)
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"  [{name}] 完成，共 {len(labels)} 样本")
        except Exception as e:
            print(f"  ! [{name}] 推理失败: {e}")

    if not results:
        print("\n! 所有推理均失败，退出")
        return

    # Parts 2-4
    plot_roc_curves(results)
    plot_doubao_score_dist(results)
    plot_category_acc(results)

    # 清理临时文件
    combined_ann.unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("  全部图表已保存至:", OUT_DIR)
    for f in sorted(OUT_DIR.iterdir()):
        if f.suffix == ".png":
            print(f"    {f.name:<40} {f.stat().st_size/1024:>7.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
