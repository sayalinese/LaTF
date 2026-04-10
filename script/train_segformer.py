"""
SegFormer 篡改定位训练脚本。
完全独立于分类模型，使用 HuggingFace SegformerForSemanticSegmentation。

Usage:
    python script/train_segformer.py                          # 纯 RGB 3ch
    python script/train_segformer.py --use_ssfr               # RGB + SSFR 10ch
    python script/train_segformer.py --model_name nvidia/segformer-b3-finetuned-ade-20k-512-512  # 换 B3

输出: outputs/segformer_rgb/ 或 outputs/segformer_ssfr/
"""
import os
import sys
import argparse
import time
import logging
import warnings
from pathlib import Path

# Windows UTF-8 console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '16'
warnings.filterwarnings('ignore', message='.*cache-system uses symlinks by default.*')
warnings.filterwarnings('ignore', message='.*Some weights of SegformerForSemanticSegmentation were not initialized.*')
warnings.filterwarnings('ignore', message='.*You should probably TRAIN this model.*')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from service.dataset_segformer import SegFormerForgeryDataset

# logging 输出到 stdout (PowerShell 对 stderr 显示红色)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ─── Loss ───────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """logits: [B,C,H,W], targets: [B,H,W] long"""
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """logits: [B,C,H,W], targets: [B,H,W] long"""
        probs = F.softmax(logits, dim=1)[:, 1]  # 取 class=1 的概率
        targets_f = targets.float()
        probs_flat = probs.reshape(-1)
        targets_flat = targets_f.reshape(-1)
        intersection = (probs_flat * targets_flat).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )


class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.fw = focal_weight
        self.dw = dice_weight

    def forward(self, logits, targets):
        return self.fw * self.focal(logits, targets) + self.dw * self.dice(logits, targets)


# ─── Metrics ────────────────────────────────────────────
@torch.no_grad()
def compute_dice(logits, targets):
    """Pixel-level Dice score. logits: [B,C,H,W], targets: [B,H,W]"""
    preds = logits.argmax(dim=1)  # [B,H,W]
    preds_flat = preds.reshape(-1).float()
    targets_flat = targets.reshape(-1).float()
    intersection = (preds_flat * targets_flat).sum()
    dice = (2.0 * intersection + 1.0) / (preds_flat.sum() + targets_flat.sum() + 1.0)
    return dice.item()


@torch.no_grad()
def compute_iou(logits, targets):
    """IoU for class=1 (tampered)."""
    preds = logits.argmax(dim=1)
    preds_flat = preds.reshape(-1).float()
    targets_flat = targets.reshape(-1).float()
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection
    iou = (intersection + 1.0) / (union + 1.0)
    return iou.item()


# ─── Model ──────────────────────────────────────────────
def build_model(model_name, num_channels=3, device='cuda'):
    from transformers import SegformerForSemanticSegmentation, logging as transformers_logging

    transformers_logging.set_verbosity_error()

    if num_channels == 3:
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=2, ignore_mismatched_sizes=True,
        )
    else:
        # 10ch: 加载预训练权重，然后手动扩展第一层
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=2, ignore_mismatched_sizes=True,
        )
        _expand_first_conv(model, num_channels)

    return model.to(device)


def _expand_first_conv(model, num_channels):
    """将 SegFormer 第一个 patch embedding 的 Conv2d 从 3ch 扩展到 num_channels ch。
    前 3ch 复制预训练权重，剩余 ch 用 xavier_uniform 初始化。"""
    # SegFormer 第一层: model.segformer.encoder.patch_embeddings[0].proj
    first_conv = model.segformer.encoder.patch_embeddings[0].proj
    old_weight = first_conv.weight.data  # [out, 3, kH, kW]
    out_ch, _, kH, kW = old_weight.shape

    new_conv = nn.Conv2d(num_channels, out_ch, kernel_size=(kH, kW),
                         stride=first_conv.stride, padding=first_conv.padding,
                         bias=first_conv.bias is not None)
    # 复制前 3ch
    new_conv.weight.data[:, :3] = old_weight
    # xavier 初始化额外通道
    nn.init.xavier_uniform_(new_conv.weight.data[:, 3:])
    if first_conv.bias is not None:
        new_conv.bias.data = first_conv.bias.data.clone()

    model.segformer.encoder.patch_embeddings[0].proj = new_conv
    logger.info(f"Expand patch_embed input channels: 3 -> {num_channels}")


# ─── Training ──────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch, freeze_encoder):
    model.train()
    if freeze_encoder:
        model.segformer.encoder.eval()
        for p in model.segformer.encoder.parameters():
            p.requires_grad = False
    else:
        for p in model.segformer.encoder.parameters():
            p.requires_grad = True

    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    progress = tqdm(
        loader,
        desc=f"Train {epoch:02d}",
        leave=False,
        dynamic_ncols=True,
        ascii=True,
    )

    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)  # [B, H, W] long

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device, enabled=scaler.is_enabled()):
            outputs = model(pixel_values=images)
            # SegFormer 输出 logits 分辨率是 H/4, W/4
            logits = outputs.logits  # [B, 2, H/4, W/4]
            # 上采样到 mask 分辨率
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode='bilinear', align_corners=False)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_dice += compute_dice(logits, masks)
        n_batches += 1
        progress.set_postfix(loss=f"{total_loss / n_batches:.4f}", dice=f"{total_dice / n_batches:.4f}")

    return total_loss / n_batches, total_dice / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = 0

    progress = tqdm(
        loader,
        desc="Val",
        leave=False,
        dynamic_ncols=True,
        ascii=True,
    )

    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(pixel_values=images)
        logits = outputs.logits
        logits = F.interpolate(logits, size=masks.shape[-2:],
                               mode='bilinear', align_corners=False)

        total_loss += criterion(logits, masks).item()
        total_dice += compute_dice(logits, masks)
        total_iou += compute_iou(logits, masks)
        n_batches += 1
        progress.set_postfix(
            loss=f"{total_loss / n_batches:.4f}",
            dice=f"{total_dice / n_batches:.4f}",
            iou=f"{total_iou / n_batches:.4f}",
        )

    return total_loss / n_batches, total_dice / n_batches, total_iou / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='nvidia/segformer-b2-finetuned-ade-512-512')
    parser.add_argument('--train_file', default=str(PROJECT_ROOT / 'annotation' / 'train_seg.txt'))
    parser.add_argument('--val_file', default=str(PROJECT_ROOT / 'annotation' / 'val_seg.txt'))
    parser.add_argument('--out_dir', default='')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--freeze_encoder_epochs', type=int, default=5)
    parser.add_argument('--encoder_lr', type=float, default=6e-5)
    parser.add_argument('--decoder_lr', type=float, default=6e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--use_ssfr', action='store_true')
    parser.add_argument('--ssfr_map_file', default=str(PROJECT_ROOT / 'dift.pt' / 'ann.txt'))
    parser.add_argument('--resume', default='')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_channels = 10 if args.use_ssfr else 3

    if not args.out_dir:
        args.out_dir = str(PROJECT_ROOT / 'outputs' / ('segformer_ssfr' if args.use_ssfr else 'segformer_rgb'))
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Mode: {'RGB+SSFR (10ch)' if args.use_ssfr else 'RGB (3ch)'}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.out_dir}")

    # ── Data ──
    data_root = PROJECT_ROOT / 'data'
    mask_dirs = [data_root / 'change' / 'masks', data_root / 'doubao' / 'masks']

    train_ds = SegFormerForgeryDataset(
        args.train_file, img_size=args.img_size, mask_dirs=mask_dirs,
        is_train=True, use_ssfr=args.use_ssfr, ssfr_map_file=args.ssfr_map_file,
    )
    val_ds = SegFormerForgeryDataset(
        args.val_file, img_size=args.img_size, mask_dirs=mask_dirs,
        is_train=False, use_ssfr=args.use_ssfr, ssfr_map_file=args.ssfr_map_file,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    persistent = args.workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=persistent)

    # ── Model ──
    model = build_model(args.model_name, num_channels=num_channels, device=device)
    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        logger.info(f"Resume from {args.resume}")

    # ── Optimizer (encoder 低 LR, decoder 高 LR) ──
    encoder_params = list(model.segformer.encoder.parameters())
    decoder_params = list(model.decode_head.parameters())
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.encoder_lr},
        {'params': decoder_params, 'lr': args.decoder_lr},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    criterion = CombinedLoss(focal_weight=0.5, dice_weight=0.5)
    scaler = GradScaler('cuda', enabled=device == 'cuda')

    # ── Training Loop ──
    best_dice = 0.0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Start training: epochs={args.epochs}, freeze_encoder_epochs={args.freeze_encoder_epochs}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Batch size: {args.batch_size}, workers: {args.workers}")
    print(f"{'='*60}\n")
    for epoch in range(1, args.epochs + 1):
        freeze_enc = epoch <= args.freeze_encoder_epochs
        t0 = time.time()

        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch, freeze_enc
        )
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_enc = optimizer.param_groups[0]['lr']
        lr_dec = optimizer.param_groups[1]['lr']

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"TrLoss {train_loss:.4f} TrDice {train_dice:.4f} | "
            f"VaLoss {val_loss:.4f} VaDice {val_dice:.4f} VaIoU {val_iou:.4f} | "
            f"LR enc={lr_enc:.2e} dec={lr_dec:.2e} | "
            f"{'FROZEN' if freeze_enc else 'FULL'} | {elapsed:.0f}s"
        )

        # Best model
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            save_path = Path(args.out_dir) / 'best.pth'
            torch.save(model.state_dict(), save_path)
            print(f"  Best Dice {best_dice:.4f} -> {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    # Save final
    torch.save(model.state_dict(), Path(args.out_dir) / 'last.pth')
    print(f"\n{'='*60}")
    print(f"Training finished. Best Val Dice: {best_dice:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
