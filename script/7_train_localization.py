#!/usr/bin/env python3
"""
FLH (Forensic Localization Head) 独立训练脚本
使用已提取的 SSFR 特征 + 亮度图 + doubao masks 训练篡改定位头

用法:
  python script/7_train_localization.py
  python script/7_train_localization.py --epochs 30 --batch_size 32
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from service.forensic_localization import ForensicLocalizationHead, FLHLoss

MASK_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def dedupe_paths(paths):
    unique = []
    seen = set()
    for path in paths:
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        unique.append(path)
    return unique


class FLHDataset(Dataset):
    """
    FLH 训练数据集
    - 从 dift.pt/ 加载已提取的 SSFR 特征 [7, 32, 32]
    - 从 data/doubao/masks/ 与 data/change/masks/ 加载篡改 mask
    - 从原图生成亮度图 [1, 32, 32]
    - 对于真图和非 doubao 假图，mask 全 0 / 全 1
    """

    def __init__(self, ann_file, feature_root, data_root, mask_root, out_size=32):
        self.feature_root = Path(feature_root)
        self.data_root = Path(data_root)
        self.mask_root = Path(mask_root)
        self.mask_roots = dedupe_paths([
            self.mask_root,
            self.data_root / 'change' / 'masks',
        ])
        self.out_size = out_size
        self.samples = []

        # 解析 feature ann.txt — 格式: img_path\tpt_path\tlabel
        ann_path = self.feature_root / ann_file
        if not ann_path.exists():
            print(f"[FLH] 特征标注文件不存在: {ann_path}")
            return

        for enc in ['utf-8', 'gbk', 'latin-1']:
            try:
                with open(ann_path, encoding=enc) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"[FLH] 无法解码 {ann_path}")
            return

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            img_path, pt_path, label = parts[0], parts[1], int(parts[2])
            if not Path(pt_path).exists():
                continue
            self.samples.append((img_path, pt_path, label))

        print(f"[FLH Dataset] 加载 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def _load_mask(self, img_path, label, ssfr_feat=None):
        """
        加载篡改 mask:
        - doubao 假图: 精确 mask
        - 其他假图: 基于 SSFR 通道异常生成伪 mask (v2)
        - 真图: 全 0
        """
        mask = np.zeros((self.out_size, self.out_size), dtype=np.float32)

        if label == 1:
            # 先尝试加载精确 mask (doubao)
            stem = Path(img_path).stem
            path_norm = str(img_path).lower().replace('\\', '/')
            candidate_roots = []
            if '/change/images/' in path_norm or '/change/fack/' in path_norm:
                candidate_roots.append(self.data_root / 'change' / 'masks')
            if '/doubao/' in path_norm:
                candidate_roots.append(self.mask_root)
            candidate_roots.extend(self.mask_roots)

            candidate_paths = []
            for root in dedupe_paths(candidate_roots):
                for ext in MASK_EXTENSIONS:
                    candidate_paths.append(root / f"{stem}{ext}")

            for mask_path in dedupe_paths(candidate_paths):
                if not mask_path.exists():
                    continue
                try:
                    data = np.fromfile(str(mask_path), dtype=np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        mask = cv2.resize(img, (self.out_size, self.out_size),
                                          interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
                        return mask
                except Exception:
                    continue

            # v2: 对非 doubao 假图，用 SSFR 通道异常生成伪 mask
            if ssfr_feat is not None:
                mask = self._generate_pseudo_mask(ssfr_feat)
            else:
                mask = np.ones((self.out_size, self.out_size), dtype=np.float32)

        return mask

    def _generate_pseudo_mask(self, ssfr_feat):
        """
        基于 SSFR 通道异常值生成伪 mask:
        多通道绝对值取均值 → 归一化 → 阈值化为软 mask
        AI 生成图全图异常时，输出接近全 1（等价于旧行为）
        局部异常区域会得到更精确的伪标注
        """
        # ssfr_feat: [7, H, W] tensor
        feat_np = ssfr_feat.numpy() if isinstance(ssfr_feat, torch.Tensor) else ssfr_feat
        # 取前 6 通道的绝对值均值 (Ch6 VAE 单独权重)
        anomaly = np.abs(feat_np[:6]).mean(axis=0)  # [H, W]
        # 加上 Ch6 VAE 重建误差 (如果有)
        if feat_np.shape[0] >= 7:
            anomaly = anomaly * 0.7 + np.abs(feat_np[6]) * 0.3

        # Min-max 归一化到 [0, 1]
        vmin, vmax = anomaly.min(), anomaly.max()
        if vmax - vmin > 1e-8:
            anomaly = (anomaly - vmin) / (vmax - vmin)
        else:
            anomaly = np.ones_like(anomaly)

        # 软阈值: 保留 > 0.3 的区域，平滑过渡
        mask = np.clip((anomaly - 0.3) / 0.4, 0.0, 1.0)
        return mask.astype(np.float32)

    def _load_luma(self, img_path):
        """从原图生成亮度图 [1, out_size, out_size]"""
        luma = np.zeros((self.out_size, self.out_size), dtype=np.float32)
        try:
            full_path = img_path
            if not Path(full_path).exists():
                full_path = str(self.data_root.parent / img_path)
            data = np.fromfile(full_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                luma = cv2.resize(img, (self.out_size, self.out_size),
                                  interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        except Exception:
            pass
        return luma

    def __getitem__(self, idx):
        img_path, pt_path, label = self.samples[idx]

        # 加载 SSFR 特征 [7, 32, 32]
        try:
            ssfr_feat = torch.load(pt_path, map_location='cpu', weights_only=True).float()
        except Exception:
            ssfr_feat = torch.zeros(7, self.out_size, self.out_size)

        # [V16] Backward compat: pad Ch6 with zeros if old 6-channel features
        if ssfr_feat.shape[0] == 6:
            ssfr_feat = torch.cat([ssfr_feat, torch.zeros(1, self.out_size, self.out_size)], dim=0)

        # 亮度图 [1, 32, 32]
        luma = self._load_luma(img_path)
        luma_t = torch.from_numpy(luma).unsqueeze(0).float()

        # 拼接 [8, 32, 32]
        feat = torch.cat([ssfr_feat, luma_t], dim=0)

        # 篡改 mask [1, 32, 32] (v2: 传入 ssfr_feat 用于伪 mask 生成)
        mask = self._load_mask(img_path, label, ssfr_feat=ssfr_feat)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        return feat, mask_t, label


def train_flh(args):
    device = args.device
    print(f"[FLH] 开始训练 — device={device}, epochs={args.epochs}, batch_size={args.batch_size}")

    # Dataset
    dataset = FLHDataset(
        ann_file='ann.txt',
        feature_root=args.feature_root,
        data_root=args.data_root,
        mask_root=args.mask_root,
    )
    if len(dataset) == 0:
        print("[FLH] 无训练数据，退出")
        return

    # 80/20 split
    n = len(dataset)
    n_val = max(1, int(n * 0.2))
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(2 if args.num_workers > 0 else None),
    )

    # Model (v2: mid_channels 48)
    model = ForensicLocalizationHead(in_channels=8, mid_channels=48).to(device)
    print(f"[FLH] 参数量: {model.count_params():,}")

    criterion = FLHLoss(focal_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.startswith('cuda')
    scaler = GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    epoch_bar = tqdm(range(args.epochs), desc='[FLH] Epoch', unit='ep')
    for epoch in epoch_bar:
        model.train()
        losses = []
        step_bar = tqdm(train_loader, desc=f'  Train {epoch+1}/{args.epochs}',
                        leave=False, unit='batch', dynamic_ncols=True)
        for feat, mask, _ in step_bar:
            feat = feat.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # === v2 增强数据增强 ===
            # 1. 随机涂黑矩形区域并标记为篡改 (30%)
            if torch.rand(1).item() < 0.3:
                b, _, h, w = feat.shape
                bh = torch.randint(4, h // 2, (1,)).item()
                bw = torch.randint(4, w // 2, (1,)).item()
                y0 = torch.randint(0, h - bh, (1,)).item()
                x0 = torch.randint(0, w - bw, (1,)).item()
                feat[:, :7, y0:y0+bh, x0:x0+bw] = 0.0
                feat[:, 7:, y0:y0+bh, x0:x0+bw] = 0.0
                mask[:, :, y0:y0+bh, x0:x0+bw] = 1.0

            # 2. 随机通道 dropout: 随机将 1-2 个 SSFR 通道置零 (20%)
            if torch.rand(1).item() < 0.2:
                n_drop = torch.randint(1, 3, (1,)).item()
                drop_chs = torch.randperm(7)[:n_drop]
                for ch in drop_chs:
                    feat[:, ch] = 0.0

            # 3. 高斯噪声扰动 (20%) — 增强对噪声特征的鲁棒性
            if torch.rand(1).item() < 0.2:
                noise = torch.randn_like(feat[:, :7]) * 0.05
                feat[:, :7] = feat[:, :7] + noise

            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits = model(feat)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            step_bar.set_postfix(loss=f'{loss.item():.4f}')

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for feat, mask, _ in val_loader:
                feat = feat.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                with autocast(enabled=use_amp):
                    logits = model(feat)
                val_losses.append(criterion(logits, mask).item())

        train_loss = np.mean(losses)
        val_loss = np.mean(val_losses) if val_losses else float('inf')
        flag = ' *' if val_loss < best_val_loss else ''
        epoch_bar.set_postfix(train=f'{train_loss:.4f}', val=f'{val_loss:.4f}')
        tqdm.write(f'  Epoch {epoch+1:>2}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}{flag}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    print(f"[FLH] 训练完成: {time.time()-t0:.1f}s, best val={best_val_loss:.4f}")
    print(f"[FLH] 模型已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="FLH 篡改定位头训练")
    parser.add_argument('--feature_root', type=str, default='dift.pt')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--mask_root', type=str, default='data/doubao/masks')
    parser.add_argument('--save_path', type=str, default='outputs/flh_localization.pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128, help='训练 batch size，显存够可设 256')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader worker 数量; Windows 建议 0（spawn 有额外开销），Linux 可设 4-8')
    args = parser.parse_args()

    train_flh(args)


if __name__ == '__main__':
    main()
