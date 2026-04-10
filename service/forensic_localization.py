"""
FLH v2 (Forensic Localization Head) — 自研篡改区域定位模块
替代 TruFor，完全基于 SSFR 7 通道物理特征 + 亮度门控

输入: [B, 8, 32, 32]  (7 SSFR + 1 亮度图)
输出: [B, 1, 32, 32]  篡改概率 logits

v2 改进:
  1. SpatialLuminanceGate — 逐像素亮度门控 (替代 v1 全局均值)
  2. ChannelInteraction    — 跨通道关联建模 (1×1 conv 捕捉通道共现)
  3. SpatialAttention      — 空间注意力 (max+avg 双池化)
  4. CBAM 风格 ResBlock    — 同时具备通道 + 空间注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SpatialAttention(nn.Module):
    """空间注意力: max+avg 双池化 → Conv → Sigmoid"""
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)       # [B, 1, H, W]
        max_out = x.max(dim=1, keepdim=True)[0]      # [B, 1, H, W]
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class ResBlock(nn.Module):
    """残差卷积块: Conv-BN-ReLU-Conv-BN + skip"""
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class CBAMBlock(nn.Module):
    """CBAM = SE 通道注意力 + 空间注意力 + 残差"""
    def __init__(self, ch, reduction=4, spatial_kernel=7):
        super().__init__()
        self.res = ResBlock(ch)
        self.se = SEBlock(ch, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.res(x)
        x = self.se(x)
        x = self.sa(x)
        return x


class SpatialLuminanceGate(nn.Module):
    """
    逐像素亮度感知门控 (v2):
    - 用 1×1 conv 从亮度图生成逐像素的通道门控权重
    - 暗区像素和亮区像素得到不同的通道缩放
    - 替代 v1 的全局均值门控 (所有像素共享同一权重)
    """
    def __init__(self, n_ssfr=7):
        super().__init__()
        self.n_ssfr = n_ssfr
        self.gate_conv = nn.Sequential(
            nn.Conv2d(1, 16, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_ssfr, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, n_ssfr+1, H, W]
        ssfr = x[:, :self.n_ssfr]   # [B, n_ssfr, H, W]
        luma = x[:, self.n_ssfr:]   # [B, 1, H, W]

        # 逐像素门控: gate shape [B, n_ssfr, H, W]
        gate = self.gate_conv(luma)
        gated_ssfr = ssfr * gate
        return torch.cat([gated_ssfr, luma], dim=1)


class ChannelInteraction(nn.Module):
    """
    跨通道关联建模:
    用 1×1 conv 让通道之间交换信息，捕捉 "Ch1+Ch4 同时异常" 等共现模式
    """
    def __init__(self, in_ch, mid_ch=16):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.mix(x))


class ForensicLocalizationHead(nn.Module):
    """
    FLH v2: 从 SSFR 物理特征 + 亮度图生成像素级篡改定位
    替代 TruFor 提供的 [B, 1, 32, 32] 定位概率图

    v2 改进:
    - SpatialLuminanceGate 逐像素门控 (v1 用全局均值)
    - ChannelInteraction 跨通道关联
    - CBAMBlock (SE + SpatialAttention) 替代纯 SE
    """
    def __init__(self, in_channels=8, mid_channels=48):
        super().__init__()
        n_ssfr = in_channels - 1
        self.luma_gate = SpatialLuminanceGate(n_ssfr=n_ssfr)
        self.ch_interact = ChannelInteraction(in_ch=in_channels, mid_ch=16)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.cbam1 = CBAMBlock(mid_channels)
        self.cbam2 = CBAMBlock(mid_channels)
        self.cbam3 = CBAMBlock(mid_channels)

        self.head = nn.Conv2d(mid_channels, 1, 1)

    def forward(self, x):
        """
        Args:
            x: [B, 8, 32, 32]  (7 SSFR channels + 1 luma channel)
        Returns:
            logits: [B, 1, 32, 32]  篡改概率 logits (未经 sigmoid)
        """
        x = self.luma_gate(x)      # 逐像素亮度门控
        x = self.ch_interact(x)    # 跨通道关联
        x = self.stem(x)           # 8 → mid_channels
        x = self.cbam1(x)          # ResBlock + SE + SpatialAttn
        x = self.cbam2(x)
        x = self.cbam3(x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class FocalLoss(nn.Module):
    """Focal Loss — 专注于难分类像素，比 BCE 更适合极端不均衡"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class DiceLoss(nn.Module):
    """Dice Loss — 解决篡改区域远小于正常区域的严重正负样本不均衡"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (probs * targets).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)


class FLHLoss(nn.Module):
    """Focal + Dice 组合损失 (v2: Focal 替代 BCE，更关注难样本)"""
    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=0.25, gamma=2.0,
                 boundary_weight=3.0, boundary_width=2):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.boundary_width = boundary_width

    @staticmethod
    def _boundary_mask(targets, width=2):
        """Extract boundary pixels from binary mask using max_pool erosion/dilation."""
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        k = 2 * width + 1
        dilated = F.max_pool2d(targets, kernel_size=k, stride=1, padding=width)
        eroded = -F.max_pool2d(-targets, kernel_size=k, stride=1, padding=width)
        boundary = (dilated - eroded).clamp(0, 1)
        return boundary.squeeze(1) if targets.dim() == 4 else boundary

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
