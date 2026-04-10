"""
水印区域检测与降权模块
基于高频边缘检测 + 连通域分析，无需额外模型

输入: [B, 3, H, W] RGB 图像 (0-1)
输出: [B, 1, out_size, out_size] 水印抑制 mask (0=水印区域需抑制, 1=正常区域)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WatermarkDetector(nn.Module):
    """
    轻量级水印区域检测器
    原理: 水印通常是半透明的高对比文字/logo，在高频域表现为
    "窄而锐利" 的纹理，与自然图像的宽频纹理有显著差异。

    检测逻辑:
    1. Laplacian 高频提取
    2. 全局均值自适应阈值
    3. 局部方差过滤 (排除自然边缘)
    4. 输出抑制 mask
    """

    def __init__(self, out_size=32, suppress_value=0.3):
        super().__init__()
        self.out_size = out_size
        self.suppress_value = suppress_value

        # Laplacian kernel (固定，不参与训练)
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('lap_kernel', lap.reshape(1, 1, 3, 3))

        # 局部方差卷积核
        self.register_buffer('var_kernel', torch.ones(1, 1, 5, 5) / 25.0)

    def forward(self, rgb):
        """
        Args:
            rgb: [B, 3, H, W] 输入图像 (0-1 range)
        Returns:
            suppress_mask: [B, 1, out_size, out_size]
                          1.0 = 正常区域 (不干预)
                          suppress_value = 水印区域 (需降权)
        """
        gray = rgb.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # 高频提取
        edge = F.conv2d(F.pad(gray, (1,1,1,1), mode='reflect'), self.lap_kernel).abs()

        # 局部均值 & 方差
        local_mean = F.conv2d(F.pad(edge, (2,2,2,2), mode='reflect'), self.var_kernel)
        local_sq = F.conv2d(F.pad(edge.pow(2), (2,2,2,2), mode='reflect'), self.var_kernel)
        local_var = (local_sq - local_mean.pow(2)).clamp(min=0)

        # 自适应阈值: 高频强但局部方差低 → 疑似水印
        # (自然边缘高频强且方差也高; 水印是均匀的高频薄层)
        global_mean = edge.mean(dim=(2, 3), keepdim=True)
        high_freq_mask = edge > (global_mean * 2.0)
        low_var_mask = local_var < (local_var.mean(dim=(2, 3), keepdim=True) * 0.5)

        watermark_mask = (high_freq_mask & low_var_mask).float()

        # 缩放到目标尺寸
        wm_small = F.interpolate(watermark_mask, (self.out_size, self.out_size),
                                 mode='bilinear', align_corners=False)

        # 生成抑制 mask: 水印区域 → suppress_value, 正常区域 → 1.0
        suppress_mask = torch.ones_like(wm_small)
        suppress_mask = suppress_mask - wm_small * (1.0 - self.suppress_value)

        return suppress_mask
