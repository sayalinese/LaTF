""
Non-Local Attention Module for V9-Lite
Adapted from PSCC-Net's NonLocalMask implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalAttention(nn.Module):
    """
    Non-Local Attention with both Spatial and Channel attention.
    Captures long-range dependencies for detecting global inconsistencies.
    
    Args:
        in_channels: Number of input channels
        reduce_scale: Spatial reduction scale for efficiency (default: 1)
        use_lare_guide: Whether to use LaRE map as attention guidance
    """
    def __init__(self, in_channels: int, reduce_scale: int = 1, use_lare_guide: bool = True):
        super(NonLocalAttention, self).__init__()
        
        self.r = reduce_scale
        self.use_lare_guide = use_lare_guide
        
        # Effective input channels after spatial reduction
        self.ic = in_channels * self.r * self.r
        
        # Middle channels (same as input for simplicity)
        self.mc = self.ic
        
        # Key, Query, Value projections
        self.g = nn.Conv2d(self.ic, self.ic, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.ic, self.mc, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.ic, self.mc, kernel_size=1, stride=1, padding=0)
        
        # Output projections for spatial and channel attention
        self.W_s = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.W_c = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
        # Learnable attention weights
        self.gamma_s = nn.Parameter(torch.ones(1))
        self.gamma_c = nn.Parameter(torch.ones(1))
        
        # LaRE guidance projection (if enabled)
        if use_lare_guide:
            self.lare_proj = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )
            self.gamma_lare = nn.Parameter(torch.ones(1))
        
        # Attention mask output head
        self.get_mask = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, lare_map: torch.Tensor = None):
        """
        Args:
            x: Input feature map [B, C, H, W]
            lare_map: LaRE reconstruction error map [B, 4, 32, 32] (optional)
        
        Returns:
            mask: Attention/anomaly mask [B, 1, H, W]
            z: Refined feature map [B, C, H, W]
        """
        b, c, h, w = x.shape
        
        # Spatial reduction for efficiency
        if self.r > 1:
            x_reduced = x.reshape(b, self.ic, h // self.r, w // self.r)
        else:
            x_reduced = x
        
        # g(x) - Value
        g_x = self.g(x_reduced).view(b, self.ic, -1)  # [B, C, HW]
        g_x = g_x.permute(0, 2, 1)  # [B, HW, C]
        
        # theta(x) - Query
        theta_x = self.theta(x_reduced).view(b, self.mc, -1)  # [B, C, HW]
        
        # phi(x) - Key
        phi_x = self.phi(x_reduced).view(b, self.mc, -1)  # [B, C, HW]
        
        # === Spatial Attention ===
        # Attention: softmax(Q^T * K) * V
        theta_x_s = theta_x.permute(0, 2, 1)  # [B, HW, C]
        phi_x_s = phi_x  # [B, C, HW]
        
        f_s = torch.matmul(theta_x_s, phi_x_s)  # [B, HW, HW]
        f_s = F.softmax(f_s, dim=-1)
        
        y_s = torch.matmul(f_s, g_x)  # [B, HW, C]
        y_s = y_s.permute(0, 2, 1).contiguous()  # [B, C, HW]
        y_s = y_s.view(b, c, h, w)
        
        # === Channel Attention ===
        theta_x_c = theta_x  # [B, C, HW]
        phi_x_c = phi_x.permute(0, 2, 1)  # [B, HW, C]
        
        f_c = torch.matmul(theta_x_c, phi_x_c)  # [B, C, C]
        f_c = F.softmax(f_c, dim=-1)
        
        # Channel Attention Map (C x C) * Value (C x HW) -> (C x HW)
        y_c = torch.matmul(f_c, g_x.permute(0, 2, 1))  # [B, C, HW]
        y_c = y_c.view(b, c, h, w)
        
        # === Fusion ===
        z = x + self.gamma_s * self.W_s(y_s) + self.gamma_c * self.W_c(y_c)
        
        # === LaRE Guidance (if enabled) ===
        if self.use_lare_guide and lare_map is not None:
            # Resize LaRE map to match feature map size
            lare_resized = F.interpolate(lare_map, size=(h, w), mode='bilinear', align_corners=False)
            lare_weight = self.lare_proj(lare_resized)  # [B, 1, H, W]
            
            # Apply LaRE-guided weighting
            z = z * (1 + self.gamma_lare * lare_weight)
        
        # === Generate Attention Mask ===
        mask = self.get_mask(z)
        
        return mask, z


class MultiScaleNonLocal(nn.Module):
    """
    Multi-scale Non-Local Attention for capturing anomalies at different scales.
    """
    def __init__(self, in_channels: int, scales: list = [1, 2, 4]):
        super(MultiScaleNonLocal, self).__init__()
        
        self.scales = scales
        self.nl_blocks = nn.ModuleList([
            NonLocalAttention(in_channels, reduce_scale=s, use_lare_guide=(s == 1))
            for s in scales
        ])
        
        # Fusion conv
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * len(scales), in_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Final mask head
        self.mask_fusion = nn.Conv2d(len(scales), 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor, lare_map: torch.Tensor = None):
        """
        Args:
            x: Input feature map [B, C, H, W]
            lare_map: LaRE map [B, 4, 32, 32]
        
        Returns:
            mask: Fused attention mask [B, 1, H, W]
            z: Refined feature map [B, C, H, W]
        """
        masks = []
        features = []
        
        for i, nl in enumerate(self.nl_blocks):
            if self.scales[i] == 1:
                mask_i, z_i = nl(x, lare_map)
            else:
                mask_i, z_i = nl(x, None)
            masks.append(mask_i)
            features.append(z_i)
        
        # Fuse features
        z = self.fusion(torch.cat(features, dim=1))
        
        # Fuse masks
        mask = self.mask_fusion(torch.cat(masks, dim=1))
        mask = torch.sigmoid(mask)
        
        return mask, z


class LocalizationHead(nn.Module):
    """
    Localization head that outputs a pixel-level anomaly map.
    Can be supervised with LaRE map as pseudo-label during training.
    """
    def __init__(self, in_channels: int, out_size: int = 448):
        super(LocalizationHead, self).__init__()
        
        self.out_size = out_size
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Feature map [B, C, H, W]
        
        Returns:
            loc_map: Localization map [B, 1, out_size, out_size]
        """
        loc = self.head(x)
        
        # Upsample to output size
        if loc.shape[-1] != self.out_size:
            loc = F.interpolate(loc, size=(self.out_size, self.out_size), 
                               mode='bilinear', align_corners=False)
        
        return loc

