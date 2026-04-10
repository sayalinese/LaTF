import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import timm
import numpy as np
from PIL import Image
from service.forensic_localization import (
    SpatialLuminanceGate, ChannelInteraction, CBAMBlock, SpatialAttention
)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LocalizationRefineBlock(nn.Module):
    """Refine coarse forensic features with high-resolution texture skip features."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up_proj = ConvBNAct(in_ch, out_ch, kernel_size=3)
        self.skip_proj = ConvBNAct(skip_ch, out_ch, kernel_size=1, padding=0)
        self.fuse = ConvBNAct(out_ch * 2, out_ch, kernel_size=3)
        self.mix = ChannelInteraction(in_ch=out_ch, mid_ch=max(16, out_ch // 2))
        self.cbam = CBAMBlock(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up_proj(x)
        shortcut = x  # residual before fusion
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        skip = self.skip_proj(skip)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.mix(x)
        x = self.cbam(x)
        return x + shortcut  # residual connection

# 伪造检测常用的 SRM 滤波器 (高频噪声增强)
class SRMConv2d(nn.Module):
    def __init__(self, in_channels=3):
        super(SRMConv2d, self).__init__()
        self.in_channels = in_channels
        # 3个基础 SRM 卷积核
        q1 = [[0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0],
              [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0]]
        
        q2 = [[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]]
        
        q3 = [[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]
        
        # 定义卷积层，不更新权重
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=5, padding=2, bias=False)
        
        # 初始化权重 (Scale to keep values reasonable)
        filters = torch.FloatTensor([q1, q2, q3]) / 4.0 
        filters = filters.unsqueeze(1).repeat(1, in_channels, 1, 1) # [3, 3, 5, 5]
        
        self.conv.weight.data = filters
        for param in self.conv.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        return self.conv(x)

class TextureBranch(nn.Module):
    """
    高频纹理分支：专门处理高分辨率输入，提取噪声特征
    默认使用 ConvNeXt-Tiny (比 ResNet50 强且快)，也可换 EfficientViT
    """
    def __init__(self, model_name='convnext_tiny', pretrained=True, use_srm=True):
        super().__init__()
        self.use_srm = use_srm
        if use_srm:
            self.srm = SRMConv2d(in_channels=3)
        
        # 加载 timm 模型
        # num_classes=0 表示移除最后的分类头，只取特征
        print(f"[V11] Initializing Texture Branch: {model_name}")
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
            self.features_only = True
            self.feature_dims = list(self.backbone.feature_info.channels())
        except Exception:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
            self.features_only = False
            dummy_in = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                dummy_features = self.backbone.forward_features(dummy_in)
            if isinstance(dummy_features, (list, tuple)):
                self.feature_dims = [feat.shape[1] for feat in dummy_features]
            else:
                self.feature_dims = [dummy_features.shape[1]]

        self.out_dim = self.feature_dims[-1]

    def forward(self, x, return_features=False):
        # x: [B, 3, H, W] (High Res, e.g., 512x512)
        
        # 1. 提取高频噪声 (可选)
        if self.use_srm:
            # SRM 提取的是噪声残差，我们把它叠加到 RGB 上，或者直接作为输入
            # 策略：RGB + SRM 融合 (简单的加法或拼接)
            noise = self.srm(x)
            # 简单的融合策略：RGB (保留语义) + Noise (增强纹理)
            x_in = x + noise 
        else:
            x_in = x
            
        # 2. Backbone 提取
        if self.features_only:
            feature_maps = self.backbone(x_in)
        else:
            raw_features = self.backbone.forward_features(x_in)
            if isinstance(raw_features, (list, tuple)):
                feature_maps = list(raw_features)
            else:
                feature_maps = [raw_features]

        global_feat = F.adaptive_avg_pool2d(feature_maps[-1], (1, 1)).flatten(1)
        if not return_features:
            return global_feat

        if len(feature_maps) == 1:
            feature_maps = [feature_maps[0], feature_maps[0], feature_maps[0]]
        elif len(feature_maps) == 2:
            feature_maps = [feature_maps[0], feature_maps[0], feature_maps[1]]

        pyramid = {
            'skip_128': feature_maps[0],
            'skip_64': feature_maps[1],
            'bridge_32': feature_maps[2],
        }
        return global_feat, pyramid

class LaREDeepFakeV11(nn.Module):
    """
    LaRE V11: Dual-Stream Fusion Network
    Stream 1: CLIP (Semantic / Content)
    Stream 2: ConvNeXt/EfficientViT (Texture / Noise)
    Stream 3: LaRE Map (Physical Reconstruction Error)
    """
    def __init__(self, num_classes=2, clip_type="RN50x64", texture_model='convnext_tiny'):
        super().__init__()
        
        # --- Stream 1: CLIP (Semantic) ---
        print(f"[V11] Loading CLIP: {clip_type}")
        self.clip_model, _ = clip.load(clip_type, device='cpu', jit=False)
        # 冻结 CLIP 以保持泛化能力，只微调最后一层 (可选)
        for param in self.clip_model.visual.parameters():
            param.requires_grad = False
        self.clip_dim = 1024 if "RN50x64" in clip_type else 512
        
        # --- Stream 2: Texture (High Freq) ---
        self.texture_branch = TextureBranch(model_name=texture_model, use_srm=True)
        self.texture_dim = self.texture_branch.out_dim
        
        # --- Stream 3: SSFR/LaRE Map Processing ---
        # [V17] 8 channels (7 SSFR + 1 luma), no external FLH
        self.luma_gate = SpatialLuminanceGate(n_ssfr=7)   # 逐像素亮度门控
        self.ch_interact = ChannelInteraction(in_ch=8, mid_ch=16)  # 跨通道关联

        self.lare_conv = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.lare_dim = 64
        
        # --- [V14] Localization 独立前处理 (不共享权重，避免 seg loss 污染分类路) ---
        self.luma_gate_loc = SpatialLuminanceGate(n_ssfr=7)
        self.ch_interact_loc = ChannelInteraction(in_ch=8, mid_ch=16)
        
        # --- [V14] Integrated Localization Head (replaces external FLH + old seg_head) ---
        self.loc_stem = nn.Sequential(
            nn.Conv2d(8, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.loc_cbam1 = CBAMBlock(48)
        self.loc_cbam2 = CBAMBlock(48)
        self.loc_cbam3 = CBAMBlock(48)
        self.loc_head = nn.Conv2d(48, 1, 1)  # Output: [B, 1, 32, 32] Logits

        # --- [V14] High-Resolution Refinement Decoder ---
        skip_64_dim = self.texture_branch.feature_dims[1] if len(self.texture_branch.feature_dims) > 1 else self.texture_branch.feature_dims[0]
        skip_128_dim = self.texture_branch.feature_dims[0]
        self.loc_refine64 = LocalizationRefineBlock(48, skip_64_dim, 32)
        self.loc_refine128 = LocalizationRefineBlock(32, skip_128_dim, 24)
        self.mid_head = nn.Conv2d(32, 1, 1)
        self.fine_head = nn.Conv2d(24, 1, 1)

        # Zero-init residual refinement heads so old checkpoints fall back to upsampled coarse maps.
        nn.init.zeros_(self.mid_head.weight)
        nn.init.zeros_(self.mid_head.bias)
        nn.init.zeros_(self.fine_head.weight)
        nn.init.zeros_(self.fine_head.bias)
        
        # --- Fusion Head ---
        fusion_dim = self.clip_dim + self.texture_dim + self.lare_dim
        print(f"[V11] Feature Fusion Dimensions: CLIP({self.clip_dim}) + Texture({self.texture_dim}) + LaRE({self.lare_dim}) = {fusion_dim}")
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        
    def forward(self, img_clip, img_highres, lare_map, luma_map=None, return_seg=False, return_seg_pyramid=False):
        """
        [V14] 内嵌定位头，不再需要外部 FLH
        Args:
            img_clip: [B, 3, 448, 448] (for CLIP)
            img_highres: [B, 3, 1024, 1024] or [B, 3, 512, 512] (for Texture Branch)
            lare_map: [B, 7, 32, 32] (SSFR features, 7 channels including VAE recon)
            luma_map: [B, 1, 32, 32] (Luminance map) - Optional
            return_seg: If True, returns the fine localization logits mapping
            return_seg_pyramid: If True, returns (coarse, mid, fine) localization logits
        """
        # 统一转为 float32，避免 bf16/fp16 混合精度导致 torch.cat dtype mismatch 或 NaN
        img_clip = img_clip.float()
        img_highres = img_highres.float()
        lare_map = lare_map.float()

        # 1. Semantic Features (CLIP)
        with torch.no_grad():
            semantic_feat = self.clip_model.encode_image(img_clip)
            semantic_feat = semantic_feat.float()
        
        # 2. Texture Features (ConvNeXt)
        # 支持梯度更新，专门学习伪造纹理
        need_seg_features = return_seg or return_seg_pyramid
        if need_seg_features:
            texture_feat, texture_pyramid = self.texture_branch(img_highres, return_features=True)
            texture_feat = texture_feat.float()
        else:
            texture_feat = self.texture_branch(img_highres).float()
            texture_pyramid = None
        
        # 3. SSFR Map + Luma Map → 共享前处理
        batch_size, n_ssfr, h, w = lare_map.shape
        # Backward compat: pad to 7 SSFR channels if old features
        if n_ssfr < 7:
            pad = torch.zeros((batch_size, 7 - n_ssfr, h, w), device=lare_map.device, dtype=lare_map.dtype)
            lare_map = torch.cat([lare_map, pad], dim=1)
        if luma_map is not None:
            luma_small = F.interpolate(luma_map, size=(h, w), mode='bilinear', align_corners=False)
        else:
            luma_small = torch.zeros((batch_size, 1, h, w), device=lare_map.device, dtype=lare_map.dtype)

        # [V14] 共享前处理: 亮度门控 + 跨通道关联
        map_input = torch.cat([lare_map, luma_small], dim=1)  # [B, 8, 32, 32]
        map_gated = self.luma_gate(map_input)      # 逐像素亮度门控 → [B, 8, 32, 32]
        map_mixed = self.ch_interact(map_gated)    # 跨通道关联     → [B, 8, 32, 32]

        # 分类路: 池化到向量
        lare_feat = self.lare_conv(map_mixed).flatten(1).float()
        
        # 4. Feature Fusion → Classification
        combined = torch.cat([semantic_feat, texture_feat, lare_feat], dim=1)
        logits = self.fusion_mlp(combined)
        
        if need_seg_features:
            # [V14] 定位路独立前处理: seg loss 不回传到分类路的 luma_gate/ch_interact
            map_gated_loc = self.luma_gate_loc(map_input)
            map_mixed_loc = self.ch_interact_loc(map_gated_loc)

            # [V14] 内嵌定位头: coarse → fine refinement
            loc_x = self.loc_stem(map_mixed_loc)
            loc_x = self.loc_cbam1(loc_x)
            loc_x = self.loc_cbam2(loc_x)
            loc_x = self.loc_cbam3(loc_x)
            coarse_logits = self.loc_head(loc_x).float()  # [B, 1, 32, 32]

            # [V14 fix] detach texture pyramid → seg loss 不回传到纹理分支，保护分类能力
            mid_x = self.loc_refine64(loc_x, texture_pyramid['skip_64'].detach().float())
            mid_base = F.interpolate(coarse_logits, size=mid_x.shape[-2:], mode='bilinear', align_corners=False)
            mid_logits = (mid_base + self.mid_head(mid_x)).float()  # [B, 1, 64, 64]

            fine_x = self.loc_refine128(mid_x, texture_pyramid['skip_128'].detach().float())
            fine_base = F.interpolate(mid_logits, size=fine_x.shape[-2:], mode='bilinear', align_corners=False)
            fine_logits = (fine_base + self.fine_head(fine_x)).float()  # [B, 1, 128, 128]

            if return_seg_pyramid:
                return logits, (coarse_logits, mid_logits, fine_logits)
            return logits, fine_logits
            
        return logits
