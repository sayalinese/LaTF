import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import timm
import numpy as np
from PIL import Image

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
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # 获取特征维度 (convnext_tiny=768, efficientnet_b0=1280, etc.)
        dummy_in = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dummy_out = self.backbone(dummy_in)
        self.out_dim = dummy_out.shape[1]
        
    def forward(self, x):
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
        features = self.backbone(x_in) # [B, out_dim]
        return features

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
        
        # --- Stream 3: LaRE Map Processing ---
        # 简单的卷积层处理 LaRE Map
        # [V13] Updated to 5 channels (4 LaRE + 1 TruFor)
        self.lare_conv = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.lare_dim = 64
        
        # --- [V12] Auxiliary Segmentation Head (Pixel-level Supervision) ---
        # [V13] Updated to 5 channels
        self.seg_head = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1) # Output: [B, 1, 32, 32] Logits
        )
        
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

        
    def forward(self, img_clip, img_highres, lare_map, trufor_map=None, return_seg=False):
        """
        Args:
            img_clip: [B, 3, 448, 448] (for CLIP)
            img_highres: [B, 3, 1024, 1024] or [B, 3, 512, 512] (for Texture Branch)
            lare_map: [B, 4, 32, 32] (LaRE features)
            trufor_map: [B, 1, 512, 512] (TruFor prob map) - Optional
            return_seg: If True, returns segmentation logits mapping
        """
        # 统一转为 float32，避免 bf16/fp16 混合精度导致 torch.cat dtype mismatch 或 NaN
        img_clip = img_clip.float()
        img_highres = img_highres.float()
        lare_map = lare_map.float()
        if trufor_map is not None:
            trufor_map = trufor_map.float()

        # 1. Semantic Features (CLIP)
        with torch.no_grad():
            semantic_feat = self.clip_model.encode_image(img_clip)
            semantic_feat = semantic_feat.float()
        
        # 2. Texture Features (ConvNeXt)
        # 支持梯度更新，专门学习伪造纹理
        texture_feat = self.texture_branch(img_highres).float()
        
        # 3. LaRE Map + TruFor Map Processing
        # Note: trufor_map could be None if user forgot to pass it
        if trufor_map is not None:
             # Resize TruFor from 512x512/HighRes to 32x32
             # Note: It might be already resized in Dataset if we did it there, 
             # but to be safe and handle different resolutions, we interpolate.
             trufor_small = F.interpolate(trufor_map, size=(32, 32), mode='bilinear', align_corners=False)
             # Concatenate: [B, 4, 32, 32] + [B, 1, 32, 32] -> [B, 5, 32, 32]
             map_input = torch.cat([lare_map, trufor_small], dim=1)
        else:
             # Fallback if no TruFor provided (e.g. old code inference)
             # Pad with zeros? Or fail? 
             # For robustness, we pad with a zero channel so indices match trained weights
             batch_size, _, h, w = lare_map.shape
             zeros = torch.zeros((batch_size, 1, h, w), device=lare_map.device, dtype=lare_map.dtype)
             map_input = torch.cat([lare_map, zeros], dim=1)

        lare_feat = self.lare_conv(map_input).flatten(1).float()
        
        # 4. Feature Fusion
        combined = torch.cat([semantic_feat, texture_feat, lare_feat], dim=1)
        logits = self.fusion_mlp(combined)
        
        if return_seg:
            # [V12] Auxiliary Segmentation Output using LaRE Map
            seg_logits = self.seg_head(map_input).float() # [B, 1, 32, 32]
            
            return logits, seg_logits
            
        return logits
