"""
LaTF (LaRE + TruFor Fusion) - TruFor Integration Module
TruFor: Detecting Image Forgeries by Localized Traces.
Provides reliability map and localization for manipulated regions.

Reference: https://www.grip.unina.it/download/prog/TruFor/
"""
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# 动态添加 TruFor 路径到环境变量，以便导入其内部模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRUFOR_ROOT = PROJECT_ROOT / '项目参考' / 'TruFor-main' / 'TruFor_train_test'

if str(TRUFOR_ROOT) not in sys.path:
    sys.path.insert(0, str(TRUFOR_ROOT))

# 尝试导入 TruFor 依赖
try:
    from lib.config import config, update_config
    if not hasattr(config, "utils_imported"):
        # 这是一个临时补丁，防止重复导入导致的配置重置
        from lib.utils import get_model
        config.utils_imported = True
    else:
        from lib.utils import get_model
except ImportError as e:
    print(f"TruFor Import Error: {e}")
    print(f"请确保 TruFor 目录结构正确: {TRUFOR_ROOT}")
    # Define dummy config to avoid NameError if import fails
    config = None
    update_config = None
    get_model = None

class TruForExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self._load_model()
        
        # TruFor 的标准预处理
        # 注意: TruFor 模型内部会对 backbone 输入做归一化，但 Noiseprint++ 分支需要 [0,1] 原始 Tensor
        # 所以这里只做 ToTensor，不做 Normalize
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

    def extract_batch(self, images_tensor):
        """
        Batch inference.
        Args:
            images_tensor (torch.Tensor): [B, C, H, W]
        Returns:
            prob_maps (np.array): [B, H, W]
        """
        if self.model is None:
            return None
            
        # Ensure tensor on device
        if images_tensor.device != self.device:
            images_tensor = images_tensor.to(self.device)
            
        with torch.no_grad():
            outputs = self.model(images_tensor)
            
            if isinstance(outputs, tuple):
                # out is usually index 0
                out = outputs[0]
            elif isinstance(outputs, dict):
                out = outputs.get('map', outputs.get('out'))
            else:
                out = outputs

            # Convert Logits to Probability
            prob_map = torch.sigmoid(out)
            
            # [B, 1, H, W] -> [B, H, W]
            prob_map = prob_map.squeeze(1).cpu().numpy()
            
            return prob_map

    def _load_model(self):
        # 1. 设置权重路径
        weights_path = TRUFOR_ROOT / 'pretrained_models' / 'trufor.pth.tar'
        if not weights_path.exists():
            print(f"⚠️ Warning: TruFor weights not found at {weights_path}")
            print("请先下载权重: https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip")
            return

        # 2. 加载配置 (复用 TruFor 默认配置)
        # 我们手动覆盖一些必要的测试参数
        config.TEST.MODEL_FILE = str(weights_path)
        
        # [Fix] 手动合并 trufor_ph3.yaml 的关键配置
        # 因为 get_model 构建模型时非常依赖 config.MODEL.EXTRA 中的参数
        # 而单纯 import config 只有 default.py 的内容，必须手动 merge yaml
        yaml_path = TRUFOR_ROOT / 'lib' / 'config' / 'trufor_ph3.yaml'
        if yaml_path.exists():
            config.merge_from_file(str(yaml_path))
        else:
            print(f"⚠️ Config not found: {yaml_path}")

        # 3. 初始化模型
        print(f"[TruFor] Loading model from {weights_path}...")
        try:
            self.model = get_model(config)
            
            # [Fix] weights_only=False for older .pth.tar files containing numpy scalars
            try:
                checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
            except TypeError:
                # Fallback for older torch versions without weights_only arg
                checkpoint = torch.load(weights_path, map_location='cpu')

            self.model.load_state_dict(checkpoint['state_dict'])
            
            # Retrieve model device - fallback to CPU if not set
            print(f"Moving model to {self.device}...")
            self.model.to(self.device)
            if hasattr(self.model, 'backbone'): self.model.backbone.to(self.device)
            if hasattr(self.model, 'dncnn'): self.model.dncnn.to(self.device)
            if hasattr(self.model, 'decode_head'): self.model.decode_head.to(self.device)
            
            # Verify
            p_device = next(self.model.parameters()).device
            print(f"[TruFor] Model loaded successfully on {p_device}")
            
        except Exception as e:
            print(f"[TruFor] Model load failed: {e}")
            import traceback
            traceback.print_exc()

    def extract_batch(self, images_tensor):
        """
        Batch inference.
        Args:
            images_tensor (torch.Tensor): [B, C, H, W]
        Returns:
            prob_maps (np.array): [B, H, W]
        """
        if self.model is None:
            return None
            
        try:
            # Ensure tensor on device
            if images_tensor.device != self.device:
                images_tensor = images_tensor.to(self.device)
            
            # [Safety Check] Ensure model is on the correct device
            # This handles cases where .to() failed silently or was reset
            param_device = next(self.model.parameters()).device
            if param_device.type != 'cuda' and 'cuda' in str(self.device):
                # print(f"Warning: Model on {param_device}, forcing to {self.device}")
                self.model.to(self.device)
                images_tensor = images_tensor.to(self.device)

            with torch.no_grad():
                outputs = self.model(images_tensor)
                
                if isinstance(outputs, tuple):
                    # out is usually index 0
                    out = outputs[0]
                elif isinstance(outputs, dict):
                    out = outputs.get('map', outputs.get('out'))
                else:
                    out = outputs

                # Convert Logits to Probability
                prob_map = torch.sigmoid(out)
                
                # Check dimensions and handle multi-class (B, 2, H, W)
                if prob_map.dim() == 4:
                    if prob_map.shape[1] == 2:
                        # Take channel 1 (Fake class)
                        prob_map = prob_map[:, 1, :, :]
                    elif prob_map.shape[1] == 1:
                        # Remove channel dim
                        prob_map = prob_map.squeeze(1)
                
                prob_map = prob_map.cpu().numpy()
                
                return prob_map
        except Exception as e:
            print(f"Extract Batch Error: {e}")
            import traceback
            traceback.print_exc()
            return None
            # TruFor model forward returns a tuple: (out, conf, det, modal_x)
            # out: Localization map (Logits)
            # conf: Confidence map
            # det: Detection score
            inputs = img_tensor
            outputs = self.model(inputs)
            
            if isinstance(outputs, tuple):
                # out is usually index 0
                out = outputs[0]
            elif isinstance(outputs, dict):
                out = outputs.get('map', outputs.get('out'))
            else:
                out = outputs

            # Convert Logits to Probability
            prob_map = torch.sigmoid(out)
            
            # Squeeze batch and channel dims: [B, C, H, W] -> [H, W]
            prob_map = prob_map.squeeze().cpu().numpy()
            
            return prob_map

if __name__ == "__main__":
    # 简单测试
    extractor = TruForExtractor()
    if extractor.model:
        print("TruFor Extractor is ready.")
