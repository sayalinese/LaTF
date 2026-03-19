import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path

from service.model import CLipClassifierWMapV9Lite, CLipClassifierWMapV7, CLipClassifierV10Fusion
from service.model_v11_fusion import LaREDeepFakeV11
from service.lare_extractor_module import LareExtractor
from service.trufor_wrapper import TruForExtractor
from service.cascade_inference import CascadeInference
from service.statistical_detector import StatisticalLocalDetector, DualModelDetector
from service.heatmap_utils import refine_map_for_visibility

class LaFTManager:
    """
    统一管理 LaRE + TruFor 以及统计检测和融合分类器的加载和预测逻辑。
    将核心逻辑从 Flask app.py 中解耦。
    """
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = ["Real Photo", "AI Generated"]
        
        self.model = None
        self.lare_extractor = None
        self.trufor_extractor = None
        self.cascade_inference = None
        self.dual_detector = None
        self.stat_detector = None
        
        self.loaded_ckpt_path = None
        self.resolved_out_dir = None
        
        self.use_cascade = os.getenv("USE_CASCADE_INFERENCE", "true").lower() == "true"
        self.cascade_threshold = float(os.getenv("CASCADE_THRESHOLD", "0.3"))
        self.model_version = os.getenv("MODEL_VERSION", "V13")
        self.ai_confidence_threshold = float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.5"))
        
    def load_models(self):
        print(f"Loading models on {self.device}...")

        # 1. Initialize LaRE Extractor
        model_type = os.getenv("LARE_MODEL_TYPE", "sdxl")  
        print(f"[Config] Using LaRE backbone: {model_type.upper()}")
        
        if self.lare_extractor is None:
            try:
                self.lare_extractor = LareExtractor(device=self.device, model_type=model_type, dtype=torch.float16)
            except Exception as ex:
                print(f"[LaRE] Failed to load: {ex}")
                self.lare_extractor = None

        # 2. Initialize TruFor Extractor
        if self.trufor_extractor is None and self.model_version in ["V13", "V11"]:
            try:
                print("[Config] Initializing TruFor Extractor...")
                self.trufor_extractor = TruForExtractor(device=self.device)
            except Exception as ex:
                print(f"[TruFor] Failed to load: {ex}")
                self.trufor_extractor = None

        # 3. Initialize Classifier
        clip_type = "RN50x64"
        print(f"[Config] Loading model version: {self.model_version}")
        
        ckpt_path = None
        
        if self.model_version == "V13" or self.model_version == "V11Fusion":
            texture_model = os.getenv("TEXTURE_MODEL", "convnext_tiny")
            model = LaREDeepFakeV11(clip_type=clip_type, num_classes=2, texture_model=texture_model)
            out_dir = os.getenv('OUT_DIR', 'outputs/v13_doubao_focused')
            print(f"[Config] Initializing V13 Fusion (TruFor Enhanced) with Texture Branch: {texture_model}")
        elif self.model_version == "V11":
            texture_model = os.getenv("TEXTURE_MODEL", "convnext_tiny")
            model = LaREDeepFakeV11(clip_type=clip_type, num_classes=2, texture_model=texture_model)
            out_dir = os.getenv('OUT_DIR', 'outputs/v11_fusion')
            print(f"[Config] Initializing V11 Fusion with Texture Branch: {texture_model}")
        elif self.model_version == "V10Fusion":
            model = CLipClassifierV10Fusion(clip_type=clip_type, num_class=2)
            out_dir = os.getenv('OUT_DIR', 'outputs/v10_fusion_stage2_local')
        elif self.model_version == "V9Lite":
            model = CLipClassifierWMapV9Lite(clip_type=clip_type, num_class=2)
            out_dir = os.getenv('OUT_DIR', 'outputs/v9lite')
        else:
            model = CLipClassifierWMapV7(clip_type=clip_type, num_class=2)
            out_dir = os.getenv('OUT_DIR', 'outputs/sdv5_v7')

        out_dir_path = Path(out_dir)
        if not out_dir_path.is_absolute():
            out_dir_path = (self.project_root / out_dir_path).resolve()
            
        self.resolved_out_dir = str(out_dir_path)
        print(f"[Config] Model weights directory: {out_dir} -> {self.resolved_out_dir}")
        
        possible_names = ['best.pth', 'Val_best.pth', 'latest.pth']
        
        if os.path.exists(self.resolved_out_dir):
            for name in possible_names:
                p = os.path.join(self.resolved_out_dir, name)
                if os.path.exists(p):
                    ckpt_path = p
                    break
        
        if ckpt_path is None:
            for root, dirs, files in os.walk(self.resolved_out_dir):
                for name in possible_names:
                    if name in files:
                        ckpt_path = os.path.join(root, name)
                        break
                if ckpt_path:
                    break

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"Load result: {msg}")
            self.loaded_ckpt_path = ckpt_path
        else:
            self.loaded_ckpt_path = None
            print(f"Warning: No checkpoint found under {self.resolved_out_dir}! Using random weights.")

        model.to(self.device)
        model.eval()
        self.model = model
        
        # 4. Cascade Inference
        if self.use_cascade and self.model_version in ("V9Lite", "V10Fusion", "V11", "V13"):
            self.cascade_inference = CascadeInference(
                model=self.model,
                lare_extractor=self.lare_extractor,
                trufor_extractor=self.trufor_extractor,
                device=self.device,
                threshold=self.cascade_threshold
            )
            print(f"[Config] Cascade inference enabled (threshold={self.cascade_threshold})")
        else:
            self.cascade_inference = None
            print("[Config] Using standard inference")
        
        # 5. Statistical Detector
        self.stat_detector = StatisticalLocalDetector()
        print("[Config] Statistical local detector initialized")
        
        # 6. Dual Model Detector
        if self.lare_extractor is not None:
            self.dual_detector = DualModelDetector(
                global_model=self.model,
                lare_extractor=self.lare_extractor,
                device=self.device,
                global_high_threshold=0.90,
                global_low_threshold=0.20,
                local_weight=0.6
            )
            print("[Config] Dual model detector initialized")
        
        print("Models loaded successfully!")

    def _generate_heatmap_base64(self, loc_map_np, img: Image.Image):
        """生成并在原图上叠加热力图，返回base64"""
        src_hwc = np.array(img.convert('RGB')) if img is not None else None
        
        # 使用 shared utility (包含 Percentile norm 等)
        loc_norm_byte = refine_map_for_visibility(
            loc_map_np,
            src_hwc,
            boost_black_fill=True,
            dark_threshold=35,
            fill_min_area_ratio=0.001
        )
        
        target_size = img.size if img else (448, 448)
        loc_norm = cv2.resize(loc_norm_byte, target_size, interpolation=cv2.INTER_LINEAR)
        alpha_mask = np.clip((loc_norm.astype(np.float32) - 36.0) / 170.0, 0.0, 1.0) * 0.78

        heatmap_color = cv2.applyColorMap(loc_norm, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        if img is not None:
            base_rgb = np.array(img.convert('RGB'), dtype=np.float32)
            heatmap_rgb = heatmap_color.astype(np.float32)
            heatmap_color = (base_rgb * (1.0 - alpha_mask[..., None]) + heatmap_rgb * alpha_mask[..., None]).astype(np.uint8)
        
        heatmap_pil = Image.fromarray(heatmap_color)
        buffered = BytesIO()
        heatmap_pil.save(buffered, format="JPEG", quality=90)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def predict(self, img: Image.Image):
        if self.model is None or self.lare_extractor is None:
            try:
                self.load_models()
            except Exception as e:
                return {
                    "is_real": False,
                    "confidence": 0.0,
                    "error": f"Failed to load models: {str(e)}",
                    "cascade_triggered": False
                }
            
        if self.lare_extractor is None:
            return {
                "is_real": False,
                "confidence": 0.0,
                "error": "LaRE_extractor (SDXL backbone) failed to load. Check GPU memory.",
                "cascade_triggered": False
            }

        try:
            if self.cascade_inference is not None:
                return self._predict_cascade(img)
            else:
                return self._predict_standard(img)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _predict_cascade(self, img: Image.Image):
        trufor_vis_map = None
        if self.trufor_extractor is not None:
            try:
                img_np = np.asarray(img, dtype=np.float32) / 255.0
                tf_input = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
                tf_np_vis = self.trufor_extractor.extract_batch(tf_input)
                if tf_np_vis is not None and len(tf_np_vis) > 0:
                    trufor_vis_map = tf_np_vis[0]
            except Exception:
                trufor_vis_map = None

        result = self.cascade_inference.inference(img, return_details=True)
        heatmap_base64 = None
        
        map_source = None
        map_source_type = None
        if trufor_vis_map is not None:
            map_source = trufor_vis_map
            map_source_type = "trufor_map"
        elif result['loc_map'] is not None:
            map_source = result['loc_map']
            map_source_type = "loc_map"
        elif 'raw_lare_map' in result and result['raw_lare_map'] is not None:
            map_source = result['raw_lare_map']
            map_source_type = "raw_lare_map"
            
        if map_source is not None and (result['pred'] == 1 or result['prob'] > 0.3):
            heatmap_base64 = self._generate_heatmap_base64(map_source, img)
        
        return {
            "class_name": result['class_name'],
            "confidence": result['prob'],
            "class_idx": result['pred'],
            "probabilities": {
                self.class_names[0]: 1 - result['prob'],
                self.class_names[1]: result['prob']
            },
            "heatmap": heatmap_base64,
            "cascade_info": {"global_prob": result['global_prob'], "local_prob": result['local_prob'], "crop_bbox": result['crop_bbox']},
            "debug": {
                "model_version": self.model_version,
                "resolved_out_dir": self.resolved_out_dir,
                "ckpt_path": self.loaded_ckpt_path,
                "cascade_enabled": True,
                "heatmap_source": map_source_type
            }
        }

    def _predict_standard(self, img: Image.Image):
        from torchvision import transforms
        preprocess_clip = transforms.Compose([
            transforms.Resize((448, 448)), transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        preprocess_highres = transforms.Compose([
            transforms.Resize((512, 512)), transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        preprocess_trufor = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])
        
        with torch.no_grad():
            loss_map = self.lare_extractor.extract_single(img).to(self.device)
            clip_input = preprocess_clip(img).unsqueeze(0).to(self.device)
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
            try: from torch.amp import autocast
            except ImportError: from torch.cuda.amp import autocast

            with autocast('cuda', dtype=amp_dtype):
                if self.model_version in ["V11", "V13"]:
                    highres_input = preprocess_highres(img).unsqueeze(0).to(self.device)
                    trufor_map = None
                    if self.trufor_extractor is not None:
                         tf_in = preprocess_trufor(img).unsqueeze(0).to(self.device)
                         tf_np = self.trufor_extractor.extract_batch(tf_in)
                         if tf_np is not None:
                             tf_tensor = torch.from_numpy(tf_np).unsqueeze(1).float()
                             tf_tensor = F.interpolate(tf_tensor, size=(448, 448), mode='bilinear', align_corners=False)
                             trufor_map = tf_tensor.to(self.device)
                             
                    try: logits = self.model(clip_input, highres_input, loss_map, trufor_map=trufor_map)
                    except TypeError: logits = self.model(clip_input, highres_input, loss_map)
                    loc_map = loss_map
                    if loc_map.shape[1] == 4: loc_map = loc_map.mean(dim=1, keepdim=True)
                elif hasattr(self.model, 'loc_head'):
                    logits, loc_map = self.model(clip_input, loss_map, return_loc=True)
                else:
                    logits = self.model(clip_input, loss_map)
                    loc_map = None
                
        probs = F.softmax(logits, dim=1)[0]
        confidence, pred_idx = torch.max(probs, 0)
        
        if probs[1].item() < self.ai_confidence_threshold:
            pred_idx = torch.tensor(0)
            confidence = torch.tensor(1.0 - probs[1].item())
            
        idx = int(pred_idx.item())
        heatmap_base64 = None
        use_heatmap = True if (self.model_version in ["V11", "V13"]) else False

        if use_heatmap and (idx == 1 or self.model_version in ["V11", "V13"]) and loc_map is not None:
            loc_np = loc_map.squeeze().cpu().float().numpy()
            # Basic Cleanup
            h, w = loc_np.shape
            if h > 8 and w > 8:
                loc_np = loc_np[1:-1, 1:-1]
                loc_np = np.pad(loc_np, ((1, 1), (1, 1)), mode='edge')
            p05, p95 = np.percentile(loc_np, [5, 95])
            loc_clipped = np.clip(loc_np, p05, p95)
            loc_norm_float = (loc_clipped - p05) / (p95 - p05) if (p95 - p05) > 1e-6 else np.zeros_like(loc_clipped)
            
            loc_byte = (loc_norm_float * 255).astype(np.uint8)
            loc_byte = cv2.morphologyEx(loc_byte, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            
            guide_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            loc_upscaled = cv2.resize(loc_byte, img.size, interpolation=cv2.INTER_LINEAR)
            
            try:
                from cv2.ximgproc import guidedFilter
                refined_map = guidedFilter(guide=guide_img, src=loc_upscaled, radius=16, eps=500, dDepth=-1)
                loc_final = cv2.normalize(refined_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            except ImportError:
                loc_final = cv2.bilateralFilter(loc_upscaled, d=9, sigmaColor=75, sigmaSpace=75)

            heatmap_color = cv2.cvtColor(cv2.applyColorMap(loc_final, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            heatmap_pil = Image.fromarray(heatmap_color)
            buffered = BytesIO()
            heatmap_pil.save(buffered, format="JPEG", quality=90)
            heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
        return {
            "class_name": self.class_names[idx], "confidence": float(confidence.item()), "class_idx": idx,
            "probabilities": {self.class_names[0]: float(probs[0]), self.class_names[1]: float(probs[1])},
            "heatmap": heatmap_base64,
            "debug": {"model_version": self.model_version, "cascade_enabled": False}
        }