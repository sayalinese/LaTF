import os
import contextlib
import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from service.model import CLipClassifierWMapV9Lite, CLipClassifierWMapV7, CLipClassifierV10Fusion
from service.model_v11_fusion import LaFT
from service.lare_extractor_module import LareExtractor
from service.ssfr_extractor_module import SsfrExtractor
from service.cascade_inference import CascadeInference
from service.statistical_detector import StatisticalLocalDetector
from service.heatmap_utils import refine_map_for_visibility

class LaFTManager:
    """
    统一管理 LaRE + FLH 以及统计检测和融合分类器的加载和预测逻辑。
    将核心逻辑从 Flask app.py 中解耦。
    """
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.class_names = ["Real", "AI Generated"]

        self.model = None
        self.lare_extractor = None
        self.cascade_inference = None
        self.dual_detector = None  # 保留字段以兼容旧前端 config 展示
        self.stat_detector = None

        # 独立定位模型 (SegFormer)
        self.localization_model = None
        self.localization_ckpt_path = None
        self.localization_enabled = os.getenv("LOCALIZATION_ENABLED", "true").lower() == "true"
        self.localization_model_name = os.getenv("LOCALIZATION_MODEL_NAME", "nvidia/segformer-b2-finetuned-ade-512-512")
        self.localization_weights = os.getenv("LOCALIZATION_CKPT", "outputs/segformer_rgb/best.pth")
        self.localization_img_size = int(os.getenv("LOCALIZATION_IMG_SIZE", "512"))
        self.localization_threshold = float(os.getenv("LOCALIZATION_THRESHOLD", "0.5"))
        self.localization_use_ssfr = os.getenv("LOCALIZATION_USE_SSFR", "false").lower() == "true"

        self.loaded_ckpt_path = None
        self.resolved_out_dir = None

        self.use_cascade = os.getenv("USE_CASCADE_INFERENCE", "true").lower() == "true"
        self.cascade_threshold = float(os.getenv("CASCADE_THRESHOLD", "0.25"))
        self.cascade_high_threshold = float(os.getenv("CASCADE_HIGH_THRESHOLD", "0.90"))
        self.model_version = os.getenv("MODEL_VERSION", "V13")
        self.ai_confidence_threshold = float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.5"))

        self.preprocess_clip = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        self.preprocess_highres = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def load_models(self):
        print(f"Loading models on {self.device}...")

        # 1. Initialize LaRE Extractor
        model_type = os.getenv("LARE_MODEL_TYPE", "sdxl")  
        print(f"[Config] Using LaRE backbone: {model_type.upper()}, Model Version: {self.model_version}")
        
        if self.lare_extractor is None:
            try:
                if self.model_version in ("V14", "V17", "V13", "V11", "V11Fusion"):
                    # V14/V17/V13: 使用 SSFR 7ch 特征提取器 (无需 SDXL UNet)
                    unet_path = os.getenv('SSFR_UNET_PATH', 'outputs/ssfr_unet.pth')
                    vae_model_id = os.getenv('VAE_MODEL_ID', '')
                    self.lare_extractor = SsfrExtractor(
                        device=self.device,
                        unet_path=unet_path,
                        vae_model_id=vae_model_id or None
                    )
                    print(f"[SSFR] Loaded SsfrExtractor (7ch, UNet={unet_path}, VAE={vae_model_id or 'None'})")
                else:
                    self.lare_extractor = LareExtractor(device=self.device, model_type=model_type, dtype=torch.float16)
            except Exception as ex:
                print(f"[LaRE/SSFR] Failed to load: {ex}")
                self.lare_extractor = None

        # 2. (TruFor 已移除)

        # 3. Initialize Classifier
        clip_type = "RN50x64"
        print(f"[Config] Loading model version: {self.model_version}")
        
        ckpt_path = None
        
        if self.model_version in ("V14", "V17"):
            texture_model = os.getenv("TEXTURE_MODEL", "convnext_tiny")
            model = LaFT(clip_type=clip_type, num_classes=2, texture_model=texture_model)
            default_out_dir = 'outputs/v14_multiscale' if self.model_version == 'V14' else 'outputs/v17_joint'
            out_dir = os.getenv('OUT_DIR', default_out_dir)
            print(f"[Config] Initializing {self.model_version} Joint (Integrated FLH) with Texture Branch: {texture_model}")
        elif self.model_version == "V13" or self.model_version == "V11Fusion":
            texture_model = os.getenv("TEXTURE_MODEL", "convnext_tiny")
            model = LaFT(clip_type=clip_type, num_classes=2, texture_model=texture_model)
            out_dir = os.getenv('OUT_DIR', 'outputs/v13_doubao_focused')
            print(f"[Config] Initializing V13 Fusion (FLH Enhanced) with Texture Branch: {texture_model}")
        elif self.model_version == "V11":
            texture_model = os.getenv("TEXTURE_MODEL", "convnext_tiny")
            model = LaFT(clip_type=clip_type, num_classes=2, texture_model=texture_model)
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
        
        # 4. Cascade Inference (仅影响分类概率，不替代独立定位模型)
        if self.use_cascade and self.model_version in ("V9Lite", "V10Fusion", "V11", "V13", "V14", "V17"):
            self.cascade_inference = CascadeInference(
                model=self.model,
                lare_extractor=self.lare_extractor,
                device=self.device,
                threshold=self.cascade_threshold,
                high_threshold=self.cascade_high_threshold
            )
            print(f"[Config] Cascade inference enabled (low={self.cascade_threshold}, high={self.cascade_high_threshold})")
        else:
            self.cascade_inference = None
            print("[Config] Using standard inference")
        
        # 5. Statistical Detector
        self.stat_detector = StatisticalLocalDetector()
        print("[Config] Statistical local detector initialized")
        
        # 6. 独立定位模型 (SegFormer)
        self._load_localization_model()

        # 7. 旧接口字段保留
        self.dual_detector = None

        print("Models loaded successfully!")

    def _resolve_path(self, maybe_relative_path: str) -> Path:
        p = Path(maybe_relative_path)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    def _expand_first_conv(self, model, num_channels: int):
        """将 SegFormer 第一层 Conv 输入从 3 扩展到 num_channels。"""
        first_conv = model.segformer.encoder.patch_embeddings[0].proj
        old_weight = first_conv.weight.data  # [out, 3, kH, kW]
        out_ch, _, k_h, k_w = old_weight.shape

        new_conv = nn.Conv2d(
            num_channels,
            out_ch,
            kernel_size=(k_h, k_w),
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_weight
            nn.init.xavier_uniform_(new_conv.weight[:, 3:])
            if first_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias.data)

        model.segformer.encoder.patch_embeddings[0].proj = new_conv

    def _load_localization_model(self):
        self.localization_model = None
        self.localization_ckpt_path = None

        if not self.localization_enabled:
            print("[Config] Independent localization disabled")
            return

        try:
            from transformers import SegformerForSemanticSegmentation, logging as transformers_logging
        except Exception as ex:
            print(f"[Config] transformers not available, localization disabled: {ex}")
            return

        ckpt_path = self._resolve_path(self.localization_weights)
        if not ckpt_path.exists():
            print(f"[Config] Localization checkpoint not found: {ckpt_path}")
            return

        transformers_logging.set_verbosity_error()
        num_channels = 10 if self.localization_use_ssfr else 3

        try:
            model = SegformerForSemanticSegmentation.from_pretrained(
                self.localization_model_name,
                num_labels=2,
                ignore_mismatched_sizes=True,
            )
            if num_channels != 3:
                self._expand_first_conv(model, num_channels)

            checkpoint = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"[Config] Localization load result: {msg}")

            model.to(self.device)
            model.eval()
            self.localization_model = model
            self.localization_ckpt_path = str(ckpt_path)
            mode = "RGB+SSFR(10ch)" if self.localization_use_ssfr else "RGB(3ch)"
            print(f"[Config] Independent localization enabled: {self.localization_model_name}, {mode}, ckpt={ckpt_path}")
        except Exception as ex:
            print(f"[Config] Failed to load independent localization model: {ex}")
            self.localization_model = None
            self.localization_ckpt_path = None

    def _get_autocast_context(self):
        if self.device != "cuda" or not torch.cuda.is_available():
            return contextlib.nullcontext()
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        try:
            from torch.amp import autocast
            return autocast(device_type='cuda', dtype=amp_dtype)
        except Exception:
            from torch.cuda.amp import autocast
            return autocast(dtype=amp_dtype)

    @staticmethod
    def _get_luma_map(img: Image.Image, size: int = 32):
        gray = np.array(img.convert('L'), dtype=np.float32) / 255.0
        luma_small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(luma_small).unsqueeze(0).unsqueeze(0).float()

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

    def _predict_localization(self, img: Image.Image):
        """独立 SegFormer 定位推理。"""
        if self.localization_model is None:
            return {
                "localization_heatmap": None,
                "localization_score": None,
                "localization_area_ratio": None,
                "localization_source": None,
            }

        rgb = np.array(img.convert('RGB'))
        h, w = rgb.shape[:2]
        resized = cv2.resize(rgb, (self.localization_img_size, self.localization_img_size), interpolation=cv2.INTER_LINEAR)

        x = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        x = (x - mean) / std

        if self.localization_use_ssfr:
            if self.lare_extractor is None:
                return {
                    "localization_heatmap": None,
                    "localization_score": None,
                    "localization_area_ratio": None,
                    "localization_source": "segformer_ssfr_missing_extractor",
                }
            ssfr = self.lare_extractor.extract_single(img).float().cpu()  # [1,7,32,32]
            ssfr = F.interpolate(ssfr, size=(self.localization_img_size, self.localization_img_size), mode='bilinear', align_corners=False)
            x = torch.cat([x, ssfr], dim=1)

        x = x.to(self.device)

        with torch.no_grad():
            with self._get_autocast_context():
                outputs = self.localization_model(pixel_values=x)
                logits = outputs.logits.float()

            logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            prob_map = torch.softmax(logits, dim=1)[:, 1].squeeze(0).cpu().numpy().astype(np.float32)

        heatmap = self._generate_heatmap_base64(prob_map, img)
        area_ratio = float((prob_map >= self.localization_threshold).mean())
        score = float(prob_map.max())

        return {
            "localization_heatmap": heatmap,
            "localization_score": score,
            "localization_area_ratio": area_ratio,
            "localization_source": "segformer_b2",
        }

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
                base = self._predict_cascade(img)
            else:
                base = self._predict_standard(img)

            # 兼容字段：heatmap 默认展示独立定位热图；保留老定位热图在 lare_heatmap
            lare_heatmap = base.pop("heatmap", None)
            localization = self._predict_localization(img)

            base["lare_heatmap"] = lare_heatmap
            base["localization_heatmap"] = localization["localization_heatmap"]
            base["localization_score"] = localization["localization_score"]
            base["localization_area_ratio"] = localization["localization_area_ratio"]
            base["heatmap"] = localization["localization_heatmap"] or lare_heatmap

            debug = base.get("debug", {})
            debug.update({
                "localization_enabled": self.localization_model is not None,
                "localization_source": localization["localization_source"],
                "localization_ckpt": self.localization_ckpt_path,
            })
            base["debug"] = debug

            return base
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _predict_cascade(self, img: Image.Image):
        result = self.cascade_inference.inference(img, return_details=True)
        heatmap_base64 = None
        
        map_source = None
        map_source_type = None
        if result['loc_map'] is not None:
            map_source = result['loc_map']
            map_source_type = "loc_map"
        elif 'raw_lare_map' in result and result['raw_lare_map'] is not None:
            map_source = result['raw_lare_map']
            map_source_type = "raw_lare_map"
            
        if map_source is not None and (result['pred'] == 1 or result['prob'] > 0.3):
            heatmap_base64 = self._generate_heatmap_base64(map_source, img)
        
        ai_prob = float(result['prob'])
        class_idx = 1 if ai_prob >= self.ai_confidence_threshold else 0

        return {
            "class_name": self.class_names[class_idx],
            "confidence": ai_prob,
            "class_idx": class_idx,
            "probabilities": {
                self.class_names[0]: 1.0 - ai_prob,
                self.class_names[1]: ai_prob,
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
        with torch.no_grad():
            loss_map = self.lare_extractor.extract_single(img).to(self.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            clip_input = self.preprocess_clip(img).unsqueeze(0).to(self.device)

            with self._get_autocast_context():
                if self.model_version in ("V14", "V17"):
                    highres_input = self.preprocess_highres(img).unsqueeze(0).to(self.device)
                    luma_map = self._get_luma_map(img).to(self.device)

                    logits, seg_logits = self.model(
                        clip_input, highres_input, loss_map,
                        luma_map=luma_map, return_seg=True
                    )
                    loc_map = torch.sigmoid(seg_logits.float())
                elif self.model_version in ["V11", "V13", "V11Fusion"]:
                    highres_input = self.preprocess_highres(img).unsqueeze(0).to(self.device)
                    luma_map = self._get_luma_map(img).to(self.device)

                    try:
                        logits = self.model(clip_input, highres_input, loss_map, luma_map=luma_map)
                    except TypeError:
                        logits = self.model(clip_input, highres_input, loss_map)
                    loc_map = loss_map
                    if loc_map.shape[1] >= 4:
                        loc_map = loc_map.mean(dim=1, keepdim=True)
                elif hasattr(self.model, 'loc_head'):
                    logits, loc_map = self.model(clip_input, loss_map, return_loc=True)
                else:
                    logits = self.model(clip_input, loss_map)
                    loc_map = None

        if isinstance(logits, tuple):
            logits = logits[0]

        probs = F.softmax(logits.float(), dim=1)[0]
        ai_prob = float(probs[1].item())
        idx = 1 if ai_prob >= self.ai_confidence_threshold else 0

        heatmap_base64 = None
        use_heatmap = self.model_version in ("V14", "V17", "V13", "V11")

        if use_heatmap and loc_map is not None:
            loc_np = loc_map.squeeze().cpu().float().numpy()
            heatmap_base64 = self._generate_heatmap_base64(loc_np, img)

        return {
            "class_name": self.class_names[idx],
            "confidence": ai_prob,
            "class_idx": idx,
            "probabilities": {self.class_names[0]: 1.0 - ai_prob, self.class_names[1]: ai_prob},
            "heatmap": heatmap_base64,
            "debug": {"model_version": self.model_version, "cascade_enabled": False}
        }