import os
import random
import json
import logging
import cv2
import numpy as np
import torch
from functools import lru_cache
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from torchvision import transforms

logger = logging.getLogger(__name__)

MASK_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def _dedupe_paths(paths):
    unique = []
    seen = set()
    for path in paths:
        path_str = str(path)
        if path_str in seen:
            continue
        seen.add(path_str)
        unique.append(path)
    return unique

class SDXLCollector:
    def __init__(self, fake_root, real_root):
        self.fake_root = Path(fake_root)
        self.real_root = Path(real_root)
    
    def collect_images(self, directory):
        images = []
        if not directory.exists():
            return images
        for img_path in sorted(directory.glob('**/*')):
            if img_path.is_file() and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                images.append(img_path)
        return images
    
    def get_fake_images(self):
        # Scan all subdirectories in fake root if specific categories aren't guaranteed
        if not self.fake_root.exists():
            return {}
        categories = [d.name for d in self.fake_root.iterdir() if d.is_dir()]
        return {cat: self.collect_images(self.fake_root / cat) for cat in categories}
    
    def get_real_images(self):
        if not self.real_root.exists():
            return {}
        categories = [d.name for d in self.real_root.iterdir() if d.is_dir()]
        return {cat: self.collect_images(self.real_root / cat) for cat in categories}

class ImageDataset(Dataset):
    def __init__(self, data_root, train_file,
                 data_size=512, val_ratio=None, split_anchor=True,
                 map_file=None,
                 transform=None,
                 is_train=True,
                 enable_v11=False, # New flag for V11 features
                 highres_size=512, # Resolution for texture branch
                 enable_luma=True, # [V15] Luma map for FLH (replaces TruFor)
                 drop_no_map=True
                 ):
        self.data_root = Path(data_root).resolve()
        self.data_size = data_size
        self.enable_v11 = enable_v11
        self.enable_luma = enable_luma
        self.highres_size = highres_size
        self.train_list = []
        self.test_list = []  # 初始化 test_list
        self.anchor_list = []
        self.isAnchor = False
        self.isVal = False # Internal flag for switching transforms
        self.split_anchor = split_anchor
        self.is_train = is_train  # 保存 is_train 参数
        self.drop_no_map = drop_no_map
        
        # [V15] TruFor removed, luma generated on-the-fly
        # (no pre-computed map directory needed)

        # Initialize Transforms
        self.transform_pipeline = transform
        
        # Define synchronized transforms (Geometric) - Apply to both Image and LossMap
        # Resize is handled separately per branch. Here we only apply shared geometric ops
        # so image, highres, mask, luma, and loss_map always use identical spatial params.

        # Heavy transforms (warp-based) — only for image/highres/mask (same resolution)
        # loss_map (7ch, 32×32) and luma (1ch, 32×32) are excluded because OpenCV
        # warpAffine/remap cannot handle 7-channel data and resolution mismatch.
        heavy_targets = {}
        if self.enable_v11:
            heavy_targets['highres'] = 'image'

        # [Phase3] Heavy warp transforms DISABLED — loss_map (7ch, 32×32) cannot be
        # spatially transformed in sync, causing train-time misalignment that degrades
        # localization. Only lightweight axis transforms (Rotate90/Flip) are safe.
        self.heavy_geo_transform = None

        # Light transforms (axis-only, safe for any channel count / resolution)
        additional_targets = {'loss_map': 'image'}
        if self.enable_v11:
            additional_targets['highres'] = 'image'
        if self.enable_luma:
            additional_targets['luma'] = 'mask'

        self.geometric_transform = A.Compose([
            A.RandomRotate90(p=0.33),
            A.HorizontalFlip(p=0.33),
        ], additional_targets=additional_targets, is_check_shapes=False, p=1.0) if is_train else None

        # Per-branch resize first, then shared geometry, then photometric on image only.
        self.image_resize_transform = A.Compose([
            A.Resize(height=self.data_size, width=self.data_size, p=1.0),
        ], p=1.0)

        self.highres_resize_transform = A.Compose([
            A.Resize(height=self.highres_size, width=self.highres_size, p=1.0),
        ], p=1.0) if self.enable_v11 else None

        if is_train:
            self.image_photo_transform = A.Compose([
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.ToGray(p=1.0),
                ], p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ], p=1.0)
        else:
            self.image_photo_transform = None
            
        # [V17 Perf] 直接 numpy→tensor 归一化，跳过 ToPILImage 中间转换
        _clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        _clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        self._clip_mean = _clip_mean
        self._clip_std  = _clip_std
        
        _inet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        _inet_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self._inet_mean = _inet_mean
        self._inet_std  = _inet_std
        
        # [V12+] Support multiple localization supervision roots.
        self.mask_roots = _dedupe_paths([
            self.data_root / 'doubao' / 'masks',
            self.data_root / 'change' / 'masks',
        ])
        self.use_masks = any(root.exists() and any(root.iterdir()) for root in self.mask_roots)
        if self.use_masks:
            active_roots = [str(root) for root in self.mask_roots if root.exists() and any(root.iterdir())]
            print(f"[Dataset V12] Mask Supervision ACTIVE. Files found in {active_roots}")
        else:
            print(f"[Dataset V12] Mask Supervision INACTIVE. (Masks missing)")

        # Load data
        self._load_data(train_file)
        
        # Load map file
        self.stem_to_pt = {}
        self.imgpath_to_pt = {}
        if map_file:
            self._load_map_file(map_file)

        # Split val
        # 仅在训练集模式下做 train->(train/val) 切分
        # 验证集模式（is_train=False）的数据已直接写入 self.test_list，不能被覆盖
        if self.is_train:
            if val_ratio is not None and val_ratio > 0:
                # Deterministic shuffle for validation split
                rng = np.random.RandomState(42)
                rng.shuffle(self.train_list)
                split_idx = int(len(self.train_list) * val_ratio)
                self.test_list = self.train_list[:split_idx]
                self.train_list = self.train_list[split_idx:]
            else:
                self.test_list = self.train_list

        # Filter maps
        self.train_list, self.train_map_paths = self._filter_with_map(self.train_list)
        self.test_list, self.test_map_paths = self._filter_with_map(self.test_list)
        self.anchor_list, self.anchor_map_paths = self._filter_with_map(self.anchor_list)

    def _get_default_transforms(self, is_train):
        # Kept for backward compatibility if needed, but not used in new pipeline
        if is_train:
            return A.Compose([
                A.Resize(height=self.data_size, width=self.data_size, p=1.0),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.ToGray(p=1.0),
                ], p=0.5),
                A.RandomRotate90(p=0.33),
                A.HorizontalFlip(p=0.33),
            ], p=1.0)
        else:
            return A.Compose([
                A.Resize(height=self.data_size, width=self.data_size, p=1.0),
            ], p=1.0)

    def _load_data(self, train_file):
        train_file_path = Path(train_file)
        if not train_file_path.exists():
            # Try relative to project root if not found
            if not train_file_path.is_absolute():
                pass 
        
        # Try UTF-8 first, fallback to GBK for Chinese filenames on Windows
        content = None
        for enc in ['utf-8', 'gbk', 'latin-1']:
            try:
                with open(train_file, encoding=enc) as f:
                    content = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            raise ValueError(f"Could not decode {train_file} with any encoding")
                
        for line in content:
            line = line.strip()
            if not line: continue
            
            if '\t' in line:
                image_path, image_label = line.rsplit('\t', 1)
            else:
                image_path, image_label = line.rsplit(' ', 1)
            
            label = int(image_label)
            if self.is_train and self.split_anchor and random.random() < 0.1 and label == 0 and len(self.anchor_list) < 100:
                self.anchor_list.append((image_path, label))
            else:
                # 如果是验证模式，将数据添加到 test_list
                if not self.is_train:
                    self.test_list.append((image_path, label))
                else:
                    self.train_list.append((image_path, label))

    def _load_map_file(self, map_file):
        map_path = Path(map_file)
        if map_path.exists():
            # Try UTF-8 first, fallback to GBK for Chinese filenames on Windows
            content = None
            for enc in ['utf-8', 'gbk', 'latin-1']:
                try:
                    with open(map_path, encoding=enc) as f:
                        content = f.readlines()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"[Warning] Could not decode {map_file} with any encoding")
                return
                
            for raw in content:
                    raw = raw.strip()
                    if not raw or raw.startswith('#'): continue
                    parts = raw.split('\t')
                    if len(parts) == 2:
                        pt_path, _ = parts
                        stem = Path(pt_path).stem
                        self.stem_to_pt[stem] = pt_path
                    elif len(parts) >= 3:
                        img_path, pt_path, _ = parts[:3]
                        img_path_norm = img_path.replace('\\', '/')
                        self.imgpath_to_pt[img_path_norm] = pt_path
                        stem = Path(img_path_norm).stem
                        self.stem_to_pt.setdefault(stem, pt_path)

    def _normalize_img_path(self, p):
        p_path = Path(p)
        if not p_path.exists():
            p_path = self.data_root / p
        return str(p_path.absolute()).replace('\\', '/')

    def _filter_with_map(self, data_list):
        filtered_list = []
        map_paths = []
        has_any_features = bool(self.stem_to_pt or self.imgpath_to_pt)
        
        for image_path, label in data_list:
            img_norm = self._normalize_img_path(image_path)
            loss_path = self.imgpath_to_pt.get(img_norm)
            if loss_path is None:
                filename = Path(image_path).stem
                loss_path = self.stem_to_pt.get(filename)
            
            if has_any_features and loss_path is None and self.drop_no_map:
                continue
            
            filtered_list.append((image_path, label))
            map_paths.append(loss_path if loss_path else "")
            
        return filtered_list, map_paths

    def _load_mask(self, image_path, label):
        mask = np.zeros((self.highres_size, self.highres_size), dtype=np.float32)
        if not (self.use_masks and label == 1):
            return mask

        stem = Path(image_path).stem
        image_path_norm = str(image_path).lower().replace('\\', '/')

        candidate_roots = []
        if '/change/images/' in image_path_norm or '/change/fack/' in image_path_norm:
            candidate_roots.append(self.data_root / 'change' / 'masks')
        if '/doubao/' in image_path_norm:
            candidate_roots.append(self.data_root / 'doubao' / 'masks')

        candidate_roots.extend(self.mask_roots)
        candidate_roots = _dedupe_paths(candidate_roots)

        candidate_paths = []
        for root in candidate_roots:
            for ext in MASK_EXTENSIONS:
                candidate_paths.append(root / f"{stem}{ext}")
        candidate_paths = _dedupe_paths(candidate_paths)

        for mask_path_png in candidate_paths:
            if not mask_path_png.exists():
                continue
            try:
                mask_data = np.fromfile(str(mask_path_png), dtype=np.uint8)
                mask_img = cv2.imdecode(mask_data, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    mask_resized = cv2.resize(
                        mask_img,
                        (self.highres_size, self.highres_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    mask = mask_resized.astype(np.float32) / 255.0
                    break
            except Exception:
                continue
        return mask

    def _generate_luma(self, image, target_hw):
        h, w = target_hw
        luma = np.zeros((h, w), dtype=np.float32)
        if not self.enable_luma:
            return luma

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
            luma = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        except Exception:
            pass
        return luma

    def __len__(self):
        if self.isAnchor: return len(self.anchor_list)
        if self.isVal: return len(self.test_list)
        return len(self.train_list)

    def __getitem__(self, index):
        if self.isAnchor:
            data_list, map_list = self.anchor_list, self.anchor_map_paths
        elif self.isVal:
            data_list, map_list = self.test_list, self.test_map_paths
        else:
            data_list, map_list = self.train_list, self.train_map_paths
            
        image_path, label = data_list[index]
        map_path = map_list[index]
        
        # Load Map  [V17 Perf: weights_only=True 加速 torch.load]
        # loss_map shape is (C, H, W) -> (7, 32, 32) for SSFR v16, or (6, 32, 32) for legacy
        if map_path and Path(map_path).exists():
            try:
                loss_map = torch.load(map_path, map_location='cpu', weights_only=True)
            except Exception:
                loss_map = torch.zeros((7, 32, 32), dtype=torch.float32)
        else:
            loss_map = torch.zeros((7, 32, 32), dtype=torch.float32)

        # [V16] Backward compat: pad Ch6 with zeros if old 6-channel features
        if loss_map.dim() == 3 and loss_map.shape[0] == 6:
            loss_map = torch.cat([loss_map, torch.zeros(1, loss_map.shape[1], loss_map.shape[2])], dim=0)
        if loss_map.dtype in [torch.bfloat16, torch.float16]:
            loss_map = loss_map.float()
        loss_map = loss_map.contiguous()

        # Load Image
        full_path = self._normalize_img_path(image_path)
        try:
            # Handle Chinese paths via numpy
            image_data = np.fromfile(full_path, dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            image = np.zeros((self.data_size, self.data_size, 3), dtype=np.uint8)

        mask = self._load_mask(image_path, label)
        raw_image = image

        # Prepare branch-specific resolutions first, then apply one shared geometric transform.
        if self.transform_pipeline:
            image = self.transform_pipeline(image=raw_image)['image']
        else:
            image = raw_image

        image = self.image_resize_transform(image=image)['image']
        highres = None
        if self.enable_v11 and self.highres_resize_transform is not None:
            highres = self.highres_resize_transform(image=raw_image)['image']

        loss_map_np = np.ascontiguousarray(loss_map.permute(1, 2, 0).cpu().numpy().astype(np.float32))
        luma_np = self._generate_luma(image, loss_map.shape[-2:]) if self.enable_luma else None

        # Apply Transforms
        if not self.isVal and not self.isAnchor:
            # Step 1: Heavy warp transforms (ShiftScaleRotate, Elastic)
            # Applied to highres (512) + mask (512) which share the same resolution.
            # Then image (448) is derived from augmented highres via resize.
            if self.heavy_geo_transform is not None and self.enable_v11 and highres is not None:
                heavy_res = self.heavy_geo_transform(image=highres, mask=mask)
                highres = heavy_res['image']
                mask = heavy_res['mask']
                # Regenerate image from augmented highres
                image = self.image_resize_transform(image=highres)['image']

            # Step 2: Light axis transforms (Rotate90, Flip) — all targets including loss_map/luma
            if self.geometric_transform is not None:
                transform_args = {
                    'image': image,
                    'mask': mask,
                    'loss_map': loss_map_np,
                }
                if self.enable_v11 and highres is not None:
                    transform_args['highres'] = highres
                if self.enable_luma and luma_np is not None:
                    transform_args['luma'] = luma_np

                res = self.geometric_transform(**transform_args)
                image = res['image']
                mask = res['mask']
                loss_map_np = res['loss_map']
                if self.enable_v11 and highres is not None:
                    highres = res['highres']
                if self.enable_luma and luma_np is not None:
                    luma_np = res['luma']

        # Photometric transforms are image-only; keep them after shared geometry.
        if self.image_photo_transform is not None and not self.isVal and not self.isAnchor:
            image = self.image_photo_transform(image=image)['image']

        loss_map = torch.from_numpy(np.ascontiguousarray(loss_map_np)).permute(2, 0, 1).float().contiguous()

        # Final Normalization for CLIP  [V17 Perf: 直接 numpy→tensor, 无 PIL 中间层]
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)
        image = (image - self._clip_mean) / self._clip_std
        
        # [V11/V12/V13] Returns
        returns = [image, label, loss_map]
        
        if self.enable_v11:
            highres = torch.from_numpy(highres.transpose(2, 0, 1).copy()).float().div_(255.0)
            highres = (highres - self._inet_mean) / self._inet_std
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
            returns.extend([highres, mask_tensor])
            
        if self.enable_luma:
            # Luma: (H, W) -> (1, H, W)
            luma_tensor = torch.from_numpy(luma_np).unsqueeze(0).float()
            returns.append(luma_tensor)

        # [Debug/Vis] Append Path
        returns.append(str(full_path))

        return tuple(returns)

    def set_val_mode(self, is_val):
        self.isVal = is_val
        self.isAnchor = False

    def set_anchor_mode(self, is_anchor):
        self.isAnchor = is_anchor
        self.isVal = False
