import os
import random
import json
import logging
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from torchvision import transforms

logger = logging.getLogger(__name__)

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
                 enable_trufor=False, # [V13] TruFor Map Support
                 drop_no_map=True
                 ):
        self.data_root = Path(data_root).resolve()
        self.data_size = data_size
        self.enable_v11 = enable_v11
        self.enable_trufor = enable_trufor
        self.highres_size = highres_size
        self.train_list = []
        self.test_list = []  # 初始化 test_list
        self.anchor_list = []
        self.isAnchor = False
        self.isVal = False # Internal flag for switching transforms
        self.split_anchor = split_anchor
        self.is_train = is_train  # 保存 is_train 参数
        self.drop_no_map = drop_no_map
        
        # [V13] TruFor Root
        # data/ -> trufor_maps/data/
        self.trufor_root = self.data_root.parent / "trufor_maps" / "data"

        # Initialize Transforms
        self.transform_pipeline = transform
        
        # Define synchronized transforms (Geometric) - Apply to both Image and LossMap
        # Note: Resize is handled separately because LossMap is already small (32x32)
        # We only flip/rotate the LossMap to match the Image orientation.
        # [V11] Added 'highres' target
        additional_targets = {'loss_map': 'image'}
        if self.enable_v11:
            additional_targets['highres'] = 'image'
        if self.enable_trufor:
            additional_targets['trufor'] = 'image'
            
        self.geometric_transform = A.Compose([
            A.RandomRotate90(p=0.33),
            A.HorizontalFlip(p=0.33),
        ], additional_targets=additional_targets, is_check_shapes=False, p=1.0) if is_train else None

        # Define Image-only transforms (Resize, Photometric)
        if is_train:
            self.image_transform = A.Compose([
                A.Resize(height=self.data_size, width=self.data_size, p=1.0),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.ToGray(p=1.0),
                ], p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ], p=1.0)
            
            # [V11] HighRes Transform (Texture Branch) - Less augmentation to preserve noise?
            # Usually we keep geometric sync, but photometric can be applied
            self.highres_transform = A.Compose([
                A.Resize(height=self.highres_size, width=self.highres_size, p=1.0),
                # texture branch might benefit from less color jitter to preserve pattern?
                # For now keeping it simple: just Resize.
            ], p=1.0)
            
        else:
            self.image_transform = A.Compose([
                A.Resize(height=self.data_size, width=self.data_size, p=1.0),
            ], p=1.0)
            
            self.highres_transform = A.Compose([
                A.Resize(height=self.highres_size, width=self.highres_size, p=1.0),
            ], p=1.0)
            
        self.clip_norm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        
        # [V11] Normalization for Texture Branch (ImageNet stats usually)
        # EfficientViT/ConvNeXt expect ImageNet mean/std
        self.highres_norm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225),
            ),
        ])
        
        # [V12] Mask Pre-loading (V12.1 Refined)
        # Scan for masks directory to enable pixel-level supervision
        self.mask_root = self.data_root / 'doubao' / 'masks'
        # V12 assumes manual mask generation, so check existence
        self.use_masks = self.mask_root.exists() and any(self.mask_root.iterdir())
        if self.use_masks:
            print(f"[Dataset V12] Mask Supervision ACTIVE. Files found in {self.mask_root}")
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
            if self.split_anchor and random.random() < 0.1 and label == 0 and len(self.anchor_list) < 100:
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
        
        # Load Map
        # loss_map shape is (C, H, W) -> (4, 32, 32)
        if map_path and Path(map_path).exists():
            try:
                loss_map = torch.load(map_path, map_location='cpu') # Load to CPU first
            except Exception:
                loss_map = torch.zeros((4, 32, 32), dtype=torch.float32)
        else:
            loss_map = torch.zeros((4, 32, 32), dtype=torch.float32)

        # Load Image
        full_path = self._normalize_img_path(image_path)
        try:
            # Handle Chinese paths via numpy
            image_data = np.fromfile(full_path, dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            image = np.zeros((self.data_size, self.data_size, 3), dtype=np.uint8)

        # [V13] Load TruFor Map (Pre-Transform)
        # Default: zeros (C=1, H, W) -> will be resized to match image
        trufor_np = np.zeros((self.data_size, self.data_size), dtype=np.float32)
        if self.enable_trufor:
             # [Fixed] Map path mirrors image path
             rel_path = Path(image_path)
             if rel_path.is_absolute():
                 try:
                    rel_path = rel_path.resolve().relative_to(self.data_root)
                 except ValueError:
                    # Fallback: find 'data' in path
                    parts = rel_path.parts
                    if 'data' in parts:
                        idx = max(i for i, p in enumerate(parts) if p == 'data')
                        rel_path = Path(*parts[idx+1:])
             tf_path = self.trufor_root / rel_path.with_suffix('.png')
             
             # Handle Chinese paths explicitly
             tf_full_path = str(tf_path.absolute())
             
             try:
                 # Check existence
                 if tf_path.exists():
                     # Use numpy fromfile to handle complex Windows paths (including Chinese)
                     # cv2.imread fails on non-ASCII paths on Windows
                     tf_data = np.fromfile(tf_full_path, dtype=np.uint8)
                     tf_img = cv2.imdecode(tf_data, cv2.IMREAD_GRAYSCALE)
                     
                     if tf_img is not None:
                         # Resize to data_size
                         trufor_np = cv2.resize(tf_img, (self.data_size, self.data_size), interpolation=cv2.INTER_LINEAR)
                         trufor_np = trufor_np.astype(np.float32) / 255.0
             except Exception:
                 pass

        # Apply Transforms
        if self.transform_pipeline:
            # Legacy/External transform support
            image = self.transform_pipeline(image=image)['image']
        else:
            # New synchronized pipeline
            # [V11] Prepare multiple versions if needed
            highres = None
            if self.enable_v11:
                # Create the highres version (Resize only)
                highres = self.highres_transform(image=image)['image']
            
            # 1. Resize Image (LossMap is already small)
            if self.image_transform:
                image = self.image_transform(image=image)['image']
            
            # 2. Geometric Transforms (Flip/Rotate) - Synced
            if self.geometric_transform and not self.isVal and not self.isAnchor:
                # [CRITICAL Fix 1] Convert loss map to float32 if needed
                if loss_map.dtype == torch.bfloat16:
                    loss_map = loss_map.float()
                    
                # [CRITICAL Fix 2] Resize loss_map to match image size
                orig_h, orig_w = loss_map.shape[1], loss_map.shape[2]
                target_h, target_w = image.shape[0], image.shape[1]  # Numpy: (H, W, C)
                
                loss_map_resized = torch.nn.functional.interpolate(
                    loss_map.unsqueeze(0), 
                    size=(target_h, target_w), 
                    mode='nearest'
                ).squeeze(0)
                
                # Convert to Numpy (H,W,C)
                loss_map_np = loss_map_resized.permute(1, 2, 0).numpy()
                
                # Prepare args for transform
                transform_args = {'image': image, 'loss_map': loss_map_np}
                if self.enable_v11:
                    transform_args['highres'] = highres
                if self.enable_trufor:
                    transform_args['trufor'] = trufor_np
                
                # Apply transform with fallback for size mismatch
                try:
                    res = self.geometric_transform(**transform_args)
                    image = res['image']
                    loss_map_np = res['loss_map']
                    if self.enable_v11: highres = res['highres']
                    if self.enable_trufor: trufor_np = res['trufor']
                except ValueError:
                    # Fallback: Transform main items only
                    res = self.geometric_transform(image=image, loss_map=loss_map_np)
                    image = res['image']
                    loss_map_np = res['loss_map']
                    # highres/trufor get NO geometric augs in fallback
                
                # Convert back to Tensor and resize back to 32x32
                loss_map_resized = torch.from_numpy(loss_map_np).permute(2, 0, 1)
                loss_map = torch.nn.functional.interpolate(
                    loss_map_resized.unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode='nearest'
                ).squeeze(0)
            
            elif self.enable_v11 and highres is None:
                 # If not training (val/anchor), we still need highres
                 highres = self.highres_transform(image=image)['image']
                
        # [V12] Mask Loading Strategy
        mask = np.zeros((self.highres_size, self.highres_size), dtype=np.float32)
        if self.use_masks and label == 1:
            stem = Path(image_path).stem
            # Supports both .png (generated) and .jpg (original name style)
            mask_path_png = self.mask_root / f"{stem}.png"
            if mask_path_png.exists():
                try:
                    # Read as grayscale
                    # Use np.fromfile for Windows Chinese path support
                    mask_data = np.fromfile(str(mask_path_png), dtype=np.uint8)
                    mask_img = cv2.imdecode(mask_data, cv2.IMREAD_GRAYSCALE)
                    
                    if mask_img is not None:
                        # Resize to match training resolution (nearest neighbor to keep sharp edges)
                        mask_resized = cv2.resize(mask_img, (self.highres_size, self.highres_size), interpolation=cv2.INTER_NEAREST)
                        # Normalize to 0.0-1.0
                        mask = mask_resized.astype(np.float32) / 255.0
                except Exception:
                    pass 

        # Final safety check: ensure loss_map is float32 and contiguous
        if loss_map.dtype in [torch.bfloat16, torch.float16]:
            loss_map = loss_map.float()
        loss_map = loss_map.contiguous()

        # Final Normalization for CLIP
        image = self.clip_norm(image)
        
        # [V11/V12/V13] Returns
        returns = [image, label, loss_map]
        
        if self.enable_v11:
            highres = self.highres_norm(highres)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
            returns.extend([highres, mask_tensor])
            
        if self.enable_trufor:
            # TruFor: (H, W) -> (1, H, W)
            trufor_tensor = torch.from_numpy(trufor_np).unsqueeze(0).float()
            returns.append(trufor_tensor)

        # [Debug/Vis] Append Path
        returns.append(str(full_path))

        return tuple(returns)

    def set_val_mode(self, is_val):
        self.isVal = is_val
        self.isAnchor = False

    def set_anchor_mode(self, is_anchor):
        self.isAnchor = is_anchor
        self.isVal = False
