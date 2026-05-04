"""
SegFormer 篡改定位专用 Dataset。
只处理 RGB 图片 + 二值 Mask，与分类模型完全解耦。

支持:
  - 3ch (RGB only) 或 10ch (RGB + SSFR 7ch) 输入模式
  - 激进数据增强 (无 SSFR 32×32 对齐限制)
  - Windows 中文路径兼容 (np.fromfile + cv2.imdecode)
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A

MASK_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ImageNet normalization (SegFormer 默认)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _read_image(path):
    """用 np.fromfile + cv2.imdecode 读取图片，兼容中文路径。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _mask_stem_candidates(stem):
    candidates = [stem]
    if stem.endswith('_mask'):
        base = stem[:-5]
        if base:
            candidates.append(base)
    else:
        candidates.append(f'{stem}_mask')
    return candidates


def _ordered_mask_dirs(image_path, mask_dirs):
    """按图片来源优先排列 mask 目录，减少同 stem 误匹配。"""
    norm = str(image_path).lower().replace('\\', '/')
    preferred = []
    for d in mask_dirs:
        d_str = str(d).lower().replace('\\', '/')
        if '/brgen_stuff_val/tp/' in norm and '/brgen_stuff_val/gt' in d_str:
            preferred.append(d)
        elif '/br-gen/forged/' in norm and '/stuff/coco/' in norm and '/mask/stuff/coco' in d_str:
            preferred.append(d)
        elif '/br-gen/forged/' in norm and '/stuff/imagenet/' in norm and '/mask/stuff/imagenet' in d_str:
            preferred.append(d)
        elif '/br-gen/forged/' in norm and '/stuff/places/' in norm and '/mask/stuff/places' in d_str:
            preferred.append(d)

    ordered = preferred + [d for d in mask_dirs if d not in preferred]
    return ordered


def _find_mask_for_image(image_path, mask_dirs):
    """在 mask_dirs 列表中按 stem + 各种扩展名搜索 mask 文件。"""
    stem = Path(image_path).stem
    stem_candidates = _mask_stem_candidates(stem)
    for d in _ordered_mask_dirs(image_path, mask_dirs):
        for stem_name in stem_candidates:
            for ext in MASK_EXTENSIONS:
                p = d / f"{stem_name}{ext}"
                if p.exists():
                    data = np.fromfile(str(p), dtype=np.uint8)
                    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        return mask
    return None


class SegFormerForgeryDataset(Dataset):
    """
    SegFormer 篡改定位 Dataset。

    标注格式: 每行 "<path>\t<label>" (label: 0=real, 1=fake)

    Args:
        ann_file: 标注文件路径
        img_size: 输出图片尺寸 (default 512)
        mask_dirs: mask 搜索目录列表
        is_train: 是否训练模式 (控制数据增强)
        use_ssfr: 是否加载 SSFR 特征 (10ch 模式)
        ssfr_map_file: SSFR ann.txt 路径 (stem → .pt 映射)
    """

    def __init__(self, ann_file, img_size=512, mask_dirs=None, is_train=True,
                 use_ssfr=False, ssfr_map_file=None):
        super().__init__()
        self.img_size = img_size
        self.is_train = is_train
        self.use_ssfr = use_ssfr

        # 解析标注
        self.samples = []
        with open(ann_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sep = '\t' if '\t' in line else ' '
                parts = line.rsplit(sep, 1)
                if len(parts) == 2:
                    self.samples.append((parts[0], int(parts[1])))

        # Mask 搜索目录
        if mask_dirs is None:
            data_root = PROJECT_ROOT / 'data'
            self.mask_dirs = [
                data_root / '新数据' / 'BR-Gen' / 'Mask' / 'Stuff' / 'COCO',
                data_root / '新数据' / 'BR-Gen' / 'Mask' / 'Stuff' / 'ImageNet',
                data_root / '新数据' / 'BR-Gen' / 'Mask' / 'Stuff' / 'Places',
                data_root / '新数据' / 'BRGen_Stuff_Val' / 'Gt',
                # legacy fallback
                data_root / 'change' / 'masks',
                data_root / 'doubao' / 'masks',
            ]
        else:
            self.mask_dirs = [Path(d) for d in mask_dirs]
        self.mask_dirs = [d for d in self.mask_dirs if d.exists()]

        # SSFR 映射表 (可选)
        self.ssfr_map = {}
        if use_ssfr and ssfr_map_file:
            self._load_ssfr_map(ssfr_map_file)

        # 数据增强
        if is_train:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(img_size, img_size),
                                    scale=(0.5, 1.0), ratio=(0.8, 1.2), p=0.5),
                A.Resize(height=img_size, width=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.OneOf([
                    A.ElasticTransform(alpha=60, sigma=60 * 0.05, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                    A.OpticalDistortion(distort_limit=0.3, p=1.0),
                ], p=0.3),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.GaussNoise(p=1.0),
                ], p=0.3),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
            ])
            
        self._init_mask_map(mask_dirs, is_train)

    def _load_ssfr_map(self, map_file):
        """从 ann.txt 加载 stem → .pt 文件路径的映射。"""
        map_root = Path(map_file).parent
        with open(map_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pt_path = map_root / line
                stem = Path(line).stem
                self.ssfr_map[stem] = str(pt_path)

    def __len__(self):
        return len(self.samples)

    def _init_mask_map(self, mask_dirs, is_train):
        # 预先扫描 mask 目录，构建从 stem 到绝对路径的内存字典
        self.mask_map = {}
        if is_train or True:  # always scan
            print("Pre-scanning mask directories into memory to avoid IO bounds...")
            for d in mask_dirs:
                d_path = Path(d)
                if not d_path.exists(): continue
                for p in d_path.iterdir():
                    if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS:
                        self.mask_map[p.stem.lower()] = str(p)

    def _find_mask_fast(self, stem):
        stem_lower = stem.lower()
        if stem_lower in self.mask_map:
            return self.mask_map[stem_lower]
        if stem_lower.endswith('_mask'):
            base = stem_lower[:-5]
            if base in self.mask_map:
                return self.mask_map[base]
        else:
            with_mask = stem_lower + '_mask'
            if with_mask in self.mask_map:
                return self.mask_map[with_mask]
        return None

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = _read_image(img_path)
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        if label == 1:
            mask_path = self._find_mask_fast(Path(img_path).stem)
            if mask_path is not None:
                data = np.fromfile(mask_path, dtype=np.uint8)
                mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if self.load_ssfr:
            ssfr_feat = self._load_ssfr(Path(img_path).stem)
        else:
            ssfr_feat = torch.zeros((7, self.img_size, self.img_size), dtype=torch.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # numpy -> tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).float() / 255.0

        return image, mask, label, ssfr_feat, str(img_path)

    def _load_ssfr(self, stem):
        """加载 SSFR 特征并上采样到 img_size。"""
        pt_path = self.ssfr_map.get(stem)
        if pt_path and Path(pt_path).exists():
            try:
                feat = torch.load(pt_path, map_location='cpu', weights_only=True)
                if feat.dim() == 3 and feat.shape[0] == 6:
                    feat = torch.cat([feat, torch.zeros(1, feat.shape[1], feat.shape[2])], dim=0)
                if feat.dtype in (torch.bfloat16, torch.float16):
                    feat = feat.float()
            except Exception:
                feat = torch.zeros(7, 32, 32)
        else:
            feat = torch.zeros(7, 32, 32)

        # 上采样到 img_size
        feat = feat.unsqueeze(0)  # [1, 7, 32, 32]
        feat = torch.nn.functional.interpolate(feat, size=(self.img_size, self.img_size),
                                                mode='bilinear', align_corners=False)
        return feat.squeeze(0)  # [7, H, W]
