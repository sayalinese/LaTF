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


def _find_mask(stem, mask_dirs):
    """在 mask_dirs 列表中按 stem + 各种扩展名搜索 mask 文件。"""
    for d in mask_dirs:
        for ext in MASK_EXTENSIONS:
            p = d / f"{stem}{ext}"
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
            data_root = Path(self.samples[0][0]).parent
            while data_root.name != 'data' and data_root != data_root.parent:
                data_root = data_root.parent
            self.mask_dirs = [
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

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        # 读取图片
        image = _read_image(img_path)
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        # 读取 mask
        if label == 1:
            stem = Path(img_path).stem
            raw_mask = _find_mask(stem, self.mask_dirs)
            if raw_mask is not None:
                mask = raw_mask
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # 确保 image 和 mask 尺寸一致 (来自不同源时分辨率可能不同)
        h, w = image.shape[:2]
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 数据增强 (图片和 mask 同步变换)
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        # 归一化图片 → float32 [C, H, W]
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(image.transpose(2, 0, 1))  # [3, H, W]

        # Mask → long [H, W], 二值 (0=背景, 1=篡改)
        mask = (mask > 127).astype(np.int64)
        mask = torch.from_numpy(mask)  # [H, W]

        # SSFR 拼接 (可选)
        if self.use_ssfr:
            stem = Path(img_path).stem
            ssfr = self._load_ssfr(stem)  # [7, H, W]
            image = torch.cat([image, ssfr], dim=0)  # [10, H, W]

        return image, mask

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
