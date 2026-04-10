import os
import random
from pathlib import Path
import h5py
import numpy as np
from dotenv import load_dotenv

# Load environment variables for Doubao weight configuration
load_dotenv(Path(__file__).resolve().parents[1] / '.env')

class SDXLDataCollector:
    def __init__(self, fake_root, real_root):
        self.fake_root = Path(fake_root)
        self.real_root = Path(real_root)
        # Special directories that bypass sampling (e.g., precious inpaint data)
        self.no_sample_dirs = {'inpaint'}
        self.mask_dir_names = {'mask', 'masks', 'mask_vis', 'masks_vis'}
        
        # Doubao weight configuration from .env
        self.doubao_weight = float(os.getenv('DOUBAO_WEIGHT', '5.0'))
        self.doubao_sample_ratio = float(os.getenv('DOUBAO_SAMPLE_RATIO', '1.0'))
        self.regular_sample_ratio = float(os.getenv('REGULAR_SAMPLE_RATIO', '0.03'))
        print(f"[Doubao Weight] Weight: {self.doubao_weight}x, Sample Ratio: {self.doubao_sample_ratio}, Regular Ratio: {self.regular_sample_ratio}")
    
    def collect_images(self, directory):
        images = []
        if not directory.exists():
            return images
        # 递归搜索
        for img_path in sorted(directory.glob('**/*')):
            if img_path.is_file() and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                images.append(img_path)
        return images
    
    def get_fake_images(self):
        # Scan directories dynamically instead of hardcoded list
        # Supports 'flux', 'sdxl', 'inpaint' and their subfolders
        fake_images = {}
        if not self.fake_root.exists():
            return fake_images
            
        # Check for top-level folders like 'flux', 'sdxl', 'inpaint'
        for model_dir in self.fake_root.iterdir():
            if model_dir.is_dir() and model_dir.name.lower() not in {'real', 'special'}: # Exclude Real and special
                # Now scan subdirectories (animals, faces, doubao, etc.)
                subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
                if subdirs:
                    for sub in subdirs:
                        # FIX: Explicitly ignore 'masks' and 'masks_vis' directories to prevent ground truth leakage
                        if sub.name.lower() in self.mask_dir_names:
                            continue
                        
                        # Key: sdxl/animals or inpaint/doubao
                        fake_images[f"{model_dir.name}/{sub.name}"] = self.collect_images(sub)
                else:
                    # No subdirectories, just images
                    fake_images[model_dir.name] = self.collect_images(model_dir)
                
        return fake_images
    
    def get_inpaint_images(self):
        """
        Get inpaint images separately (for special handling - no sampling).
        These are precious local-edit samples that should always be fully used.
        """
        inpaint_images = []
        inpaint_dir = self.fake_root / 'inpaint'
        if inpaint_dir.exists():
            inpaint_images = self.collect_images(inpaint_dir)
            print(f"[Inpaint] Found {len(inpaint_images)} precious inpaint samples (will use ALL)")
        return inpaint_images
    
    def get_real_images(self):
        # Scan all subdirectories in real_root
        real_images = {}
        if not self.real_root.exists():
            return real_images
            
        for item in self.real_root.iterdir():
            if item.is_dir():
                real_images[item.name] = self.collect_images(item)
        
        return real_images
    
    def sample(self, images, ratio, seed=42):
        rng = random.Random(seed)
        if ratio >= 1.0:
            return images
        sample_size = max(1, int(len(images) * ratio))
        return rng.sample(images, sample_size)
    
    def create_annotations(self, train_ratio=1.0, val_ratio=1.0, test_ratio=1.0, seed=42, local_only=False):
        fake_data_dict = self.get_fake_images()
        real_data_dict = self.get_real_images()

        # 1. Separate data types
        regular_fake = []
        inpaint_fake = []
        inpaint_real = []

        for cat, imgs in fake_data_dict.items():
            cat_lower = cat.lower()
            # Doubao Real -> Precious Real
            if 'doubao' in cat_lower and 'real' in cat_lower:
                inpaint_real.extend(imgs)
                print(f"[Data] Found {len(imgs)} precious real images from {cat}")
                continue
            # Doubao / change local-edit fake sets bypass heavy downsampling.
            if (
                'doubao' in cat_lower
                or cat_lower.startswith('inpaint')
                or cat_lower.startswith('change/')
                or cat_lower == 'change'
            ):
                inpaint_fake.extend(imgs)
            else:
                regular_fake.extend(imgs)

        regular_real = []
        for cat, imgs in real_data_dict.items():
            regular_real.extend(imgs)

        # Doubao权重增强和采样参数
        doubao_weight = getattr(self, 'doubao_weight', 5.0)
        doubao_sample_ratio = getattr(self, 'doubao_sample_ratio', 1.0)
        regular_sample_ratio = getattr(self, 'regular_sample_ratio', 0.03)

        # Stage 2 mode: Only use local inpaint + balanced Real
        if local_only:
            print(f"[LOCAL ONLY MODE] Using only inpaint data for stage 2 fine-tuning")
            target_real_size = min(len(inpaint_fake) * 2, len(regular_real) + len(inpaint_real))
            all_available_real = inpaint_real + regular_real
            all_real = self.sample(all_available_real, target_real_size / len(all_available_real), seed=seed)
            all_fake = inpaint_fake
            print(f"Inpaint Fake: {len(inpaint_fake)}")
            print(f"Balanced Real: {len(all_real)}")
        else:
            print(f"Regular Fake Pool (flux/sdxl): {len(regular_fake)}")
            print(f"Inpaint Fake Pool (doubao/change etc): {len(inpaint_fake)} [NO SAMPLING - ALL USED]")
            print(f"Inpaint Real Pool (doubao src): {len(inpaint_real)} [NO SAMPLING - ALL USED]")
            print(f"Regular Real Pool: {len(regular_real)}")

            rng = random.Random(seed)

            # 2. 分别采样
            sampled_regular_fake = self.sample(regular_fake, regular_sample_ratio, seed=seed+1)
            sampled_regular_real = self.sample(regular_real, regular_sample_ratio, seed=seed+2)
            # Doubao数据采样（通常为1.0，全部用）
            sampled_inpaint_fake = self.sample(inpaint_fake, doubao_sample_ratio, seed=seed+3)
            sampled_inpaint_real = self.sample(inpaint_real, doubao_sample_ratio, seed=seed+4)

            # 先不做复制加权：避免将重复样本带入 val/test，后续仅对 train 做加权
            all_fake = sampled_regular_fake + sampled_inpaint_fake
            all_real = sampled_regular_real + sampled_inpaint_real

            print(f"Sampled Regular Fake: {len(sampled_regular_fake)}")
            print(f"Sampled Regular Real: {len(sampled_regular_real)}")
            print(f"Doubao Fake (pre-weight): {len(sampled_inpaint_fake)}")
            print(f"Doubao Real (pre-weight): {len(sampled_inpaint_real)}")
            print(f"Total Fake (combined): {len(all_fake)}")
            print(f"Total Real (combined): {len(all_real)}")

        rng = random.Random(seed)
        # Label 1: Fake, Label 0: Real
        dataset = []
        for p in all_fake:
            dataset.append((str(p.absolute()).replace('\\', '/'), '1'))
        for p in all_real:
            dataset.append((str(p.absolute()).replace('\\', '/'), '0'))
            
        # 4. Group-aware splitting (prevent leakage)
        # Group by stem (filename without extension) to ensure paired fake/real images stay in same split
        groups = {}
        for path_str, label in dataset:
            p = Path(path_str)
            stem = p.stem
            # For paired data (doubao), stem should match (e.g. '1', '10_牛油果')
            if stem not in groups:
                groups[stem] = []
            groups[stem].append((path_str, label))
            
        # Shuffle groups
        group_keys = list(groups.keys())
        rng.shuffle(group_keys)
        
        # Split groups
        total_groups = len(group_keys)
        n_train = int(total_groups * 0.8)
        n_val = int(total_groups * 0.1)
        
        train_keys = group_keys[:n_train]
        val_keys = group_keys[n_train:n_train+n_val]
        test_keys = group_keys[n_train+n_val:]
        
        def flatten(keys):
            res = []
            for k in keys:
                res.extend(groups[k])
            return res
            
        train_data = flatten(train_keys)
        val_data = flatten(val_keys)
        test_data = flatten(test_keys)

        # 仅对训练集做局部编辑样本加权复制，避免 val/test 出现重复样本污染评估
        weight_int = max(1, int(doubao_weight))
        if weight_int > 1:
            local_edit_train = [
                item for item in train_data
                if any(
                    token in item[0].lower().replace('\\', '/')
                    for token in ('/doubao/',)  # Removed change duplication to avoid over-fitting and overwhelming the dataset
                )
            ]
            if local_edit_train:
                train_data = train_data + local_edit_train * (weight_int - 1)
                print(f"[Train-only Weight] Added {len(local_edit_train) * (weight_int - 1)} duplicated local-edit samples (weight={weight_int})")
        
        print(f"[Split] Total groups: {total_groups}. Train: {len(train_keys)}, Val: {len(val_keys)}, Test: {len(test_keys)}")
        
        return train_data, val_data, test_data

def save_as_hdf5(output_txt_file, annotations):
    output_txt_file = Path(output_txt_file)
    paths = []
    labels = []
    for line in annotations:
        parts = line.strip().split()
        if len(parts) >= 2:
            paths.append(parts[0])
            labels.append(int(parts[1]))
    
    h5_file = output_txt_file.with_suffix('.h5')
    try:
        with h5py.File(h5_file, 'w') as f:
            f.create_dataset('image_paths', 
                           data=np.array(paths, dtype=h5py.string_dtype(encoding='utf-8')),
                           compression='gzip', compression_opts=9)
            f.create_dataset('labels', 
                           data=np.array(labels, dtype=np.uint8),
                           compression='gzip', compression_opts=9)
            f.attrs['total_images'] = len(paths)
            f.attrs['ai_images'] = sum(labels)
            f.attrs['real_images'] = len(labels) - sum(labels)
        print(f"HDF5 saved: {h5_file.name}")
    except Exception as e:
        print(f"HDF5 save failed: {e}")

