"""
批量生成 TruFor 特征图脚本
遍历数据集，使用 TruFor 提取篡改定位热力图，并保存为灰度 PNG 以供训练使用。
"""
import os
import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from dotenv import load_dotenv

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / '.env')

from service.heatmap_utils import refine_map_for_visibility

from service.trufor_wrapper import TruForExtractor

# Configuration
# 我们现在根据 'annotation' 文件夹中的 txt 列表进行生成，不再随机采样
# 这样能保证训练/验证集中用到的所有图片都有对应的 TruFor Map
ANNOTATION_FILES = [
    'annotation/train_sdxl.txt',
    'annotation/val_sdxl.txt', 
    'annotation/test_sdxl.txt',
]

# [V13] strict alignment mode
# 如果设为 True，则优先从 LARE_MAP_FILE 读取图片列表，确保 TruFor 和 LaRE 特征一一对应
STRICT_ALIGNMENT = True
LARE_MAP_FILE = os.getenv('LARE_MAP_FILE', os.getenv('MAP_FILE', 'dift.pt/ann.txt'))

BATCH_SIZE = 8
NUM_WORKERS = 4
INPUT_RESIZE = (1024, 1024) # Resize for consistent GPU batching

EXCLUDED_DIR_NAMES = {'masks', 'masks_vis'}


def should_skip_path(path_like) -> bool:
    p = Path(path_like)
    parts_lower = {part.lower() for part in p.parts}
    return not parts_lower.isdisjoint(EXCLUDED_DIR_NAMES)


class TruForListDataset(Dataset):
    def __init__(self, file_list, root_dir, output_dir, transform=None, overwrite=False):
        self.files = file_list
        # root_dir is PROJECT_ROOT -> D:\三创\LaRE-main
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.transform = transform
        self.overwrite = overwrite
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # file_path is something like D:\三创\LaRE-main\data\doubao\masks\000.png
        img_path_str = self.files[idx]
        img_path = Path(img_path_str)
        
        # Calculate relative path to project root
        # e.g. data\doubao\masks\000.png
        # Check if 'data' is part of path
        p_str = str(img_path)
        if 'data' in p_str:
            # Finding right-most or left-most?
            # Usually D:\...\data\...
            idx = p_str.find('data')
            rel_str = p_str[idx:]
            save_path = self.output_dir / Path(rel_str).with_suffix('.png')
        else:
            # Fallback
            try:
                rel = img_path.relative_to(self.root_dir)
                save_path = self.output_dir / rel.with_suffix('.png')
            except ValueError:
                # Can't map, skip
                return None

        # Check if exists
        try:
            if save_path.exists() and not self.overwrite:
                 return None # Skip signal

            # Ensure parent dir exists
            # We do this in main loop, but here is also safe? 
            # save_path.parent.mkdir(parents=True, exist_ok=True)
            
            img = Image.open(str(img_path)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return {'img': img, 'save_path': str(save_path)}
        except Exception:
            return None

def parse_annotation_files(file_paths, strict_alignment: bool, lare_map_file: str):
    all_images = []
    
    # 1. Full Scan for Doubao (Critical)
    # [Mod] Doubao scan now optional if STRICT_ALIGNMENT is on
    if not strict_alignment:
        doubao_path = PROJECT_ROOT / 'data/doubao'
        if doubao_path.exists():
             print("Scanning data/doubao directory for full coverage...")
             # rglob matches recursively
             doubao_files = [
                 str(f) for f in doubao_path.rglob('*')
                 if f.suffix.lower() in {'.png', '.jpg', '.jpeg'} and not should_skip_path(f)
             ]
             print(f"  Found {len(doubao_files)} Doubao images")
             all_images.extend(doubao_files)

    # 2. Parse from txt files
    file_list_to_scan = []
    if strict_alignment:
        print(f"[Strict Alignment] Scanning only from {lare_map_file}")
        file_list_to_scan = [lare_map_file]
    else:
        print(f"Parsing annotation files {file_paths}...")
        file_list_to_scan = file_paths

    for fp_rel in file_list_to_scan:
        p = PROJECT_ROOT / fp_rel
        if not p.exists():
            print(f"Skipping missing list: {p}")
            continue
            
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split('\t') # try tab split first
                if len(parts) == 1:
                     parts = line.strip().split() # then try space

                if len(parts) >= 1:
                    raw_path = parts[0]
                    # Handle "D:/涓夊垱/..." Mojibake or just Windows absolute paths
                    # Format in map file is usually absolute like "D:/三创/..."
                    # Check if exists directly first
                    abs_p = Path(raw_path)
                    if abs_p.exists():
                        if not should_skip_path(abs_p):
                            all_images.append(str(abs_p))
                        continue
                    
                    # If not, try relative fix
                    if 'data/' in raw_path or 'data\\' in raw_path:
                        n_path = raw_path.replace('\\', '/')
                        idx = n_path.find('data/')
                        # Extract "data/Real/FFHQ/..."
                        rel_path = n_path[idx:]
                        # Reconstruct "D:\三创\LaRE-main\data\Real\FFHQ\..."
                        abs_path = PROJECT_ROOT / rel_path
                        if abs_path.exists():
                            if not should_skip_path(abs_path):
                                all_images.append(str(abs_path))
                        # else:
                        #     print(f"Missing: {abs_path}")

    # Remove duplicates
    unique_images = list(set(all_images))
    print(f"Total unique images to process: {len(unique_images)}")
    return unique_images

def custom_collate(batch):
    # Filter out Nones (skipped or error)
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    imgs = torch.stack([item['img'] for item in batch])
    paths = [item['save_path'] for item in batch]
    return imgs, paths





def process_loader(extractor, loader, boost_black_fill, dark_threshold, fill_min_area_ratio):
    for batch in tqdm(loader, desc="Processing Batch"):
        if batch is None:
            continue
            
        imgs, paths = batch
        
        try:
            # Batch Extract
            # maps: [B, H, W]
            prob_maps = extractor.extract_batch(imgs)
            
            if prob_maps is None:
                continue

            for i, save_p in enumerate(paths):
                prob_map = prob_maps[i]
                src_img = imgs[i].cpu().numpy() if i < imgs.shape[0] else None
                norm_map = refine_map_for_visibility(
                    prob_map,
                    src_img,
                    boost_black_fill=boost_black_fill,
                    dark_threshold=dark_threshold,
                    fill_min_area_ratio=fill_min_area_ratio,
                )
                
                # Resize output to 512x512 standard
                resized_map = cv2.resize(norm_map, (512, 512), interpolation=cv2.INTER_LINEAR)
                
                # Ensure directory exists
                p = Path(save_p)
                p.parent.mkdir(parents=True, exist_ok=True)
                
                # Use PIL to save to support unicode paths (OpenCV has issues with Chinese paths on Windows)
                try:
                    Image.fromarray(resized_map).save(str(p))
                except Exception as e:
                    print(f"Failed to write {p}: {e}")

        except Exception as e:
            print(f"Batch Error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--strict_alignment', action='store_true', default=STRICT_ALIGNMENT)
    parser.add_argument('--no_strict_alignment', action='store_false', dest='strict_alignment')
    parser.add_argument('--lare_map_file', type=str, default=LARE_MAP_FILE)
    parser.add_argument('--annotation_files', nargs='*', default=ANNOTATION_FILES)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--input_resize', nargs=2, type=int, default=list(INPUT_RESIZE))
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--black_fill_boost', action='store_true', default=True)
    parser.add_argument('--no_black_fill_boost', action='store_false', dest='black_fill_boost')
    parser.add_argument('--dark_threshold', type=int, default=35)
    parser.add_argument('--fill_min_area_ratio', type=float, default=0.001)
    args = parser.parse_args()
    
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing TruFor Extractor on {device}...")
    
    extractor = TruForExtractor(device=device)
    if extractor.model is None:
        print(" Layer 8 Error: Model failed.")
        return

    # Preprocessing for Batch
    # TruFor requires Tensor input. We also Resize to standard size for Batching
    transform = transforms.Compose([
        transforms.Resize(tuple(args.input_resize)), # Resize for batch consistency
        transforms.ToTensor(),
    ])

    # 1. Gather all files
    # If STRICT_ALIGNMENT = True, parse_annotation_files will use LARE_MAP_FILE
    print("Gathering files...")
    all_files = parse_annotation_files(args.annotation_files, strict_alignment=args.strict_alignment, lare_map_file=args.lare_map_file)
    if args.max_images and args.max_images > 0:
        all_files = sorted(all_files)[:args.max_images]
        print(f"Limiting to max_images={args.max_images}. Now processing: {len(all_files)} images")
    
    # 2. Add Doubao & Real full scan if they are critical and possibly missed
    # Doubao full scan is already inside parse_annotation_files function now
    
    # 3. Create Dataset and Loader
    base_out = PROJECT_ROOT / 'trufor_maps'
    
    if len(all_files) == 0:
        print("No files found!")
        return

    dataset = TruForListDataset(all_files, PROJECT_ROOT, base_out, transform=transform, overwrite=args.overwrite)
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=custom_collate
    )
    
    # 4. Process
    print(f"Starting processing of {len(all_files)} images...")
    process_loader(
        extractor,
        loader,
        boost_black_fill=args.black_fill_boost,
        dark_threshold=args.dark_threshold,
        fill_min_area_ratio=args.fill_min_area_ratio,
    )
            
    print("\n✅ All TruFor features generated!")

if __name__ == "__main__":
    main()
