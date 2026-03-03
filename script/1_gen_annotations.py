#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Set encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / '.env')

from service.data_prep import SDXLDataCollector, save_as_hdf5

def main():
    parser = argparse.ArgumentParser(description='Generate SDXL Annotations')
    parser.add_argument('--output_name', type=str, default='sdxl')
    parser.add_argument('--ratio_train', type=float, default=float(os.getenv('SAMPLE_RATIO_TRAIN', '1.0')))
    parser.add_argument('--ratio_val', type=float, default=float(os.getenv('SAMPLE_RATIO_VAL', '1.0')))
    parser.add_argument('--ratio_test', type=float, default=float(os.getenv('SAMPLE_RATIO_TEST', '1.0')))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--local_only', action='store_true', help='Only use local inpaint (doubao) + Real for stage 2 fine-tuning')
    args = parser.parse_args()

    sdxl_root = PROJECT_ROOT / os.getenv('DATA_ROOT', 'data')
    output_dir = PROJECT_ROOT / "annotation"
    output_dir.mkdir(exist_ok=True)
    
    # Updated paths for new data structure:
    # data/
    #   flux/ (fake)
    #   sdxl/ (fake)
    #   special/ (fake)
    #   Real/ (real)
    fake_root = sdxl_root
    real_root = sdxl_root / "Real"
    
    if not fake_root.exists() or not real_root.exists():
        print(f"Error: Data paths do not exist at {sdxl_root}")
        print(f"Check: {fake_root} and {real_root}")
        return

    collector = SDXLDataCollector(fake_root, real_root)
    train_data, val_data, test_data = collector.create_annotations(
        train_ratio=args.ratio_train,
        val_ratio=args.ratio_val,
        test_ratio=args.ratio_test,
        seed=args.seed,
        local_only=args.local_only
    )
    
    splits = [
        ('train', train_data),
        ('val', val_data),
        ('test', test_data)
    ]
    
    for split_name, data in splits:
        file_path = output_dir / f'{split_name}_{args.output_name}.txt'
        with open(file_path, 'w', encoding='utf-8') as f:
            for img_path, label in data:
                f.write(f"{img_path}\t{label}\n")
        
        save_as_hdf5(file_path, [f"{p}\t{l}" for p, l in data])
        
        fake_count = sum(1 for _, l in data if l == '1')
        print(f"{split_name.upper()}: {len(data)} total ({fake_count} fake, {len(data)-fake_count} real)")

if __name__ == "__main__":
    main()

