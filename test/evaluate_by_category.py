import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import collections

import shutil
from PIL import Image, ImageDraw, ImageFont

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / '.env')

from service.dataset import ImageDataset
from service.model_v11_fusion import LaREDeepFakeV11

def evaluate_category_accuracy(model_path=None):
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Specify the Model Checkpoint here
    if model_path:
        ckpt_path = model_path
    else:
        ckpt_path = "outputs/v13_trufor_retrain/best.pth"
    print(f"Testing Model: {ckpt_path}")
    
    # Preview Dir for Misclassified Images
    PREVIEW_DIR = PROJECT_ROOT / "test" / "预览数据"
    DIR_REAL = PREVIEW_DIR / "real"
    DIR_FACK = PREVIEW_DIR / "fack"
    DIR_FUSION = PREVIEW_DIR / "融合"

    if PREVIEW_DIR.exists():
        try:
            shutil.rmtree(PREVIEW_DIR)
        except Exception as e:
            print(f"Warning: Could not delete preview dir: {e}")
            
    for d in [PREVIEW_DIR, DIR_REAL, DIR_FACK, DIR_FUSION]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 2. Load Validation & Test Lists (Combine them for robust evaluation)
    # We deliberately EXCLUDE training set to avoid "cheating"
    # Try multiple possible annotation files
    possible_files = [
        "annotation/val_sdxl.txt",
        "annotation/test_sdxl.txt",
        "annotation/val_doubao_focused.txt",
        "annotation/test_doubao_focused.txt",
        "annotation/val_sdxl_local.txt",
        "annotation/test_sdxl_local.txt"
    ]
    
    files_to_test = []
    for f in possible_files:
        if os.path.exists(f):
            files_to_test.append(f)
            print(f"Found annotation file: {f}")
    
    if not files_to_test:
        print("Error: No annotation files found!")
        print("Please generate annotation files first:")
        print("  python script/1_gen_annotations.py --output_name doubao_focused --ratio_train 0.03 --ratio_val 0.03 --ratio_test 0.03")
        return
    
    # 3. Categorize Images
    # Categories: Real, SDXL, Flux, Doubao
    # Structure: {'Real': [], 'SDXL': [], 'Flux': [], 'Doubao': []}
    dataset_map = collections.defaultdict(list)
    
    print("Parsing dataset files...")
    total_count = 0
    
    for fpath in files_to_test:
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found.")
            continue
            
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 支持 tab/空格 两种分隔
                if '\t' in line:
                    parts = line.rsplit('\t', 1)
                else:
                    parts = line.rsplit(None, 1)
                if len(parts) != 2:
                    continue
                path, label = parts[0], int(parts[1])
                path_norm = path.replace('\\', '/')
                
                # Determine Category based on path
                if 'data/Real' in path_norm or label == 0:
                    # Note: Doubao Real is also Label 0, but lives in data/doubao
                    # We want to separate Doubao completely
                    if 'doubao' in path_norm.lower():
                        dataset_map['Doubao_Real'].append((path, label))
                    else:
                        dataset_map['Real'].append((path, label))
                elif 'sdxl' in path_norm.lower():
                    dataset_map['SDXL'].append((path, label))
                elif 'flux' in path_norm.lower():
                    dataset_map['Flux'].append((path, label))
                else:
                    # Catch-all for other fakes (including Doubao Fake)
                    if 'doubao' in path_norm.lower():
                        # Specifically separate Doubao Real vs Fake
                        if label == 0:
                            dataset_map['Doubao_Real'].append((path, label))
                        else:
                            dataset_map['Doubao_Fake'].append((path, label))
                    else:
                        dataset_map['Other'].append((path, label))
                    
    # Double check Doubao logic: Doubao contains both Real(0) and Fake(1)
    
    for cat, items in dataset_map.items():
        print(f"-> Found {len(items)} images for {cat}")
        total_count += len(items)
        
    print(f"Total Test Images: {total_count}")

    # [Added] Print sampled details for Doubao (avoid massive logs)
    for cat in ['Doubao_Fake', 'Doubao_Real']:
        if cat in dataset_map:
            print(f"\n--- {cat} Images ({len(dataset_map[cat])}) ---")
            for path, label in dataset_map[cat][:10]:
                print(f"  {path}")
            if len(dataset_map[cat]) > 10:
                print(f"  ... ({len(dataset_map[cat]) - 10} more)")
            print("-----------------------------------")
    
    # 4. Model Setup
    model = LaREDeepFakeV11(num_classes=2, clip_type="RN50x64", texture_model="convnext_tiny")
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=False)
    else:
        print("Checkpoint not found! Running with random weights (Expect ~50%)")
        
    model.to(device)
    model.eval()
    
    # 5. Run Evaluation Per Category
    results = {}
    
    for cat_name, file_list in dataset_map.items():
        if not file_list: continue
        
        print(f"\nEvaluating Category: {cat_name} ...")
        
        # Create Temp Annotation File
        temp_ann = f"annotation/temp_eval_{cat_name}.txt"
        with open(temp_ann, 'w', encoding='utf-8') as f:
            for p, l in file_list:
                f.write(f"{p}\t{l}\n")
                
        # Loader
        dataset = ImageDataset(
            data_root="data",
            train_file=temp_ann,
            map_file=os.getenv('MAP_FILE', 'dift.pt/ann.txt'),
            data_size=448,
            highres_size=512,
            enable_v11=True, # V12
            enable_trufor=True, # [V13] Enable TruFor
            is_train=False,
            drop_no_map=False
        )
        dataset.set_val_mode(True)
        
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2) # Small workers for small batches
        if len(loader) == 0:
            print(f"[Warning] {cat_name} loader is empty, skip.")
            results[cat_name] = {
                'Accuracy': 0.0,
                'Total': 0,
                'Real_Acc': 0.0,
                'Fake_Acc': 0.0,
                'Real_Count': 0,
                'Fake_Count': 0,
            }
            continue
        
        correct = 0
        total = 0
        cat_correct = {0: 0, 1: 0} # Per label correct
        cat_total = {0: 0, 1: 0}   # Per label total
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=cat_name):
                # Handle varying unpacking (V12 vs V11)
                # V12: img, label, map, highres, mask
                # V11: img, label, map, highres
                # V13: img, label, map, highres, mask, trufor
                img_clip = batch[0].to(device)
                labels = batch[1].to(device)
                loss_maps = batch[2].to(device)
                
                img_highres = None
                if len(batch) >= 4:
                    img_highres = batch[3].to(device)
                    
                trufor = None
                if len(batch) >= 6:
                    trufor = batch[5].to(device)
                
                # Forward
                logits = model(img_clip, img_highres, loss_maps, trufor_map=trufor)
                preds = torch.argmax(logits, dim=1)
                
                # Stats
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Copy misclassified images (All Categories)
                if True: # Was: if cat_name in ['Doubao_Real', 'Doubao_Fake']:
                    # Get paths (last element)
                    img_paths = batch[-1]
                    for i, (p, l) in enumerate(zip(preds, labels)):
                        if p != l:
                            try:
                                src_path_str = img_paths[i]
                                src_path = Path(src_path_str)
                                fname = src_path.name
                                stem = src_path.stem
                                
                                # 1. Categorize to real/fack folders based on Ground Truth
                                is_real = (l.item() == 0)
                                target_dir = DIR_REAL if is_real else DIR_FACK
                                shutil.copy2(src_path, target_dir / fname)
                                
                                # 2. Fusion (Comparison) - Only for Doubao
                                if cat_name not in ['Doubao_Real', 'Doubao_Fake']:
                                    continue

                                # Find counterpart in opposite folder
                                # If current is Real -> Look in Fake folder
                                # If current is Fake -> Look in Real folder
                                
                                if is_real:
                                    search_dir = PROJECT_ROOT / "data/doubao/fack"
                                else:
                                    search_dir = PROJECT_ROOT / "data/doubao/real"
                                
                                counterpart_path = None
                                if search_dir.exists():
                                    # Try exact name first
                                    cand = search_dir / fname
                                    if cand.exists():
                                        counterpart_path = cand
                                    else:
                                        # Try stem matching
                                        for f in search_dir.iterdir():
                                            if f.stem == stem:
                                                counterpart_path = f
                                                break
                                
                                if counterpart_path:
                                    # Create Fusion
                                    img_main = Image.open(src_path).convert("RGB")
                                    img_ref = Image.open(counterpart_path).convert("RGB")
                                    
                                    # Resize to same height (512)
                                    h_target = 512
                                    
                                    def resize_h(img, h):
                                        w, old_h = img.size
                                        ratio = h / old_h
                                        new_w = int(w * ratio)
                                        return img.resize((new_w, h), Image.Resampling.LANCZOS)
                                    
                                    img_main = resize_h(img_main, h_target)
                                    img_ref = resize_h(img_ref, h_target)
                                    
                                    # Layout: [Real] | [Fake]
                                    if is_real:
                                        # Main is Real (Left), Ref is Fake (Right)
                                        left_img = img_main
                                        right_img = img_ref
                                    else:
                                        # Main is Fake (Right), Ref is Real (Left)
                                        left_img = img_ref
                                        right_img = img_main
                                    
                                    # Combine
                                    w_total = left_img.width + right_img.width
                                    fusion = Image.new("RGB", (w_total, h_target))
                                    fusion.paste(left_img, (0, 0))
                                    fusion.paste(right_img, (left_img.width, 0))
                                    
                                    # Draw Text
                                    draw = ImageDraw.Draw(fusion)
                                    try:
                                        font = ImageFont.truetype("arial.ttf", 40)
                                    except:
                                        font = ImageFont.load_default()
                                        
                                    text_color = (255, 0, 0)
                                    # Draw "Real" on Left
                                    draw.text((10, 10), "Real", fill=(0, 255, 0), font=font)
                                    # Draw "Fake" on Right
                                    draw.text((left_img.width + 10, 10), "Fake", fill=(0, 255, 0), font=font)
                                    
                                    # Add "MISCLASSIFIED" label
                                    if is_real:
                                        draw.text((10, 60), "MISCLASSIFIED (Pred: Fake)", fill=text_color, font=font)
                                    else:
                                        draw.text((left_img.width + 10, 60), "MISCLASSIFIED (Pred: Real)", fill=text_color, font=font)

                                    fusion_name = f"Fusion_{stem}.jpg"
                                    fusion.save(DIR_FUSION / fusion_name)
                                    
                            except Exception as e:
                                print(f"Error processing {src_path}: {e}")

                # Detailed Stats
                for p, l in zip(preds, labels):
                    l_item = l.item()
                    cat_total[l_item] += 1
                    if p.item() == l_item:
                        cat_correct[l_item] += 1
        
        acc = correct / total * 100 if total > 0 else 0
        rec_acc = cat_correct[0] / cat_total[0] * 100 if cat_total[0] > 0 else 0
        fake_acc = cat_correct[1] / cat_total[1] * 100 if cat_total[1] > 0 else 0
        
        results[cat_name] = {
            'Accuracy': acc,
            'Total': total,
            'Real_Acc': rec_acc, # Recall for Real
            'Fake_Acc': fake_acc, # Recall for Fake
            'Real_Count': cat_total[0],
            'Fake_Count': cat_total[1]
        }
        
    # 6. Final Report
    print("\n" + "="*60)
    print(f"{'Category':<10} | {'Total':<6} | {'Acc (%)':<8} | {'Real Acc':<8} | {'Fake Acc':<8}")
    print("-" * 60)
    
    for cat, res in results.items():
        real_str = f"{res['Real_Acc']:.1f}" if res['Real_Count'] > 0 else "N/A"
        fake_str = f"{res['Fake_Acc']:.1f}" if res['Fake_Count'] > 0 else "N/A"
        print(f"{cat:<10} | {res['Total']:<6} | {res['Accuracy']:<8.2f} | {real_str:<8} | {fake_str:<8}")
        
    print("="*60)
    print("Note: 'Real Acc' means how many Real images were correctly classified as Real.")
    print("Note: 'Fake Acc' means how many AI images were correctly classified as Fake.")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    args = parser.parse_args()
    evaluate_category_accuracy(model_path=args.model)
