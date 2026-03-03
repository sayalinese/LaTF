
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import requests
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from service.model_v11_fusion import LaREDeepFakeV11
from service.lare_extractor_module import LareExtractor
from service.trufor_wrapper import TruForExtractor
from service.cascade_inference import CascadeInference

def load_map_dict(map_file):
    map_dict = {}
    path = Path(map_file)
    if not path.exists():
        return map_dict
    
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    # Key: Image filename stem (e.g., "1" from "1.jpg")
                    # Value: Map path
                    img_path = parts[0]
                    map_path = parts[1]
                    stem = Path(img_path).stem
                    map_dict[stem] = map_path
    except Exception as e:
        print(f"Error loading map dict: {e}")
    return map_dict

def test_consistency():
    print("=== Running Consistency Test V2 (With TruFor Stats) ===")
    only_web = os.getenv("ONLY_WEB", "0") == "1"
    only_local = os.getenv("ONLY_LOCAL", "0") == "1"
    cache_path = PROJECT_ROOT / "web_probs_cache.json"
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        r = requests.get("http://127.0.0.1:5000/health", timeout=2)
        if r.status_code == 200 and device == "cuda":
            device = "cpu"
    except Exception:
        pass
    if os.getenv("FORCE_GPU", "0") == "1" and torch.cuda.is_available():
        device = "cuda"
    print(f"Using device: {device}")

    model = None
    lare_extractor = None
    trufor_extractor = None
    if not only_web:
        print("Loading models locally...")
        model = LaREDeepFakeV11(
            num_classes=2,
            clip_type="RN50x64",
            texture_model="convnext_tiny"
        ).to(device)
        
        checkpoint_path = PROJECT_ROOT / "outputs" / "v13_trufor_retrain" / "best.pth"
        if not checkpoint_path.exists():
             checkpoint_path = PROJECT_ROOT / "outputs" / "v13_trufor_fusion" / "best.pth"

        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
                
            model.load_state_dict(new_state_dict, strict=False)
            print("Model loaded.")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            return

        try:
            lare_extractor = LareExtractor(device=device, dtype=torch.float32)
            trufor_extractor = TruForExtractor(device=device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                device = "cpu"
                model.to(device)
                lare_extractor = LareExtractor(device=device, dtype=torch.float32)
                trufor_extractor = TruForExtractor(device=device)
                print("Switched to CPU due to CUDA OOM.")
            else:
                raise

    # 2. Get Test Images
    folder_override = os.getenv("FOLDER_PATH")
    test_items = []
    if folder_override:
        folder_path = Path(folder_override)
        if not folder_path.is_absolute():
            folder_path = PROJECT_ROOT / folder_path
        if folder_path.exists():
            exts = {".jpg", ".jpeg", ".png", ".webp"}
            for p in sorted(folder_path.rglob("*")):
                if p.suffix.lower() in exts:
                    test_items.append((p, 0))
        print(f"Found {len(test_items)} images in folder: {folder_path}")
    else:
        # Parse from annotation files (Doubao 20 images)
        val_file = PROJECT_ROOT / "annotation" / "val_sdxl.txt"
        test_file = PROJECT_ROOT / "annotation" / "test_sdxl.txt"
        
        image_paths = []
        
        # Helper to find Doubao images
        def find_doubao_images(file_path):
            found = []
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if "doubao" in line.lower():
                            parts = line.strip().split('\t') if '\t' in line else line.strip().split(' ')
                            img_path_str = parts[0]
                            label = int(parts[1])
                            
                            # Handle absolute or relative paths
                            if os.path.isabs(img_path_str):
                                full_path = Path(img_path_str)
                            else:
                                full_path = PROJECT_ROOT / img_path_str
                                
                            if full_path.exists():
                                found.append((full_path, label))
            return found

        image_paths.extend(find_doubao_images(val_file))
        image_paths.extend(find_doubao_images(test_file))
        
        # Deduplicate
        unique_paths = {}
        for p, l in image_paths:
            unique_paths[str(p)] = (p, l)
        
        test_items = list(unique_paths.values())
        print(f"Found {len(test_items)} Doubao images.")
        if len(test_items) > 20:
            test_items = test_items[:20]
        print(f"Using {len(test_items)} Doubao images for test.")

        # Save list to file for user
        with open(PROJECT_ROOT / "doubao_test_list.txt", "w", encoding="utf-8") as f:
            f.write("Path\tLabel\n")
            for p, l in test_items:
                f.write(f"{p}\t{l}\n")
        print(f"Saved Doubao image list to {PROJECT_ROOT / 'doubao_test_list.txt'}")
    
    # Load Precomputed Map Index
    map_file = PROJECT_ROOT / "features" / "lightning_map.txt"
    if not map_file.exists():
        print(f"Warning: Map file not found at {map_file}. Skipping precomputed map comparison.")
        precomputed_maps = {}
    else:
        precomputed_maps = load_map_dict(map_file)

    # Results table
    print(f"\n{'Image':<30} | {'Label':<5} | {'Web Prob':<10} | {'Local Prob':<10} | {'L-Map Diff':<10} | {'TF-Mean':<10} | {'TF-Max':<10} | {'Status'}")
    print("-" * 120)

    cascade = None
    if not only_web:
        cascade = CascadeInference(
            model=model,
            lare_extractor=lare_extractor,
            trufor_extractor=trufor_extractor,
            device=device
        )

    total = 0
    web_ok = 0
    local_ok = 0
    mismatch_count = 0
    web_err = 0
    local_err = 0
    cache_data = {}
    if only_local and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        except Exception:
            cache_data = {}

    for img_path, label in test_items:
        img_stem = img_path.stem
        total += 1
        
        # --- Web API Inference ---
        web_prob = "N/A"
        try:
            if only_local:
                if str(img_path) in cache_data:
                    web_prob = f"{float(cache_data[str(img_path)]):.4f}"
            else:
                with open(img_path, 'rb') as f:
                    files = {'file': f}
                    response = requests.post('http://127.0.0.1:5000/predict', files=files, timeout=60)
                    if response.status_code == 200:
                        res_json = response.json()
                        prob_val = res_json.get('prob')
                        if prob_val is None:
                            prob_val = res_json.get('confidence')
                        if prob_val is None:
                            probs = res_json.get('probabilities', {})
                            prob_val = probs.get('AI Generated', probs.get('AI', 0.0))
                        web_prob = f"{float(prob_val):.4f}"
                        cache_data[str(img_path)] = float(prob_val)
        except Exception as e:
            web_prob = "Err" # e.g. ConnectionError if server down

        # --- Local Simulation ---
        tf_mean = "N/A"
        tf_max = "N/A"
        l_diff = "N/A"
        local_prob = "N/A"
        if not only_web:
            try:
                raw_pil = Image.open(img_path).convert("RGB")
                
                res = cascade.inference(raw_pil)
                local_prob = f"{res['prob']:.4f}"
                
                if trufor_extractor:
                    tf_transform = transforms.Compose([
                        transforms.Resize((1024, 1024)),
                        transforms.ToTensor(),
                    ])
                    print(f"Image Size: {raw_pil.size}")
                    tf_input = tf_transform(raw_pil).unsqueeze(0).to(device)
                    tf_np = trufor_extractor.extract_batch(tf_input)
                    if tf_np is not None:
                        tf_mean = f"{tf_np.mean():.4f}"
                        tf_max = f"{tf_np.max():.4f}"

                local_map = lare_extractor.extract_single(raw_pil, ensemble_size=4).to(device)
                
                if img_stem in precomputed_maps:
                    pt_path = PROJECT_ROOT / precomputed_maps[img_stem]
                    if pt_path.exists():
                        gt_map = torch.load(pt_path, map_location=device)
                        if local_map.shape != gt_map.shape:
                             local_map_resized = F.interpolate(local_map, size=gt_map.shape[-2:], mode='bilinear')
                             diff = (local_map_resized - gt_map).abs().mean().item()
                        else:
                             diff = (local_map - gt_map).abs().mean().item()
                        l_diff = f"{diff:.4f}"
                
            except Exception as e:
                local_prob = f"Err: {e}"
                l_diff = "Err"

        status = "MATCH"
        try:
            if web_prob != "Err" and local_prob != "Err":
                wp = float(web_prob)
                lp = float(local_prob)
                if abs(wp - lp) > 0.1:
                    status = "MISMATCH"
                    mismatch_count += 1
        except:
            pass

        try:
            if web_prob != "Err":
                wp = float(web_prob)
                web_pred = 1 if wp >= 0.5 else 0
                if web_pred == int(label):
                    web_ok += 1
            else:
                web_err += 1
        except:
            web_err += 1

        try:
            if local_prob != "Err":
                lp = float(local_prob)
                local_pred = 1 if lp >= 0.5 else 0
                if local_pred == int(label):
                    local_ok += 1
            else:
                local_err += 1
        except:
            local_err += 1

        print(f"{img_stem:<30} | {label:<5} | {web_prob:<10} | {local_prob:<10} | {l_diff:<10} | {tf_mean:<10} | {tf_max:<10} | {status}", flush=True)
        
        if not only_web:
            torch.cuda.empty_cache()

    if not only_local and cache_data:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

    if total > 0:
        web_acc = (web_ok / max(total - web_err, 1)) * 100
        print("\n=== Summary ===")
        if only_web:
            print(f"Total: {total} | Web Errors: {web_err}")
            print(f"Web Accuracy: {web_acc:.2f}%")
        else:
            local_acc = (local_ok / max(total - local_err, 1)) * 100
            print(f"Total: {total} | Web Errors: {web_err} | Local Errors: {local_err}")
            print(f"Web Accuracy: {web_acc:.2f}% | Local Accuracy: {local_acc:.2f}%")
            print(f"Web-Local Mismatches (>0.1): {mismatch_count}")

if __name__ == "__main__":
    test_consistency()
