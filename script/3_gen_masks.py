import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from multiprocessing import Pool, cpu_count, freeze_support

# Global variables for worker processes
ROOT_DIR = Path('data/doubao')
FAKE_DIR = ROOT_DIR / 'fack'
REAL_DIR = ROOT_DIR / 'real'
MASK_DIR = ROOT_DIR / 'masks'
VIS_DIR = ROOT_DIR / 'masks_vis'

def process_single_image(fake_path_str):
    try:
        fake_path = Path(fake_path_str)
        filename = fake_path.name
        
        # Determine Real Path
        real_path = REAL_DIR / filename
        if not real_path.exists():
             possible_reals = list(REAL_DIR.glob(f"{fake_path.stem}.*"))
             if possible_reals:
                 real_path = possible_reals[0]
             else:
                 return ('skipped', filename)

        # Read Images
        # Use numpy fromfile for Windows path compatibility
        img_fake_data = np.fromfile(str(fake_path), dtype=np.uint8)
        img_fake = cv2.imdecode(img_fake_data, cv2.IMREAD_COLOR)
        
        img_real_data = np.fromfile(str(real_path), dtype=np.uint8)
        img_real = cv2.imdecode(img_real_data, cv2.IMREAD_COLOR)

        if img_fake is None or img_real is None:
            return 'error'

        # Resize validation
        if img_fake.shape != img_real.shape:
             img_real = cv2.resize(img_real, (img_fake.shape[1], img_fake.shape[0]))

        # [Check] Identity Check (Data Integrity)
        # If images are exactly the same, skip mask generation (empty mask)
        diff_temp = cv2.absdiff(img_fake, img_real)
        if np.sum(diff_temp) == 0:
            return ('skipped', filename)

        # SSIM Calculation
        gray_fake = cv2.cvtColor(img_fake, cv2.COLOR_BGR2GRAY)
        gray_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY)

        # score, diff = ssim(..., full=True)
        # diff range 0.0 - 1.0 (1.0 = same)
        (score, diff) = ssim(gray_real, gray_fake, full=True)
        
        # Convert to 0-255 uint8
        diff = (diff * 255).astype("uint8")
        
        # Thresholding (Similarity < 0.9 -> Difference)
        # High SSIM usually > 0.9. We want regions where similarity is LOW.
        # Threshold: 230 (~0.9)
        thresh_value = 230 
        _, mask = cv2.threshold(diff, thresh_value, 255, cv2.THRESH_BINARY_INV)

        # Post-processing
        # 1. Connected Components (Filter small noise)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 100: # Filter small dots
                cleaned_mask[labels == i] = 255
        mask = cleaned_mask

        # 2. Morphology (Close gaps)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Save Mask (Use opencv imencode for unicode paths support)
        save_path = MASK_DIR / f"{fake_path.stem}.png"
        is_success, buffer = cv2.imencode(".png", mask)
        if is_success:
            buffer.tofile(str(save_path))

        # Save Visualization (Only for some to check quality)
        # Force visualization for first 20 images to check quality immediately
        if np.random.rand() < 0.2: # 20% sample
             vis = img_fake.copy()
             # Green overlay
             vis[mask > 0] = vis[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
             
             # Layout: [Real Image] | [Fake Image] | [Mask Overlay]
             # easier to spot what changed
             combined = np.hstack([img_real, img_fake, vis])
             
             vis_save_path = VIS_DIR / f"vis_{filename}"
             is_success, buffer = cv2.imencode(".png", combined)
             if is_success:
                buffer.tofile(str(vis_save_path))

        return 'success'

    except Exception as e:
        return f'error: {e}'

def generate_masks():
    # Directories
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Gather Files
    print(f"Scanning {FAKE_DIR}...")
    fake_images = sorted(list(FAKE_DIR.glob('*.jpg')) + list(FAKE_DIR.glob('*.png')))
    file_list = [str(p) for p in fake_images]
    
    print(f"[SSIM Multi-Process] Found {len(file_list)} images.")
    print(f"Using {cpu_count()} CPU cores.")
    
    # Run Pool
    count_success = 0
    count_skipped = 0
    count_error = 0
    deleted_list = []
    skipped_list = []
    
    with Pool(processes=cpu_count()) as pool:
        # Use imap to get progress bar
        for result in tqdm(pool.imap_unordered(process_single_image, file_list), total=len(file_list)):
            if isinstance(result, tuple):
                tag, val = result[0], result[1]
                if tag == 'deleted':
                    deleted_list.append(val)
                    count_skipped += 1
                elif tag == 'skipped':
                    skipped_list.append(val)
                    count_skipped += 1
                elif tag == 'error_deleting':
                    print(f"[Delete Error] {val}")
                    count_error += 1
                else:
                    count_error += 1
            else:
                if result == 'success':
                    count_success += 1
                else:
                    count_error += 1
                
    print(f"\nProcessing Complete!")
    print(f"- Generated: {count_success}")
    print(f"- Skipped: {count_skipped}")
    print(f"- Deleted Identical Pairs: {len(deleted_list)}")
    print(f"- Errors: {count_error}")
    
    if len(deleted_list) > 0:
        print(f"\n[Cleanup Report] Deleted {len(deleted_list)} identical Real/Fake pairs.")
        # print("Files deleted:")
        # for fname in deleted_list:
        #     print(f" -> {fname}")
    if len(skipped_list) > 0:
        print(f"\n[Skipped Report] {len(skipped_list)} fake images had no matching real image:")
        for fname in skipped_list:
            print(f" -> {fname}")
            
    print(f"- Masks saved to: {MASK_DIR}")

if __name__ == "__main__":
    freeze_support()
    generate_masks()
