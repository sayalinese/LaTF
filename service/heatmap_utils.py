import cv2
import numpy as np

def refine_map_for_visibility(
    prob_map: np.ndarray,
    src_img: np.ndarray | None,
    boost_black_fill: bool = True,
    dark_threshold: int = 35,
    fill_min_area_ratio: float = 0.001,
) -> np.ndarray:
    """
    Refine a probability map for better visual clarity, especially in dark tampered regions.
    
    Args:
        prob_map: [H, W] float array in [0, 1]
        src_img: [H, W, 3] uint8/float RGB image (HWC) OR [3, H, W] (CHW)
        boost_black_fill: Whether to detect and fill "forensic silence" (black/smooth regions)
        dark_threshold: Threshold for "dark" pixel (0-255)
        fill_min_area_ratio: Minimum area for filling holes
    """
    p = np.asarray(prob_map, dtype=np.float32)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)
    
    # 1. Percentile-based Contrast Stretching
    lo, hi = np.percentile(p, [2.0, 99.5])
    if hi - lo > 1e-6:
        p = (p - lo) / (hi - lo)
    p = np.clip(p, 0.0, 1.0)
    
    # Gamma correction to pull up weak signals
    p = np.power(p, 0.85).astype(np.float32)
    m8 = (p * 255.0).astype(np.uint8)

    if not boost_black_fill:
        return m8

    # --- Forensic Silence Detection ---
    # Detect regions that are "too smooth" and "too dark"
    # These are often artifacts of pure black painting/erasing that hide noise traces.
    silence_mask = np.zeros_like(m8, dtype=bool)
    if src_img is not None:
        # Auto-detect CHW vs HWC
        if src_img.shape[0] == 3 and len(src_img.shape) == 3:
            # CHW -> HWC
            src_hwc = np.transpose(src_img, (1, 2, 0))
        else:
            src_hwc = src_img
            
        # Ensure uint8
        if src_hwc.dtype != np.uint8:
            src_hwc = np.clip(src_hwc * 255.0, 0.0, 255.0).astype(np.uint8)

        gray = cv2.cvtColor(src_hwc, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray = cv2.resize(gray, (m8.shape[1], m8.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Local Variance (Noise Floor)
        mean = cv2.blur(gray, (3, 3))
        var = cv2.blur(gray**2, (3, 3)) - mean**2
        
        # Thresholds: Very dark and Extremely smooth (Var < 0.5)
        # Note: Natural dark regions have sensor noise (Var > 1-2). 0.5 is very strict.
        silence_mask = (gray < dark_threshold) & (var < 0.5)
        
        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        silence_mask = cv2.morphologyEx(silence_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0

    # 2. Identify Seed for Boosting
    # High TruFor response OR Forensic Silence
    seed = cv2.threshold(m8, 180, 255, cv2.THRESH_BINARY)[1]
    if np.any(silence_mask):
        seed[silence_mask] = 255

    if seed.max() == 0:
        return m8

    # 3. Filling Holes / Growing Regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = np.zeros_like(seed)
    min_area = max(1.0, float(fill_min_area_ratio) * float(seed.shape[0] * seed.shape[1]))
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(fill, [c], -1, 255, thickness=-1)

    # 4. Boost regions that are both Filled and Dark
    if src_img is not None:
        gray = cv2.cvtColor(src_hwc, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (m8.shape[1], m8.shape[0]), interpolation=cv2.INTER_LINEAR)
        dark_mask = gray <= int(dark_threshold)
        boost_mask = (fill > 0) & dark_mask
    else:
        boost_mask = fill > 0

    if np.any(boost_mask):
        m8 = m8.copy()
        # Ensure at least a medium-high visibility (170/255)
        m8[boost_mask] = np.maximum(m8[boost_mask], 170)
        
    return m8
