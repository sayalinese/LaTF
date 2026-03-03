"""分析doubao样本和Real样本的LaRE特征对比"""
import torch
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from service.statistical_detector import StatisticalLocalDetector

detector = StatisticalLocalDetector()

def analyze_folder(feature_dir, name, limit=15):
    files = [f for f in os.listdir(feature_dir) if f.endswith('.pt')][:limit]
    detected = 0
    
    print(f'=== {name} ({len(files)}张) ===\n')
    
    for f in files:
        lare = torch.load(os.path.join(feature_dir, f), map_location='cpu')
        if lare.dtype == torch.bfloat16:
            lare = lare.float()
        
        stats = detector.analyze(lare)
        is_tampered, score, _ = detector.detect(lare)
        
        if is_tampered:
            detected += 1
        
        print(f'{f}: peak={stats["peak_ratio"]:.2f}, std_r={stats["std_ratio"]:.2f}, conc={stats["concentration"]*100:.1f}% -> {is_tampered}')
    
    print(f'\n检测率: {detected}/{len(files)} ({detected/len(files)*100:.1f}%)\n')
    return detected, len(files)

# 测试Doubao (应该检测出)
d1, t1 = analyze_folder('features_local/train/doubao', 'Doubao局部篡改')

# 测试Real (不应该误报)
d2, t2 = analyze_folder('features_local/train/FFHQ', 'FFHQ真实图')
d3, t3 = analyze_folder('features_local/train/FORLAB', 'FORLAB真实图')

print('=== 总结 ===')
print(f'Doubao检测率: {d1}/{t1} ({d1/t1*100:.1f}%)')
print(f'FFHQ误报率: {d2}/{t2} ({d2/t2*100:.1f}%)')
print(f'FORLAB误报率: {d3}/{t3} ({d3/t3*100:.1f}%)')
print(f'\n当前阈值: peak>{detector.peak_threshold}, std_ratio>{detector.min_std_ratio}')
