import torch
import sys
import os
from pathlib import Path

# Add root to path
sys.path.append(os.getcwd())

from service.model import CLipClassifierWMapV9Lite

def check_acc():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Checking model on device: {device}")
    
    # Paths to check
    ckpt_paths = [
        "outputs/v9lite/best.pth",
        "outputs/sdv5_v7/best.pth"
    ]
    
    for ckpt_path in ckpt_paths:
        if not os.path.exists(ckpt_path):
            continue
            
        print(f"\n--- Analyzing {ckpt_path} ---")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            
            if isinstance(checkpoint, dict):
                epoch = checkpoint.get('epoch', 'N/A')
                best_acc = checkpoint.get('best_acc', 'N/A')
                print(f"Epoch: {epoch}")
                print(f"Recorded Best Accuracy: {best_acc}")
                
                if best_acc != 'N/A':
                    if isinstance(best_acc, float):
                        print(f"Accuracy Percentage: {best_acc * 100:.2f}%")
                    else:
                         print(f"Accuracy Value: {best_acc}")
            else:
                print("Checkpoint is a raw state_dict (no metadata available).")
                
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")

if __name__ == "__main__":
    check_acc()
