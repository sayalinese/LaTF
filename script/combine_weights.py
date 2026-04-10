import torch
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from service.model_v11_fusion import LaREDeepFakeV11

def create_fused_start_point(base_path, flh_path, out_path):
    print("Instantiating empty LaREDeepFakeV11 model...")
    model = LaREDeepFakeV11(num_classes=2, clip_type='RN50x64', texture_model='convnext_tiny')
    state = model.state_dict()
    
    print(f"Loading Base model: {base_path}")
    checkpoint = torch.load(base_path, map_location='cpu')
    base_state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Load base weights (will keep texture_branch, clip, fusion_mlp)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    print(f"Loaded Base. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    print(f"Loading FLH model: {flh_path}")
    flh_state = torch.load(flh_path, map_location='cpu')
    
    FLH_KEY_MAP = {
        'luma_gate.': 'luma_gate_loc.',
        'ch_interact.': 'ch_interact_loc.',
        'stem.': 'loc_stem.',
        'cbam1.': 'loc_cbam1.',
        'cbam2.': 'loc_cbam2.',
        'cbam3.': 'loc_cbam3.',
        'head.': 'loc_head.'
    }
    
    # Inject FLH state into the model
    injected_count = 0
    current_model_state = model.state_dict()
    for flh_k, v in flh_state.items():
        base_k = flh_k
        for old, new in FLH_KEY_MAP.items():
            if base_k.startswith(old):
                base_k = base_k.replace(old, new)
                break
        if base_k in current_model_state:
            if current_model_state[base_k].shape == v.shape:
                current_model_state[base_k].copy_(v)
                injected_count += 1
            else:
                print(f"Shape mismatch: {base_k}")
        else:
            print(f"Not found: {base_k}")
            
    print(f"Success! Injected {injected_count} keys from FLH into Base model.")
    
    out_dict = {'epoch': 0, 'state_dict': current_model_state, 'best_acc': checkpoint.get('best_acc', 0.0)}
    torch.save(out_dict, out_path)
    print(f"Saved fused weights to: {out_path}")

if __name__ == '__main__':
    create_fused_start_point(
        'outputs/v13_doubao_focused/best.pth',
        'outputs/flh_localization_v2.pth',
        'outputs/v14_fused_ready.pth'
    )
