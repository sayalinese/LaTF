import os
import sys
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / '.env')

import service.model as models
from service.lare_extractor_module import LareExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, model_path, device='cuda', clip_type="RN50x64", extract_features=False, model_class="CLipClassifierWMapV7"):
        self.device = device
        self.clip_type = clip_type
        self.extract_features = extract_features
        self.model_class = model_class
        
        # Load Model
        self.model = self._load_model(model_path)
        
        # Determine input resolution
        if "RN50x64" in clip_type:
            self.input_res = 448
        elif "RN50x16" in clip_type:
            self.input_res = 384
        elif "RN50x4" in clip_type:
            self.input_res = 288
        else:
            self.input_res = 224
            
        print(f"Model Tester: Using input resolution {self.input_res} for {clip_type}")
        
        # Load Extractor if needed
        if extract_features:
            # Determine dtype based on device capability
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
            self.extractor = LareExtractor(device=device, dtype=dtype)
            
        # Standard Preprocessing (CLIP)
        from torchvision import transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((self.input_res, self.input_res)), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def _load_model(self, path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")
            
        # Model Class (dynamic by --model_class arg)
        ModelClass = getattr(models, self.model_class, None)
        if ModelClass is None:
            raise ValueError(f"Model class {self.model_class} not found in service.model")
        model = ModelClass(num_class=2, clip_type=self.clip_type)
        
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Fix keys
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path, feature_path=None):
        image_path = Path(image_path)
        img = Image.open(image_path).convert('RGB')
        
        # 1. Prepare Image Input
        img_input = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # 2. Prepare Feature Input (Loss Map)
        if feature_path:
            loss_map = torch.load(feature_path, map_location=self.device)
            if loss_map.dim() == 3:
                loss_map = loss_map.unsqueeze(0)
        elif self.extract_features:
            loss_map = self.extractor.extract_single(img).to(self.device) # [1, 4, 32, 32]
        else:
            raise ValueError("Must provide feature_path or enable extract_features")
            
        # Determine dtype for autocast (consistent with training)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # 3. Predict
        with torch.no_grad():
            try:
                from torch.amp import autocast
            except ImportError:
                from torch.cuda.amp import autocast
                
            with autocast('cuda', dtype=amp_dtype):
                outputs = self.model(img_input, loss_map)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                probs = F.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
            
        return {
            'image': image_path.name,
            'prediction': 'AI' if pred_idx == 1 else 'Real',
            'confidence': probs[0, pred_idx].item(),
            'probs': probs[0].tolist()
        }

    def evaluate_folder(self, folder, feature_dir=None):
        folder = Path(folder)
        results = []
        
        for img_file in folder.glob('*'):
            if img_file.suffix.lower() not in {'.jpg', '.png', '.jpeg', '.webp'}:
                continue
                
            feature_path = None
            if feature_dir:
                # Try finding matching feature file
                cand = Path(feature_dir) / (img_file.stem + '.pt')
                if cand.exists():
                    feature_path = cand
            
            try:
                res = self.predict(img_file, feature_path)
                logger.info(f"{res['image']}: {res['prediction']} ({res['confidence']:.2f})")
                results.append(res)
            except Exception as e:
                logger.error(f"Failed {img_file.name}: {e}")
                
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--dir', type=str)
    
    # Auto-find model if not specified
    default_model = os.getenv('MODEL_PATH')
    if not default_model:
        # Try common paths
        candidates = [
            PROJECT_ROOT / "outputs/model/best.pth",
            PROJECT_ROOT / "outputs/model/latest.pth",
            PROJECT_ROOT / "outputs/sdv5_v7/Val_best.pth"
        ]
        for c in candidates:
            if c.exists():
                default_model = str(c)
                break
                
    parser.add_argument('--model', type=str, default=default_model, help="Path to model checkpoint")
    parser.add_argument('--model_class', type=str, default='CLipClassifierWMapV7', help="Model class name (CLipClassifierWMapV7, CLipClassifierWMapV9Lite, CLipClassifierV10Fusion)")
    parser.add_argument('--feature', type=str)
    parser.add_argument('--feature_dir', type=str)
    parser.add_argument('--extract_features', action='store_true')
    args = parser.parse_args()
    
    if not args.model:
        logger.error("No model found. Please specify --model or set MODEL_PATH env var.")
        # return

    tester = ModelTester(args.model, extract_features=args.extract_features, model_class=args.model_class)
    
    if args.image:
        res = tester.predict(args.image, args.feature)
        print(res)
    elif args.dir:
        tester.evaluate_folder(args.dir, args.feature_dir)

if __name__ == '__main__':
    main()

