from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / '.env')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from service.model import CLipClassifierWMapV7
from service.lare_extractor_module import LareExtractor

app = Flask(__name__)
CORS(app)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 双模型集成
MODEL_SD15 = None
MODEL_SDXL_CLASSIFIER = None
LARE_SD15 = None
LARE_SDXL = None

CLASS_NAMES = ["Real Photo", "AI Generated"]


def load_uploaded_image(file_storage) -> Image.Image:
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    return img.convert('RGB')

def load_models():
    global MODEL_SD15, MODEL_SDXL_CLASSIFIER, LARE_SD15, LARE_SDXL
    
    print(f"[Ensemble] Loading dual-backbone models on {DEVICE}...")
    
    # SD1.5 分支（检测老图）
    print("[Ensemble] Loading SD1.5 branch...")
    LARE_SD15 = LareExtractor(device=DEVICE, model_type="sd15")
    MODEL_SD15 = CLipClassifierWMapV7(clip_type="RN50x64", num_class=2)
    
    # Load SD1.5 checkpoint from .env configuration
    out_dir = os.getenv('OUT_DIR', 'outputs/sdv5_v7')
    ckpt_sd15 = None
    if os.path.exists(out_dir):
        # Look for Val_best.pth in subdirectories
        for root, dirs, files in os.walk(out_dir):
            if 'Val_best.pth' in files:
                ckpt_sd15 = os.path.join(root, 'Val_best.pth')
                break
    
    if ckpt_sd15 and os.path.exists(ckpt_sd15):
        print(f"[Ensemble] Loading SD1.5 weights from {ckpt_sd15}")
        checkpoint = torch.load(ckpt_sd15, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        MODEL_SD15.load_state_dict(state_dict, strict=False)
    else:
        print("[Ensemble] No SD1.5 checkpoint found, using random weights")
    MODEL_SD15.to(DEVICE).eval()
    
    # SDXL 分支（检测新图，如豆包）
    print("[Ensemble] Loading SDXL branch...")
    try:
        LARE_SDXL = LareExtractor(device=DEVICE, model_type="sdxl")
        MODEL_SDXL_CLASSIFIER = CLipClassifierWMapV7(clip_type="RN50x64", num_class=2)
        
        # Try to load SDXL-trained weights, fallback to SD1.5 weights if not found
        ckpt_sdxl = os.getenv('CKPT_SDXL', '')  # Can be overridden via .env
        if not ckpt_sdxl or not os.path.exists(ckpt_sdxl):
            # Search in common output directories for SDXL checkpoint
            outputs_dirs = [
                'outputs/sdxl_full',
                'outputs/sdv5_sdxl',
                os.getenv('OUT_DIR', 'outputs/sdv5_v7'),
            ]
            for dir_path in outputs_dirs:
                if os.path.exists(dir_path):
                    for root, dirs, files in os.walk(dir_path):
                        if 'Val_best.pth' in files:
                            ckpt_sdxl = os.path.join(root, 'Val_best.pth')
                            break
                    if ckpt_sdxl:
                        break
        
        if ckpt_sdxl and os.path.exists(ckpt_sdxl):
            print(f"[Ensemble] Loading SDXL-trained classifier from {ckpt_sdxl}")
            checkpoint = torch.load(ckpt_sdxl, map_location=DEVICE, weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            MODEL_SDXL_CLASSIFIER.load_state_dict(state_dict, strict=False)
            print("[Ensemble] Using SDXL-trained classifier")
        else:
            # 回退到SD15权重
            MODEL_SDXL_CLASSIFIER.load_state_dict(MODEL_SD15.state_dict())
            print("[Ensemble] SDXL classifier using SD15 weights (fallback)")
            
        MODEL_SDXL_CLASSIFIER.to(DEVICE).eval()
    except Exception as e:
        print(f"[Ensemble] SDXL loading failed: {e}, using SD15 only")
        LARE_SDXL = None
        MODEL_SDXL_CLASSIFIER = None
    
    print("[Ensemble] Models ready!")

with app.app_context():
    try:
        load_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "device": DEVICE,
        "sd15_loaded": MODEL_SD15 is not None,
        "sdxl_loaded": MODEL_SDXL_CLASSIFIER is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        img = load_uploaded_image(file)
        
        with torch.no_grad():
            # 双模型预测
            predictions = []
            
            # SD1.5 预测
            loss_map_sd15 = LARE_SD15.extract_single(img)
            clip_input_sd15 = MODEL_SD15.preprocess(img).unsqueeze(0).to(DEVICE)
            model_dtype = next(MODEL_SD15.parameters()).dtype
            loss_map_sd15 = loss_map_sd15.to(device=DEVICE, dtype=model_dtype)
            clip_input_sd15 = clip_input_sd15.to(dtype=model_dtype)
            
            logits_sd15 = MODEL_SD15(clip_input_sd15, loss_map_sd15)
            probs_sd15 = F.softmax(logits_sd15, dim=1)[0]
            predictions.append(probs_sd15)
            
            # SDXL 预测（如果可用）
            if LARE_SDXL is not None and MODEL_SDXL_CLASSIFIER is not None:
                loss_map_sdxl = LARE_SDXL.extract_single(img)
                clip_input_sdxl = MODEL_SDXL_CLASSIFIER.preprocess(img).unsqueeze(0).to(DEVICE)
                model_dtype_sdxl = next(MODEL_SDXL_CLASSIFIER.parameters()).dtype
                loss_map_sdxl = loss_map_sdxl.to(device=DEVICE, dtype=model_dtype_sdxl)
                clip_input_sdxl = clip_input_sdxl.to(dtype=model_dtype_sdxl)
                
                logits_sdxl = MODEL_SDXL_CLASSIFIER(clip_input_sdxl, loss_map_sdxl)
                probs_sdxl = F.softmax(logits_sdxl, dim=1)[0]
                predictions.append(probs_sdxl)
            
            # 集成：加权平均（SDXL权重更高，因为更适合现代AI）
            if len(predictions) == 2:
                probs = 0.3 * predictions[0] + 0.7 * predictions[1]  # SDXL 70%权重
            else:
                probs = predictions[0]

        confidence, pred_idx = torch.max(probs, 0)
        idx = int(pred_idx.item())
        
        result = {
            "class_name": CLASS_NAMES[idx],
            "confidence": float(confidence.item()),
            "class_idx": idx,
            "probabilities": {
                CLASS_NAMES[0]: float(probs[0]),
                CLASS_NAMES[1]: float(probs[1])
            },
            "ensemble_method": "SD15+SDXL" if len(predictions) == 2 else "SD15_only"
        }
        
        print(f"Prediction: {result}")
        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
