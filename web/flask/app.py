from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
from io import BytesIO
import base64
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / '.env')
load_dotenv(PROJECT_ROOT / 'web' / '.env', override=True)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Support Chinese characters

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/laft')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Import models and initialize db and migrate
from models import db, migrate
db.init_app(app)
migrate.init_app(app, db)

CORS(app)  # Allow Cross-Origin requests

# 导入重构后的模块 (位于 web/flask/flask_service 下)
try:
    from flask_service.laft.manager import LaFTManager
    from flask_service.vllm.qwen_vl import (
        OllamaConnectionError,
        OllamaTimeoutError,
        explain_logic_with_qwen,
        explain_logic_with_qwen_stream,
    )
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running this from the correct environment and paths are set.")
    sys.exit(1)

# 初始化统一的预测管理器
LAFT_MANAGER = LaFTManager(project_root=str(PROJECT_ROOT))

def load_uploaded_image(file_storage) -> Image.Image:
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    return img.convert('RGB')

# Load model on startup
with app.app_context():
    try:
        LAFT_MANAGER.load_models()
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "device": LAFT_MANAGER.device,
        "model_loaded": LAFT_MANAGER.model is not None,
        "model_version": LAFT_MANAGER.model_version,
        "cascade_enabled": LAFT_MANAGER.cascade_inference is not None,
        "resolved_out_dir": LAFT_MANAGER.resolved_out_dir,
        "ckpt_path": LAFT_MANAGER.loaded_ckpt_path,
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
        response = LAFT_MANAGER.predict(img)
        print(f"Prediction: {response['class_name']} ({response['confidence']:.2f})")
        return jsonify(response)
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/config', methods=['GET'])
def get_config():
    """Return current model configuration"""
    return jsonify({
        "model_version": LAFT_MANAGER.model_version,
        "cascade_enabled": LAFT_MANAGER.cascade_inference is not None,
        "cascade_threshold": LAFT_MANAGER.cascade_threshold,
        "ai_confidence_threshold": LAFT_MANAGER.ai_confidence_threshold,
        "lare_model_type": os.getenv("LARE_MODEL_TYPE", "sdxl"),
        "device": LAFT_MANAGER.device,
        "dual_detector_enabled": LAFT_MANAGER.dual_detector is not None,
        "resolved_out_dir": LAFT_MANAGER.resolved_out_dir,
        "ckpt_path": LAFT_MANAGER.loaded_ckpt_path,
    })

@app.route('/predict_dual', methods=['POST'])
def predict_dual():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        if LAFT_MANAGER.dual_detector is None:
            return jsonify({"error": "Dual detector not initialized"}), 500

        img = load_uploaded_image(file)
        result = LAFT_MANAGER.dual_detector.predict(img)
        
        heatmap_base64 = None
        if result['loc_map'] is not None:
            loc_np = result['loc_map']
            loc_np = cv2.GaussianBlur(loc_np.astype(np.float32), (5, 5), 0)
            if loc_np.max() > loc_np.min():
                loc_norm = ((loc_np - loc_np.min()) / (loc_np.max() - loc_np.min()) * 255).astype(np.uint8)
            else:
                loc_norm = np.zeros_like(loc_np, dtype=np.uint8)
            loc_norm = cv2.resize(loc_norm, (448, 448), interpolation=cv2.INTER_LINEAR)
            heatmap_color = cv2.applyColorMap(loc_norm, cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            heatmap_pil = Image.fromarray(heatmap_color)
            buffered = BytesIO()
            heatmap_pil.save(buffered, format="JPEG")
            heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        response = {
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "detection_type": result['detection_type'],
            "prob_global": result['prob_global'],
            "prob_local": result['prob_local'],
            "explanation": result['explanation'],
            "heatmap": heatmap_base64
        }
        
        print(f"[Dual] {result['prediction']} (global={result['prob_global']:.2f}, local={result['prob_local']})")
        return jsonify(response)

    except Exception as e:
        print(f"Dual prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/analyze_lare', methods=['POST'])
def analyze_lare():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    try:
        img = load_uploaded_image(file)
        
        if LAFT_MANAGER.lare_extractor is None:
            return jsonify({"error": "LaRE extractor not loaded"}), 500
        
        loss_map = LAFT_MANAGER.lare_extractor.extract_single(img)
        
        if LAFT_MANAGER.stat_detector is None:
            return jsonify({"error": "Statistical detector not initialized"}), 500
        
        stats = LAFT_MANAGER.stat_detector.analyze(loss_map)
        is_tampered, score, _ = LAFT_MANAGER.stat_detector.detect(loss_map)
        explanation = LAFT_MANAGER.stat_detector.get_explanation(loss_map)
        
        return jsonify({
            "is_local_tampered": is_tampered,
            "anomaly_score": score,
            "statistics": {
                "mean": stats['mean'],
                "std": stats['std'],
                "min": stats['min'],
                "max": stats['max'],
                "peak_ratio": stats['peak_ratio'],
                "concentration": stats['concentration'],
                "border_ratio": stats['border_ratio'],
                "num_peaks": stats['num_peaks']
            },
            "explanation": explanation
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/explain_logic_stream', methods=['POST'])
@app.route('/explain_logic_stream', methods=['POST'])
def explain_logic_stream():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "缺少图片数据"}), 400
        
    base64_image = data['image']
    is_fake = data.get('is_fake', False)
    
    def generate():
        try:
            for chunk in explain_logic_with_qwen_stream(base64_image, is_fake):
                yield chunk
        except Exception as e:
            yield f"\n[发生错误] {str(e)}\n"
            
    return Response(stream_with_context(generate()), mimetype='text/plain')

@app.route('/api/explain_logic', methods=['POST'])
@app.route('/explain_logic', methods=['POST'])
def explain_logic():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "缺少图片数据"}), 400
            
        base64_image = data['image']
        is_fake = data.get('is_fake', False)
        
        result = explain_logic_with_qwen(base64_image, is_fake)
        return jsonify(result)

    except OllamaTimeoutError as e:
        return jsonify({"error": str(e)}), 504

    except OllamaConnectionError as e:
        return jsonify({"error": str(e)}), 503

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
