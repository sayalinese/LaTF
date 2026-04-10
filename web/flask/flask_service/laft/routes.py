"""Blueprint: AI 检测相关路由 (/predict, /predict_dual, /analyze_lare, /detect_message)"""
import os
import traceback
import base64
from io import BytesIO

import cv2
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from PIL import Image, ImageOps
import uuid as _uuid

from flask_service.utils import load_uploaded_image

laft_bp = Blueprint('laft', __name__)


@laft_bp.route('/health', methods=['GET'])
def health_check():
    mgr = current_app.laft_manager
    return jsonify({
        "status": "healthy",
        "device": mgr.device,
        "model_loaded": mgr.model is not None,
        "model_version": mgr.model_version,
        "cascade_enabled": mgr.cascade_inference is not None,
        "resolved_out_dir": mgr.resolved_out_dir,
        "ckpt_path": mgr.loaded_ckpt_path,
    })


@laft_bp.route('/config', methods=['GET'])
def get_config():
    mgr = current_app.laft_manager
    return jsonify({
        "model_version": mgr.model_version,
        "cascade_enabled": mgr.cascade_inference is not None,
        "cascade_threshold": mgr.cascade_threshold,
        "ai_confidence_threshold": mgr.ai_confidence_threshold,
        "lare_model_type": os.getenv("LARE_MODEL_TYPE", "sdxl"),
        "device": mgr.device,
        "dual_detector_enabled": mgr.dual_detector is not None,
        "resolved_out_dir": mgr.resolved_out_dir,
        "ckpt_path": mgr.loaded_ckpt_path,
    })


@laft_bp.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    try:
        img = load_uploaded_image(file)
        response = current_app.laft_manager.predict(img)
        print(f"Prediction: {response['class_name']} ({response['confidence']:.2f})")
        return jsonify(response)
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@laft_bp.route('/predict_dual', methods=['POST'])
def predict_dual():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    try:
        mgr = current_app.laft_manager
        if mgr.dual_detector is None:
            return jsonify({"error": "Dual detector not initialized"}), 500

        img = load_uploaded_image(file)
        result = mgr.dual_detector.predict(img)

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
            "heatmap": heatmap_base64,
        }
        print(f"[Dual] {result['prediction']} (global={result['prob_global']:.2f}, local={result['prob_local']})")
        return jsonify(response)
    except Exception as e:
        print(f"Dual prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@laft_bp.route('/analyze_lare', methods=['POST'])
def analyze_lare():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    try:
        img = load_uploaded_image(file)
        mgr = current_app.laft_manager

        if mgr.lare_extractor is None:
            return jsonify({"error": "LaRE extractor not loaded"}), 500
        loss_map = mgr.lare_extractor.extract_single(img)

        if mgr.stat_detector is None:
            return jsonify({"error": "Statistical detector not initialized"}), 500
        stats = mgr.stat_detector.analyze(loss_map)
        is_tampered, score, _ = mgr.stat_detector.detect(loss_map)
        explanation = mgr.stat_detector.get_explanation(loss_map)

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
                "num_peaks": stats['num_peaks'],
            },
            "explanation": explanation,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@laft_bp.route('/detect_message', methods=['POST'])
def detect_message():
    from models import Message, db

    data = request.json or {}
    message_id = data.get('message_id')
    use_vl = data.get('use_vl', False)

    if not message_id:
        return jsonify({"success": False, "message": "缺少 message_id"}), 400

    msg = Message.query.get(_uuid.UUID(message_id))
    if not msg:
        return jsonify({"success": False, "message": "消息不存在"}), 404
    if msg.content_type != 'image':
        return jsonify({"success": False, "message": "该消息不是图片"}), 400

    img_url = msg.content
    filename = img_url.split('/uploads/')[-1] if '/uploads/' in img_url else None
    if not filename:
        return jsonify({"success": False, "message": "无法解析图片路径"}), 400

    filepath = os.path.join(current_app.config['UPLOAD_DIR'], filename)
    if not os.path.isfile(filepath):
        return jsonify({"success": False, "message": "图片文件不存在"}), 404

    try:
        img = Image.open(filepath)
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')

        laft_result = current_app.laft_manager.predict(img)
        detect_data = {
            "class_name": laft_result.get("class_name", "Unknown"),
            "confidence": round(laft_result.get("confidence", 0), 4),
            "probabilities": laft_result.get("probabilities", {}),
            "heatmap": laft_result.get("heatmap"),
        }

        vl_report = None
        if use_vl:
            try:
                from flask_service.vllm.qwen_vl import explain_logic_with_qwen
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                b64_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
                is_fake = detect_data["class_name"] != "Real"
                vl_result = explain_logic_with_qwen(b64_img, is_fake)
                vl_report = vl_result.get("report", "")
            except Exception as vl_err:
                vl_report = f"VL 分析不可用: {str(vl_err)}"
        detect_data["vl_report"] = vl_report

        msg.ai_detect_data = detect_data
        db.session.commit()
        return jsonify({"success": True, "detect": detect_data})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "message": f"检测失败: {str(e)}"}), 500
