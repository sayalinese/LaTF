"""Blueprint: Qwen 视觉大模型解释路由 (/explain_logic, /explain_logic_stream)"""
import traceback
from flask import Blueprint, request, jsonify, Response, stream_with_context

from flask_service.vllm.qwen_vl import (
    OllamaConnectionError,
    OllamaTimeoutError,
    explain_logic_with_qwen,
    explain_logic_with_qwen_stream,
)

vllm_bp = Blueprint('vllm', __name__)


@vllm_bp.route('/api/explain_logic_stream', methods=['POST'])
@vllm_bp.route('/explain_logic_stream', methods=['POST'])
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


@vllm_bp.route('/api/explain_logic', methods=['POST'])
@vllm_bp.route('/explain_logic', methods=['POST'])
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
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
