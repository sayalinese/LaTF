from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory, session as flask_session
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
import uuid as uuid_mod
from werkzeug.utils import secure_filename

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / '.env')
load_dotenv(PROJECT_ROOT / 'web' / '.env', override=True)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Support Chinese characters
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'lare-dev-secret-key-2026')
# Session cookie 配置（开发环境）
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False   # HTTP 开发环境不需要 HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_PATH'] = '/'

# 上传文件配置
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/laft')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Import models and initialize db and migrate
from models import db, migrate, User, DisputeSession, Message
db.init_app(app)
migrate.init_app(app, db)

CORS(app, supports_credentials=True, origins=[
    'http://localhost:3000', 'http://127.0.0.1:3000',
    'http://localhost:5173', 'http://127.0.0.1:5173',
    r'https://.*\.ngrok-free\.app',   # ngrok 免费域名
    r'https://.*\.ngrok\.io',         # ngrok 旧域名
])  # Flask-CORS 6.x 必须指定 origins，否则 credentials 会被浏览器拒绝

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

# RAG 模块（可选，依赖不存在时不影响启动）
try:
    from flask_service.rag.retriever import get_retriever
    from flask_service.rag.prompts import build_rag_system_prompt
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False
    def get_retriever():  # type: ignore
        return None
    def build_rag_system_prompt(base, docs, platform_hint=None):  # type: ignore
        return base

# LLM 风险定责审查模块
try:
    from flask_service.vllm.llm import generate_forensics_report_data
    _LLM_REPORT_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] LLM report module not available: {e}")
    _LLM_REPORT_AVAILABLE = False
    def generate_forensics_report_data(*args, **kwargs):
        return {"riskyStatements": [], "ruleViolations": []}

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

@app.route('/', methods=['GET'])
def index():
    return jsonify({"name": "LaRE API", "status": "running", "docs": "/health"})

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

# --- 认证相关接口 ---
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    nickname = data.get('nickname', username)

    if not username or not password:
        return jsonify({"success": False, "message": "用户名和密码不能为空"}), 400

    existing_user = User.query.filter_by(id=username).first()
    if existing_user:
        return jsonify({"success": False, "message": "用户名已存在"}), 400

    new_user = User(id=username, nickname=nickname, password=password, avatar=username[0].upper())
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({"success": True, "message": "注册成功"})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(id=username, password=password).first()
    if user:
        flask_session['user_id'] = user.id
        disputes_count = DisputeSession.query.filter((DisputeSession.buyer_id == user.id) | (DisputeSession.seller_id == user.id)).count()
        history_count = disputes_count * 3 + 5 
        
        return jsonify({
            "success": True, 
            "message": "登录成功", 
            "user": {
                "id": user.id,
                "username": user.id, 
                "nickname": user.nickname, 
                "avatar": user.avatar or user.nickname[0].upper(),
                "historyCount": history_count,
                "disputesCount": disputes_count,
                "accuracy": "97.5%"
            }
        })
    else:
        return jsonify({"success": False, "message": "用户名或密码错误"}), 401

@app.route('/logout', methods=['POST'])
def logout():
    flask_session.pop('user_id', None)
    return jsonify({"success": True, "message": "已退出登录"})

@app.route('/me', methods=['GET'])
def get_current_user():
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "message": "未登录"}), 401
    user = User.query.get(user_id)
    if not user:
        flask_session.pop('user_id', None)
        return jsonify({"success": False, "message": "用户不存在"}), 401
    disputes_count = DisputeSession.query.filter((DisputeSession.buyer_id == user.id) | (DisputeSession.seller_id == user.id)).count()
    history_count = disputes_count * 3 + 5
    return jsonify({
        "success": True,
        "user": {
            "id": user.id,
            "username": user.id,
            "nickname": user.nickname,
            "avatar": user.avatar or user.nickname[0].upper(),
            "historyCount": history_count,
            "disputesCount": disputes_count,
            "accuracy": "97.5%"
        }
    })

@app.route('/update_profile', methods=['POST'])
def update_profile():
    user_id = flask_session.get('user_id')
    if not user_id:
        return jsonify({"success": False, "message": "未登录"}), 401
    user = User.query.get(user_id)
    if not user:
        return jsonify({"success": False, "message": "用户不存在"}), 401

    # 处理头像上传
    if 'avatar' in request.files:
        file = request.files['avatar']
        if file.filename and allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_name = f"avatar_{uuid_mod.uuid4().hex}.{ext}"
            file.save(os.path.join(UPLOAD_DIR, unique_name))
            user.avatar = f"/api/uploads/{unique_name}"

    # 处理昵称更新
    nickname = request.form.get('nickname')
    if nickname and nickname.strip():
        user.nickname = nickname.strip()

    db.session.commit()
    return jsonify({
        "success": True,
        "message": "资料更新成功",
        "user": {
            "nickname": user.nickname,
            "avatar": user.avatar or user.nickname[0].upper()
        }
    })

# --- 会话相关工具函数 ---
def _session_to_dict(s):
    last_msg = Message.query.filter_by(session_id=s.id).order_by(Message.created_at.desc()).first()
    desc = "等待对方上传证据图片..."
    if last_msg:
        if last_msg.content_type == 'image':
            desc = f"[{last_msg.sender_role}发了图片]"
        else:
            desc = last_msg.content[:15] + "..." if len(last_msg.content) > 15 else last_msg.content
    return {
        "id": str(s.id),
        "title": s.topic_name,
        "desc": desc,
        "status": s.status,
        "platform": s.platform or 'taobao',
        "buyer_id": s.buyer_id or '',
        "seller_id": s.seller_id or '',
        "order_id": s.order_id or '',
        "product_name": s.product_name or '',
        "product_price": s.product_price or '',
        "product_image": s.product_image or '',
        "time": s.created_at.strftime('%H:%M') if s.created_at else ''
    }

def _check_session(session_id_str, user_id=None, owner_only=False):
    """返回 (session, error_tuple)，error_tuple 为 None 表示成功。"""
    import uuid as _uuid
    try:
        sid = _uuid.UUID(session_id_str)
    except Exception:
        return None, (jsonify({"success": False, "message": "无效的会话ID"}), 400)
    s = DisputeSession.query.get(sid)
    if not s:
        return None, (jsonify({"success": False, "message": "会话不存在"}), 404)
    if user_id:
        if owner_only and s.buyer_id != user_id:
            return None, (jsonify({"success": False, "message": "仅会话创建者可执行此操作"}), 403)
        elif not owner_only and s.buyer_id != user_id and s.seller_id != user_id:
            return None, (jsonify({"success": False, "message": "无权访问此会话"}), 403)
    return s, None

# --- 会话相关接口 ---
@app.route('/sessions', methods=['GET'])
def get_sessions():
    current_user_id = flask_session.get('user_id')
    if not current_user_id:
        return jsonify({"success": True, "sessions": []})
    sessions = DisputeSession.query.filter(
        (DisputeSession.buyer_id == current_user_id) | (DisputeSession.seller_id == current_user_id)
    ).order_by(DisputeSession.updated_at.desc()).all()
    return jsonify({"success": True, "sessions": [_session_to_dict(s) for s in sessions]})

@app.route('/sessions', methods=['POST'])
def create_session():
    current_user_id = flask_session.get('user_id')
    if not current_user_id:
        return jsonify({"success": False, "message": "请先登录"}), 401
    data = request.json or {}
    topic_name = (data.get('topic_name') or '未命名交易纠纷').strip() or '未命名交易纠纷'
    platform = (data.get('platform') or 'taobao').strip()
    # 验证 platform 值
    if platform not in ['taobao', 'xianyu', 'jd', 'pdd']:
        platform = 'taobao'
    s = DisputeSession(topic_name=topic_name, platform=platform, buyer_id=current_user_id, status='open')
    # 订单/商品信息（可选）
    s.order_id = (data.get('order_id') or '').strip() or None
    s.product_name = (data.get('product_name') or '').strip() or None
    s.product_price = (data.get('product_price') or '').strip() or None
    s.product_image = (data.get('product_image') or '').strip() or None
    db.session.add(s)
    db.session.commit()
    return jsonify({"success": True, "session": _session_to_dict(s)})

@app.route('/sessions/<session_id>', methods=['PUT'])
def update_session(session_id):
    current_user_id = flask_session.get('user_id')
    if not current_user_id:
        return jsonify({"success": False, "message": "请先登录"}), 401
    s, err = _check_session(session_id, current_user_id, owner_only=True)
    if err: return err
    data = request.json or {}
    topic_name = (data.get('topic_name') or '').strip()
    platform = (data.get('platform') or '').strip()
    if topic_name:
        s.topic_name = topic_name
    if platform and platform in ['taobao', 'xianyu', 'jd', 'pdd']:
        s.platform = platform
    # 订单/商品信息编辑
    if 'order_id' in data:
        s.order_id = (data['order_id'] or '').strip() or None
    if 'product_name' in data:
        s.product_name = (data['product_name'] or '').strip() or None
    if 'product_price' in data:
        s.product_price = (data['product_price'] or '').strip() or None
    if 'product_image' in data:
        s.product_image = (data['product_image'] or '').strip() or None
    db.session.commit()
    return jsonify({"success": True, "session": _session_to_dict(s)})

@app.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    current_user_id = flask_session.get('user_id')
    if not current_user_id:
        return jsonify({"success": False, "message": "请先登录"}), 401
    s, err = _check_session(session_id, current_user_id, owner_only=True)
    if err: return err
    db.session.delete(s)
    db.session.commit()
    return jsonify({"success": True})

@app.route('/sessions/<session_id>/join', methods=['POST'])
def join_session(session_id):
    current_user_id = flask_session.get('user_id')
    if not current_user_id:
        return jsonify({"success": False, "message": "请先登录后再加入会话"}), 401
    import uuid as _uuid
    try:
        s = DisputeSession.query.get(_uuid.UUID(session_id))
    except Exception:
        return jsonify({"success": False, "message": "无效的邀请码"}), 400
    if not s:
        return jsonify({"success": False, "message": "会话不存在，请检查邀请码"}), 404
    if s.buyer_id == current_user_id:
        return jsonify({"success": False, "message": "你是该会话的创建者，无需加入"}), 400
    if s.seller_id:
        if s.seller_id != current_user_id:
            return jsonify({"success": False, "message": "该会话对方席位已被占用"}), 400
    else:
        s.seller_id = current_user_id
        db.session.commit()
    return jsonify({"success": True, "session": _session_to_dict(s)})

@app.route('/sessions/<session_id>/messages', methods=['GET'])
def get_session_messages(session_id):
    current_user_id = flask_session.get('user_id')
    s, err = _check_session(session_id, current_user_id)
    if err: return err

    after_id = request.args.get('after')
    query = Message.query.filter_by(session_id=session_id)
    if after_id:
        import uuid as _uuid
        try:
            after_msg = Message.query.get(_uuid.UUID(after_id))
            if after_msg:
                query = query.filter(Message.created_at > after_msg.created_at)
        except Exception:
            pass

    msgs = query.order_by(Message.created_at.asc()).all()
    res = []
    for m in msgs:
        sender = User.query.get(m.sender_id)
        sender_name = sender.nickname if sender else m.sender_id
        sender_avatar = (sender.avatar or sender_name[0].upper()) if sender else m.sender_id[0].upper()
        res.append({
            "id": str(m.id),
            "role": m.sender_role,
            "sender_id": m.sender_id,
            "name": sender_name,
            "avatar": sender_avatar,
            "content": m.content,
            "type": m.content_type,
            "hasBeenDetected": True if m.ai_detect_data else False,
            "created_at": m.created_at.isoformat() if m.created_at else None
        })
    return jsonify({"success": True, "messages": res})

# --- 静态上传文件服务 ---
@app.route('/uploads/<path:filename>', methods=['GET'])
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# --- 发送消息（文本 / 图片） ---
@app.route('/sessions/<session_id>/messages', methods=['POST'])
def send_message(session_id):
    current_user_id = flask_session.get('user_id')
    if not current_user_id:
        return jsonify({"success": False, "message": "请先登录"}), 401
    s, err = _check_session(session_id, current_user_id)
    if err: return err

    # 从会话成员关系自动推导角色，防止客户端伪造
    if s.buyer_id == current_user_id:
        sender_role = 'buyer'
    elif s.seller_id == current_user_id:
        sender_role = 'seller'
    else:
        return jsonify({"success": False, "message": "你不是此会话成员"}), 403

    content_type = request.form.get('content_type', 'text')
    text_content = request.form.get('content', '')

    import uuid
    sid = uuid.UUID(session_id)

    if content_type == 'image':
        if 'file' not in request.files:
            return jsonify({"success": False, "message": "未上传图片文件"}), 400
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"success": False, "message": "不支持的文件格式"}), 400
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_name = f"{uuid_mod.uuid4().hex}.{ext}"
        file.save(os.path.join(UPLOAD_DIR, unique_name))
        saved_content = f"/api/uploads/{unique_name}"
    else:
        if not text_content.strip():
            return jsonify({"success": False, "message": "消息内容不能为空"}), 400
        saved_content = text_content

    msg = Message(
        session_id=sid,
        sender_id=current_user_id,
        sender_role=sender_role,
        content_type=content_type,
        content=saved_content
    )
    db.session.add(msg)
    # 更新 updated_at 使会话排在列表前面
    s.updated_at = __import__('datetime').datetime.utcnow()
    db.session.commit()

    sender = User.query.get(current_user_id)
    sender_name = sender.nickname if sender else current_user_id
    sender_avatar = (sender.avatar or sender_name[0].upper()) if sender else current_user_id[0].upper()

    return jsonify({
        "success": True,
        "message": {
            "id": str(msg.id),
            "role": msg.sender_role,
            "sender_id": current_user_id,
            "name": sender_name,
            "avatar": sender_avatar,
            "content": msg.content,
            "type": msg.content_type,
            "hasBeenDetected": False
        }
    })

# --- AI 鉴定：对会话中的图片消息执行 LaFT 检测 ---
@app.route('/detect_message', methods=['POST'])
def detect_message():
    data = request.json or {}
    message_id = data.get('message_id')
    use_vl = data.get('use_vl', False)  # 是否使用 Qwen-VL 生成报告

    if not message_id:
        return jsonify({"success": False, "message": "缺少 message_id"}), 400

    import uuid as _uuid
    msg = Message.query.get(_uuid.UUID(message_id))
    if not msg:
        return jsonify({"success": False, "message": "消息不存在"}), 404
    if msg.content_type != 'image':
        return jsonify({"success": False, "message": "该消息不是图片"}), 400

    # 解析图片路径 - 本地文件
    img_url = msg.content  # e.g. "/api/uploads/xxx.jpg"
    filename = img_url.split('/uploads/')[-1] if '/uploads/' in img_url else None
    if not filename:
        return jsonify({"success": False, "message": "无法解析图片路径"}), 400

    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.isfile(filepath):
        return jsonify({"success": False, "message": "图片文件不存在"}), 404

    try:
        img = Image.open(filepath)
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')

        # LaFT 检测
        laft_result = LAFT_MANAGER.predict(img)

        # 组装检测数据
        detect_data = {
            "class_name": laft_result.get("class_name", "Unknown"),
            "confidence": round(laft_result.get("confidence", 0), 4),
            "probabilities": laft_result.get("probabilities", {}),
            "heatmap": laft_result.get("heatmap"),  # base64
        }

        # 可选：Qwen-VL 解释报告
        vl_report = None
        if use_vl:
            try:
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                b64_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
                is_fake = detect_data["class_name"] != "Real"
                vl_result = explain_logic_with_qwen(b64_img, is_fake)
                vl_report = vl_result.get("report", "")
            except Exception as vl_err:
                vl_report = f"VL 分析不可用: {str(vl_err)}"

        detect_data["vl_report"] = vl_report

        # 存入数据库
        msg.ai_detect_data = detect_data
        db.session.commit()

        return jsonify({"success": True, "detect": detect_data})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"检测失败: {str(e)}"}), 500


# --- 悬浮 AI 小助手：基于会话上下文的智能分析 ---
@app.route('/sessions/<session_id>/assistant_chat', methods=['POST'])
def assistant_chat(session_id):
    current_user_id = flask_session.get('user_id')
    if not current_user_id:
        return jsonify({"success": False, "message": "请先登录"}), 401

    s, err = _check_session(session_id, current_user_id)
    if err:
        return err

    data = request.json or {}
    user_message = (data.get('message') or '').strip()
    if not user_message:
        return jsonify({"success": False, "message": "消息不能为空"}), 400

    # 对话历史（前端传来的助手对话记录，保持上下文连贯）
    history = data.get('history', [])
    # 使用会话绑定的平台，或前端可选覆盖
    rag_platform = (data.get('platform') or s.platform or 'taobao').strip()

    # 确定当前用户角色
    if s.buyer_id == current_user_id:
        user_role = '买家'
        opponent_role = '卖家'
    else:
        user_role = '卖家'
        opponent_role = '买家'

    # 获取会话聊天记录作为背景上下文
    chat_messages = Message.query.filter_by(session_id=s.id).order_by(Message.created_at.asc()).all()
    chat_context_lines = []
    for m in chat_messages[-30:]:  # 最近 30 条
        sender = User.query.get(m.sender_id)
        name = sender.nickname if sender else m.sender_id
        is_me = (m.sender_id == current_user_id)
        role_label = '买家' if m.sender_id == s.buyer_id else '卖家'
        side_label = '【我方】' if is_me else '【对方】'
        if m.content_type == 'image':
            content_desc = '[发送了一张图片]'
            if m.ai_detect_data:
                det = m.ai_detect_data
                cls = det.get('class_name', '未知')
                conf = det.get('confidence', 0)
                content_desc += f' (AI鉴定结果: {cls}, 置信度: {conf:.1%})'
        else:
            content_desc = m.content
        chat_context_lines.append(f"{side_label}({role_label}/{name}): {content_desc}")

    chat_context = '\n'.join(chat_context_lines) if chat_context_lines else '(暂无聊天记录)'

    sys_prompt = (
        "你是一个交易纠纷 AI 小助手，帮助用户分析纠纷会话中的可疑行为和风险点。\n\n"
        f"## 身份说明\n"
        f"- 当前使用助手的用户（即\"我方\"）身份是：**{user_role}**\n"
        f"- 对方身份是：**{opponent_role}**\n"
        f"- 聊天记录中每条消息已用【我方】或【对方】明确标注\n\n"
        "## 你的职责\n"
        "1. 分析【对方】的聊天行为，判断是否存在可疑行为（虚假承诺、前后矛盾、回避关键问题等）\n"
        "2. 如果聊天中有图片且已被 AI 鉴定为伪造，结合鉴定结果给出风险提示\n"
        "3. 站在【我方】角度，提供维权建议和沟通策略\n"
        "4. 回答用户关于纠纷处理的疑问\n\n"
        "## 注意事项\n"
        "- 回答时「你」或「我方」始终指当前用户（即使用本助手的人）\n"
        "- 「对方」始终指交易另一方\n"
        "- 请简明扼要，重点突出，使用 Markdown 格式，控制在 300 字以内\n\n"
        "===== 当前纠纷会话聊天记录 =====\n"
        f"{chat_context}\n"
        "===== 聊天记录结束 ====="
    )

    # RAG 增强：检索相关平台规则条款
    _retriever = get_retriever() if _RAG_AVAILABLE else None
    if _retriever is not None:
        rag_docs = _retriever.retrieve(user_message, platform=rag_platform, k=3)
        sys_prompt = build_rag_system_prompt(sys_prompt, rag_docs, platform_hint=rag_platform)
    

    # 构建调用消息: system + history + user
    api_messages = [{"role": "system", "content": sys_prompt}]
    for h in history[-10:]:  # 保留最近 10 轮助手对话
        api_messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    api_messages.append({"role": "user", "content": user_message})

    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    base_url = os.getenv("SILICONFLOW_URL", "https://api.siliconflow.cn/v1")
    if base_url.endswith("/chat/completions"):
        base_url = base_url.replace("/chat/completions", "")
    model = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")

    if not api_key or api_key == "your_siliconflow_api_key_here":
        return jsonify({"success": False, "message": "AI 服务未配置 API Key"}), 503

    def generate():
        try:
            from openai import OpenAI as _OpenAI
            client = _OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=api_messages,
                stream=True,
                timeout=60
            )
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as exc:
            yield f"\n[AI 服务请求失败]: {str(exc)}\n"

    return Response(stream_with_context(generate()), mimetype='text/plain')


@app.route("/api/generate_comprehensive_report", methods=["POST"])
@app.route("/generate_comprehensive_report", methods=["POST"])
def api_generate_comprehensive_report():
    data = request.json
    session_id = data.get("session_id")
    # image_id = data.get("image_id")
    is_fake_str = data.get("is_fake", "true") 
    is_fake = is_fake_str.lower() == "true" if isinstance(is_fake_str, str) else bool(is_fake_str)
    fake_reason = data.get("fake_reason", "")
    
    # 1. Provide chat history
    messages = Message.query.filter_by(session_id=session_id).order_by(Message.created_at.asc()).all()
    session = DisputeSession.query.get(session_id)
    platform = session.platform if session else "taobao"

    chat_context_lines = []
    for msg in messages:
        role_label = "买家" if msg.sender_role == "buyer" else "商家"
        side_label = "【我方】" if msg.sender_role == "buyer" else "【对方】"
        header = f"{side_label}({role_label})"
        if msg.content_type == "text":
            chat_context_lines.append(f"{header}: {msg.content}")
        elif msg.content_type == "image":
            chat_context_lines.append(f"{header}: [发送了一张图片]")
    full_chat = "\n".join(chat_context_lines)
    
    # 2. Get RAG info
    rules_text = ""
    if _RAG_AVAILABLE:
        try:
            retriever = get_retriever()
            if retriever is not None:
                rag_docs = retriever.retrieve(f"用户在{platform}平台上发生了纠纷，对方声称商品没问题或平台不管。有哪些相关规定？", platform)
                rules_text = "\n".join([r.get("content", "") for r in rag_docs])
        except Exception as e:
            print(f"[WARN] RAG retrieval failed: {e}")
    
    # 3. Call text LLM for rules and risky statements
    print(f"[INFO] 综合报告: session={session_id}, chat_lines={len(chat_context_lines)}, rules_len={len(rules_text)}, is_fake={is_fake}")
    try:
        report_data = generate_forensics_report_data(full_chat, rules_text, is_fake, fake_reason)
        print(f"[INFO] LLM返回: riskyStatements={len(report_data.get('riskyStatements', []))}, ruleViolations={len(report_data.get('ruleViolations', []))}")
    except Exception as e:
        print(f"[ERROR] generate_forensics_report_data failed: {e}")
        import traceback; traceback.print_exc()
        report_data = {"riskyStatements": [], "ruleViolations": [], "actionSuggestions": []}
    
    # 构建聊天记录摘录（带时间戳），嵌入报告
    chat_log = []
    for msg in messages:
        role_label = "买家" if msg.sender_role == "buyer" else "商家"
        time_str = msg.created_at.strftime('%H:%M') if msg.created_at else ""
        content_str = msg.content if msg.content_type == "text" else "[图片]"
        chat_log.append({
            "time": time_str,
            "role": role_label,
            "side": "我方" if msg.sender_role == "buyer" else "对方",
            "content": content_str
        })
    report_data["chatLog"] = chat_log

    return jsonify({
        "status": "success",
        "data": report_data
    })



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)  # threaded=True: 流式响应不阻塞其他请求
