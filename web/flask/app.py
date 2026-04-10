from flask import Flask, request, jsonify, send_from_directory, session as flask_session
from flask_cors import CORS
from datetime import timedelta, datetime
import os
import sys
import traceback
import uuid as uuid_mod
from pathlib import Path
from dotenv import load_dotenv

from flask_service.utils import allowed_file, check_session

# ------------------------------------------------------------------
# 环境 & 路径
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / '.env')
load_dotenv(PROJECT_ROOT / 'web' / '.env', override=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# ------------------------------------------------------------------
# Flask app 创建 & 配置
# ------------------------------------------------------------------
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'lare-dev-secret-key-2026')
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_PATH'] = '/'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_REFRESH_EACH_REQUEST'] = False

# 上传文件
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config['UPLOAD_DIR'] = UPLOAD_DIR

# 数据库
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/laft')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from models import db, migrate, User, DisputeSession, Message
db.init_app(app)
migrate.init_app(app, db)

CORS(app, supports_credentials=True, origins=[
    'http://localhost:3000', 'http://127.0.0.1:3000',
    'http://localhost:5173', 'http://127.0.0.1:5173',
    r'https://.*\.ngrok-free\.app',
    r'https://.*\.ngrok\.io',
])

# ------------------------------------------------------------------
# AI 检测管理器 (挂载到 app 上供蓝图访问)
# ------------------------------------------------------------------
from flask_service.laft.manager import LaFTManager
app.laft_manager = LaFTManager(project_root=str(PROJECT_ROOT))

with app.app_context():
    try:
        app.laft_manager.load_models()
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()

# ------------------------------------------------------------------
# 注册蓝图: laft(检测) / vllm(视觉大模型) / rag(RAG助手+报告)
# ------------------------------------------------------------------
from flask_service.laft.routes import laft_bp
from flask_service.vllm.routes import vllm_bp
from flask_service.rag.routes import rag_bp

app.register_blueprint(laft_bp)
app.register_blueprint(vllm_bp)
app.register_blueprint(rag_bp)

# ------------------------------------------------------------------
# 基础路由
# ------------------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return jsonify({"name": "LaRE API", "status": "running", "docs": "/health"})


# ===================================================================
# 认证相关接口
# ===================================================================
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
        flask_session.permanent = True
        flask_session['user_id'] = user.id
        disputes_count = DisputeSession.query.filter(
            (DisputeSession.buyer_id == user.id) | (DisputeSession.seller_id == user.id)
        ).count()
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
    disputes_count = DisputeSession.query.filter(
        (DisputeSession.buyer_id == user.id) | (DisputeSession.seller_id == user.id)
    ).count()
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

    if 'avatar' in request.files:
        file = request.files['avatar']
        if file.filename and allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_name = f"avatar_{uuid_mod.uuid4().hex}.{ext}"
            file.save(os.path.join(UPLOAD_DIR, unique_name))
            user.avatar = f"/api/uploads/{unique_name}"

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


# ===================================================================
# 会话管理
# ===================================================================
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
    if platform not in ['taobao', 'xianyu', 'jd', 'pdd']:
        platform = 'taobao'
    s = DisputeSession(topic_name=topic_name, platform=platform, buyer_id=current_user_id, status='open')
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
    s, err = check_session(session_id, current_user_id, owner_only=True)
    if err:
        return err
    data = request.json or {}
    topic_name = (data.get('topic_name') or '').strip()
    platform = (data.get('platform') or '').strip()
    if topic_name:
        s.topic_name = topic_name
    if platform and platform in ['taobao', 'xianyu', 'jd', 'pdd']:
        s.platform = platform
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
    s, err = check_session(session_id, current_user_id, owner_only=True)
    if err:
        return err
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


# ===================================================================
# 消息
# ===================================================================
@app.route('/sessions/<session_id>/messages', methods=['GET'])
def get_session_messages(session_id):
    current_user_id = flask_session.get('user_id')
    s, err = check_session(session_id, current_user_id)
    if err:
        return err

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


@app.route('/uploads/<path:filename>', methods=['GET'])
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route('/sessions/<session_id>/messages', methods=['POST'])
def send_message(session_id):
    current_user_id = flask_session.get('user_id')
    if not current_user_id:
        return jsonify({"success": False, "message": "请先登录"}), 401
    s, err = check_session(session_id, current_user_id)
    if err:
        return err

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
        content=saved_content,
    )
    db.session.add(msg)
    s.updated_at = datetime.utcnow()
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
            "hasBeenDetected": False,
        }
    })


# ===================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('FLASK_PORT', '5555')), debug=True, threaded=True)
