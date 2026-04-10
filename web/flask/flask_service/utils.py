"""Shared utility functions for Flask service modules."""
import uuid as _uuid
from flask import jsonify
from PIL import Image, ImageOps

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_uploaded_image(file_storage) -> Image.Image:
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    return img.convert('RGB')


def check_session(session_id_str, user_id=None, owner_only=False):
    """Validate and retrieve a DisputeSession.

    Returns (session, error_tuple). error_tuple is None on success.
    """
    from models import DisputeSession

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
