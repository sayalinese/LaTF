"""Blueprint: RAG 增强路由 (AI 助手聊天 + 综合鉴定报告)"""
import os
import traceback
from flask import Blueprint, request, jsonify, Response, stream_with_context, session as flask_session

from flask_service.utils import check_session

rag_bp = Blueprint('rag', __name__)


# ------------------------------------------------------------------
# 可选依赖的懒加载包装
# ------------------------------------------------------------------
def _get_rag():
    """Return (get_retriever, build_rag_system_prompt) or (None, None)."""
    try:
        from flask_service.rag.retriever import get_retriever
        from flask_service.rag.prompts import build_rag_system_prompt
        return get_retriever, build_rag_system_prompt
    except ImportError:
        return None, None


def _get_llm_report():
    """Return generate_forensics_report_data or None."""
    try:
        from flask_service.vllm.llm import generate_forensics_report_data
        return generate_forensics_report_data
    except ImportError:
        return None


# ------------------------------------------------------------------
# 悬浮 AI 小助手：基于会话上下文的智能分析
# ------------------------------------------------------------------
@rag_bp.route('/sessions/<session_id>/assistant_chat', methods=['POST'])
def assistant_chat(session_id):
    from models import User, Message

    current_user_id = flask_session.get('user_id')
    if not current_user_id:
        return jsonify({"success": False, "message": "请先登录"}), 401

    s, err = check_session(session_id, current_user_id)
    if err:
        return err

    data = request.json or {}
    user_message = (data.get('message') or '').strip()
    if not user_message:
        return jsonify({"success": False, "message": "消息不能为空"}), 400

    history = data.get('history', [])
    rag_platform = (data.get('platform') or s.platform or 'taobao').strip()

    # 角色判定
    if s.buyer_id == current_user_id:
        user_role, opponent_role = '买家', '卖家'
    else:
        user_role, opponent_role = '卖家', '买家'

    # 构建聊天上下文
    chat_messages = Message.query.filter_by(session_id=s.id).order_by(Message.created_at.asc()).all()
    chat_context_lines = []
    for m in chat_messages[-30:]:
        sender = User.query.get(m.sender_id)
        name = sender.nickname if sender else m.sender_id
        role_label = '买家' if m.sender_id == s.buyer_id else '卖家'
        side_label = '【我方】' if m.sender_id == current_user_id else '【对方】'
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

    # RAG 增强
    get_retriever, build_rag_system_prompt = _get_rag()
    if get_retriever is not None:
        _retriever = get_retriever()
        if _retriever is not None:
            rag_docs = _retriever.retrieve(user_message, platform=rag_platform, k=3)
            sys_prompt = build_rag_system_prompt(sys_prompt, rag_docs, platform_hint=rag_platform)

    # 构建 LLM 调用消息
    api_messages = [{"role": "system", "content": sys_prompt}]
    for h in history[-10:]:
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
                model=model, messages=api_messages, stream=True, timeout=60,
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


# ------------------------------------------------------------------
# 综合鉴定报告
# ------------------------------------------------------------------
@rag_bp.route("/api/generate_comprehensive_report", methods=["POST"])
@rag_bp.route("/generate_comprehensive_report", methods=["POST"])
def api_generate_comprehensive_report():
    from models import Message, DisputeSession

    data = request.json
    session_id = data.get("session_id")
    is_fake_str = data.get("is_fake", "true")
    is_fake = is_fake_str.lower() == "true" if isinstance(is_fake_str, str) else bool(is_fake_str)
    fake_reason = data.get("fake_reason", "")

    # 1. 获取聊天记录
    messages = Message.query.filter_by(session_id=session_id).order_by(Message.created_at.asc()).all()
    session_obj = DisputeSession.query.get(session_id)
    platform = session_obj.platform if session_obj else "taobao"

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

    # 2. RAG 检索平台规则
    rules_text = ""
    get_retriever, _ = _get_rag()
    if get_retriever is not None:
        try:
            retriever = get_retriever()
            if retriever is not None:
                rag_docs = retriever.retrieve(
                    f"用户在{platform}平台上发生了纠纷，对方声称商品没问题或平台不管。有哪些相关规定？",
                    platform,
                )
                rules_text = "\n".join([r.get("content", "") for r in rag_docs])
        except Exception as e:
            print(f"[WARN] RAG retrieval failed: {e}")

    # 3. LLM 风险定责分析
    _gen_report = _get_llm_report()
    if _gen_report is None:
        _gen_report = lambda *a, **kw: {"riskyStatements": [], "ruleViolations": [], "actionSuggestions": []}

    print(f"[INFO] 综合报告: session={session_id}, chat_lines={len(chat_context_lines)}, rules_len={len(rules_text)}, is_fake={is_fake}")
    try:
        report_data = _gen_report(full_chat, rules_text, is_fake, fake_reason)
        print(f"[INFO] LLM返回: riskyStatements={len(report_data.get('riskyStatements', []))}, ruleViolations={len(report_data.get('ruleViolations', []))}")
    except Exception as e:
        print(f"[ERROR] generate_forensics_report_data failed: {e}")
        traceback.print_exc()
        report_data = {"riskyStatements": [], "ruleViolations": [], "actionSuggestions": []}

    # 4. 构建聊天记录摘录
    chat_log = []
    for msg in messages:
        role_label = "买家" if msg.sender_role == "buyer" else "商家"
        time_str = msg.created_at.strftime('%H:%M') if msg.created_at else ""
        content_str = msg.content if msg.content_type == "text" else "[图片]"
        chat_log.append({
            "time": time_str,
            "role": role_label,
            "side": "我方" if msg.sender_role == "buyer" else "对方",
            "content": content_str,
        })
    report_data["chatLog"] = chat_log

    return jsonify({"status": "success", "data": report_data})
