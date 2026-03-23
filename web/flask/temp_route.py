@app.route("/api/generate_comprehensive_report", methods=["POST"])
def api_generate_comprehensive_report():
    data = request.json
    session_id = data.get("session_id")
    # image_id = data.get("image_id")
    is_fake_str = data.get("is_fake", "true") 
    is_fake = is_fake_str.lower() == "true" if isinstance(is_fake_str, str) else bool(is_fake_str)
    fake_reason = data.get("fake_reason", "")
    
    # 1. Provide chat history
    messages = Message.query.filter_by(session_id=session_id).order_by(Message.timestamp).all()
    session = DisputeSession.query.get(session_id)
    platform = session.platform if session else "taobao"

    chat_context_lines = []
    for msg in messages:
        role_label = "买家" if msg.role == "buyer" else "商家"
        side_label = "【我方】" if msg.role == "buyer" else "【对方】"
        header = f"{side_label}({role_label})"
        if msg.content_type == "text":
            chat_context_lines.append(f"{header}: {msg.content}")
        elif msg.content_type == "image":
            chat_context_lines.append(f"{header}: [发送了一张图片]")
    full_chat = "\n".join(chat_context_lines)
    
    # 2. Get RAG info (use the global _retriever from app.py)
    rag_docs = _retriever.retrieve(f"用户在{platform}平台上发生了纠纷，对方声称商品没问题或平台不管。有哪些相关规定？", platform)
    rules_text = "\n".join([r.get("content", "") for r in rag_docs])
    
    # 3. Call text LLM for rules and risky statements
    try:
        report_data = generate_forensics_report_data(full_chat, rules_text, is_fake, fake_reason)
    except Exception as e:
        print(f"Error in generate_forensics_report_data: {e}")
        report_data = {"riskyStatements": [], "ruleViolations": []}
    
    return jsonify({
        "status": "success",
        "data": report_data
    })

