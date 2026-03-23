import os
from openai import OpenAI
import json
import re


def generate_forensics_report_data(chat_history: str, platform_rules: str, is_fake: bool, fake_reason: str = None) -> dict:
    """调用 LLM 对聊天记录做全局风险定责审查，返回违规言论与平台规则违规项。"""

    detection_result = "伪造/篡改" if is_fake else "真实"
    reason_line = f"\n伪造原因：{fake_reason}" if fake_reason else ""

    prompt = f"""你是一位电商平台纠纷取证专家。以下是一段交易纠纷聊天记录，其中【我方】是买家，【对方】是卖家。
系统已对争议图片进行AI鉴定，结果为：{detection_result}。{reason_line}

平台相关规则：
{platform_rules if platform_rules else "（暂无检索到的平台规则）"}

聊天记录：
{chat_history}

请你完成以下三项分析任务：

任务一：从【对方】（卖家）的发言中，提取所有存在法律或平台风险的言论，包括但不限于：威胁恐吓、承认造假、推卸责任、拒绝退换货、侮辱买家等。
对每条给出：
- context：引发该言论的上文背景（我方说了什么或什么情况下对方说出此话），用简短一句话概括
- quote：对方的原话
- riskLevel：风险等级（high/medium/low）
- comment：风险分析点评

任务二：结合聊天内容和图片鉴定结果，判断对方违反了哪些平台规则，给出规则名称和违规说明。

任务三：基于以上分析结果，给出 3-5 条具体可执行的维权操作建议（如发起退款、平台投诉、客服介入、12315举报等），每条建议包含：
- action：建议操作名称（简短）
- detail：具体操作步骤说明

请严格按以下JSON格式回复，不要输出任何其他文字：
{{
  "riskyStatements": [
    {{"context": "我方指出商品存在问题后", "quote": "对方原话", "riskLevel": "high", "comment": "风险分析"}}
  ],
  "ruleViolations": [
    {{"ruleName": "违反的规则名称", "explanation": "违规说明"}}
  ],
  "actionSuggestions": [
    {{"action": "操作名称", "detail": "具体步骤"}}
  ]
}}"""

    timeout_sec = int(os.getenv("VLM_TIMEOUT_SECONDS", "120"))
    empty_result = {"riskyStatements": [], "ruleViolations": [], "actionSuggestions": []}

    # 策略：优先用 Kimi K2.5 做文本逻辑分析，其次 SiliconFlow 文本模型，最后 Ollama
    kimi_key = os.getenv("KIMI_API_KEY", "")
    if kimi_key:
        return _call_kimi(prompt, kimi_key, timeout_sec, empty_result)

    sf_key = os.getenv("SILICONFLOW_API_KEY", "")
    if sf_key:
        return _call_siliconflow(prompt, sf_key, timeout_sec, empty_result)

    return _call_ollama(prompt, timeout_sec, empty_result)


def _call_kimi(prompt: str, api_key: str, timeout_sec: int, fallback: dict) -> dict:
    """Kimi K2.5 —— 专职文本逻辑分析。K2.5是思考型模型，需要更长超时。"""
    base_url = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
    model = os.getenv("KIMI_MODEL", "kimi-k2.5")
    kimi_timeout = max(timeout_sec, 300)  # K2.5 思考型模型至少给 5 分钟

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=kimi_timeout)
    try:
        print(f"[LLM] 调用 Kimi {model} 进行风险定责审查...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是电商平台纠纷取证AI，必须且只能输出纯 JSON，不要输出任何其他文字。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
        )
        raw = response.choices[0].message.content or ""
        print(f"[LLM] Kimi 返回 {len(raw)} 字符")
        return _parse_json_response(raw, fallback)
    except Exception as e:
        print(f"[LLM] Kimi API error: {e}")
        # Kimi 失败后尝试 SiliconFlow 降级
        sf_key = os.getenv("SILICONFLOW_API_KEY", "")
        if sf_key:
            print("[LLM] 降级到 SiliconFlow...")
            return _call_siliconflow(prompt, sf_key, timeout_sec, fallback)
        return fallback


def _call_siliconflow(prompt: str, api_key: str, timeout_sec: int, fallback: dict) -> dict:
    """硬基流动 SiliconFlow 文本模型。"""
    base_url = os.getenv("SILICONFLOW_URL", "https://api.siliconflow.cn/v1")
    if base_url.endswith("/chat/completions"):
        base_url = base_url.replace("/chat/completions", "")
    model = os.getenv("SILICONFLOW_TEXT_MODEL",
                      os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-7B-Instruct"))

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_sec)
    try:
        print(f"[LLM] 调用 SiliconFlow {model}...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是纠纷取证AI，必须且只能输出JSON。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content or ""
        return _parse_json_response(raw, fallback)
    except Exception as e:
        print(f"[LLM] SiliconFlow API error: {e}")
        return fallback


def _call_ollama(prompt: str, timeout_sec: int, fallback: dict) -> dict:
    """本地 Ollama 回退。"""
    import requests as req_lib
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3-vl:2b")

    payload = {
        "model": ollama_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json"
    }
    try:
        print(f"[LLM] 调用 Ollama {ollama_model}...")
        res = req_lib.post(ollama_url, json=payload, timeout=timeout_sec)
        raw = res.json().get("message", {}).get("content", "")
        return _parse_json_response(raw, fallback)
    except Exception as e:
        print(f"[LLM] Ollama error: {e}")
        return fallback


def _parse_json_response(raw: str, fallback: dict) -> dict:
    """从 LLM 返回文本中提取 JSON，即使被 markdown 代码块包裹也能处理。"""
    raw = raw.strip()
    # 去除 ```json ... ``` 包裹
    m = re.search(r'```(?:json)?\s*(\{.*\})\s*```', raw, re.DOTALL)
    if m:
        raw = m.group(1)
    # 直接尝试找最外层花括号
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start:end + 1])
            if "riskyStatements" in data or "ruleViolations" in data:
                return data
        except json.JSONDecodeError:
            pass
    print(f"[LLM] Failed to parse response: {raw[:300]}")
    return fallback

