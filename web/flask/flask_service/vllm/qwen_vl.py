import os
from openai import OpenAI
import requests

class OllamaTimeoutError(Exception):
    pass


class OllamaConnectionError(Exception):
    pass


def _get_ollama_timeout_seconds() -> float:
    raw_value = os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")
    try:
        return max(float(raw_value), 1.0)
    except ValueError:
        return 180.0


def _build_prompts(is_fake: bool):
    """
    统一构建 VL 模型的系统提示词和用户提示词，确保输出精简、结构化、专注辅助判断。
    """
    sys_prompt = (
        "你是一个辅助底层深度学习模型(AIGC检测器)的高级图像物证鉴定专家。\n"
        "由于底层系统已经完成了像素级和频域分析，你的任务是：基于人类视觉和逻辑层面，给出简明、结构化的分析支持。\n"
        "!!请严格控制字数在150字以内，直接给出核心结论，禁止长篇大论!!"
    )
    
    if is_fake:
        user_prompt = (
            "【底层系统状态】：已检测出此图为【AI伪造】。\n"
            "【指令】：请快速观察画面光影、物理空间、人体结构(手/眼/脸)或材质纹理。找出支撑伪造结论的破绽。\n"
            "【输出要求】请使用严格的 Markdown 格式：\n"
            "### 1. 核心判定\n(一句话说明倾向)\n"
            "### 2. 视觉破绽\n- (列出1~3点具体的荒谬/不合理之处)\n"
            "### 3. 协同结论\n(一句话总结：视觉逻辑如何印证底层判定)"
        )
    else:
        user_prompt = (
            "【底层系统状态】：已检测出此图为【真实照片】。\n"
            "【指令】：请快速观察画面物理连贯性、纹理自然度和背景细节，判断是否有未被察觉的异常。\n"
            "【输出要求】请使用严格的 Markdown 格式：\n"
            "### 1. 核心判定\n(一句话说明倾向)\n"
            "### 2. 逻辑分析\n- (列出符合自然规律的特征，或极其微小的疑点)\n"
            "### 3. 协同结论\n(一句话总结：视觉分析结果与底层得出真实判断是否吻合)"
        )
    return sys_prompt, user_prompt

def explain_logic_with_qwen_stream(base64_image: str, is_fake: bool):
    """
    返回一个生成器，用于流式（SSE）返回本地或云端的解释分析结果
    """
    if "," in base64_image:
        base64_image = base64_image.split(",")[1]
        
    sys_prompt, user_prompt = _build_prompts(is_fake)

    provider = os.getenv("VLM_PROVIDER", "ollama").lower()
    
    if provider == "siliconflow":
        return _call_siliconflow_stream(sys_prompt, user_prompt, base64_image)
    else:
        return _call_ollama_stream(sys_prompt, user_prompt, base64_image)

def _call_siliconflow_stream(sys_prompt: str, user_prompt: str, base64_image: str):
    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    base_url = os.getenv("SILICONFLOW_URL", "https://api.siliconflow.cn/v1")
    if base_url.endswith("/chat/completions"):
        base_url = base_url.replace("/chat/completions", "")
        
    model = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")
    timeout_seconds = _get_ollama_timeout_seconds()

    if not api_key or api_key == "your_siliconflow_api_key_here":
        yield " [配置错误] 选择了 siliconflow 作为 VLM_PROVIDER，但未配置 SILICONFLOW_API_KEY。\n"
        return

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        image_url = f"data:image/jpeg;base64,{base64_image}"
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            stream=True,
            timeout=timeout_seconds
        )
        for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except Exception as exc:
        yield f"\n [云端请求失败]: {str(exc)}\n"

def _call_ollama_stream(sys_prompt: str, user_prompt: str, base64_image: str):
    import json
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3-vl:2b")
    timeout_seconds = _get_ollama_timeout_seconds()
    
    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user", 
                "content": user_prompt,
                "images": [base64_image] 
            }
        ],
        "stream": True 
    }

    try:
        with requests.post(ollama_url, json=payload, timeout=timeout_seconds, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            content = data.get('message', {}).get('content', '')
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"\n [Ollama 本地请求失败]: 状态码 {response.status_code}\n"
    except requests.exceptions.ReadTimeout:
        yield f"\n [Ollama 超时]: 请确认模型已加载或增大超时设置。\n"
    except requests.exceptions.ConnectionError:
        yield f"\n [Ollama 连接失败]: 请确认 Ollama 已启动 ({ollama_url})。\n"
    except Exception as exc:
        yield f"\n [Ollama 发生异常]: {str(exc)}\n"


def explain_logic_with_qwen(base64_image: str, is_fake: bool) -> dict:
    """
    调用本地 Ollama 或 硅基流动的 VL 模型给出深度逻辑分析报告
    """
    # 确保没有携带头部信息 (data:image/jpeg;base64,)
    if "," in base64_image:
        base64_image = base64_image.split(",")[1]
        
    sys_prompt, user_prompt = _build_prompts(is_fake)

    provider = os.getenv("VLM_PROVIDER", "ollama").lower()
    
    if provider == "siliconflow":
        return _call_siliconflow(sys_prompt, user_prompt, base64_image)
    else:
        return _call_ollama(sys_prompt, user_prompt, base64_image)


def _call_siliconflow(sys_prompt: str, user_prompt: str, base64_image: str) -> dict:
    import builtins
    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    base_url = os.getenv("SILICONFLOW_URL", "https://api.siliconflow.cn/v1")
    # 支持类似 https://api.siliconflow.cn/v1/chat/completions 形式向下兼容
    if base_url.endswith("/chat/completions"):
        base_url = base_url.replace("/chat/completions", "")
        
    model = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")
    timeout_seconds = _get_ollama_timeout_seconds()

    if not api_key or api_key == "your_siliconflow_api_key_here":
        raise Exception("选择了 siliconflow 作为 VLM_PROVIDER，但未配置 SILICONFLOW_API_KEY。")

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        # 传递 base64 给 OpenAI 兼容协议
        image_url = f"data:image/jpeg;base64,{base64_image}"
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            stream=False,
            timeout=timeout_seconds
        )
        report = response.choices[0].message.content
        return {"report": report}
        
    except Exception as exc:
        raise Exception(f"请求 SiliconFlow 失败: {str(exc)}")


def _call_ollama(sys_prompt: str, user_prompt: str, base64_image: str) -> dict:
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3-vl:2b")
    timeout_seconds = _get_ollama_timeout_seconds()
    
    payload = {
        "model": ollama_model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user", 
                "content": user_prompt,
                "images": [base64_image] 
            }
        ],
        "stream": False 
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=timeout_seconds)
    except requests.exceptions.ReadTimeout as exc:
        raise OllamaTimeoutError(
            f"本地 Ollama 推理超时（>{timeout_seconds:.0f}s）。请确认模型 {ollama_model} 已拉起，或增大 OLLAMA_TIMEOUT_SECONDS。"
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise OllamaConnectionError(
            f"无法连接到本地 Ollama 服务：{ollama_url}。请确认 Ollama 已启动且模型 {ollama_model} 可用。"
        ) from exc
    
    if response.status_code == 200:
        result = response.json()
        return {"report": result['message']['content']}
    else:
        raise Exception(
            f"请求本地 Ollama 失败，状态码: {response.status_code}, 模型: {ollama_model}, 内容: {response.text}"
        )
