import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

# 显式加载当前目录下的 .env
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

def test_siliconflow():
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not api_key or "your_siliconflow_api_key_here" in api_key:
        print(" 错误：请先在 .env 文件中填入真实的 SILICONFLOW_API_KEY。")
        return

    base_url = os.environ.get("SILICONFLOW_URL", "https://api.siliconflow.cn/v1")
    if base_url.endswith("/chat/completions"):
        base_url = base_url.replace("/chat/completions", "")
        
    model = os.environ.get("SILICONFLOW_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")

    print(f"📌 Base URL: {base_url}")
    print(f"📌 模型: {model}")
    print(f"🔑 API Key: {api_key[:5]}...{api_key[-4:] if len(api_key)>8 else ''}")

    image_path = r"D:\三创\LaRE-main\data\doubao\fack\0001.jpg"
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    print("\n[1] 图片加载成功，准备向云端发送请求 (可能需要几秒到十几秒)......")

    client = OpenAI(api_key=api_key, base_url=base_url)
    image_url = f"data:image/jpeg;base64,{base64_image}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "这是一个测试请求。请用一句话简要描述这张图片的主体内容。"},
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
            timeout=60
        )
        print("\n 请求成功！SiliconFlow 响应结果:")
        print("-" * 50)
        print(response.choices[0].message.content)
        print("-" * 50)
    except Exception as e:
        print(f"\n 测试失败，抛出异常: {e}")

if __name__ == "__main__":
    test_siliconflow()
