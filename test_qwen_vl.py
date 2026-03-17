import base64
import requests
import time

def test_qwen_vl():
    image_path = r"D:\三创\LaRE-main\data\doubao\fack\0001.jpg"
    
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
    print(f"[1] 成功读取图片: {image_path} (大小: {len(base64_image)} bytes)")
    
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": "qwen3-vl:2b",
        "messages": [
            {
                "role": "user",
                "content": "请描述这张图片里的内容。",
                "images": [base64_image]
            }
        ],
        "stream": False
    }
    
    print("[2] 正在向本地 Ollama 发送请求...")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        end_time = time.time()
        print(f"[3] 请求完成，耗时: {end_time - start_time:.2f} 秒")
        
        if response.status_code == 200:
            result = response.json()
            message = result.get('message', {}).get('content', '')
            print("-" * 50)
            print("Qwen3-VL 2B 返回结果:")
            print(message)
            print("-" * 50)
            print(f"总生成 tokens: {result.get('eval_count', 'N/A')} ")
        else:
            print(f"请求失败. 状态码: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"发生异常: {e}")

if __name__ == "__main__":
    test_qwen_vl()
