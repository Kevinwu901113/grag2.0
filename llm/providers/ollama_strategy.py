import requests
import json

class OllamaStrategy:
    def __init__(self, config: dict):
        self.model = config["model_name"]
        self.host = config["host"]
        self.options = config.get("options", {})
        self.embedding_model = config.get("embedding_model", self.model)

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": self.options,
        }
        response = requests.post(f"{self.host}/api/generate", json=payload, stream=True)
        full_text = ""

        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    full_text += data.get("response", "")
                except json.JSONDecodeError as e:
                    print("[解析失败行]", line)
        except (requests.RequestException, ConnectionError) as e:
            print(f"[Ollama 请求失败] 网络错误: {e}")
        except Exception as e:
            print(f"[Ollama 请求失败] 未知错误: {e}")

        return full_text.strip()

    def embed(self, texts):
        embeddings = []
        
        # 直接使用配置的嵌入模型
        model_to_use = self.embedding_model
        
        for i, text in enumerate(texts):
            # 检查文本有效性
            if not text or not text.strip():
                print(f"[Ollama embed] 跳过空文本 (索引: {i})")
                continue
                
            payload = {
                "model": model_to_use,
                "prompt": text,
                "stream": False,
                "options": self.options,
            }
            
            # 打印调试信息（减少冗余输出）
            if i == 0 or i % 10 == 0:  # 只在第一个和每10个文本时打印
                print(f"[Ollama embed] 使用模型: {model_to_use} 处理第 {i+1}/{len(texts)} 条文本")
            
            try:
                r = requests.post(f"{self.host}/api/embeddings", json=payload, timeout=30)
                r.raise_for_status()  # 检查HTTP状态码
            except requests.exceptions.Timeout:
                print(f"[Ollama embed] 请求超时 (文本索引: {i})")
                raise RuntimeError(f"嵌入请求超时: 文本索引 {i}")
            except requests.exceptions.RequestException as e:
                print(f"[Ollama embed] 网络请求失败 (文本索引: {i}): {e}")
                raise RuntimeError(f"嵌入网络请求失败: {e}")

            try:
                result = r.json()
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[Ollama embed] JSON解析失败 (文本索引: {i}): {e}, 响应内容: {r.text[:200]}...")
                raise RuntimeError(f"嵌入响应解析失败: {e}")
            except Exception as e:
                print(f"[Ollama embed] 未知错误 (文本索引: {i}): {e}, 响应内容: {r.text[:200]}...")
                raise RuntimeError(f"嵌入处理未知错误: {e}")

            embedding = result.get("embedding")
            if not embedding:
                print(f"[Ollama embed] 文本返回空嵌入 (索引: {i}), 响应: {r.text[:200]}...")
                raise RuntimeError(f"嵌入返回为空: 文本索引 {i}")
            
            # 基本的向量有效性检查
            if not isinstance(embedding, list) or len(embedding) == 0:
                raise RuntimeError(f"嵌入格式无效: 文本索引 {i}")

            embeddings.append(embedding)

        if not embeddings:
            raise RuntimeError("所有文本的嵌入生成都失败")
            
        return embeddings
