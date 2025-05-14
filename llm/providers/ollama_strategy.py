import requests
import json

class OllamaStrategy:
    def __init__(self, config: dict):
        self.model = config["llm"]["model_name"]
        self.host = config["llm"]["host"]
        self.options = config["llm"].get("options", {})
        self.config = config  # ⬅️ 添加 config 以便获取 embedding_model

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
        except Exception as e:
            print("[Ollama 请求失败]", e)

        return full_text.strip()

    def embed(self, texts):
        model = self.config["llm"].get("embedding_model", self.model)
        embeddings = []

        for text in texts:
            payload = {
                "model": model,
                "prompt": text,
                "stream": False,
                "options": self.options,
            }
            r = requests.post(f"{self.host}/api/embeddings", json=payload)

            try:
                result = r.json()
            except Exception as e:
                print("[Ollama embed] 无法解析响应:", r.text)
                raise e

            embedding = result.get("embedding")
            if not embedding:
                print("[Ollama embed] 某条文本返回为空，响应原文:", r.text)
                continue

            embeddings.append(embedding)

        return embeddings
