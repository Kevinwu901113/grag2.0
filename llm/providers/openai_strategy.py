from typing import List
from openai import OpenAI

class OpenAIStrategy:
    def __init__(self, config: dict):
        self.model_name = config["model_name"]
        self.embedding_model = config.get("embedding_model", self.model_name)
        self.client = OpenAI(
            api_key=config["openai_api_key"],
            base_url=config.get("openai_base_url", "https://api.openai.com/v1")
        )

    def generate(self, prompt: str) -> str:
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content

    def embed(self, texts: List[str]) -> List[List[float]]:
        res = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [e.embedding for e in res.data]
