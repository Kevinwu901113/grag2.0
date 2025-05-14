import sys
from typing import List, Tuple
from llm.prompt import (
    get_entity_extraction_prompt,
    get_relation_extraction_prompt
)
from llm.providers.ollama_strategy import OllamaStrategy
from llm.providers.openai_strategy import OpenAIStrategy

class LLMClient:
    def __init__(self, config: dict):
        self.provider = config["llm"]["provider"]
        self.config = config
        self.strategy = self._load_strategy(self.provider, config)

    def _load_strategy(self, provider: str, config: dict):
        if provider == "ollama":
            return OllamaStrategy(config)
        elif provider == "openai":
            return OpenAIStrategy(config)
        else:
            raise ValueError(f"不支持的 provider: {provider}")

    def generate(self, prompt: str) -> str:
        return self.strategy.generate(prompt)

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.strategy.embed(texts)
        if not embeddings:
            print("[LLMClient.embed] 警告：嵌入返回为空，可能是 LLM 接口调用失败或输入为空")
        elif not isinstance(embeddings[0], list):
            raise ValueError("嵌入返回格式异常，应为二维 list[float]")
        else:
            print(f"[LLMClient.embed] 成功嵌入 {len(embeddings)} 条文本，每条维度 {len(embeddings[0])}")
        return embeddings

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        prompt = get_entity_extraction_prompt(text)
        response = self.generate(prompt)
        return self._parse_entity_response(response)

    def extract_relations(self, text: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
        prompt = get_relation_extraction_prompt(text, entities)
        response = self.generate(prompt)
        return self._parse_relation_response(response)

    def _parse_entity_response(self, text: str) -> List[Tuple[str, str]]:
        lines = text.strip().splitlines()
        pairs = []
        for line in lines:
            line = line.strip().strip("()")
            if "，" not in line:
                continue
            parts = line.split("，")
            if len(parts) == 2:
                ent, ent_type = parts
                pairs.append((ent.strip(), ent_type.strip()))
        return pairs

    def _parse_relation_response(self, text: str) -> List[Tuple[str, str, str]]:
        lines = text.strip().splitlines()
        triples = []
        for line in lines:
            line = line.strip().strip("() ").strip("<END_OF_OUTPUT>")
            if "|" not in line:
                continue
            parts = [p.strip().strip('"') for p in line.split("|")]
            if len(parts) >= 4:
                src = parts[1]
                tgt = parts[2]
                rel = parts[3]
                triples.append((src, rel, tgt))
        return triples
