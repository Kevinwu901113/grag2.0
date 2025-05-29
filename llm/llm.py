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
        # 如果直接传入了llm配置，则直接使用；否则尝试从config中获取llm字段
        self.gen_config = config if "provider" in config else config.get("llm", {})
        self.embed_config = config.get("embedding", self.gen_config)

        self.strategy = self._load_strategy(self.gen_config.get("provider"), self.gen_config)
        self.embed_strategy = self._load_strategy(self.embed_config.get("provider"), self.embed_config)

    def _load_strategy(self, provider: str, config: dict):
        if provider == "ollama":
            return OllamaStrategy(config)
        elif provider == "openai":
            return OpenAIStrategy(config)
        else:
            raise ValueError(f"不支持的 provider: {provider}")

    def generate(self, prompt: str) -> str:
        return self.strategy.generate(prompt)

    def embed(self, texts, normalize_text: bool = True, validate_dim: bool = True) -> List[List[float]]:
        """
        统一的嵌入向量生成接口
        
        Args:
            texts: 文本或文本列表，支持 str 或 List[str]
            normalize_text: 是否进行文本预处理
            validate_dim: 是否验证向量维度
            
        Returns:
            嵌入向量列表，格式为 List[List[float]]
        """
        from utils.common import normalize_text as text_normalizer, validate_embedding_dimension
        
        # 统一处理输入格式
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise ValueError("输入必须是字符串或字符串列表")
            
        if not texts:
            raise ValueError("输入文本列表为空")
        
        # 文本预处理
        if normalize_text:
            texts = [text_normalizer(text) for text in texts]
            # 过滤空文本
            texts = [text for text in texts if text.strip()]
            if not texts:
                raise ValueError("预处理后文本列表为空")
        
        try:
            embeddings = self.embed_strategy.embed(texts)
        except Exception as e:
            raise RuntimeError(f"嵌入生成失败: {e}")
        
        if not embeddings:
            raise RuntimeError("嵌入返回为空，LLM接口调用失败")
        
        if not isinstance(embeddings[0], list):
            raise ValueError("嵌入返回格式异常，应为二维 list[float]")
        
        # 向量维度验证
        if validate_dim:
            expected_dim = len(embeddings[0]) if embeddings else None
            for i, embedding in enumerate(embeddings):
                if not validate_embedding_dimension(embedding, expected_dim):
                    raise ValueError(f"第{i}个嵌入向量维度验证失败，期望维度: {expected_dim}，实际维度: {len(embedding) if embedding else 0}")
        
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
