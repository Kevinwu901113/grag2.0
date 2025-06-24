import sys
from typing import List, Tuple
from functools import lru_cache
import hashlib
from llm.prompt import (
    get_entity_extraction_prompt,
    get_relation_extraction_prompt
)
from llm.providers.ollama_strategy import OllamaStrategy
from llm.providers.openai_strategy import OpenAIStrategy
from llm.providers.huggingface_strategy import HuggingFaceStrategy

class LLMClient:
    def __init__(self, config: dict):
        # 如果直接传入了llm配置，则直接使用；否则尝试从config中获取llm字段
        self.gen_config = config if "provider" in config else config.get("llm", {})
        self.embed_config = config.get("embedding", self.gen_config)

        self.strategy = self._load_strategy(self.gen_config.get("provider"), self.gen_config)
        self.embed_strategy = self._load_strategy(self.embed_config.get("provider"), config)
        
        # 嵌入缓存配置
        self.enable_embed_cache = config.get("embedding", {}).get("cache_embeddings", True)
        self.embed_cache_size = config.get("embedding", {}).get("cache_size", 10000)
        
        # 初始化嵌入缓存字典
        self._embed_cache = {} if self.enable_embed_cache else None

    def _load_strategy(self, provider: str, config: dict):
        if provider == "ollama":
            return OllamaStrategy(config)
        elif provider == "openai":
            return OpenAIStrategy(config)
        elif provider == "huggingface":
            return HuggingFaceStrategy(config)
        else:
            raise ValueError(f"不支持的 provider: {provider}")

    def generate(self, prompt: str) -> str:
        return self.strategy.generate(prompt)

    def _get_text_hash(self, text: str) -> str:
        """生成文本的哈希值作为缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _manage_cache_size(self):
        """管理缓存大小，使用LRU策略"""
        if self._embed_cache and len(self._embed_cache) > self.embed_cache_size:
            # 简单的LRU实现：删除最旧的条目
            oldest_key = next(iter(self._embed_cache))
            del self._embed_cache[oldest_key]
    
    def embed(self, texts, normalize_text: bool = True, validate_dim: bool = True) -> List[List[float]]:
        """
        统一的嵌入向量生成接口，支持缓存机制
        
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
        processed_texts = texts
        if normalize_text:
            processed_texts = [text_normalizer(text) for text in texts]
            # 过滤空文本
            processed_texts = [text for text in processed_texts if text.strip()]
            if not processed_texts:
                raise ValueError("预处理后文本列表为空")
        
        # 缓存检查和处理
        embeddings = []
        texts_to_embed = []
        cache_indices = []
        
        if self.enable_embed_cache and self._embed_cache is not None:
            for i, text in enumerate(processed_texts):
                text_hash = self._get_text_hash(text)
                if text_hash in self._embed_cache:
                    # 缓存命中
                    embeddings.append(self._embed_cache[text_hash])
                    cache_indices.append(i)
                else:
                    # 缓存未命中，需要计算
                    texts_to_embed.append(text)
            
            cache_hits = len(cache_indices)
            total_texts = len(processed_texts)
            if cache_hits > 0:
                print(f"[LLMClient.embed] 缓存命中 {cache_hits}/{total_texts} 条文本")
        else:
            texts_to_embed = processed_texts
        
        # 对未缓存的文本进行嵌入计算
        if texts_to_embed:
            try:
                new_embeddings = self.embed_strategy.embed(texts_to_embed)
            except Exception as e:
                raise RuntimeError(f"嵌入生成失败: {e}")
            
            if not new_embeddings:
                raise RuntimeError("嵌入返回为空，LLM接口调用失败")
            
            if not isinstance(new_embeddings[0], list):
                raise ValueError("嵌入返回格式异常，应为二维 list[float]")
            
            # 将新计算的嵌入添加到缓存
            if self.enable_embed_cache and self._embed_cache is not None:
                embed_idx = 0
                for i, text in enumerate(processed_texts):
                    if i not in cache_indices:
                        text_hash = self._get_text_hash(text)
                        self._embed_cache[text_hash] = new_embeddings[embed_idx]
                        self._manage_cache_size()  # 管理缓存大小
                        embed_idx += 1
            
            # 合并缓存和新计算的嵌入结果
            if self.enable_embed_cache and cache_indices:
                # 重新排序，保持原始顺序
                final_embeddings = []
                cache_idx = 0
                new_idx = 0
                for i in range(len(processed_texts)):
                    if i in cache_indices:
                        final_embeddings.append(embeddings[cache_idx])
                        cache_idx += 1
                    else:
                        final_embeddings.append(new_embeddings[new_idx])
                        new_idx += 1
                embeddings = final_embeddings
            else:
                embeddings = new_embeddings
        
        # 向量维度验证
        if validate_dim and embeddings:
            expected_dim = len(embeddings[0]) if embeddings else None
            for i, embedding in enumerate(embeddings):
                if not validate_embedding_dimension(embedding, expected_dim):
                    raise ValueError(f"第{i}个嵌入向量维度验证失败，期望维度: {expected_dim}，实际维度: {len(embedding) if embedding else 0}")
        
        computed_count = len(texts_to_embed)
        cached_count = len(processed_texts) - computed_count
        print(f"[LLMClient.embed] 成功处理 {len(embeddings)} 条文本 (新计算: {computed_count}, 缓存: {cached_count})，每条维度 {len(embeddings[0]) if embeddings else 0}")
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
