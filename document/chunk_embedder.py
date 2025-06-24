#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本块嵌入器

负责使用 HuggingFace 模型对文本块进行批量嵌入：
1. 支持 GPU 运行和批量处理
2. 嵌入后进行 L2 归一化
3. 与 config.yaml 中 embedding 部分保持兼容
4. 支持嵌入缓存以提高性能

作者: AI Assistant
日期: 2024
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
from sklearn.preprocessing import normalize

from embedding.hf_embedder import HuggingFaceEmbedder


class ChunkEmbedder:
    """
    文本块嵌入器
    
    使用 HuggingFace 模型对文本块进行批量嵌入处理
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化文本块嵌入器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 从配置中获取嵌入参数
        embedding_config = config.get('embedding', {})
        self.model_name = embedding_config.get('model_name', 'BAAI/bge-m3')
        self.device = embedding_config.get('device', 'auto')
        self.batch_size = embedding_config.get('batch_size', 32)
        self.normalize = embedding_config.get('normalize', True)
        self.local_files_only = embedding_config.get('local_files_only', False)
        
        # 缓存配置
        self.cache_embeddings = embedding_config.get('cache_embeddings', True)
        self.cache_size = embedding_config.get('cache_size', 10000)
        
        # 初始化嵌入器
        try:
            self.embedder = HuggingFaceEmbedder(
                model_name=self.model_name,
                device=self.device,
                batch_size=self.batch_size,
                local_files_only=self.local_files_only
            )
            self.logger.info(f"嵌入器初始化成功: {self.model_name}")
        except Exception as e:
            self.logger.error(f"嵌入器初始化失败: {e}")
            raise
        
        # 嵌入缓存
        self.embedding_cache = {} if self.cache_embeddings else None
        
        self.logger.info(f"文本块嵌入器初始化完成")
        self.logger.info(f"模型: {self.model_name}, 设备: {self.device}, 批量大小: {self.batch_size}")
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """
        对文本块列表进行批量嵌入
        
        Args:
            chunks: 文本块列表，每个块包含 'text' 字段
            
        Returns:
            嵌入向量矩阵，形状为 (n_chunks, embedding_dim)
            如果失败则返回 None
        """
        if not chunks:
            self.logger.warning("输入文本块列表为空")
            return None
        
        self.logger.info(f"开始对 {len(chunks)} 个文本块进行批量嵌入")
        
        try:
            # 提取文本内容，处理不同数据格式的兼容性
            texts = []
            for chunk in chunks:
                if 'text' in chunk:
                    texts.append(chunk['text'])
                elif 'sentences' in chunk:
                    # 处理static_chunk_processor生成的格式
                    text = "\n".join(chunk['sentences']) if isinstance(chunk['sentences'], list) else str(chunk['sentences'])
                    texts.append(text)
                else:
                    self.logger.warning(f"块 {chunk.get('id', 'unknown')} 缺少文本内容，跳过处理")
                    continue
            
            # 检查缓存
            if self.cache_embeddings:
                cached_embeddings, uncached_indices, uncached_texts = self._check_cache(texts)
            else:
                cached_embeddings = None
                uncached_indices = list(range(len(texts)))
                uncached_texts = texts
            
            # 对未缓存的文本进行嵌入
            if uncached_texts:
                self.logger.info(f"需要计算 {len(uncached_texts)} 个新嵌入")
                new_embeddings = self._compute_embeddings(uncached_texts)
                
                if new_embeddings is None:
                    self.logger.error("嵌入计算失败")
                    return None
                
                # 更新缓存
                if self.cache_embeddings:
                    self._update_cache(uncached_texts, new_embeddings)
            else:
                new_embeddings = np.array([])
            
            # 合并缓存和新计算的嵌入
            if cached_embeddings is not None:
                all_embeddings = self._merge_embeddings(
                    cached_embeddings, new_embeddings, uncached_indices, len(texts)
                )
            else:
                all_embeddings = new_embeddings
            
            # L2 归一化
            if self.normalize and all_embeddings is not None:
                all_embeddings = normalize(all_embeddings, norm='l2')
                self.logger.debug("嵌入向量已进行 L2 归一化")
            
            self.logger.info(f"批量嵌入完成，形状: {all_embeddings.shape}")
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"批量嵌入失败: {e}")
            return None
    
    def _compute_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        计算文本列表的嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量矩阵
        """
        try:
            # 批量处理
            all_embeddings = []
            
            for i in tqdm(range(0, len(texts), self.batch_size), desc="计算嵌入"):
                batch_texts = texts[i:i + self.batch_size]
                
                # 调用 HuggingFace 嵌入器
                batch_embeddings = self.embedder.encode(batch_texts)
                
                if batch_embeddings is None:
                    self.logger.error(f"批次 {i//self.batch_size + 1} 嵌入失败")
                    return None
                
                # 转换为 numpy 数组
                if isinstance(batch_embeddings, list):
                    batch_embeddings = np.array(batch_embeddings)
                
                all_embeddings.append(batch_embeddings)
            
            # 合并所有批次的嵌入
            if all_embeddings:
                final_embeddings = np.vstack(all_embeddings)
                self.logger.debug(f"嵌入计算完成，形状: {final_embeddings.shape}")
                return final_embeddings
            else:
                self.logger.error("没有成功计算任何嵌入")
                return None
                
        except Exception as e:
            self.logger.error(f"嵌入计算过程中出错: {e}")
            return None
    
    def _check_cache(self, texts: List[str]) -> tuple:
        """
        检查嵌入缓存
        
        Args:
            texts: 文本列表
            
        Returns:
            (cached_embeddings, uncached_indices, uncached_texts)
        """
        if not self.embedding_cache:
            return None, list(range(len(texts))), texts
        
        cached_embeddings = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            if text_hash in self.embedding_cache:
                cached_embeddings.append(self.embedding_cache[text_hash])
            else:
                cached_embeddings.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        self.logger.debug(f"缓存命中: {len(texts) - len(uncached_texts)}/{len(texts)}")
        
        return cached_embeddings, uncached_indices, uncached_texts
    
    def _update_cache(self, texts: List[str], embeddings: np.ndarray) -> None:
        """
        更新嵌入缓存
        
        Args:
            texts: 文本列表
            embeddings: 对应的嵌入向量
        """
        if not self.cache_embeddings or embeddings is None:
            return
        
        for text, embedding in zip(texts, embeddings):
            text_hash = self._get_text_hash(text)
            
            # 如果缓存已满，删除一些旧条目
            if len(self.embedding_cache) >= self.cache_size:
                # 简单的 FIFO 策略
                keys_to_remove = list(self.embedding_cache.keys())[:len(self.embedding_cache) // 4]
                for key in keys_to_remove:
                    del self.embedding_cache[key]
            
            self.embedding_cache[text_hash] = embedding
        
        self.logger.debug(f"缓存已更新，当前大小: {len(self.embedding_cache)}")
    
    def _merge_embeddings(self, cached_embeddings: List, new_embeddings: np.ndarray, 
                         uncached_indices: List[int], total_count: int) -> np.ndarray:
        """
        合并缓存的嵌入和新计算的嵌入
        
        Args:
            cached_embeddings: 缓存的嵌入列表（包含 None）
            new_embeddings: 新计算的嵌入矩阵
            uncached_indices: 未缓存文本的索引列表
            total_count: 总文本数量
            
        Returns:
            合并后的嵌入矩阵
        """
        if len(new_embeddings) == 0:
            # 全部来自缓存
            return np.array([emb for emb in cached_embeddings if emb is not None])
        
        # 获取嵌入维度
        embedding_dim = new_embeddings.shape[1]
        result = np.zeros((total_count, embedding_dim))
        
        # 填入新计算的嵌入
        new_idx = 0
        for i in uncached_indices:
            result[i] = new_embeddings[new_idx]
            new_idx += 1
        
        # 填入缓存的嵌入
        for i, cached_emb in enumerate(cached_embeddings):
            if cached_emb is not None:
                result[i] = cached_emb
        
        return result
    
    def _get_text_hash(self, text: str) -> str:
        """
        获取文本的哈希值用于缓存键
        
        Args:
            text: 文本内容
            
        Returns:
            文本哈希值
        """
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_embedding_dimension(self) -> Optional[int]:
        """
        获取嵌入向量的维度
        
        Returns:
            嵌入维度，如果未知则返回 None
        """
        try:
            if hasattr(self.embedder, 'embedding_dim') and self.embedder.embedding_dim:
                return self.embedder.embedding_dim
            
            # 通过测试文本获取维度
            test_embedding = self.embedder.encode(["测试文本"])
            if test_embedding is not None:
                return test_embedding.shape[1]
            
            return None
        except Exception as e:
            self.logger.error(f"获取嵌入维度失败: {e}")
            return None
    
    def clear_cache(self) -> None:
        """
        清空嵌入缓存
        """
        if self.embedding_cache:
            self.embedding_cache.clear()
            self.logger.info("嵌入缓存已清空")