#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚类块管理器

负责将聚类结果合并为主题块：
1. 相同聚类标签的小块合并为一个主题块
2. 每个主题块保留 ID、来源文档信息、小块列表
3. 可选支持为每个主题块生成摘要
4. 输出格式与原主题池逻辑兼容

作者: AI Assistant
日期: 2024
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
from tqdm import tqdm

# 导入 LLM 客户端用于摘要生成
try:
    from llm.llm import LLMClient
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMClient = None


class ClusteredBlockManager:
    """
    聚类块管理器
    
    负责将聚类后的文本块合并为主题块，并可选生成摘要
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化聚类块管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 从配置中获取参数
        doc_config = config.get('document_processing', {})
        self.enable_summary = doc_config.get('enable_summary', True)
        self.max_summary_length = doc_config.get('max_summary_length', 200)
        self.min_block_size = doc_config.get('min_block_size', 2)  # 最小块大小（文本块数量）
        self.max_block_size = doc_config.get('max_block_size', 50)  # 最大块大小
        
        # 初始化 LLM 客户端用于摘要生成
        self.llm_client = None
        if self.enable_summary and LLM_AVAILABLE:
            try:
                self.llm_client = LLMClient(config)
                self.logger.info("LLM 客户端初始化成功，将生成主题摘要")
            except Exception as e:
                self.logger.warning(f"LLM 客户端初始化失败: {e}，将跳过摘要生成")
                self.enable_summary = False
        elif self.enable_summary:
            self.logger.warning("LLM 不可用，将跳过摘要生成")
            self.enable_summary = False
        
        self.block_id_counter = 0
        
        self.logger.info(f"聚类块管理器初始化完成")
        self.logger.info(f"摘要生成: {self.enable_summary}, 最小块大小: {self.min_block_size}")
    
    def create_blocks(self, chunks: List[Dict[str, Any]], cluster_labels: np.ndarray, 
                     embeddings: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        根据聚类结果创建主题块
        
        Args:
            chunks: 原始文本块列表
            cluster_labels: 聚类标签数组
            embeddings: 嵌入向量矩阵（可选，用于计算块中心）
            
        Returns:
            主题块列表
        """
        if len(chunks) != len(cluster_labels):
            self.logger.error(f"文本块数量 ({len(chunks)}) 与聚类标签数量 ({len(cluster_labels)}) 不匹配")
            return []
        
        self.logger.info(f"开始创建主题块，输入 {len(chunks)} 个文本块")
        
        # 按聚类标签分组
        cluster_groups = self._group_by_cluster(chunks, cluster_labels)
        
        # 过滤小聚类
        filtered_groups = self._filter_small_clusters(cluster_groups)
        
        # 分割大聚类
        split_groups = self._split_large_clusters(filtered_groups)
        
        # 创建主题块
        blocks = []
        for cluster_id, chunk_group in tqdm(split_groups.items(), desc="创建主题块"):
            block = self._create_single_block(cluster_id, chunk_group, embeddings)
            if block:
                blocks.append(block)
        
        self.logger.info(f"主题块创建完成，共 {len(blocks)} 个块")
        return blocks
    
    def _group_by_cluster(self, chunks: List[Dict[str, Any]], 
                         cluster_labels: np.ndarray) -> Dict[int, List[Dict[str, Any]]]:
        """
        按聚类标签分组文本块
        
        Args:
            chunks: 文本块列表
            cluster_labels: 聚类标签数组
            
        Returns:
            按聚类标签分组的字典
        """
        cluster_groups = defaultdict(list)
        
        for chunk, label in zip(chunks, cluster_labels):
            # 为每个块添加聚类信息
            chunk_with_cluster = chunk.copy()
            chunk_with_cluster['cluster_label'] = int(label)
            cluster_groups[int(label)].append(chunk_with_cluster)
        
        self.logger.debug(f"按聚类分组完成，共 {len(cluster_groups)} 个聚类")
        return dict(cluster_groups)
    
    def _filter_small_clusters(self, cluster_groups: Dict[int, List[Dict[str, Any]]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        过滤掉过小的聚类
        
        Args:
            cluster_groups: 聚类分组字典
            
        Returns:
            过滤后的聚类分组字典
        """
        filtered_groups = {}
        small_clusters = []
        
        for cluster_id, chunks in cluster_groups.items():
            if len(chunks) >= self.min_block_size:
                filtered_groups[cluster_id] = chunks
            else:
                small_clusters.extend(chunks)
        
        # 将小聚类合并为一个特殊聚类
        if small_clusters:
            # 使用负数作为合并聚类的 ID
            merged_cluster_id = -1
            filtered_groups[merged_cluster_id] = small_clusters
            self.logger.info(f"合并了 {len(small_clusters)} 个来自小聚类的文本块")
        
        self.logger.debug(f"过滤小聚类完成，剩余 {len(filtered_groups)} 个聚类")
        return filtered_groups
    
    def _split_large_clusters(self, cluster_groups: Dict[int, List[Dict[str, Any]]]) -> Dict[int, List[Dict[str, Any]]]:
        """
        分割过大的聚类
        
        Args:
            cluster_groups: 聚类分组字典
            
        Returns:
            分割后的聚类分组字典
        """
        split_groups = {}
        split_counter = 0
        
        for cluster_id, chunks in cluster_groups.items():
            if len(chunks) <= self.max_block_size:
                split_groups[cluster_id] = chunks
            else:
                # 分割大聚类
                self.logger.info(f"聚类 {cluster_id} 包含 {len(chunks)} 个块，需要分割")
                
                # 简单的顺序分割策略
                for i in range(0, len(chunks), self.max_block_size):
                    sub_chunks = chunks[i:i + self.max_block_size]
                    if len(sub_chunks) >= self.min_block_size:
                        # 使用原聚类 ID + 分割后缀
                        split_id = f"{cluster_id}_split_{split_counter}"
                        split_groups[split_id] = sub_chunks
                        split_counter += 1
        
        self.logger.debug(f"分割大聚类完成，共 {len(split_groups)} 个聚类")
        return split_groups
    
    def _create_single_block(self, cluster_id: Any, chunks: List[Dict[str, Any]], 
                           embeddings: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        创建单个主题块
        
        Args:
            cluster_id: 聚类 ID
            chunks: 属于该聚类的文本块列表
            embeddings: 嵌入向量矩阵（可选）
            
        Returns:
            主题块字典
        """
        if not chunks:
            return None
        
        try:
            # 生成块 ID
            block_id = f"block_{self.block_id_counter:06d}"
            self.block_id_counter += 1
            
            # 收集基本信息，处理不同数据格式的兼容性
            all_texts = []
            for chunk in chunks:
                if 'text' in chunk:
                    all_texts.append(chunk['text'])
                elif 'sentences' in chunk:
                    # 处理static_chunk_processor生成的格式
                    text = "\n".join(chunk['sentences']) if isinstance(chunk['sentences'], list) else str(chunk['sentences'])
                    all_texts.append(text)
                else:
                    print(f"警告: 块 {chunk.get('id', 'unknown')} 缺少文本内容，跳过处理")
                    continue
            source_files = list(set(chunk['source_file'] for chunk in chunks))
            total_chars = sum(len(text) for text in all_texts)
            
            # 生成摘要
            summary = ""
            if self.enable_summary:
                summary = self._generate_summary(all_texts)
            
            # 计算块中心（如果有嵌入向量）
            block_center = None
            if embeddings is not None:
                try:
                    # 假设 chunks 的顺序与 embeddings 一致
                    chunk_indices = [i for i, chunk in enumerate(chunks) if 'cluster_label' in chunk]
                    if chunk_indices:
                        block_embeddings = embeddings[chunk_indices]
                        block_center = np.mean(block_embeddings, axis=0).tolist()
                except Exception as e:
                    self.logger.warning(f"计算块中心失败: {e}")
            
            # 创建主题块
            block = {
                "id": block_id,
                "cluster_id": cluster_id,
                "chunks": chunks,
                "summary": summary,
                "metadata": {
                    "chunk_count": len(chunks),
                    "total_chars": total_chars,
                    "avg_chars_per_chunk": total_chars / len(chunks),
                    "source_files": source_files,
                    "source_file_count": len(source_files),
                    "processing_method": "static_clustering",
                    "block_center": block_center
                }
            }
            
            self.logger.debug(f"创建主题块 {block_id}，包含 {len(chunks)} 个文本块")
            return block
            
        except Exception as e:
            self.logger.error(f"创建主题块失败: {e}")
            return None
    
    def _generate_summary(self, texts: List[str]) -> str:
        """
        为文本列表生成摘要
        
        Args:
            texts: 文本列表
            
        Returns:
            生成的摘要
        """
        if not self.llm_client or not texts:
            return ""
        
        try:
            # 合并文本
            combined_text = "\n\n".join(texts)
            
            # 如果文本过长，截取前部分
            max_input_length = 2000  # 限制输入长度
            if len(combined_text) > max_input_length:
                combined_text = combined_text[:max_input_length] + "..."
            
            # 构建摘要提示
            prompt = f"""请为以下文本内容生成一个简洁的主题摘要，不超过{self.max_summary_length}字：

{combined_text}

摘要："""
            
            # 调用 LLM 生成摘要
            response = self.llm_client.generate(prompt)
            
            if response and response.strip():
                summary = response.strip()
                # 确保摘要长度不超过限制
                if len(summary) > self.max_summary_length:
                    summary = summary[:self.max_summary_length] + "..."
                self.logger.debug(f"成功生成摘要: {summary[:50]}...")
                return summary
            else:
                self.logger.warning("LLM 返回空摘要")
                return ""
                
        except ConnectionError as e:
            self.logger.error(f"LLM连接失败: {e}")
            return ""
        except ValueError as e:
            self.logger.warning(f"LLM返回空响应: {e}")
            return ""
        except Exception as e:
            self.logger.error(f"生成摘要失败: {e}")
            return ""
    
    def get_block_statistics(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取主题块统计信息
        
        Args:
            blocks: 主题块列表
            
        Returns:
            统计信息字典
        """
        if not blocks:
            return {}
        
        try:
            chunk_counts = [block['metadata']['chunk_count'] for block in blocks]
            char_counts = [block['metadata']['total_chars'] for block in blocks]
            
            stats = {
                "total_blocks": len(blocks),
                "total_chunks": sum(chunk_counts),
                "total_chars": sum(char_counts),
                "avg_chunks_per_block": np.mean(chunk_counts),
                "avg_chars_per_block": np.mean(char_counts),
                "min_chunks_per_block": min(chunk_counts),
                "max_chunks_per_block": max(chunk_counts),
                "blocks_with_summary": sum(1 for block in blocks if block.get('summary', '').strip()),
                "unique_source_files": len(set(
                    file for block in blocks 
                    for file in block['metadata']['source_files']
                ))
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"计算统计信息失败: {e}")
            return {}
    
    def validate_blocks(self, blocks: List[Dict[str, Any]]) -> bool:
        """
        验证主题块的完整性
        
        Args:
            blocks: 主题块列表
            
        Returns:
            验证是否通过
        """
        try:
            for i, block in enumerate(blocks):
                # 检查必需字段
                required_fields = ['id', 'cluster_id', 'chunks', 'metadata']
                for field in required_fields:
                    if field not in block:
                        self.logger.error(f"块 {i} 缺少必需字段: {field}")
                        return False
                
                # 检查块内容
                if not block['chunks']:
                    self.logger.error(f"块 {i} 包含空的文本块列表")
                    return False
                
                # 检查元数据
                metadata = block['metadata']
                if metadata['chunk_count'] != len(block['chunks']):
                    self.logger.error(f"块 {i} 的元数据不一致")
                    return False
            
            self.logger.info(f"主题块验证通过，共 {len(blocks)} 个块")
            return True
            
        except Exception as e:
            self.logger.error(f"主题块验证失败: {e}")
            return False