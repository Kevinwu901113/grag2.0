#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
静态批量文档处理器

该模块实现了新的文档处理方案：
1. 文档预切分为小块（chunk）
2. 批量嵌入所有小块
3. 对嵌入向量进行聚类
4. 将聚类结果合并为主题块

相比原有的主题池机制，该方案采用静态批量处理，
可以显著提升处理速度并减少随文档变大而出现的性能下降。

作者: AI Assistant
日期: 2024
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .chunk_splitter import ChunkSplitter
from .chunk_embedder import ChunkEmbedder
from .clusterer import Clusterer
from .clustered_block import ClusteredBlockManager
from utils.config_manager import ConfigManager
from utils.performance_monitor import performance_monitor


class StaticChunkProcessor:
    """
    静态批量文档处理器
    
    负责整个静态批量处理流程的控制和协调：
    1. 文档切分
    2. 批量嵌入
    3. 聚类分析
    4. 结果合并
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化静态批量处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 从配置中获取参数
        doc_config = config.get('document_processing', {})
        self.chunk_length = doc_config.get('chunk_length', 200)
        self.cluster_count = doc_config.get('cluster_count', 50)
        self.enable_summary = doc_config.get('enable_summary', True)
        
        # 初始化各个组件
        self.chunk_splitter = ChunkSplitter(config)
        self.chunk_embedder = ChunkEmbedder(config)
        self.clusterer = Clusterer(config)
        self.block_manager = ClusteredBlockManager(config)
        
        self.logger.info(f"静态批量处理器初始化完成")
        self.logger.info(f"块长度: {self.chunk_length}, 聚类数: {self.cluster_count}")
    
    @performance_monitor(include_args=True, include_memory=True, include_cpu=True)
    def process_documents(self, input_files: List[str], output_dir: str) -> Dict[str, Any]:
        """
        处理文档列表
        
        Args:
            input_files: 输入文件路径列表
            output_dir: 输出目录
            
        Returns:
            处理结果统计信息
        """
        self.logger.info(f"开始静态批量处理 {len(input_files)} 个文档")
        
        # 步骤1: 文档切分
        self.logger.info("步骤1: 文档切分")
        all_chunks = []
        
        for file_path in tqdm(input_files, desc="切分文档"):
            try:
                chunks = self.chunk_splitter.split_document(file_path)
                all_chunks.extend(chunks)
                self.logger.debug(f"文档 {file_path} 切分为 {len(chunks)} 个块")
            except Exception as e:
                self.logger.error(f"切分文档 {file_path} 失败: {e}")
                continue
        
        self.logger.info(f"总共切分得到 {len(all_chunks)} 个文本块")
        
        if not all_chunks:
            self.logger.warning("没有有效的文本块，处理结束")
            return {"total_chunks": 0, "total_blocks": 0}
        
        # 步骤2: 批量嵌入
        self.logger.info("步骤2: 批量嵌入")
        embeddings = self.chunk_embedder.embed_chunks(all_chunks)
        
        if embeddings is None or len(embeddings) == 0:
            self.logger.error("嵌入生成失败")
            return {"total_chunks": len(all_chunks), "total_blocks": 0}
        
        # 步骤3: 聚类分析
        self.logger.info("步骤3: 聚类分析")
        cluster_labels = self.clusterer.cluster_embeddings(embeddings)
        
        if cluster_labels is None:
            self.logger.error("聚类分析失败")
            return {"total_chunks": len(all_chunks), "total_blocks": 0}
        
        # 步骤4: 合并为主题块
        self.logger.info("步骤4: 合并为主题块")
        clustered_blocks = self.block_manager.create_blocks(
            all_chunks, cluster_labels, embeddings
        )
        
        # 步骤5: 保存结果
        self.logger.info("步骤5: 保存结果")
        self._save_results(clustered_blocks, output_dir)
        
        # 统计信息
        stats = {
            "total_chunks": len(all_chunks),
            "total_blocks": len(clustered_blocks),
            "unique_clusters": len(set(cluster_labels)),
            "avg_chunks_per_block": len(all_chunks) / len(clustered_blocks) if clustered_blocks else 0
        }
        
        self.logger.info(f"静态批量处理完成: {stats}")
        return stats
    
    def _save_results(self, clustered_blocks: List[Dict[str, Any]], output_dir: str) -> None:
        """
        保存处理结果
        
        Args:
            clustered_blocks: 聚类后的主题块列表
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为JSONL格式，与原主题池逻辑兼容
        output_file = os.path.join(output_dir, "clustered_blocks.jsonl")
        
        # 准备兼容格式的块列表
        compatible_blocks = []
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for block in clustered_blocks:
                # 转换为与原主题池兼容的格式
                compatible_block = {
                    "id": block["id"],
                    "sentences": [chunk["text"] for chunk in block["chunks"]],
                    "source_files": list(set(chunk["source_file"] for chunk in block["chunks"])),
                    "cluster_id": block["cluster_id"],
                    "summary": block.get("summary", ""),
                    "metadata": {
                        "chunk_count": len(block["chunks"]),
                        "total_chars": sum(len(chunk["text"]) for chunk in block["chunks"]),
                        "processing_method": "static_clustering"
                    }
                }
                compatible_blocks.append(compatible_block)
                f.write(json.dumps(compatible_block, ensure_ascii=False) + "\n")
        
        self.logger.info(f"结果已保存到: {output_file}")
        
        # 同时保存为chunks.json格式，确保与其他模块兼容
        chunks_file = os.path.join(output_dir, "chunks.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(compatible_blocks, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"兼容格式已保存到: {chunks_file}")
        
        # 保存统计信息
        stats_file = os.path.join(output_dir, "processing_stats.json")
        stats = {
            "total_blocks": len(clustered_blocks),
            "processing_method": "static_clustering",
            "config": {
                "chunk_length": self.chunk_length,
                "cluster_count": self.cluster_count,
                "enable_summary": self.enable_summary
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"统计信息已保存到: {stats_file}")


def run_static_chunk_processing(config: Dict[str, Any], work_dir: str, logger: logging.Logger) -> None:
    """
    运行静态批量文档处理
    
    Args:
        config: 配置字典
        work_dir: 工作目录
        logger: 日志记录器
    """
    try:
        # 创建处理器
        processor = StaticChunkProcessor(config)
        
        # 获取输入文件
        input_dir = config.get('document', {}).get('input_dir', './data')
        allowed_types = config.get('document', {}).get('allowed_types', ['.txt', '.json', '.jsonl', '.docx'])
        
        input_files = []
        if os.path.exists(input_dir):
            for file_name in os.listdir(input_dir):
                file_path = os.path.join(input_dir, file_name)
                if os.path.isfile(file_path) and any(file_name.endswith(ext) for ext in allowed_types):
                    input_files.append(file_path)
        
        if not input_files:
            logger.warning(f"在目录 {input_dir} 中未找到有效的输入文件")
            return
        
        logger.info(f"找到 {len(input_files)} 个输入文件")
        
        # 设置输出目录
        output_dir = work_dir
        
        # 执行处理
        stats = processor.process_documents(input_files, output_dir)
        
        logger.info(f"静态批量处理完成: {stats}")
        
    except Exception as e:
        logger.error(f"静态批量处理失败: {e}")
        raise