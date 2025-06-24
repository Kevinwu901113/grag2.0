#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的图构建器
解决图构建过程中的性能瓶颈和GPU利用率问题
"""

import logging
import time
import gc
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import networkx as nx

from .entity_extractor import EntityExtractor
from .performance_monitor import (
    monitor_performance, 
    memory_management_context, 
    check_gpu_memory_usage, 
    optimize_for_memory,
    performance_monitor
)
from llm.llm import LLMClient

class OptimizedGraphBuilder:
    """
    优化的图构建器
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph_config = config.get('knowledge_graph', {})
        self.entity_config = self.graph_config.get('entity_extraction', {})
        
        # 性能参数
        self.max_entities_per_chunk = self.graph_config.get('max_entities_per_chunk', 30)
        self.max_relations_per_chunk = self.graph_config.get('max_relations_per_chunk', 50)
        self.batch_size = self.graph_config.get('batch_size', 8)
        self.max_text_length = self.graph_config.get('max_text_length', 800)
        self.enable_parallel = self.graph_config.get('enable_parallel_processing', False)
        
        # 初始化组件
        self.entity_extractor = None
        self.llm_client = None
        self.graph = nx.Graph()
        
        # 统计信息
        self.stats = {
            'total_entities': 0,
            'total_relations': 0,
            'processed_chunks': 0,
            'skipped_chunks': 0,
            'processing_time': 0
        }
        
    def initialize_components(self, entity_extractor: EntityExtractor, llm_client: LLMClient):
        """初始化组件"""
        self.entity_extractor = entity_extractor
        self.llm_client = llm_client
        logging.info("[图构建器] 组件初始化完成")
        
    @monitor_performance("文本预处理")
    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 移除多余空白字符
        text = ' '.join(text.split())
        
        # 长度限制
        if len(text) > self.max_text_length:
            logging.warning(f"文本过长({len(text)}字符)，截取前{self.max_text_length}字符")
            text = text[:self.max_text_length]
            
        return text
        
    @monitor_performance("实体抽取")
    def extract_entities_optimized(self, text: str) -> List[Dict[str, Any]]:
        """优化的实体抽取"""
        if not self.entity_extractor:
            logging.error("实体抽取器未初始化")
            return []
            
        with memory_management_context("实体抽取处理"):
            # 检查GPU内存
            if check_gpu_memory_usage():
                logging.warning("GPU内存使用率过高，执行内存清理")
                performance_monitor.force_garbage_collection()
                
            entities = self.entity_extractor.extract_entities(text)
            
            # 限制实体数量
            if len(entities) > self.max_entities_per_chunk:
                entities = sorted(entities, key=lambda x: x.get('confidence', 0), reverse=True)[:self.max_entities_per_chunk]
                logging.info(f"实体数量限制: {len(entities)}/{self.max_entities_per_chunk}")
                
            return entities
            
    @monitor_performance("关系抽取")
    def extract_relations_optimized(self, text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """优化的关系抽取"""
        if len(entities) < 2:
            return []
            
        # 实体数量检查
        if len(entities) > 20:
            logging.info(f"实体数量过多({len(entities)})，跳过LLM关系抽取")
            return []
            
        with memory_management_context("关系抽取处理"):
            # GPU内存检查
            if check_gpu_memory_usage():
                logging.warning("GPU内存不足，跳过LLM关系抽取")
                return []
                
            try:
                relations = self.entity_extractor.extract_relations_with_llm(text, entities, self.llm_client)
                
                # 过滤自环关系
                filtered_relations = []
                for rel in relations:
                    if len(rel) == 3 and rel[0].lower() != rel[2].lower():
                        filtered_relations.append(rel)
                        
                # 限制关系数量
                if len(filtered_relations) > self.max_relations_per_chunk:
                    filtered_relations = filtered_relations[:self.max_relations_per_chunk]
                    logging.info(f"关系数量限制: {len(filtered_relations)}/{self.max_relations_per_chunk}")
                    
                return filtered_relations
                
            except Exception as e:
                logging.error(f"关系抽取失败: {e}")
                return []
                
    def add_entities_to_graph(self, entities: List[Dict[str, Any]], topic_id: str = None):
        """将实体添加到图中"""
        for entity in entities:
            entity_text = entity['text']
            entity_type = entity.get('type', 'UNKNOWN')
            confidence = entity.get('confidence', 0.0)
            
            # 添加实体节点
            if not self.graph.has_node(entity_text):
                self.graph.add_node(entity_text, 
                                  type=entity_type, 
                                  confidence=confidence,
                                  topic_id=topic_id)
                self.stats['total_entities'] += 1
            else:
                # 更新置信度（取最高值）
                current_conf = self.graph.nodes[entity_text].get('confidence', 0.0)
                if confidence > current_conf:
                    self.graph.nodes[entity_text]['confidence'] = confidence
                    
    def add_relations_to_graph(self, relations: List[Tuple[str, str, str]]):
        """将关系添加到图中"""
        for head, relation, tail in relations:
            # 确保实体节点存在
            if not self.graph.has_node(head):
                self.graph.add_node(head, type='UNKNOWN', confidence=0.5)
            if not self.graph.has_node(tail):
                self.graph.add_node(tail, type='UNKNOWN', confidence=0.5)
                
            # 添加关系边
            if not self.graph.has_edge(head, tail):
                self.graph.add_edge(head, tail, relation=relation)
                self.stats['total_relations'] += 1
            else:
                # 如果边已存在，可以添加多重关系
                existing_relations = self.graph[head][tail].get('relation', [])
                if isinstance(existing_relations, str):
                    existing_relations = [existing_relations]
                if relation not in existing_relations:
                    existing_relations.append(relation)
                    self.graph[head][tail]['relation'] = existing_relations
                    
    @monitor_performance("单个文档处理")
    def process_single_document(self, text: str, topic_id: str = None) -> Dict[str, Any]:
        """处理单个文档"""
        start_time = time.time()
        
        try:
            # 文本预处理
            processed_text = self.preprocess_text(text)
            
            # 实体抽取
            entities = self.extract_entities_optimized(processed_text)
            if not entities:
                logging.warning("未抽取到任何实体")
                self.stats['skipped_chunks'] += 1
                return {'entities': 0, 'relations': 0, 'processing_time': time.time() - start_time}
                
            # 关系抽取
            entity_texts = [entity['text'] for entity in entities]
            relations = self.extract_relations_optimized(processed_text, entity_texts)
            
            # 添加到图中
            self.add_entities_to_graph(entities, topic_id)
            self.add_relations_to_graph(relations)
            
            self.stats['processed_chunks'] += 1
            processing_time = time.time() - start_time
            self.stats['processing_time'] += processing_time
            
            logging.info(f"文档处理完成: {len(entities)}个实体, {len(relations)}个关系, 耗时{processing_time:.2f}s")
            
            return {
                'entities': len(entities),
                'relations': len(relations),
                'processing_time': processing_time
            }
            
        except Exception as e:
            logging.error(f"文档处理失败: {e}")
            self.stats['skipped_chunks'] += 1
            return {'entities': 0, 'relations': 0, 'processing_time': time.time() - start_time}
            
    @monitor_performance("批量文档处理")
    def process_documents_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量处理文档"""
        total_start_time = time.time()
        batch_results = []
        
        logging.info(f"开始批量处理 {len(documents)} 个文档")
        
        for i, doc in enumerate(documents):
            try:
                # 定期检查内存使用情况
                if i % 10 == 0 and i > 0:
                    if check_gpu_memory_usage():
                        logging.warning(f"处理第{i}个文档时GPU内存不足，执行清理")
                        performance_monitor.force_garbage_collection()
                        
                # 获取文档内容
                text = doc.get('content', doc.get('text', ''))
                topic_id = doc.get('topic_id', doc.get('id', f'doc_{i}'))
                
                if not text.strip():
                    logging.warning(f"文档{i}内容为空，跳过")
                    continue
                    
                # 处理单个文档
                result = self.process_single_document(text, topic_id)
                batch_results.append(result)
                
                # 进度报告
                if (i + 1) % 50 == 0:
                    avg_time = sum(r['processing_time'] for r in batch_results) / len(batch_results)
                    logging.info(f"已处理 {i+1}/{len(documents)} 个文档，平均耗时 {avg_time:.2f}s/文档")
                    
            except Exception as e:
                logging.error(f"处理文档{i}时出错: {e}")
                continue
                
        total_time = time.time() - total_start_time
        
        # 统计结果
        total_entities = sum(r['entities'] for r in batch_results)
        total_relations = sum(r['relations'] for r in batch_results)
        avg_processing_time = sum(r['processing_time'] for r in batch_results) / len(batch_results) if batch_results else 0
        
        logging.info(f"批量处理完成:")
        logging.info(f"  处理文档: {len(batch_results)}/{len(documents)}")
        logging.info(f"  总实体数: {total_entities}")
        logging.info(f"  总关系数: {total_relations}")
        logging.info(f"  总耗时: {total_time:.2f}s")
        logging.info(f"  平均耗时: {avg_processing_time:.2f}s/文档")
        
        return {
            'processed_documents': len(batch_results),
            'total_documents': len(documents),
            'total_entities': total_entities,
            'total_relations': total_relations,
            'total_time': total_time,
            'average_time_per_doc': avg_processing_time
        }
        
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'connected_components': nx.number_connected_components(self.graph),
            'processing_stats': self.stats.copy()
        }
        
    def save_graph(self, filepath: str):
        """保存图"""
        try:
            nx.write_gexf(self.graph, filepath)
            logging.info(f"图已保存到: {filepath}")
        except Exception as e:
            logging.error(f"保存图失败: {e}")
            
    def load_graph(self, filepath: str):
        """加载图"""
        try:
            self.graph = nx.read_gexf(filepath)
            logging.info(f"图已从 {filepath} 加载")
        except Exception as e:
            logging.error(f"加载图失败: {e}")