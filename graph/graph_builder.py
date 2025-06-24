import os
import re
import json
import logging
import networkx as nx
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from llm.llm import LLMClient
from networkx.readwrite import json_graph
from utils.io import load_json, save_graph
from graph.entity_extractor import EntityExtractor, extract_relations_with_llm
from graph.performance_monitor import monitor_performance, memory_management_context, check_gpu_memory_usage, optimize_for_memory

@monitor_performance("实体关系抽取")
def extract_entities_and_relations(text: str, llm_client: LLMClient, entity_extractor: EntityExtractor = None, config: dict = None):
    """
    使用改进的实体抽取器和LLM抽取实体与关系 - 优化版本
    
    Args:
        text: 输入文本
        llm_client: LLM客户端
        entity_extractor: 实体抽取器实例
        config: 配置信息
        
    Returns:
        关系三元组列表 (头实体, 关系, 尾实体)
    """
    if entity_extractor is None:
        # 正确初始化EntityExtractor，传递完整配置
        entity_extractor = EntityExtractor(config)
    
    try:
        # 检查GPU内存使用情况
        if check_gpu_memory_usage():
            optimize_for_memory()
            
        # 文本预处理，移除过多空白字符
        text = ' '.join(text.split())
        
        # 1. 使用改进的实体抽取器识别实体
        with memory_management_context("实体抽取"):
            entities_info = entity_extractor.extract_entities(text)
            entities = [entity['text'] for entity in entities_info]
        
        # 获取最大实体数量限制
        max_entities_per_topic = config.get('graph_construction', {}).get('max_entities_per_topic', 50) if config else 50
        
        # 限制实体数量
        if len(entities) > max_entities_per_topic:
            # 按置信度排序，保留最高置信度的实体
            entities_with_conf = sorted(entities_info, key=lambda x: x.get('confidence', 0), reverse=True)[:max_entities_per_topic]
            entities = [entity['text'] for entity in entities_with_conf]
            print(f"[⚠️ 实体数量限制]: {len(entities)}/{max_entities_per_topic}")
        
        if len(entities) < 2:
            print(f"[⚠️ 实体数量不足]: 仅识别到 {len(entities)} 个实体")
            return []
        
        print(f"[✅ 实体识别]: 识别到 {len(entities)} 个实体: {entities[:10]}{'...' if len(entities) > 10 else ''}")
        
        # 2. 使用LLM抽取实体间的关系（限制实体数量以提高效率）
        if len(entities) <= 20:  # 限制LLM处理的实体数量
            with memory_management_context("关系抽取"):
                # 再次检查GPU内存
                if check_gpu_memory_usage():
                    print(f"[⚠️ GPU内存不足]: 跳过LLM关系抽取")
                    triples = []
                else:
                    triples = extract_relations_with_llm(text, entities, llm_client)
        else:
            print(f"[⚠️ 实体数量过多({len(entities)})]: 跳过LLM关系抽取")
            triples = []
        
        # 过滤自环关系
        filtered_triples = []
        for triple in triples:
            if len(triple) == 3:
                head, rel, tail = triple
                if head.lower() != tail.lower():  # 跳过自环关系
                    filtered_triples.append(triple)
        
        print(f"[✅ 关系抽取]: 抽取到 {len(filtered_triples)} 个关系三元组")
        
        return filtered_triples
        
    except Exception as e:
        print(f"❌ 实体关系抽取失败: {e}")
        return []


def build_graph(chunks: list[dict], llm_client: LLMClient, config: dict = None, logger=None):
    """
    构建知识图谱
    
    Args:
        chunks: 文档块列表
        llm_client: LLM客户端
        config: 配置信息
        
    Returns:
        构建的知识图谱
    """
    G = nx.DiGraph()
    
    # 如果没有提供logger，使用默认的print输出
    if logger is None:
        class DefaultLogger:
            def warning(self, msg): print(f"WARNING: {msg}")
            def info(self, msg): print(f"INFO: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
        logger = DefaultLogger()
    
    # 初始化实体抽取器和图构建配置
    graph_config = config.get("graph_construction", {}) if config else {}
    entity_extractor = EntityExtractor(config)
    
    # 图构建参数
    enable_reverse_links = graph_config.get("enable_reverse_links", True)
    max_entities_per_topic = graph_config.get("max_entities_per_topic", 50)
    relation_confidence_threshold = graph_config.get("relation_confidence_threshold", 0.6)
    
    # 统计信息
    total_entities = set()
    total_relations = 0
    
    for i, chunk in enumerate(chunks):
        # 处理不同数据格式的兼容性
        if "text" in chunk:
            text = chunk["text"]
        elif "sentences" in chunk:
            # 处理static_chunk_processor生成的格式
            text = "\n".join(chunk["sentences"]) if isinstance(chunk["sentences"], list) else str(chunk["sentences"])
        else:
            logger.warning(f"块 {i} 缺少文本内容，跳过处理")
            continue
            
        chunk_id = chunk["id"]
        summary = chunk.get("summary", chunk_id)
        topic_node_id = f"topic::{chunk_id}"

        # 创建主题节点，添加更多属性
        G.add_node(topic_node_id, 
                  type="topic", 
                  label=summary,
                  chunk_id=chunk_id,
                  topic_id=chunk_id,  # 添加topic_id属性用于图谱检索
                  text_length=len(text))

        # 抽取实体三元组
        triples = extract_entities_and_relations(text, llm_client, entity_extractor, config)
        
        if not triples:
            print(f"[⚠️ 块 {i+1}/{len(chunks)}]: 未抽取到有效关系")
            continue
            
        print(f"[📊 块 {i+1}/{len(chunks)}]: 抽取到 {len(triples)} 个关系")
        total_relations += len(triples)

        # 记录当前块的实体
        chunk_entities = set()
        
        for head, relation, tail in triples:
            # 添加实体节点（如果不存在）
            if not G.has_node(head):
                G.add_node(head, type="entity", label=head)
            if not G.has_node(tail):
                G.add_node(tail, type="entity", label=tail)
            
            # 记录实体
            chunk_entities.add(head)
            chunk_entities.add(tail)
            total_entities.add(head)
            total_entities.add(tail)

            # 添加实体间的关系边
            if G.has_edge(head, tail):
                # 如果边已存在，更新关系信息
                existing_relations = G[head][tail].get('relations', [])
                if relation not in existing_relations:
                    existing_relations.append(relation)
                    G[head][tail]['relations'] = existing_relations
            else:
                G.add_edge(head, tail, relation=relation, relations=[relation])

            # 连接实体到主题节点（双向链接）
            if not G.has_edge(topic_node_id, head):
                G.add_edge(topic_node_id, head, relation="包含")
            if not G.has_edge(topic_node_id, tail):
                G.add_edge(topic_node_id, tail, relation="包含")
                
            # 添加实体→主题的反向链接（如果启用）
            if enable_reverse_links:
                if not G.has_edge(head, topic_node_id):
                    G.add_edge(head, topic_node_id, relation="属于主题")
                if not G.has_edge(tail, topic_node_id):
                    G.add_edge(tail, topic_node_id, relation="属于主题")
        
        # 为主题节点添加实体计数信息和限制检查
        entity_count = len(chunk_entities)
        G.nodes[topic_node_id]['entity_count'] = entity_count
        
        # 如果实体数量超过限制，记录警告
        if entity_count > max_entities_per_topic:
            print(f"[⚠️ 警告]: 主题 {topic_node_id} 包含 {entity_count} 个实体，超过限制 {max_entities_per_topic}")
            G.nodes[topic_node_id]['entity_overflow'] = True
    
    print(f"[🎯 图构建完成]: 总实体数 {len(total_entities)}, 总关系数 {total_relations}")
    return G

# save_graph函数已移至utils.io模块

def run_graph_construction(config: dict, work_dir: str, logger):
    from utils.io import load_chunks
    
    output_dir = os.path.join(work_dir)

    logger.info("加载文本块...")
    chunks = load_chunks(work_dir)

    llm_client = LLMClient(config["llm"])

    logger.info("开始实体图构建...")
    G = build_graph(chunks, llm_client, config["graph"], logger)

    logger.info(f"图构建完成，节点数: {len(G.nodes)}, 边数: {len(G.edges)}")
    save_graph(G, output_dir)
    logger.info(f"图已保存至 {output_dir}/graph.json 和 graph.graphml")
