import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
import networkx as nx
from llm.llm import LLMClient
from utils.logger import setup_logger
from utils.model_cache import model_cache

def run_entity_vector_indexer(config: dict, work_dir: str, graph: nx.DiGraph = None, logger=None):
    """
    实体向量索引器，为知识图谱中的实体节点构建向量索引
    
    Args:
        config: 配置信息
        work_dir: 工作目录
        graph: 知识图谱，如果为None则从文件加载
        logger: 日志对象，如果为None则创建新的日志对象
    """
    if logger is None:
        logger = setup_logger(work_dir)
    
    from utils.io import load_graph, save_entity_vector_index, file_exists
    
    # 加载知识图谱
    if graph is None:
        graph_path = os.path.join(work_dir, "graph.json")
        if not file_exists(graph_path):
            logger.error("未找到 graph.json，无法构建实体向量索引。请先运行图构建模块。")
            return
        graph = load_graph(work_dir)
    
    # 提取实体节点
    entity_nodes = [node for node, data in graph.nodes(data=True) if data.get('type') == 'entity']
    
    if not entity_nodes:
        logger.warning("图中未找到实体节点，跳过实体向量索引构建")
        return
    
    logger.info(f"找到 {len(entity_nodes)} 个实体节点，开始构建向量索引")
    
    # 获取嵌入配置
    embedding_config = config.get("embedding", {})
    provider = embedding_config.get("provider", "ollama")
    model_name = embedding_config.get("model_name", "bge-m3")
    
    # 初始化LLM客户端
    llm = LLMClient(config)
    
    # 准备实体文本和上下文信息
    entity_texts, entity_contexts = prepare_entity_texts(graph, entity_nodes)
    
    # 获取批处理大小
    batch_size = config.get("vector", {}).get("batch_size", 100)
    logger.info(f"开始生成实体嵌入，使用{provider}的{model_name}模型，批处理大小: {batch_size}")

    # 使用批处理生成嵌入
    all_embeddings = []
    from utils.logger import get_progress_bar
    
    # 创建进度条
    pbar = get_progress_bar(total=len(entity_texts), desc="生成实体嵌入")
    
    try:
        for i in range(0, len(entity_texts), batch_size):
            batch_texts = entity_texts[i:i+batch_size]
            # 启用文本预处理和维度验证
            batch_embeddings = llm.embed(batch_texts, normalize_text=True, validate_dim=True)
            
            # 验证批次嵌入
            if not batch_embeddings:
                logger.warning(f"实体批次 {i//batch_size + 1} 嵌入生成失败，跳过")
                pbar.update(len(batch_texts))
                continue
                
            # 检查嵌入维度一致性
            if all_embeddings and len(batch_embeddings[0]) != len(all_embeddings[0]):
                logger.error(f"实体批次 {i//batch_size + 1} 嵌入维度不一致: 期望 {len(all_embeddings[0])}, 实际 {len(batch_embeddings[0])}")
                pbar.close()
                return
                
            all_embeddings.extend(batch_embeddings)
            pbar.update(len(batch_texts))
    except (ValueError, ConnectionError, TimeoutError, RuntimeError) as e:
        logger.error(f"生成实体嵌入时出错: {e}")
        pbar.close()
        return
    except Exception as e:
        logger.error(f"生成实体嵌入时发生未知错误: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        pbar.close()
        return
    
    pbar.close()
    
    if not all_embeddings:
        logger.error("生成的实体嵌入为空，无法构建索引")
        return
        
    # 构建FAISS索引
    dim = len(all_embeddings[0])
    logger.info(f"实体嵌入维度: {dim}, 有效嵌入数量: {len(all_embeddings)}")
    
    # 验证所有实体嵌入向量
    from utils.common import validate_embedding_dimension
    invalid_count = 0
    for i, embedding in enumerate(all_embeddings):
        if not validate_embedding_dimension(embedding, dim):
            logger.warning(f"第 {i} 个实体嵌入向量无效")
            invalid_count += 1
    
    if invalid_count > 0:
        logger.warning(f"发现 {invalid_count} 个无效实体嵌入向量")
        if invalid_count > len(all_embeddings) * 0.1:  # 如果超过10%的向量无效
            logger.error("无效实体嵌入向量过多，停止构建索引")
            return
    
    embeddings_np = np.array(all_embeddings).astype('float32')
    
    # 检查数组中的无效值
    if np.any(np.isnan(embeddings_np)) or np.any(np.isinf(embeddings_np)):
        logger.error("实体嵌入数组包含NaN或无穷大值，无法构建索引")
        return

    vec_config = config.get("vector", {})
    index_type = vec_config.get("index_type", "Flat").upper()
    nlist = vec_config.get("nlist", 100)
    nprobe = vec_config.get("nprobe", 10)
    efSearch = vec_config.get("ef_search", 32)

    if index_type == "IVF":
        # 确保nlist不超过训练数据数量
        num_samples = len(embeddings_np)
        adjusted_nlist = min(nlist, num_samples)
        if adjusted_nlist != nlist:
            logger.warning(f"实体数量({num_samples})少于配置的nlist({nlist})，自动调整为{adjusted_nlist}")
        
        logger.info(f"构建实体 IVF 索引（nlist={adjusted_nlist}）...")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, adjusted_nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings_np)
        index.nprobe = min(nprobe, adjusted_nlist)
    elif index_type == "HNSW":
        logger.info(f"构建实体 HNSW 索引（efSearch={efSearch}）...")
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = efSearch
    else:
        logger.info(f"构建实体 Flat 索引...")
        index = faiss.IndexFlatIP(dim)

    # 归一化向量以确保内积等于余弦相似度
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)

    # 保存实体向量索引
    entity_id_map = {i: entity_nodes[i] for i in range(len(entity_nodes))}
    entity_metadata = {
        'contexts': entity_contexts,
        'texts': entity_texts,
        'graph_stats': get_entity_graph_stats(graph, entity_nodes)
    }
    
    save_entity_vector_index(index, entity_id_map, entity_metadata, work_dir)

    logger.info(f"✅ 实体向量索引构建完成：{index_type}，维度 {dim}，共 {len(all_embeddings)} 个实体")
    logger.info(f"实体索引文件已保存至工作目录 {work_dir}")

def prepare_entity_texts(graph: nx.DiGraph, entity_nodes: List[str]) -> Tuple[List[str], List[Dict]]:
    """
    为实体节点准备文本表示和上下文信息
    
    Args:
        graph: 知识图谱
        entity_nodes: 实体节点列表
        
    Returns:
        实体文本列表和上下文信息列表
    """
    entity_texts = []
    entity_contexts = []
    
    for entity in entity_nodes:
        # 获取实体的基本信息
        entity_data = graph.nodes[entity]
        entity_label = entity_data.get('label', entity)
        
        # 获取实体的关系信息
        relations = []
        
        # 出边关系
        for _, target, edge_data in graph.out_edges(entity, data=True):
            relation = edge_data.get('relation', '相关')
            target_label = graph.nodes[target].get('label', target)
            relations.append(f"{entity_label} {relation} {target_label}")
        
        # 入边关系
        for source, _, edge_data in graph.in_edges(entity, data=True):
            relation = edge_data.get('relation', '相关')
            source_label = graph.nodes[source].get('label', source)
            relations.append(f"{source_label} {relation} {entity_label}")
        
        # 构建实体的文本表示
        if relations:
            entity_text = f"实体: {entity_label}\n相关关系: {'; '.join(relations[:5])}"  # 限制关系数量
        else:
            entity_text = f"实体: {entity_label}"
        
        entity_texts.append(entity_text)
        
        # 保存上下文信息
        context = {
            'entity': entity,
            'label': entity_label,
            'degree': graph.degree(entity),
            'in_degree': graph.in_degree(entity),
            'out_degree': graph.out_degree(entity),
            'relations_count': len(relations)
        }
        entity_contexts.append(context)
    
    return entity_texts, entity_contexts

def get_entity_graph_stats(graph: nx.DiGraph, entity_nodes: List[str]) -> Dict[str, Any]:
    """
    获取实体在图中的统计信息
    
    Args:
        graph: 知识图谱
        entity_nodes: 实体节点列表
        
    Returns:
        统计信息字典
    """
    stats = {
        'total_entities': len(entity_nodes),
        'total_nodes': graph.number_of_nodes(),
        'total_edges': graph.number_of_edges(),
        'avg_degree': sum(graph.degree(entity) for entity in entity_nodes) / len(entity_nodes) if entity_nodes else 0
    }
    
    # 计算度数分布
    degrees = [graph.degree(entity) for entity in entity_nodes]
    if degrees:
        stats['min_degree'] = min(degrees)
        stats['max_degree'] = max(degrees)
        stats['median_degree'] = sorted(degrees)[len(degrees)//2]
    
    return stats