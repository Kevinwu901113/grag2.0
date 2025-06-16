import json
import networkx as nx
import re
from networkx.readwrite import json_graph
import jieba
from utils.io import load_graph
from functools import lru_cache
import time
import os
import sys

# 添加config目录到路径
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
if config_path not in sys.path:
    sys.path.append(config_path)

try:
    from graph_retrieval_config import GraphRetrievalConfig
except ImportError:
    # 如果配置文件不存在，使用默认配置
    class GraphRetrievalConfig:
        ENTITY_MATCHING = {
            'enable_fuzzy_match': True,
            'enable_partial_match': True,
            'min_similarity': 0.6,
            'max_fuzzy_results': 20,
            'min_token_length': 2,
            'max_length_diff': 3,
        }
        SYNONYM_MAP = {
            '深圳': ['深圳市', '鹏城'],
            '宝安': ['宝安区', '宝安县'],
            '村': ['村委会', '村民委员会'],
            '区': ['区政府', '行政区'],
            '市': ['市政府', '地级市'],
            '县': ['县政府', '县级市'],
            '镇': ['镇政府', '乡镇'],
            '街道': ['街道办', '街道办事处'],
            '社区': ['社区居委会', '居委会']
        }
        @classmethod
        def get_entity_matching_config(cls):
            return cls.ENTITY_MATCHING.copy()
        @classmethod
        def get_synonym_map(cls):
            return cls.SYNONYM_MAP.copy()

def extract_entity_names(graph: nx.Graph) -> set[str]:
    return {
        node for node, data in graph.nodes(data=True)
        if data.get("type") == "entity"
    }

@lru_cache(maxsize=1000)
def _cached_entity_match(query: str, entity_names_tuple: tuple, config_hash: str) -> tuple:
    """
    缓存的实体匹配函数（内部使用）
    """
    entity_names = set(entity_names_tuple)
    return tuple(match_entities_in_query_impl(query, entity_names))

def match_entities_in_query(query: str, entity_names: set[str], 
                            enable_fuzzy_match: bool = None,
                            enable_partial_match: bool = None,
                            min_similarity: float = None,
                            use_cache: bool = True) -> set[str]:
    """
    改进的实体匹配函数，支持多种匹配策略和缓存
    
    Args:
        query: 查询文本
        entity_names: 实体名称集合
        enable_fuzzy_match: 是否启用模糊匹配（None时使用配置文件）
        enable_partial_match: 是否启用部分匹配（None时使用配置文件）
        min_similarity: 最小相似度阈值（None时使用配置文件）
        use_cache: 是否使用缓存
        
    Returns:
        匹配到的实体集合
    """
    if not query or not entity_names:
        return set()
    
    # 获取配置
    config = GraphRetrievalConfig.get_entity_matching_config()
    if enable_fuzzy_match is None:
        enable_fuzzy_match = config['enable_fuzzy_match']
    if enable_partial_match is None:
        enable_partial_match = config['enable_partial_match']
    if min_similarity is None:
        min_similarity = config['min_similarity']
    
    # 如果启用缓存，使用缓存版本
    if use_cache:
        entity_names_tuple = tuple(sorted(entity_names))
        config_hash = f"{enable_fuzzy_match}_{enable_partial_match}_{min_similarity}"
        try:
            cached_result = _cached_entity_match(query, entity_names_tuple, config_hash)
            return set(cached_result)
        except Exception:
            # 缓存失败时回退到非缓存版本
            pass
    
    return match_entities_in_query_impl(query, entity_names, enable_fuzzy_match, 
                                       enable_partial_match, min_similarity)

def match_entities_in_query_impl(query: str, entity_names: set[str], 
                                enable_fuzzy_match: bool = True,
                                enable_partial_match: bool = True,
                                min_similarity: float = 0.6) -> set[str]:
    """
    实体匹配的具体实现
    """
    start_time = time.time()
    matched = set()
    tokens = list(jieba.cut(query))
    joined = "".join(tokens)
    query_lower = query.lower()
    
    # 获取配置
    config = GraphRetrievalConfig.get_entity_matching_config()
    synonym_map = GraphRetrievalConfig.get_synonym_map()
    
    # 1. 精确匹配（原有逻辑）
    exact_matched = {ent for ent in entity_names if ent in joined}
    matched.update(exact_matched)
    
    # 2. 部分匹配 - 检查实体名称是否包含查询中的关键词
    if enable_partial_match:
        min_token_length = config.get('min_token_length', 2)
        for token in tokens:
            if len(token) >= min_token_length:
                token_lower = token.lower()
                partial_matched = {
                    ent for ent in entity_names 
                    if token_lower in ent.lower() or ent.lower() in token_lower
                }
                matched.update(partial_matched)
    
    # 3. 优化的模糊匹配
    max_entities_for_fuzzy = config.get('max_entities_for_fuzzy', 50)
    if enable_fuzzy_match and len(matched) < max_entities_for_fuzzy:
        try:
            from difflib import SequenceMatcher
            
            fuzzy_candidates = set()
            max_fuzzy_results = config.get('max_fuzzy_results', 20)
            max_length_diff = config.get('max_length_diff', 3)
            min_token_length = config.get('min_token_length', 2)
            
            for token in tokens:
                if len(token) >= min_token_length:
                    token_lower = token.lower()
                    for entity in entity_names:
                        entity_lower = entity.lower()
                        # 长度差异过大时跳过
                        if abs(len(entity_lower) - len(token_lower)) > max(len(token_lower), max_length_diff):
                            continue
                            
                        # 快速预检查：是否有公共字符
                        if not set(token_lower) & set(entity_lower):
                            continue
                            
                        # 计算相似度
                        similarity = SequenceMatcher(None, token_lower, entity_lower).ratio()
                        if similarity >= min_similarity:
                            fuzzy_candidates.add(entity)
                            
            # 限制模糊匹配结果数量
            if len(fuzzy_candidates) <= max_fuzzy_results:
                matched.update(fuzzy_candidates)
                
        except ImportError:
            print("警告: difflib模块不可用，跳过模糊匹配")
    
    # 4. 同义词和缩写匹配
    for token in tokens:
        if token in synonym_map:
            synonyms = synonym_map[token]
            for synonym in synonyms:
                synonym_matched = {
                    ent for ent in entity_names 
                    if synonym in ent or ent in synonym
                }
                matched.update(synonym_matched)
    
    # 性能监控
    elapsed_time = time.time() - start_time
    if elapsed_time > 0.1:  # 超过100ms时记录警告
        print(f"⚠️ 实体匹配耗时较长: {elapsed_time:.3f}秒, 查询: '{query}', 匹配结果: {len(matched)}个")
    
    return matched

def extract_subgraph(graph, entities, depth=1):
    nodes = set()
    for entity in entities:
        if entity in graph:
            nodes.add(entity)
            for neighbor in nx.single_source_shortest_path_length(graph, entity, cutoff=depth):
                nodes.add(neighbor)
    return graph.subgraph(nodes).copy()

def summarize_subgraph(subgraph, max_length: int = 300):
    """
    生成子图的简洁摘要，支持长度限制
    
    Args:
        subgraph: 子图
        max_length: 最大摘要长度
        
    Returns:
        简洁的子图摘要
    """
    if not subgraph.edges():
        return "未发现相关实体关系。"
    
    summary = []
    edge_count = 0
    max_edges = 10  # 限制边的数量
    
    # 按边的重要性排序（可以根据节点度数或其他指标）
    edges_with_weight = []
    for u, v, data in subgraph.edges(data=True):
        # 计算边的权重（基于节点度数）
        weight = subgraph.degree(u) + subgraph.degree(v)
        edges_with_weight.append((u, v, data, weight))
    
    # 按权重排序，选择最重要的边
    edges_with_weight.sort(key=lambda x: x[3], reverse=True)
    
    for u, v, data, _ in edges_with_weight[:max_edges]:
        if edge_count >= max_edges:
            break
            
        relation = data.get("relation", "关联")
        summary.append(f"{u}与{v}存在{relation}关系")
        edge_count += 1
        
        # 检查长度限制
        current_summary = "；".join(summary) + "。"
        if len(current_summary) > max_length:
            # 移除最后一个关系，确保不超过长度限制
            summary.pop()
            break
    
    if not summary:
        return "发现实体关联但关系复杂。"
    
    result = "；".join(summary) + "。"
    
    # 如果还有更多关系未显示，添加提示
    if len(edges_with_weight) > len(summary):
        remaining = len(edges_with_weight) - len(summary)
        if len(result) + 20 < max_length:  # 确保有空间添加提示
            result += f"另有{remaining}个相关关系。"
    
    return result

def retrieve_by_entity(graph: nx.Graph, matched_entities: set[str], chunks: list) -> list:
    """
    基于图谱实体检索相关文档块
    
    Args:
        graph: 知识图谱
        matched_entities: 匹配到的实体集合
        chunks: 所有文档块列表
        
    Returns:
        检索到的文档块列表，包含相似度信息
    """
    if not matched_entities or not chunks:
        return []
        
    # 收集所有相关的主题节点ID
    topic_ids = set()
    entity_weights = {}  # 实体权重，可用于后续PageRank等算法
    
    for entity in matched_entities:
        if entity not in graph:
            continue
            
        # 计算实体的度中心性作为权重
        entity_weights[entity] = graph.degree(entity)
        
        # 获取实体连接的所有节点
        for neighbor in graph.neighbors(entity):
            neighbor_data = graph.nodes[neighbor]
            # 如果邻居节点是主题节点，收集其ID
            if neighbor_data.get("type") == "topic":
                topic_id = neighbor_data.get("topic_id")
                if topic_id is not None:
                    topic_ids.add(topic_id)
    
    if not topic_ids:
        return []
        
    # 根据主题ID检索对应的文档块
    retrieved_chunks = []
    for chunk in chunks:
        # 检查多个可能的ID字段
        chunk_topic_id = chunk.get('topic_id') or chunk.get('id')
        if chunk_topic_id in topic_ids:
            # 计算基于图谱的相似度分数
            # 这里使用简单的二进制匹配，可以根据需要扩展为更复杂的计算
            graph_score = 1.0
            
            # 如果chunk关联的实体有权重信息，可以加权
            max_entity_weight = 0
            for entity in matched_entities:
                if entity in entity_weights:
                    max_entity_weight = max(max_entity_weight, entity_weights[entity])
            
            # 根据实体重要性调整分数
            if max_entity_weight > 0:
                # 归一化权重到0.5-1.5范围
                normalized_weight = min(1.5, 0.5 + max_entity_weight / 10.0)
                graph_score *= normalized_weight
            
            result = chunk.copy()
            result['similarity'] = graph_score
            result['retrieval_type'] = 'graph_entity'
            result['matched_entities'] = list(matched_entities)
            retrieved_chunks.append(result)
    
    # 按相似度排序
    retrieved_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
    
    return retrieved_chunks

def calculate_entity_pagerank(graph: nx.Graph, max_iter: int = 100, alpha: float = 0.85) -> dict:
    """
    计算图谱中实体节点的PageRank值，用于实体重要性评估
    
    Args:
        graph: 知识图谱
        max_iter: 最大迭代次数
        alpha: 阻尼系数
        
    Returns:
        实体节点的PageRank值字典
    """
    try:
        # 只对实体节点计算PageRank
        entity_nodes = [node for node, data in graph.nodes(data=True) 
                       if data.get("type") == "entity"]
        
        if not entity_nodes:
            return {}
            
        # 创建只包含实体节点的子图
        entity_subgraph = graph.subgraph(entity_nodes)
        
        # 计算PageRank
        pagerank_scores = nx.pagerank(entity_subgraph, max_iter=max_iter, alpha=alpha)
        
        return pagerank_scores
        
    except Exception as e:
        print(f"计算PageRank失败: {e}")
        return {}

if __name__ == "__main__":
    # 测试示例
    g = load_graph("./result/your_run_dir")
    names = extract_entity_names(g)
    print("图谱实体样本：", list(names)[:10])
    matched = match_entities_in_query("党委和陈义妹的历史背景", names)
    print("匹配到实体：", matched)
    subg = extract_subgraph(g, matched)
    print(summarize_subgraph(subg))
