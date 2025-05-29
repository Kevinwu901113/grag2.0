import json
import networkx as nx
import re
from networkx.readwrite import json_graph
import jieba
from utils.io import load_graph

def extract_entity_names(graph: nx.Graph) -> set[str]:
    return {
        node for node, data in graph.nodes(data=True)
        if data.get("type") == "entity"
    }

def match_entities_in_query(query: str, entity_names: set[str]) -> set[str]:
    tokens = list(jieba.cut(query))
    joined = "".join(tokens)
    matched = {ent for ent in entity_names if ent in joined}
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
        chunk_topic_id = chunk.get('topic_id')
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
