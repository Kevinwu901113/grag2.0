import networkx as nx
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
import heapq

def extract_multi_hop_subgraph(graph: nx.Graph, entities: Set[str], max_depth: int = 2) -> nx.Graph:
    """
    提取包含多个实体的多跳子图，用于发现实体间的中介路径
    
    Args:
        graph: 知识图谱
        entities: 查询中匹配到的实体集合
        max_depth: 最大搜索深度
        
    Returns:
        包含多跳路径的子图
    """
    if not entities or len(entities) < 2:
        # 如果实体少于2个，使用原有的单跳逻辑
        from graph.graph_utils import extract_subgraph
        return extract_subgraph(graph, entities, depth=1)
    
    nodes_to_include = set()
    
    # 添加所有查询实体
    for entity in entities:
        if entity in graph:
            nodes_to_include.add(entity)
    
    # 寻找实体间的最短路径和中介节点
    entity_list = list(entities)
    for i in range(len(entity_list)):
        for j in range(i + 1, len(entity_list)):
            entity1, entity2 = entity_list[i], entity_list[j]
            if entity1 in graph and entity2 in graph:
                try:
                    # 寻找最短路径
                    paths = list(nx.all_shortest_paths(graph, entity1, entity2))
                    for path in paths[:3]:  # 限制路径数量避免过多
                        if len(path) <= max_depth + 1:  # 路径长度限制
                            nodes_to_include.update(path)
                except nx.NetworkXNoPath:
                    # 如果没有直接路径，添加各自的邻居节点
                    for entity in [entity1, entity2]:
                        neighbors = list(graph.neighbors(entity))
                        nodes_to_include.update(neighbors[:5])  # 限制邻居数量
    
    # 为每个实体添加其直接邻居（深度1）
    for entity in entities:
        if entity in graph:
            neighbors = list(graph.neighbors(entity))
            nodes_to_include.update(neighbors[:10])  # 限制邻居数量
    
    return graph.subgraph(nodes_to_include).copy()

def find_intermediate_entities(graph: nx.Graph, source_entities: Set[str], 
                             target_entities: Set[str] = None) -> List[Dict[str, Any]]:
    """
    发现连接源实体和目标实体的中介实体
    
    Args:
        graph: 知识图谱
        source_entities: 源实体集合
        target_entities: 目标实体集合，如果为None则在source_entities内部寻找连接
        
    Returns:
        中介实体信息列表，包含路径和权重
    """
    if target_entities is None:
        target_entities = source_entities
    
    intermediate_paths = []
    
    for source in source_entities:
        if source not in graph:
            continue
            
        for target in target_entities:
            if target not in graph or source == target:
                continue
                
            try:
                # 寻找长度为2的路径（即有一个中介节点）
                paths = nx.all_simple_paths(graph, source, target, cutoff=2)
                for path in paths:
                    if len(path) == 3:  # 源-中介-目标
                        intermediate = path[1]
                        # 获取边的关系信息
                        rel1 = graph.get_edge_data(path[0], path[1], {}).get('relation', '关联')
                        rel2 = graph.get_edge_data(path[1], path[2], {}).get('relation', '关联')
                        
                        intermediate_paths.append({
                            'source': source,
                            'intermediate': intermediate,
                            'target': target,
                            'path': path,
                            'relations': [rel1, rel2],
                            'weight': graph.degree(intermediate)  # 使用度中心性作为权重
                        })
            except nx.NetworkXNoPath:
                continue
    
    # 按中介节点的重要性排序
    intermediate_paths.sort(key=lambda x: x['weight'], reverse=True)
    
    return intermediate_paths

def generate_concise_graph_summary(subgraph: nx.Graph, matched_entities: Set[str], 
                                 max_summary_length: int = 500) -> str:
    """
    生成简洁的图谱摘要，包含中介实体链条说明
    
    Args:
        subgraph: 子图
        matched_entities: 查询匹配的实体
        max_summary_length: 最大摘要长度
        
    Returns:
        简洁的图谱摘要
    """
    if not subgraph.edges():
        return "未发现相关实体关系。"
    
    summary_parts = []
    
    # 1. 直接关系摘要
    direct_relations = []
    for u, v, data in subgraph.edges(data=True):
        if u in matched_entities and v in matched_entities:
            relation = data.get('relation', '关联')
            direct_relations.append(f"{u}与{v}存在{relation}关系")
    
    if direct_relations:
        summary_parts.append("直接关系：" + "；".join(direct_relations[:3]))
    
    # 2. 中介实体链条说明
    intermediate_info = find_intermediate_entities(subgraph, matched_entities)
    if intermediate_info:
        # 统计最重要的中介实体
        intermediate_counter = Counter()
        intermediate_examples = {}
        
        for info in intermediate_info[:10]:  # 限制处理数量
            intermediate = info['intermediate']
            intermediate_counter[intermediate] += info['weight']
            
            if intermediate not in intermediate_examples:
                source, target = info['source'], info['target']
                rel1, rel2 = info['relations']
                intermediate_examples[intermediate] = f"{source}通过{intermediate}与{target}关联（{rel1}-{rel2}）"
        
        # 选择最重要的中介实体
        top_intermediates = intermediate_counter.most_common(3)
        if top_intermediates:
            intermediate_descriptions = []
            for intermediate, _ in top_intermediates:
                if intermediate in intermediate_examples:
                    intermediate_descriptions.append(intermediate_examples[intermediate])
            
            if intermediate_descriptions:
                summary_parts.append("中介关系：" + "；".join(intermediate_descriptions))
    
    # 3. 重要实体说明
    entity_degrees = {}
    for node in subgraph.nodes():
        node_data = subgraph.nodes[node]
        if node_data.get('type') == 'entity':
            entity_degrees[node] = subgraph.degree(node)
    
    if entity_degrees:
        # 找出度最高的实体（除了查询实体）
        other_important = {k: v for k, v in entity_degrees.items() 
                          if k not in matched_entities and v >= 2}
        if other_important:
            top_other = sorted(other_important.items(), key=lambda x: x[1], reverse=True)[:2]
            other_descriptions = [f"{entity}（连接{degree}个节点）" for entity, degree in top_other]
            if other_descriptions:
                summary_parts.append("相关实体：" + "、".join(other_descriptions))
    
    # 组合摘要并控制长度
    full_summary = "。".join(summary_parts) + "。"
    
    # 如果摘要过长，进行截断
    if len(full_summary) > max_summary_length:
        # 优先保留直接关系和中介关系
        priority_parts = []
        if len(summary_parts) > 0:
            priority_parts.append(summary_parts[0])  # 直接关系
        if len(summary_parts) > 1:
            priority_parts.append(summary_parts[1])  # 中介关系
        
        truncated_summary = "。".join(priority_parts) + "。"
        if len(truncated_summary) <= max_summary_length:
            return truncated_summary
        else:
            # 进一步截断
            return truncated_summary[:max_summary_length-3] + "..."
    
    return full_summary

def extract_shortest_paths_summary(graph: nx.Graph, entities: Set[str], 
                                  max_paths: int = 5) -> List[Dict[str, Any]]:
    """
    提取实体间的最短路径并生成摘要
    
    Args:
        graph: 知识图谱
        entities: 实体集合
        max_paths: 最大路径数量
        
    Returns:
        路径摘要列表
    """
    path_summaries = []
    entity_list = list(entities)
    
    for i in range(len(entity_list)):
        for j in range(i + 1, len(entity_list)):
            source, target = entity_list[i], entity_list[j]
            
            if source not in graph or target not in graph:
                continue
                
            try:
                # 获取最短路径
                shortest_paths = list(nx.all_shortest_paths(graph, source, target))
                
                for path_idx, path in enumerate(shortest_paths[:2]):  # 限制每对实体的路径数
                    if len(path_summaries) >= max_paths:
                        break
                        
                    # 构建路径描述
                    path_description = []
                    for k in range(len(path) - 1):
                        u, v = path[k], path[k + 1]
                        relation = graph.get_edge_data(u, v, {}).get('relation', '关联')
                        path_description.append(f"{u}-[{relation}]->{v}")
                    
                    path_summaries.append({
                        'source': source,
                        'target': target,
                        'path': path,
                        'length': len(path) - 1,
                        'description': " → ".join(path_description),
                        'weight': sum(graph.degree(node) for node in path)  # 路径权重
                    })
                    
            except nx.NetworkXNoPath:
                continue
    
    # 按路径长度和权重排序，优先选择短路径和高权重路径
    path_summaries.sort(key=lambda x: (x['length'], -x['weight']))
    
    return path_summaries[:max_paths]

def get_multi_hop_retrieval_summary(graph: nx.Graph, matched_entities: Set[str], 
                                   chunks: List[Dict], max_summary_length: int = 400) -> str:
    """
    获取多跳检索的完整摘要，整合子图信息和路径信息
    
    Args:
        graph: 知识图谱
        matched_entities: 匹配的实体
        chunks: 文档块列表
        max_summary_length: 最大摘要长度
        
    Returns:
        完整的多跳检索摘要
    """
    if len(matched_entities) < 2:
        # 单实体情况，使用简单摘要
        from graph.graph_utils import extract_subgraph, summarize_subgraph
        subgraph = extract_subgraph(graph, matched_entities, depth=1)
        return summarize_subgraph(subgraph)
    
    # 多实体情况，使用多跳分析
    multi_hop_subgraph = extract_multi_hop_subgraph(graph, matched_entities, max_depth=2)
    
    if not multi_hop_subgraph.edges():
        return f"发现{len(matched_entities)}个相关实体：{', '.join(matched_entities)}，但它们之间没有直接关联。"
    
    # 生成简洁摘要
    summary = generate_concise_graph_summary(multi_hop_subgraph, matched_entities, max_summary_length)
    
    return summary