import os
import json
import faiss
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import jieba
from sklearn.metrics.pairwise import cosine_similarity

from llm.llm import LLMClient
from graph.graph_utils import extract_entity_names, match_entities_in_query
from utils.io import load_entity_vector_index, load_graph

class EnhancedGraphRetriever:
    """
    增强的图检索器，结合向量相似度和PageRank算法
    """
    
    def __init__(self, work_dir: str, llm_client: LLMClient, config: dict = None):
        self.work_dir = work_dir
        self.llm_client = llm_client
        self.config = config or {}
        
        # 加载知识图谱
        self.graph = load_graph(work_dir)
        if self.graph is None:
            print("⚠️ 未找到知识图谱，图检索功能将不可用")
            return
            
        # 加载实体向量索引
        self.entity_index, self.entity_id_map, self.entity_metadata = load_entity_vector_index(work_dir)
        
        # 计算PageRank分数
        self.pagerank_scores = self._compute_pagerank_scores()
        
        # 构建实体到索引的映射
        self.entity_to_idx = {entity: idx for idx, entity in self.entity_id_map.items()}
        
        print(f"✅ 图检索器初始化完成: {len(self.entity_id_map)} 个实体")
    
    def _compute_pagerank_scores(self) -> Dict[str, float]:
        """
        计算图中所有节点的PageRank分数
        
        Returns:
            节点到PageRank分数的映射
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            return {}
            
        try:
            # 只对实体节点计算PageRank
            entity_subgraph = self.graph.subgraph([
                node for node, data in self.graph.nodes(data=True) 
                if data.get('type') == 'entity'
            ])
            
            if entity_subgraph.number_of_nodes() == 0:
                return {}
                
            pagerank_scores = nx.pagerank(entity_subgraph, alpha=0.85, max_iter=100)
            return pagerank_scores
        except Exception as e:
            print(f"⚠️ PageRank计算失败: {e}")
            return {}
    
    def retrieve_by_entity_similarity(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        基于向量相似度检索相关实体
        
        Args:
            query: 查询文本
            top_k: 返回的实体数量
            
        Returns:
            (实体名, 相似度分数)的列表
        """
        if self.entity_index is None or not self.entity_id_map:
            return []
            
        try:
            # 生成查询向量（启用文本预处理和维度验证）
            query_embeddings = self.llm_client.embed([query], normalize_text=True, validate_dim=True)
            if not query_embeddings:
                print(f"实体查询向量生成失败: {query}")
                return {'entities': [], 'scores': [], 'total_found': 0}
                
            query_embedding = query_embeddings[0]
            query_vector = np.array([query_embedding]).astype('float32')
            
            # 检查向量有效性
            if query_vector.shape[1] == 0:
                print(f"实体查询向量为空: {query}")
                return {'entities': [], 'scores': [], 'total_found': 0}
            
            faiss.normalize_L2(query_vector)
            
            # 检查归一化后的向量
            if np.any(np.isnan(query_vector)) or np.any(np.isinf(query_vector)):
                print(f"实体查询向量包含无效值: {query}")
                return {'entities': [], 'scores': [], 'total_found': 0}
            
            # 搜索最相似的实体
            scores, indices = self.entity_index.search(query_vector, min(top_k * 2, len(self.entity_id_map)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS返回-1表示无效索引
                    continue
                    
                # 验证相似度分数
                if not np.isfinite(score) or score < -1.1 or score > 1.1:
                    print(f"实体检索异常相似度分数: {score}, 跳过结果")
                    continue
                    
                entity = self.entity_id_map.get(str(idx))
                if entity:
                    # 结合PageRank分数
                    pagerank_score = self.pagerank_scores.get(entity, 0.0)
                    combined_score = 0.7 * score + 0.3 * pagerank_score  # 权重可配置
                    results.append((entity, combined_score))
            
            # 按组合分数排序
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"⚠️ 实体向量检索失败: {e}")
            return []
    
    def retrieve_by_entity_matching(self, query: str) -> List[str]:
        """
        基于字符串匹配检索实体
        
        Args:
            query: 查询文本
            
        Returns:
            匹配的实体列表
        """
        if self.graph is None:
            return []
            
        try:
            matched_entities = match_entities_in_query(query, self.graph)
            return matched_entities
        except Exception as e:
            print(f"⚠️ 实体匹配失败: {e}")
            return []
    
    def get_entity_neighborhood(self, entity: str, hops: int = 2) -> Dict[str, Any]:
        """
        获取实体的邻域信息
        
        Args:
            entity: 实体名
            hops: 跳数
            
        Returns:
            邻域信息字典
        """
        if self.graph is None or entity not in self.graph:
            return {}
            
        try:
            # 获取多跳邻居
            neighbors = set([entity])
            current_level = {entity}
            
            for _ in range(hops):
                next_level = set()
                for node in current_level:
                    # 获取所有邻居（入边和出边）
                    next_level.update(self.graph.neighbors(node))
                    next_level.update(self.graph.predecessors(node))
                
                current_level = next_level - neighbors
                neighbors.update(current_level)
                
                if not current_level:  # 没有更多邻居
                    break
            
            # 提取子图
            subgraph = self.graph.subgraph(neighbors)
            
            # 收集关系信息
            relations = []
            for source, target, data in subgraph.edges(data=True):
                relation = data.get('relation', '相关')
                relations.append({
                    'source': source,
                    'target': target,
                    'relation': relation
                })
            
            return {
                'center_entity': entity,
                'neighbors': list(neighbors),
                'relations': relations,
                'subgraph_size': len(neighbors)
            }
            
        except Exception as e:
            print(f"⚠️ 获取实体邻域失败: {e}")
            return {}
    
    def find_entity_paths(self, source_entities: List[str], target_entities: List[str], max_length: int = 3) -> List[Dict[str, Any]]:
        """
        查找实体间的路径
        
        Args:
            source_entities: 源实体列表
            target_entities: 目标实体列表
            max_length: 最大路径长度
            
        Returns:
            路径信息列表
        """
        if self.graph is None:
            return []
            
        paths = []
        
        try:
            for source in source_entities:
                for target in target_entities:
                    if source == target:
                        continue
                        
                    if source not in self.graph or target not in self.graph:
                        continue
                    
                    try:
                        # 查找最短路径
                        if nx.has_path(self.graph, source, target):
                            path = nx.shortest_path(self.graph, source, target)
                            if len(path) <= max_length + 1:  # +1因为路径包含起点和终点
                                # 收集路径上的关系
                                path_relations = []
                                for i in range(len(path) - 1):
                                    edge_data = self.graph.get_edge_data(path[i], path[i+1])
                                    relation = edge_data.get('relation', '相关') if edge_data else '相关'
                                    path_relations.append(relation)
                                
                                paths.append({
                                    'source': source,
                                    'target': target,
                                    'path': path,
                                    'relations': path_relations,
                                    'length': len(path) - 1
                                })
                    except nx.NetworkXNoPath:
                        continue
                        
        except Exception as e:
            print(f"⚠️ 查找实体路径失败: {e}")
            
        # 按路径长度排序
        paths.sort(key=lambda x: x['length'])
        return paths
    
    def get_top_entities_by_importance(self, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        根据重要性（PageRank分数）获取顶级实体
        
        Args:
            top_k: 返回的实体数量
            
        Returns:
            (实体名, 重要性分数)的列表
        """
        if not self.pagerank_scores:
            return []
            
        sorted_entities = sorted(self.pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_entities[:top_k]
    
    def enhanced_entity_retrieval(self, query: str, top_k: int = 10, use_vector: bool = True, use_matching: bool = True) -> Dict[str, Any]:
        """
        增强的实体检索，结合多种方法
        
        Args:
            query: 查询文本
            top_k: 返回的实体数量
            use_vector: 是否使用向量检索
            use_matching: 是否使用字符串匹配
            
        Returns:
            检索结果字典
        """
        results = {
            'vector_entities': [],
            'matched_entities': [],
            'combined_entities': [],
            'entity_neighborhoods': {},
            'entity_paths': []
        }
        
        # 向量检索
        if use_vector and self.entity_index is not None:
            vector_entities = self.retrieve_by_entity_similarity(query, top_k)
            results['vector_entities'] = vector_entities
        
        # 字符串匹配
        if use_matching:
            matched_entities = self.retrieve_by_entity_matching(query)
            results['matched_entities'] = matched_entities
        
        # 合并结果
        all_entities = set()
        entity_scores = {}
        
        # 添加向量检索结果
        for entity, score in results['vector_entities']:
            all_entities.add(entity)
            entity_scores[entity] = entity_scores.get(entity, 0) + score
        
        # 添加匹配结果（给予固定分数）
        for entity in results['matched_entities']:
            all_entities.add(entity)
            entity_scores[entity] = entity_scores.get(entity, 0) + 1.0  # 匹配实体给予额外分数
        
        # 按分数排序
        combined_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results['combined_entities'] = combined_entities
        
        # 获取实体邻域
        for entity, _ in combined_entities[:5]:  # 只为前5个实体获取邻域
            neighborhood = self.get_entity_neighborhood(entity, hops=2)
            if neighborhood:
                results['entity_neighborhoods'][entity] = neighborhood
        
        # 查找实体间路径
        if len(combined_entities) >= 2:
            top_entities = [entity for entity, _ in combined_entities[:3]]
            paths = self.find_entity_paths(top_entities[:2], top_entities[1:], max_length=3)
            results['entity_paths'] = paths
        
        return results