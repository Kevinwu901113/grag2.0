#!/usr/bin/env python3
import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from sentence_transformers import SentenceTransformer

from llm.llm import LLMClient
from query.reranker import SimpleReranker
from query.query_enhancer import get_query_enhancer
from graph.graph_utils import extract_entity_names, match_entities_in_query, extract_subgraph, summarize_subgraph, retrieve_by_entity
from graph.graph_subgraph_extractor import (
    extract_multi_hop_subgraph, 
    find_intermediate_entities, 
    generate_concise_graph_summary,
    get_multi_hop_retrieval_summary
)
from query.enhanced_graph_retriever import EnhancedGraphRetriever
from utils.io import load_chunks, load_vector_index, load_graph

class BM25Retriever:
    """
    简化的BM25检索器实现
    """
    
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_texts = [chunk['text'] for chunk in chunks]
        self.doc_lengths = [len(jieba.lcut(text)) for text in self.doc_texts]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # 构建词频统计
        self.term_freq = []
        self.doc_freq = defaultdict(int)
        
        # 导入改进的分词函数
        from utils.common import improved_tokenize
        
        for text in self.doc_texts:
            terms = improved_tokenize(text)
            term_count = defaultdict(int)
            for term in terms:
                term_count[term] += 1
                
            self.term_freq.append(dict(term_count))
            
            # 统计文档频率
            for term in set(terms):
                self.doc_freq[term] += 1
                
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        BM25检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        # 使用改进的分词函数
        from utils.common import improved_tokenize
        query_terms = improved_tokenize(query)
        scores = []
        
        for i, doc_tf in enumerate(self.term_freq):
            score = 0.0
            doc_len = self.doc_lengths[i]
            
            for term in query_terms:
                if term in doc_tf:
                    tf = doc_tf[term]
                    df = self.doc_freq[term]
                    idf = np.log((len(self.chunks) - df + 0.5) / (df + 0.5))
                    
                    # BM25公式
                    score += idf * (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                    )
                    
            scores.append((score, i))
            
        # 排序并返回top_k结果
        scores.sort(reverse=True, key=lambda x: x[0])
        results = []
        
        for score, idx in scores[:top_k]:
            if score > 0:  # 只返回有匹配的结果
                result = self.chunks[idx].copy()
                result['similarity'] = float(score)
                result['retrieval_type'] = 'bm25'
                results.append(result)
                
        return results

class EnhancedRetriever:
    """
    增强检索器，整合向量检索、BM25检索和查询扩展
    """
    
    def __init__(self, config: Dict[str, Any], work_dir: str):
        self.config = config
        self.work_dir = work_dir
        self.llm_client = LLMClient(config)
        
        # 加载数据和索引
        self.chunks = self._load_chunks()
        self.vector_index, self.id_map = self._load_vector_index()
        
        # 初始化检索器
        self.bm25_retriever = BM25Retriever(self.chunks) if self.chunks else None
        self.reranker = SimpleReranker(config)
        
        # 初始化增强图检索器
        try:
            self.graph_retriever = EnhancedGraphRetriever(work_dir, self.llm_client, config)
        except Exception as e:
            print(f"⚠️ 图检索器初始化失败: {e}")
            self.graph_retriever = None
        
        # 检索配置
        retrieval_config = config.get('enhanced_retrieval', {})
        self.vector_top_k = retrieval_config.get('vector_top_k', 10)
        self.bm25_top_k = retrieval_config.get('bm25_top_k', 5)
        self.final_top_k = retrieval_config.get('final_top_k', 5)
        self.use_enhanced_graph = retrieval_config.get('use_enhanced_graph', True)
        self.enable_query_expansion = retrieval_config.get('enable_query_expansion', True)
        self.enable_bm25 = retrieval_config.get('enable_bm25', True)
        self.enable_graph_retrieval = retrieval_config.get('enable_graph_retrieval', True)
        self.enable_reranking = retrieval_config.get('enable_reranking', True)
        
        # 加载图谱相关数据
        self.graph = None
        self.entity_names = set()
        if self.enable_graph_retrieval:
            self._load_graph_data()
        
    def _load_chunks(self) -> List[Dict[str, Any]]:
        """加载文档块"""
        return load_chunks(self.work_dir)
            
    def _load_vector_index(self) -> Tuple[Any, Dict[str, Any]]:
        """加载向量索引"""
        return load_vector_index(self.work_dir)
            
    def _load_graph_data(self):
        """
        加载图谱数据
        """
        try:
            self.graph = load_graph(self.work_dir)
            if self.graph:
                self.entity_names = extract_entity_names(self.graph)
                print(f"✅ 成功加载图谱: {len(self.entity_names)}个实体")
            else:
                print("⚠️ 图谱文件不存在，跳过图谱检索")
                self.entity_names = set()
        except Exception as e:
            print(f"❌ 加载图谱数据失败: {e}")
            self.graph = None
            self.entity_names = set()
            
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        增强检索主函数
        
        Args:
            query: 查询文本
            top_k: 最终返回的结果数量
            
        Returns:
            检索结果列表
        """
        if top_k is None:
            top_k = self.final_top_k
            
        all_candidates = []
        
        # 1. 查询扩展
        queries = [query]
        if self.enable_query_expansion:
            try:
                query_enhancer = get_query_enhancer(self.config)
                queries = query_enhancer.enhance_query(query)
                print(f"🔍 查询扩展: {len(queries)}个查询")
            except Exception as e:
                print(f"查询扩展失败: {e}")
                
        # 2. 向量检索
        vector_candidates = self._vector_retrieval(queries)
        all_candidates.extend(vector_candidates)
        
        # 3. BM25检索
        bm25_candidates = []
        if self.enable_bm25 and self.bm25_retriever:
            bm25_candidates = self._bm25_retrieval(queries)
            all_candidates.extend(bm25_candidates)
            
        # 4. 图谱实体检索
        graph_candidates = []
        if self.enable_graph_retrieval and self.graph and self.entity_names:
            graph_candidates = self._graph_entity_retrieval(query)
            all_candidates.extend(graph_candidates)
            
        # 5. 增强图检索（新增）
        enhanced_graph_candidates = []
        if self.use_enhanced_graph and self.graph_retriever:
            enhanced_graph_candidates = self._enhanced_graph_retrieval(query)
            all_candidates.extend(enhanced_graph_candidates)
            
        # 6. 去重合并
        unique_candidates = self._deduplicate_candidates(all_candidates)
        
        # 6.5. 得分归一化（新增）
        if unique_candidates:
            from utils.common import normalize_scores
            unique_candidates = normalize_scores(unique_candidates, 'similarity', 'minmax')
        
        # 7. 重排序
        if unique_candidates and self.enable_reranking:
            final_results = self.reranker.rerank(query, unique_candidates, top_k)
        else:
            # 使用归一化后的得分排序
            unique_candidates.sort(key=lambda x: x.get('normalized_similarity', x.get('similarity', 0)), reverse=True)
            final_results = unique_candidates[:top_k] if unique_candidates else []
            
        print(f"📊 检索统计: 向量{len(vector_candidates)}条, BM25{len(bm25_candidates)}条, 图谱{len(graph_candidates)}条, 增强图{len(enhanced_graph_candidates)}条, 去重后{len(unique_candidates)}条, 最终{len(final_results)}条")
        
        return final_results
        
    def _vector_retrieval(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        向量检索
        """
        if not self.vector_index or not queries:
            return []
            
        all_results = []
        
        for query in queries:
            try:
                # 生成查询向量（启用文本预处理和维度验证）
                query_embeddings = self.llm_client.embed([query], normalize_text=True, validate_dim=True)
                if not query_embeddings:
                    print(f"查询向量生成失败: {query}")
                    continue
                    
                query_embedding = query_embeddings[0]
                query_vector = np.array([query_embedding], dtype=np.float32)
                
                # 检查向量有效性
                if query_vector.shape[1] == 0:
                    print(f"查询向量为空: {query}")
                    continue
                
                # 检查查询向量幅度（异常检测）
                vector_magnitude = np.linalg.norm(query_vector)
                if vector_magnitude < 1e-6:
                    print(f"⚠️ 查询向量幅度过小: {vector_magnitude}, 可能存在异常")
                    continue
                    
                # 归一化查询向量以确保计算余弦相似度
                import faiss
                faiss.normalize_L2(query_vector)
                
                # 检查归一化后的向量
                if np.any(np.isnan(query_vector)) or np.any(np.isinf(query_vector)):
                    print(f"查询向量包含无效值: {query}")
                    continue
                
                # 向量检索
                similarities, indices = self.vector_index.search(query_vector, self.vector_top_k)
                
                # 转换结果
                for sim, idx in zip(similarities[0], indices[0]):
                    if idx != -1 and str(idx) in self.id_map:
                        chunk_id = self.id_map[str(idx)]
                        chunk = next((c for c in self.chunks if c['id'] == chunk_id), None)
                        if chunk:
                            # 验证相似度分数
                            if not np.isfinite(sim) or sim < -1.1 or sim > 1.1:
                                print(f"异常相似度分数: {sim}, 跳过结果")
                                continue
                                
                            result = chunk.copy()
                            result['similarity'] = float(sim)
                            result['retrieval_type'] = 'vector'
                            result['query_variant'] = query
                            all_results.append(result)
                            
            except Exception as e:
                print(f"向量检索失败 (查询: {query}): {e}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
                
        return all_results
        
    def _bm25_retrieval(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        BM25检索
        """
        if not self.bm25_retriever or not queries:
            return []
            
        all_results = []
        
        for query in queries:
            try:
                results = self.bm25_retriever.search(query, self.bm25_top_k)
                for result in results:
                    result['query_variant'] = query
                    all_results.append(result)
            except Exception as e:
                print(f"BM25检索失败 (查询: {query}): {e}")
                
        return all_results
        
    def _graph_entity_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """
        基于图谱实体的检索，支持多跳图推理
        """
        if not self.graph or not self.entity_names or not self.chunks:
            return []
            
        try:
            # 匹配查询中的实体
            matched_entities = match_entities_in_query(query, self.entity_names)
            
            if not matched_entities:
                return []
                
            print(f"🔍 图谱检索匹配到实体: {matched_entities}")
            
            # 基于匹配的实体检索相关文档块
            graph_results = retrieve_by_entity(self.graph, matched_entities, self.chunks)
            
            # 如果匹配到多个实体，使用多跳图推理增强检索结果
            if len(matched_entities) >= 2:
                try:
                    # 生成多跳图推理摘要
                    multi_hop_summary = get_multi_hop_retrieval_summary(
                        self.graph, matched_entities, self.chunks, max_summary_length=400
                    )
                    
                    print(f"📊 多跳图推理摘要: {multi_hop_summary[:100]}...")
                    
                    # 将摘要信息添加到检索结果中
                    for result in graph_results:
                        if 'graph_summary' not in result:
                            result['graph_summary'] = multi_hop_summary
                            result['multi_hop_entities'] = list(matched_entities)
                            
                    # 寻找中介实体路径
                    intermediate_info = find_intermediate_entities(self.graph, matched_entities)
                    if intermediate_info:
                        print(f"🔗 发现{len(intermediate_info)}条中介实体路径")
                        
                        # 将中介实体信息添加到结果中
                        for result in graph_results:
                            result['intermediate_paths'] = intermediate_info[:5]  # 限制数量
                            
                except Exception as e:
                    print(f"多跳图推理处理失败: {e}")
            
            return graph_results
            
        except Exception as e:
            print(f"图谱实体检索失败: {e}")
            return []
        
    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        去重合并候选结果，支持基于ID和内容相似度的双重去重
        """
        if not candidates:
            return []
            
        # 第一步：按文档ID分组去重
        grouped_by_id = defaultdict(list)
        for candidate in candidates:
            doc_id = candidate.get('id', '')
            grouped_by_id[doc_id].append(candidate)
            
        # 合并同一文档的多个检索结果
        id_deduplicated = []
        for doc_id, group in grouped_by_id.items():
            if not group:
                continue
                
            # 选择最高分数的结果作为代表
            best_candidate = max(group, key=lambda x: x.get('similarity', 0))
            
            # 记录所有检索类型
            retrieval_types = list(set(c.get('retrieval_type', '') for c in group))
            best_candidate['retrieval_types'] = retrieval_types
            
            # 如果同时被多种方法检索到，给予额外加分（但不超过1.0）
            if len(retrieval_types) > 1:
                current_sim = best_candidate['similarity']
                if 'graph_entity' in retrieval_types:
                    best_candidate['similarity'] = min(1.0, current_sim + 0.1)
                else:
                    best_candidate['similarity'] = min(1.0, current_sim + 0.05)
                
            id_deduplicated.append(best_candidate)
        
        # 第二步：基于内容相似度的去重（处理相同内容但不同ID的情况）
        final_candidates = []
        similarity_threshold = 0.95  # 内容相似度阈值
        
        for candidate in id_deduplicated:
            is_duplicate = False
            candidate_text = candidate.get('text', '').strip()
            
            # 检查是否与已有候选结果内容过于相似
            for existing in final_candidates:
                existing_text = existing.get('text', '').strip()
                
                # 计算文本相似度（简单的字符级相似度）
                if candidate_text and existing_text:
                    # 使用Jaccard相似度进行快速比较
                    set1 = set(candidate_text)
                    set2 = set(existing_text)
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    jaccard_sim = intersection / union if union > 0 else 0
                    
                    if jaccard_sim > similarity_threshold:
                        # 内容过于相似，保留得分更高的
                        if candidate.get('similarity', 0) > existing.get('similarity', 0):
                            # 替换现有结果
                            final_candidates.remove(existing)
                            final_candidates.append(candidate)
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                final_candidates.append(candidate)
            
        return final_candidates
    
    def _enhanced_graph_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """
        增强图检索方法
        
        Args:
            query: 查询文本
            
        Returns:
            检索结果列表
        """
        if not self.graph_retriever:
            return []
            
        try:
            # 使用增强图检索器进行检索
            retrieval_results = self.graph_retriever.enhanced_entity_retrieval(
                query, 
                top_k=self.vector_top_k,  # 使用相同的top_k设置
                use_vector=True,
                use_matching=True
            )
            
            candidates = []
            
            # 处理检索到的实体
            combined_entities = retrieval_results.get('combined_entities', [])
            entity_neighborhoods = retrieval_results.get('entity_neighborhoods', {})
            
            for entity, score in combined_entities:
                # 查找包含该实体的文档块
                related_chunks = self._find_chunks_by_entity(entity)
                
                for chunk in related_chunks:
                    result = chunk.copy()
                    result['similarity'] = float(score * 0.8)  # 调整分数权重
                    result['retrieval_type'] = 'enhanced_graph'
                    result['matched_entity'] = entity
                    
                    # 添加实体邻域信息
                    if entity in entity_neighborhoods:
                        neighborhood = entity_neighborhoods[entity]
                        result['entity_context'] = {
                            'neighbors_count': neighborhood.get('subgraph_size', 0),
                            'relations_count': len(neighborhood.get('relations', []))
                        }
                    
                    candidates.append(result)
            
            # 处理实体路径信息
            entity_paths = retrieval_results.get('entity_paths', [])
            if entity_paths:
                # 为包含路径实体的文档块增加额外分数
                path_entities = set()
                for path_info in entity_paths:
                    path_entities.update(path_info.get('path', []))
                
                for candidate in candidates:
                    matched_entity = candidate.get('matched_entity', '')
                    if matched_entity in path_entities:
                        candidate['similarity'] = min(1.0, candidate['similarity'] + 0.1)
                        candidate['has_entity_path'] = True
            
            print(f"🔗 增强图检索: 找到 {len(combined_entities)} 个相关实体, {len(candidates)} 个候选文档")
            return candidates
            
        except Exception as e:
            print(f"⚠️ 增强图检索失败: {e}")
            return []
    
    def _find_chunks_by_entity(self, entity: str) -> List[Dict[str, Any]]:
        """
        根据实体查找相关的文档块
        
        Args:
            entity: 实体名称
            
        Returns:
            相关文档块列表
        """
        if not self.chunks:
            return []
            
        related_chunks = []
        
        # 简单的文本匹配查找
        for chunk in self.chunks:
            chunk_text = chunk.get('text', '').lower()
            if entity.lower() in chunk_text:
                related_chunks.append(chunk)
        
        return related_chunks