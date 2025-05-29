from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba

class SimpleReranker:
    """
    简单的重排序器，结合多种评分策略对检索结果进行重排序
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        对候选结果进行重排序
        
        Args:
            query: 查询文本
            candidates: 候选结果列表，每个包含text和similarity字段
            top_k: 返回的最大结果数量
            
        Returns:
            重排序后的结果列表
        """
        if not candidates:
            return []
        
        # 提取候选文本
        candidate_texts = [c.get('text', '') for c in candidates]
        
        # 计算TF-IDF相似度
        tfidf_scores = self._calculate_tfidf_similarity(query, candidate_texts)
        
        # 计算关键词重叠度
        overlap_scores = self._calculate_keyword_overlap(query, candidate_texts)
        
        # 计算多样性分数
        diversity_scores = self._calculate_diversity_scores(candidate_texts)
        
        # 综合评分
        final_scores = []
        for i, candidate in enumerate(candidates):
            # 优先使用归一化后的相似度，否则使用原始相似度
            vector_score = candidate.get('normalized_similarity', candidate.get('similarity', 0.0))
            
            # TF-IDF权重
            tfidf_score = tfidf_scores[i] if i < len(tfidf_scores) else 0.0
            
            # 关键词重叠权重
            overlap_score = overlap_scores[i] if i < len(overlap_scores) else 0.0
            
            # 多样性权重
            diversity_score = diversity_scores[i] if i < len(diversity_scores) else 0.0
            
            # 综合评分（可配置权重）
            weights = self.config.get('rerank_weights', {
                'vector': 0.4,
                'tfidf': 0.3,
                'overlap': 0.2,
                'diversity': 0.1
            })
            
            final_score = (
                vector_score * weights.get('vector', 0.4) +
                tfidf_score * weights.get('tfidf', 0.3) +
                overlap_score * weights.get('overlap', 0.2) +
                diversity_score * weights.get('diversity', 0.1)
            )
            
            final_scores.append((i, final_score))
        
        # 按综合分数排序
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回重排序后的结果
        reranked_results = []
        for i, (orig_idx, score) in enumerate(final_scores[:top_k]):
            result = candidates[orig_idx].copy()
            result['rerank_score'] = score
            result['rerank_position'] = i + 1
            reranked_results.append(result)
        
        return reranked_results
    
    def _calculate_tfidf_similarity(self, query: str, candidate_texts: List[str]) -> List[float]:
        """计算TF-IDF相似度"""
        try:
            # 将查询和候选文本一起向量化
            all_texts = [query] + candidate_texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # 计算查询与每个候选的相似度
            query_vector = tfidf_matrix[0:1]
            candidate_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, candidate_vectors)[0]
            return similarities.tolist()
        except Exception:
            return [0.0] * len(candidate_texts)
    
    def _calculate_keyword_overlap(self, query: str, candidate_texts: List[str]) -> List[float]:
        """
        计算关键词重叠度
        """
        # 使用改进的分词函数
        from utils.common import improved_tokenize
        query_words = set(improved_tokenize(query))
        
        overlap_scores = []
        for text in candidate_texts:
            text_words = set(improved_tokenize(text))
            
            # 计算Jaccard相似度
            intersection = len(query_words & text_words)
            union = len(query_words | text_words)
            
            if union == 0:
                overlap_scores.append(0.0)
            else:
                overlap_scores.append(intersection / union)
                
        return overlap_scores
    
    def _calculate_diversity_scores(self, candidate_texts: List[str]) -> List[float]:
        """计算多样性分数，避免结果过于相似"""
        if len(candidate_texts) <= 1:
            return [1.0] * len(candidate_texts)
        
        try:
            # 计算候选文本之间的相似度矩阵
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(candidate_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            diversity_scores = []
            for i in range(len(candidate_texts)):
                # 计算与其他候选的平均相似度，相似度越低多样性越高
                other_similarities = [similarity_matrix[i][j] for j in range(len(candidate_texts)) if i != j]
                avg_similarity = np.mean(other_similarities) if other_similarities else 0.0
                diversity_score = 1.0 - avg_similarity  # 多样性 = 1 - 相似度
                diversity_scores.append(max(0.0, diversity_score))
            
            return diversity_scores
        except Exception:
            return [1.0] * len(candidate_texts)

class LLMReranker:
    """
    使用LLM进行重排序的高级重排序器
    """
    
    def __init__(self, llm_client, config: Dict[str, Any] = None):
        self.llm_client = llm_client
        self.config = config or {}
    
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        使用LLM对候选结果进行重排序
        
        Args:
            query: 查询文本
            candidates: 候选结果列表
            top_k: 返回的最大结果数量
            
        Returns:
            重排序后的结果列表
        """
        if not candidates or len(candidates) <= 1:
            return candidates[:top_k]
        
        try:
            # 构建LLM重排序提示
            candidate_list = []
            for i, candidate in enumerate(candidates):
                text_preview = candidate.get('text', '')[:200] + "..."
                candidate_list.append(f"{i+1}. {text_preview}")
            
            candidates_text = "\n".join(candidate_list)
            
            prompt = f"""
请根据查询问题对以下候选文档进行重排序，按照相关性从高到低排列。
只需要返回重排序后的编号，用逗号分隔。

查询问题：{query}

候选文档：
{candidates_text}

重排序结果（只返回编号）：
"""
            
            response = self.llm_client.generate(prompt)
            
            # 解析LLM返回的排序结果
            try:
                order_nums = [int(x.strip()) for x in response.strip().split(',')]
                reranked_results = []
                
                for rank, num in enumerate(order_nums[:top_k]):
                    if 1 <= num <= len(candidates):
                        result = candidates[num-1].copy()
                        result['llm_rerank_position'] = rank + 1
                        reranked_results.append(result)
                
                return reranked_results
            except (ValueError, IndexError):
                # 如果解析失败，返回原始顺序
                return candidates[:top_k]
                
        except Exception:
            # 如果LLM重排序失败，返回原始顺序
            return candidates[:top_k]
