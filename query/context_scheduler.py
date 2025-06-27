#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优先级感知上下文调度器 (Priority-Aware Context Scheduler)

该模块实现了一个智能的上下文调度机制，根据相关度、结构权重和上下文多样性
动态决定拼接哪些片段作为 LLM 输入上下文。
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

from llm.llm import LLMClient
from utils.common import normalize_scores, improved_tokenize, safe_divide


class PriorityContextScheduler:
    """
    优先级上下文调度器
    
    根据检索相关度、结构权重和多样性动态选择最优的上下文片段组合
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化调度器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.scheduler_config = config.get('context_scheduler', {})
        
        # 调度器开关
        self.enabled = self.scheduler_config.get('enabled', True)
        
        # 权重配置
        weights = self.scheduler_config.get('weights', {})
        self.relevance_weight = weights.get('relevance', 0.5)  # 相关度权重
        self.structure_weight = weights.get('structure', 0.3)  # 结构权重
        self.diversity_weight = weights.get('diversity', 0.2)  # 多样性权重
        
        # Token限制配置
        self.max_tokens = self.scheduler_config.get('max_tokens', 8000)
        self.min_candidates = self.scheduler_config.get('min_candidates', 3)
        self.max_candidates = self.scheduler_config.get('max_candidates', 10)
        
        # 多样性配置
        self.diversity_threshold = self.scheduler_config.get('diversity_threshold', 0.85)
        self.min_diversity_score = self.scheduler_config.get('min_diversity_score', 0.3)
        
        # 结构权重配置
        self.pagerank_bonus = self.scheduler_config.get('pagerank_bonus', 0.2)
        self.multi_source_bonus = self.scheduler_config.get('multi_source_bonus', 0.1)
        self.graph_entity_bonus = self.scheduler_config.get('graph_entity_bonus', 0.15)
        
        # 初始化LLM客户端用于向量计算
        self.llm_client = LLMClient(config)
        
        print(f"📋 优先级调度器初始化: 启用={self.enabled}, 权重=[相关度:{self.relevance_weight}, 结构:{self.structure_weight}, 多样性:{self.diversity_weight}]")
    
    def schedule_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对候选片段进行优先级排序与筛选
        
        Args:
            candidates: 候选片段列表
            
        Returns:
            最终用于上下文拼接的片段集合
        """
        if not self.enabled:
            # 如果未启用调度器，返回原有逻辑（前5个）
            return candidates[:5]
        
        if not candidates:
            return []
        
        print(f"\n📋 [优先级调度] 开始处理 {len(candidates)} 个候选片段")
        
        # 1. 计算每个候选片段的优先级分数
        scored_candidates = self._compute_priority_scores(candidates)
        
        # 2. 基于优先级和多样性选择最优组合
        selected_candidates = self._select_optimal_combination(scored_candidates)
        
        # 3. 验证token限制
        final_candidates = self._enforce_token_limit(selected_candidates)
        
        print(f"📋 [优先级调度] 最终选择 {len(final_candidates)} 个片段")
        
        return final_candidates
    
    def _compute_priority_scores(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        计算每个候选片段的优先级分数
        
        Args:
            candidates: 候选片段列表
            
        Returns:
            包含优先级分数的候选片段列表
        """
        scored_candidates = []
        
        for candidate in candidates:
            # 1. 检索相关度分数
            relevance_score = self._compute_relevance_score(candidate)
            
            # 2. 结构权重分数
            structure_score = self._compute_structure_score(candidate)
            
            # 3. 计算综合优先级分数（暂不考虑多样性，后续在选择阶段处理）
            priority_score = (
                self.relevance_weight * relevance_score +
                self.structure_weight * structure_score
            )
            
            # 添加分数信息到候选片段
            enhanced_candidate = candidate.copy()
            enhanced_candidate.update({
                'relevance_score': relevance_score,
                'structure_score': structure_score,
                'priority_score': priority_score
            })
            
            scored_candidates.append(enhanced_candidate)
        
        # 按优先级分数排序
        scored_candidates.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return scored_candidates
    
    def _compute_relevance_score(self, candidate: Dict[str, Any]) -> float:
        """
        计算检索相关度分数
        
        Args:
            candidate: 候选片段
            
        Returns:
            相关度分数 (0-1)
        """
        # 优先使用归一化相似度，其次使用原始相似度
        if 'normalized_similarity' in candidate:
            base_score = candidate['normalized_similarity']
        elif 'similarity' in candidate:
            base_score = candidate['similarity']
        else:
            base_score = 0.5  # 默认中等相关度
        
        # 确保分数在合理范围内
        base_score = max(0.0, min(1.0, float(base_score)))
        
        # 如果有LLM重排序分数，给予额外加权
        if candidate.get('source') == 'llm_rerank' or 'rerank_score' in candidate:
            rerank_bonus = 0.1
            base_score = min(1.0, base_score + rerank_bonus)
        
        return base_score
    
    def _compute_structure_score(self, candidate: Dict[str, Any]) -> float:
        """
        计算结构权重分数
        
        Args:
            candidate: 候选片段
            
        Returns:
            结构权重分数 (0-1)
        """
        structure_score = 0.0
        
        # 1. PageRank实体加权
        if self._has_high_pagerank_entities(candidate):
            structure_score += self.pagerank_bonus
        
        # 2. 多源检索命中加权
        if self._is_multi_source_hit(candidate):
            structure_score += self.multi_source_bonus
        
        # 3. 图谱实体相关性加权
        if self._has_graph_entities(candidate):
            structure_score += self.graph_entity_bonus
        
        # 4. 检索类型加权
        retrieval_type_bonus = self._get_retrieval_type_bonus(candidate)
        structure_score += retrieval_type_bonus
        
        # 确保分数在合理范围内
        return max(0.0, min(1.0, structure_score))
    
    def _has_high_pagerank_entities(self, candidate: Dict[str, Any]) -> bool:
        """
        检查候选片段是否包含高PageRank实体
        
        Args:
            candidate: 候选片段
            
        Returns:
            是否包含高PageRank实体
        """
        # 检查是否有PageRank相关信息
        if 'pagerank_score' in candidate:
            return candidate['pagerank_score'] > 0.1
        
        # 检查检索类型是否为图谱相关
        retrieval_types = candidate.get('retrieval_types', [])
        if isinstance(retrieval_types, list):
            return any('graph' in rt.lower() or 'entity' in rt.lower() for rt in retrieval_types)
        
        retrieval_type = candidate.get('retrieval_type', '')
        return 'graph' in retrieval_type.lower() or 'entity' in retrieval_type.lower()
    
    def _is_multi_source_hit(self, candidate: Dict[str, Any]) -> bool:
        """
        检查是否为多源检索命中
        
        Args:
            candidate: 候选片段
            
        Returns:
            是否为多源检索命中
        """
        # 检查是否有多个检索类型
        retrieval_types = candidate.get('retrieval_types', [])
        if isinstance(retrieval_types, list) and len(retrieval_types) > 1:
            return True
        
        # 检查是否有多源标记
        return candidate.get('multi_source', False)
    
    def _has_graph_entities(self, candidate: Dict[str, Any]) -> bool:
        """
        检查是否包含图谱实体
        
        Args:
            candidate: 候选片段
            
        Returns:
            是否包含图谱实体
        """
        # 检查检索类型
        retrieval_type = candidate.get('retrieval_type', '')
        retrieval_types = candidate.get('retrieval_types', [])
        
        if isinstance(retrieval_types, list):
            graph_types = ['graph', 'entity', 'enhanced_graph']
            return any(any(gt in rt.lower() for gt in graph_types) for rt in retrieval_types)
        
        graph_keywords = ['graph', 'entity', 'enhanced_graph']
        return any(keyword in retrieval_type.lower() for keyword in graph_keywords)
    
    def _get_retrieval_type_bonus(self, candidate: Dict[str, Any]) -> float:
        """
        根据检索类型给予加权
        
        Args:
            candidate: 候选片段
            
        Returns:
            检索类型加权分数
        """
        retrieval_type = candidate.get('retrieval_type', '').lower()
        retrieval_types = candidate.get('retrieval_types', [])
        
        # 检索类型权重映射
        type_weights = {
            'enhanced_graph': 0.15,
            'graph': 0.1,
            'entity': 0.1,
            'vector': 0.05,
            'bm25': 0.05,
            'llm_rerank': 0.1
        }
        
        max_bonus = 0.0
        
        # 检查单一检索类型
        for type_name, weight in type_weights.items():
            if type_name in retrieval_type:
                max_bonus = max(max_bonus, weight)
        
        # 检查多检索类型
        if isinstance(retrieval_types, list):
            for rt in retrieval_types:
                rt_lower = rt.lower()
                for type_name, weight in type_weights.items():
                    if type_name in rt_lower:
                        max_bonus = max(max_bonus, weight)
        
        return max_bonus
    
    def _select_optimal_combination(self, scored_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于优先级和多样性选择最优组合
        
        Args:
            scored_candidates: 已评分的候选片段列表
            
        Returns:
            选择的最优片段组合
        """
        if not scored_candidates:
            return []
        
        selected = []
        remaining = scored_candidates.copy()
        
        # 首先选择优先级最高的片段
        selected.append(remaining.pop(0))
        
        # 逐步添加片段，考虑多样性
        while remaining and len(selected) < self.max_candidates:
            best_candidate = None
            best_score = -1
            best_idx = -1
            
            for idx, candidate in enumerate(remaining):
                # 计算与已选片段的多样性分数
                diversity_score = self._compute_diversity_score(candidate, selected)
                
                # 综合优先级和多样性计算最终分数
                final_score = (
                    (self.relevance_weight + self.structure_weight) * candidate['priority_score'] +
                    self.diversity_weight * diversity_score
                )
                
                if final_score > best_score:
                    best_score = final_score
                    best_candidate = candidate
                    best_idx = idx
            
            # 添加最佳候选片段
            if best_candidate and best_score > self.min_diversity_score:
                selected.append(best_candidate)
                remaining.pop(best_idx)
            else:
                break
        
        # 确保至少有最小数量的候选片段
        while len(selected) < self.min_candidates and remaining:
            selected.append(remaining.pop(0))
        
        return selected
    
    def _compute_diversity_score(self, candidate: Dict[str, Any], selected: List[Dict[str, Any]]) -> float:
        """
        计算候选片段与已选片段的多样性分数
        
        Args:
            candidate: 候选片段
            selected: 已选择的片段列表
            
        Returns:
            多样性分数 (0-1，越高表示越多样)
        """
        if not selected:
            return 1.0
        
        candidate_text = candidate.get('text', '')
        if not candidate_text:
            return 0.5
        
        # 计算与所有已选片段的相似度
        similarities = []
        for selected_item in selected:
            selected_text = selected_item.get('text', '')
            if selected_text:
                similarity = self._compute_text_similarity(candidate_text, selected_text)
                similarities.append(similarity)
        
        if not similarities:
            return 1.0
        
        # 使用最大相似度来计算多样性（最大相似度越低，多样性越高）
        max_similarity = max(similarities)
        diversity_score = 1.0 - max_similarity
        
        # 如果相似度过高，给予惩罚
        if max_similarity > self.diversity_threshold:
            diversity_score *= 0.5
        
        return max(0.0, min(1.0, diversity_score))
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            # 使用简单的词汇重叠计算相似度
            tokens1 = set(improved_tokenize(text1))
            tokens2 = set(improved_tokenize(text2))
            
            if not tokens1 or not tokens2:
                return 0.0
            
            # 计算Jaccard相似度
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            
            return safe_divide(intersection, union, 0.0)
            
        except Exception as e:
            print(f"⚠️ 文本相似度计算失败: {e}")
            return 0.5
    
    def _enforce_token_limit(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        根据token限制筛选候选片段
        
        Args:
            candidates: 候选片段列表
            
        Returns:
            符合token限制的片段列表
        """
        if not candidates:
            return []
        
        selected = []
        total_tokens = 0
        
        for candidate in candidates:
            # 估算文本token数量（简单估算：中文字符数 + 英文单词数）
            text = candidate.get('text', '')
            estimated_tokens = self._estimate_tokens(text)
            
            # 检查是否超过限制
            if total_tokens + estimated_tokens <= self.max_tokens:
                selected.append(candidate)
                total_tokens += estimated_tokens
            else:
                # 如果添加当前片段会超过限制，检查是否至少有最小数量
                if len(selected) >= self.min_candidates:
                    break
                else:
                    # 如果还没达到最小数量，尝试截断当前片段
                    remaining_tokens = self.max_tokens - total_tokens
                    if remaining_tokens > 100:  # 至少保留100个token
                        truncated_candidate = self._truncate_candidate(candidate, remaining_tokens)
                        selected.append(truncated_candidate)
                        break
        
        print(f"📋 [Token限制] 总计约 {total_tokens} tokens，选择 {len(selected)} 个片段")
        
        return selected
    
    def _estimate_tokens(self, text: str) -> int:
        """
        估算文本的token数量
        
        Args:
            text: 输入文本
            
        Returns:
            估算的token数量
        """
        if not text:
            return 0
        
        # 简单估算：中文字符按1.5个token计算，英文单词按1个token计算
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_words = len(text.split()) - chinese_chars  # 粗略估算英文单词数
        
        estimated_tokens = int(chinese_chars * 1.5 + english_words)
        return max(estimated_tokens, len(text) // 4)  # 最少按4字符1token计算
    
    def _truncate_candidate(self, candidate: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """
        截断候选片段以符合token限制
        
        Args:
            candidate: 候选片段
            max_tokens: 最大token数量
            
        Returns:
            截断后的候选片段
        """
        text = candidate.get('text', '')
        if not text:
            return candidate
        
        # 简单截断：按字符数估算
        max_chars = max_tokens * 3  # 粗略估算
        if len(text) <= max_chars:
            return candidate
        
        # 截断文本，尽量在句号处截断
        truncated_text = text[:max_chars]
        last_period = truncated_text.rfind('。')
        if last_period > max_chars * 0.7:  # 如果句号位置合理
            truncated_text = truncated_text[:last_period + 1]
        else:
            truncated_text += '...'
        
        # 创建截断后的候选片段
        truncated_candidate = candidate.copy()
        truncated_candidate['text'] = truncated_text
        truncated_candidate['truncated'] = True
        
        return truncated_candidate