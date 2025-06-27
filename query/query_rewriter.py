#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Rewriter Module

该模块实现查询改写功能，通过LLM将用户原始查询改写为更清晰、完整或语义丰富的版本，
从而提高图结构检索和语义匹配的效果。

主要功能：
1. 支持多种改写策略：clarify（补充背景）、expand（引入相关表达）、simplify（消歧义）
2. 基于上下文的多轮改写策略
3. 改写效果评估机制
4. 与现有LLM调用逻辑完全复用
"""

from typing import Dict, List, Optional, Tuple
from llm.llm import LLMClient
from llm.prompt import get_query_rewrite_prompt
import time


class QueryRewriter:
    """查询改写器"""
    
    def __init__(self, config: dict):
        """
        初始化查询改写器
        
        Args:
            config: 配置字典，包含LLM配置和改写策略配置
        """
        self.config = config
        self.llm_client = LLMClient(config)
        self.rewrite_config = config.get("query_rewrite", {})
        
        # 改写策略配置
        self.enabled_strategies = self.rewrite_config.get("strategies", ["clarify"])
        self.max_iterations = self.rewrite_config.get("max_iterations", 1)
        self.enable_context_rewrite = self.rewrite_config.get("enable_context_rewrite", False)
        self.enable_evaluation = self.rewrite_config.get("enable_evaluation", False)
        
    def rewrite_query(self, 
                     original_query: str, 
                     strategy: str = "auto", 
                     context: Optional[str] = None,
                     entities: Optional[List[str]] = None) -> Optional[Dict]:
        """
        改写查询
        
        Args:
            original_query: 原始查询
            strategy: 改写策略，可选值：auto, clarify, expand, simplify
            context: 上下文信息（用于多轮改写）
            entities: 相关实体列表（用于实体感知改写）
            
        Returns:
            包含改写结果的字典，如果改写失败则返回None
        """
        try:
            start_time = time.time()
            
            # 自动选择策略
            if strategy == "auto":
                strategy = self._select_strategy(original_query)
                
            # 执行改写
            rewritten_query = self._execute_rewrite(
                original_query, strategy, context, entities
            )
            
            # 确保改写结果不为None
            if rewritten_query is None:
                print(f"⚠️ 改写执行返回None，使用原始查询")
                rewritten_query = original_query
            
            # 评估改写效果
            evaluation = None
            if self.enable_evaluation:
                evaluation = self._evaluate_rewrite(original_query, rewritten_query)
                
            result = {
                "original_query": original_query,
                "rewritten_query": rewritten_query,
                "strategy": strategy,
                "processing_time": time.time() - start_time,
                "evaluation": evaluation,
                "context_used": context is not None,
                "entities_used": entities is not None
            }
            
            return result
            
        except Exception as e:
            print(f"❌ 查询改写过程中发生异常: {e}")
            return None
    
    def rewrite_with_context(self, 
                           original_query: str,
                           retrieved_context: str,
                           strategy: str = "context_aware") -> Dict:
        """
        基于检索上下文的改写
        
        Args:
            original_query: 原始查询
            retrieved_context: 检索到的上下文信息
            strategy: 改写策略
            
        Returns:
            改写结果字典
        """
        if not self.enable_context_rewrite:
            return {
                "original_query": original_query,
                "rewritten_query": original_query,
                "strategy": "disabled",
                "processing_time": 0,
                "evaluation": None,
                "context_used": False,
                "entities_used": False
            }
            
        return self.rewrite_query(
            original_query, 
            strategy="context_aware", 
            context=retrieved_context
        )
    
    def multi_round_rewrite(self, 
                          original_query: str,
                          max_rounds: Optional[int] = None) -> List[Dict]:
        """
        多轮改写策略
        
        Args:
            original_query: 原始查询
            max_rounds: 最大改写轮数
            
        Returns:
            每轮改写结果的列表
        """
        if max_rounds is None:
            max_rounds = self.max_iterations
            
        results = []
        current_query = original_query
        
        for round_num in range(max_rounds):
            # 选择当前轮次的策略
            if round_num < len(self.enabled_strategies):
                strategy = self.enabled_strategies[round_num]
            else:
                strategy = "refine"
                
            # 执行改写
            result = self.rewrite_query(current_query, strategy)
            results.append({
                "round": round_num + 1,
                "strategy": strategy,
                **result
            })
            
            # 更新当前查询为改写后的查询
            current_query = result["rewritten_query"]
            
            # 如果改写后查询与原查询相同，提前结束
            if current_query == result["original_query"]:
                break
                
        return results
    
    def _select_strategy(self, query: str) -> str:
        """
        自动选择改写策略
        
        Args:
            query: 查询文本
            
        Returns:
            选择的策略名称
        """
        query_length = len(query)
        
        # 基于查询长度和复杂度的简单策略选择
        if query_length < 10:
            return "expand"  # 短查询需要扩展
        elif query_length > 100:
            return "simplify"  # 长查询需要简化
        elif "什么" in query or "如何" in query or "为什么" in query:
            return "clarify"  # 疑问句需要澄清
        else:
            return "clarify"  # 默认策略
    
    def _execute_rewrite(self, 
                        query: str, 
                        strategy: str,
                        context: Optional[str] = None,
                        entities: Optional[List[str]] = None) -> str:
        """
        执行查询改写
        
        Args:
            query: 原始查询
            strategy: 改写策略
            context: 上下文信息
            entities: 相关实体
            
        Returns:
            改写后的查询，确保不返回None
        """
        try:
            # 构建改写提示词
            prompt = get_query_rewrite_prompt(
                query, strategy, context, entities
            )
            
            # 调用LLM进行改写
            response = self.llm_client.generate(prompt)
            
            # 确保response不为None
            if response is None:
                print(f"⚠️ LLM返回None响应，返回原查询: {query}")
                return query
            
            # 解析改写结果
            rewritten_query = self._parse_rewrite_response(response)
            
            # 验证改写结果
            if not rewritten_query or rewritten_query.strip() == "":
                print(f"⚠️ 改写结果为空，返回原查询: {query}")
                return query
                
            return rewritten_query
            
        except Exception as e:
            print(f"❌ 查询改写失败: {e}")
            return query
    
    def _parse_rewrite_response(self, response: str) -> str:
        """
        解析LLM改写响应
        
        Args:
            response: LLM响应文本
            
        Returns:
            解析出的改写查询，确保不返回None或空字符串
        """
        # 确保输入不为None
        if response is None:
            return ""
            
        # 移除常见的标记和格式
        response = response.strip()
        
        # 如果响应为空，直接返回空字符串
        if not response:
            return ""
        
        # 查找改写后的查询（通常在特定标记之间）
        markers = ["改写后的查询：", "改写结果：", "重写查询：", "优化查询："]
        for marker in markers:
            if marker in response:
                parts = response.split(marker, 1)
                if len(parts) > 1:
                    result = parts[1].strip().strip('"').strip("'")
                    if result:  # 确保结果不为空
                        return result
        
        # 如果没有找到标记，尝试提取第一行非空内容
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            result = lines[0].strip('"').strip("'")
            if result:  # 确保结果不为空
                return result
            
        # 最后的fallback，返回原始响应（如果不为空）
        return response if response else ""
    
    def _evaluate_rewrite(self, original: str, rewritten: str) -> Dict:
        """
        评估改写效果
        
        Args:
            original: 原始查询
            rewritten: 改写后查询
            
        Returns:
            评估结果字典
        """
        try:
            # 基本指标
            length_change = len(rewritten) - len(original)
            similarity = self._calculate_similarity(original, rewritten)
            
            # 复杂度评估
            original_complexity = self._calculate_complexity(original)
            rewritten_complexity = self._calculate_complexity(rewritten)
            complexity_change = rewritten_complexity - original_complexity
            
            evaluation = {
                "length_change": length_change,
                "similarity": similarity,
                "complexity_change": complexity_change,
                "is_meaningful_change": abs(similarity - 1.0) > 0.1,  # 相似度变化超过10%
                "recommendation": "accept" if similarity > 0.3 and similarity < 0.9 else "reject"
            }
            
            return evaluation
            
        except Exception as e:
            print(f"⚠️ 改写评估失败: {e}")
            return {"error": str(e)}
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数 (0-1)
        """
        # 简单的字符级相似度计算
        if not text1 or not text2:
            return 0.0
            
        # 使用Jaccard相似度
        set1 = set(text1)
        set2 = set(text2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_complexity(self, text: str) -> float:
        """
        计算文本复杂度
        
        Args:
            text: 输入文本
            
        Returns:
            复杂度分数
        """
        if not text:
            return 0.0
            
        # 基于长度、词汇多样性等的简单复杂度计算
        length_score = len(text) / 100.0  # 长度归一化
        unique_chars = len(set(text))
        diversity_score = unique_chars / len(text) if len(text) > 0 else 0
        
        return length_score * 0.7 + diversity_score * 0.3


def is_query_rewrite_enabled(config: dict) -> bool:
    """
    检查是否启用查询改写功能
    
    Args:
        config: 配置字典
        
    Returns:
        是否启用改写功能
    """
    return config.get("query_rewrite", {}).get("enabled", False)