#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级查询分类器模块
替代原有的BERT微调流程，使用简单规则和LLM zero-shot判断
"""

import re
from typing import Tuple
from llm.llm import LLMClient

class LightweightQueryClassifier:
    """轻量级查询分类器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.llm_client = None
        
        # 精确查询关键词
        self.precise_keywords = {
            '具体', '准确', '精确', '详细', '明确', '确切', '特定',
            '数字', '数据', '统计', '比例', '百分比', '时间', '日期',
            '名称', '地址', '电话', '邮箱', '网址', '价格', '费用',
            '定义', '含义', '概念', '原理', '公式', '算法', '步骤',
            '什么是', '如何', '怎么', '为什么', '哪里', '何时', '谁'
        }
        
        # 模糊查询关键词
        self.imprecise_keywords = {
            '大概', '大致', '约', '左右', '差不多', '类似', '相关',
            '总结', '概述', '简介', '介绍', '背景', '历史', '发展',
            '趋势', '前景', '影响', '意义', '作用', '优缺点',
            '比较', '对比', '区别', '联系', '关系', '分析'
        }
    
    def _count_keywords(self, query: str) -> int:
        """统计查询中的关键词数量"""
        # 简单的关键词计数：中文字符、英文单词、数字
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', query))
        english_words = len(re.findall(r'\b[a-zA-Z]+\b', query))
        numbers = len(re.findall(r'\d+', query))
        
        return chinese_chars + english_words + numbers
    
    def _rule_based_classify(self, query: str) -> Tuple[str, bool]:
        """基于规则的分类"""
        query_lower = query.lower()
        
        # 检查精确查询关键词
        precise_count = sum(1 for keyword in self.precise_keywords 
                          if keyword in query_lower)
        
        # 检查模糊查询关键词
        imprecise_count = sum(1 for keyword in self.imprecise_keywords 
                            if keyword in query_lower)
        
        # 关键词数量判断
        keyword_count = self._count_keywords(query)
        
        # 判断逻辑
        if precise_count > imprecise_count:
            return "hybrid_precise", True
        elif imprecise_count > precise_count:
            return "hybrid_imprecise", False
        else:
            # 根据关键词数量判断
            if keyword_count <= 5:
                return "hybrid_precise", True
            else:
                return "hybrid_imprecise", False
    
    def _llm_zero_shot_classify(self, query: str) -> Tuple[str, bool]:
        """使用LLM进行zero-shot分类"""
        if self.llm_client is None:
            try:
                self.llm_client = LLMClient(self.config)
            except Exception:
                # LLM不可用时回退到规则分类
                return self._rule_based_classify(query)
        
        prompt = f"""请判断以下查询是否需要精确回答：

查询：{query}

判断标准：
- 精确查询：需要具体数据、明确定义、准确信息的问题
- 模糊查询：需要概述、分析、比较、总结类的问题

请只回答"精确"或"模糊"，不要其他内容。"""
        
        try:
            response = self.llm_client.generate(prompt).strip()
            if "精确" in response:
                return "hybrid_precise", True
            else:
                return "hybrid_imprecise", False
        except Exception:
            # LLM调用失败时回退到规则分类
            return self._rule_based_classify(query)
    
    def classify(self, query: str, use_llm: bool = False) -> Tuple[str, bool]:
        """分类查询
        
        Args:
            query: 查询文本
            use_llm: 是否使用LLM进行zero-shot判断
            
        Returns:
            (mode, is_precise): 分类模式和是否精确的标志
        """
        if not query or not query.strip():
            return "hybrid_imprecise", False
        
        query = query.strip()
        
        # 根据配置决定使用哪种分类方法
        if use_llm and self.config.get("classifier", {}).get("use_llm_fallback", False):
            return self._llm_zero_shot_classify(query)
        else:
            return self._rule_based_classify(query)


def classify_query_lightweight(query: str, config: dict = None) -> Tuple[str, bool]:
    """轻量级查询分类函数
    
    兼容原有接口的便捷函数
    
    Args:
        query: 查询文本
        config: 配置字典
        
    Returns:
        (mode, is_precise): 分类模式和是否精确的标志
    """
    classifier = LightweightQueryClassifier(config)
    use_llm = config.get("classifier", {}).get("use_llm_fallback", False) if config else False
    return classifier.classify(query, use_llm=use_llm)


# 兼容原有BERT分类器接口
def classify_query_bert(query: str, model=None, tokenizer=None, label_encoder=None):
    """兼容原有BERT分类器接口的函数
    
    为了保持向后兼容性，这个函数会调用轻量级分类器
    """
    return classify_query_lightweight(query)