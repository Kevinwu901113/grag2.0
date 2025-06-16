#!/usr/bin/env python3
import re
from typing import List, Dict, Any
from llm.llm import LLMClient

# 实体查询模式的正则表达式
ENTITY_QUERY_PATTERNS = [
    r'(.+?)是谁',  # 某人是谁
    r'谁是(.+?)(的)?',  # 谁是某职位/某人
    r'(.+?)(的)?(.+?)是(谁|什么|哪个|哪位)',  # 某地的某职位是谁
    r'(.+?)在(哪里|哪儿|哪个|什么地方)',  # 某物在哪里
    r'(.+?)(的)?(地址|位置|电话|联系方式)',  # 某地的地址/位置/电话
    r'(.+?)是(什么|哪个)',  # 某物是什么
]

class QueryEnhancer:
    """
    查询增强器，负责生成查询的扩展形式
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm_client = LLMClient(config)
        
    def enhance_query(self, query: str) -> List[str]:
        """
        生成查询的扩展形式
        
        Args:
            query: 原始查询
            
        Returns:
            包含原始查询和扩展查询的列表
        """
        enhanced_queries = [query]  # 始终包含原始查询
        
        try:
            # 使用LLM生成扩展查询
            expansion_prompt = self._build_expansion_prompt(query)
            response = self.llm_client.generate(expansion_prompt)
            
            # 解析LLM响应
            expanded_queries = self._parse_expansion_response(response)
            enhanced_queries.extend(expanded_queries)
            
        except Exception as e:
            print(f"查询扩展失败: {e}")
            # 如果LLM扩展失败，使用基础规则扩展
            enhanced_queries.extend(self._basic_query_expansion(query))
            
        # 去重并限制数量
        unique_queries = list(dict.fromkeys(enhanced_queries))  # 保持顺序的去重
        return unique_queries[:5]  # 最多返回5个查询
        
    def _build_expansion_prompt(self, query: str) -> str:
        """
        构建查询扩展的提示词
        """
        return f"""请为以下查询生成3-4个扩展形式，包括：
1. 同义词替换
2. 关键词提取
3. 英文翻译（如果原文是中文）
4. 相关概念扩展

原始查询：{query}

请直接输出扩展查询，每行一个，不要编号或其他格式："""
        
    def _parse_expansion_response(self, response: str) -> List[str]:
        """
        解析LLM的扩展响应
        """
        lines = response.strip().split('\n')
        expanded_queries = []
        
        for line in lines:
            line = line.strip()
            # 移除可能的编号
            line = re.sub(r'^\d+[.、]\s*', '', line)
            if line and line not in expanded_queries:
                expanded_queries.append(line)
                
        return expanded_queries
        
    def _basic_query_expansion(self, query: str) -> List[str]:
        """
        基础的查询扩展规则
        """
        expanded = []
        
        # 提取关键词（简单的分词）
        import jieba
        keywords = list(jieba.cut(query))
        if len(keywords) > 1:
            # 组合关键词
            key_combination = ' '.join([w for w in keywords if len(w) > 1])
            if key_combination != query:
                expanded.append(key_combination)
                
        return expanded

def is_entity_query(query):
    """
    判断查询是否为实体查询
    
    Args:
        query: 用户查询字符串
        
    Returns:
        bool: 是否为实体查询
    """
    for pattern in ENTITY_QUERY_PATTERNS:
        if re.search(pattern, query):
            return True
    return False

def enhance_query_classification(query, original_mode, original_precise):
    """
    增强查询分类结果，对特定类型的实体查询强制使用hybrid模式
    
    Args:
        query: 用户查询字符串
        original_mode: 原始分类模式 (norag, hybrid_precise, hybrid_imprecise)
        original_precise: 原始精确标志
        
    Returns:
        tuple: (增强后的模式, 增强后的精确标志)
    """
    # 如果原始分类已经是hybrid模式，保持不变
    if original_mode.startswith('hybrid'):
        return original_mode, original_precise
    

    
    # 其他情况保持原始分类不变
    return original_mode, original_precise

# 全局查询增强器实例
_query_enhancer = None

def get_query_enhancer(config: Dict[str, Any]) -> QueryEnhancer:
    """
    获取查询增强器实例（单例模式）
    """
    global _query_enhancer
    if _query_enhancer is None:
        _query_enhancer = QueryEnhancer(config)
    return _query_enhancer