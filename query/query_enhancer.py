import re

# 实体查询模式的正则表达式
ENTITY_QUERY_PATTERNS = [
    r'(.+?)是谁',  # 某人是谁
    r'谁是(.+?)(的)?',  # 谁是某职位/某人
    r'(.+?)(的)?(.+?)是(谁|什么|哪个|哪位)',  # 某地的某职位是谁
    r'(.+?)在(哪里|哪儿|哪个|什么地方)',  # 某物在哪里
    r'(.+?)(的)?(地址|位置|电话|联系方式)',  # 某地的地址/位置/电话
    r'(.+?)是(什么|哪个)',  # 某物是什么
]

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
    
    # 如果是norag模式，检查是否为实体查询
    if original_mode == 'norag' and is_entity_query(query):
        # 强制转为hybrid_imprecise模式
        return 'hybrid_imprecise', False
    
    # 其他情况保持原始分类不变
    return original_mode, original_precise