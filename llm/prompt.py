# llm/prompt.py

def get_entity_extraction_prompt(text: str) -> str:
    return f"""
-目标-
请从以下文本中识别出所有具名实体，并为每个实体指定一个类型。实体类型包括：
- 人物（Person）
- 组织（Organization）
- 地点（Location）
- 时间（Time）
- 事件（Event）
- 产品（Product）
- 案例（Case）
- 法律（Law）
- 文档（Document）
- 角色（Role）

-步骤-
1. 对于每个识别出的实体，提取以下信息：
   - 实体名称（entity_name）
   - 实体类型（entity_type）
   - 实体描述（entity_description）：对实体的属性和活动的简要描述
   格式如下：
   ("entity" | 实体名称 | 实体类型 | 实体描述)

2. 将所有实体信息整理为一个列表返回。
3. 在输出的最后添加标记 <END_OF_OUTPUT>

-文本-
{text}
"""

def get_relation_extraction_prompt(text: str, entities: list[tuple[str, str]]) -> str:
    entity_list = "，".join([f"{name}（{etype}）" for name, etype in entities])
    return f"""
-目标-
根据以下文本和已识别的实体，找出实体之间的关系。

-已识别的实体-
{entity_list}

-步骤-
1. 对于每对存在关系的实体，提取以下信息：
   - 源实体名称（source_entity）
   - 目标实体名称（target_entity）
   - 关系描述（relationship_description）：对关系的简要说明
   - 关系强度（relationship_strength）：表示关系强度的数值评分
   格式如下：
   ("relationship" | 源实体名称 | 目标实体名称 | 关系描述 | 关系强度)

2. 将所有关系信息整理为一个列表返回。
3. 在输出的最后添加标记 <END_OF_OUTPUT>

-文本-
{text}
"""

def get_query_rewrite_prompt(query: str, strategy: str, context: str = None, entities: list = None) -> str:
    """
    生成查询改写提示词
    
    Args:
        query: 原始查询
        strategy: 改写策略
        context: 上下文信息
        entities: 相关实体列表
        
    Returns:
        改写提示词
    """
    base_prompt = f"""
-目标-
请根据指定的策略对用户查询进行改写，使其更清晰、完整或语义丰富，以提高检索和匹配效果。

-原始查询-
{query}

-改写策略-
{_get_strategy_description(strategy)}
"""
    
    # 添加上下文信息
    if context:
        base_prompt += f"""

-参考上下文-
{context[:500]}...  # 截取前500字符避免过长
"""
    
    # 添加实体信息
    if entities:
        entity_list = "、".join(entities[:10])  # 最多显示10个实体
        base_prompt += f"""

-相关实体-
{entity_list}
"""
    
    base_prompt += """

-改写要求-
1. 保持原始查询的核心意图不变
2. 根据策略要求进行相应的改写
3. 确保改写后的查询语法正确、表达清晰
4. 避免过度改写导致意思偏离
5. 直接输出改写后的查询，不需要额外解释

改写后的查询：
"""
    
    return base_prompt

def _get_strategy_description(strategy: str) -> str:
    """
    获取改写策略的描述
    
    Args:
        strategy: 策略名称
        
    Returns:
        策略描述
    """
    strategy_descriptions = {
        "clarify": "澄清策略：在保持查询开放性的前提下，适度补充必要信息以明确查询意图，避免过度限定查询范围",
        "expand": "扩展策略：引入相关表达和同义词，丰富查询的语义表示，增加检索覆盖面",
        "simplify": "简化策略：消除歧义，简化复杂表达，突出核心关键词和主要意图",
        "context_aware": "上下文感知策略：结合提供的上下文信息，生成更精准的查询表达",
        "refine": "精炼策略：在保持意图的基础上，优化表达方式，提高查询质量",
        "auto": "自动策略：根据查询特点自动选择最适合的改写方式"
    }
    
    return strategy_descriptions.get(strategy, "通用策略：优化查询表达，提高检索效果")
