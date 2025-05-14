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
