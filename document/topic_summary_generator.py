import re
import jieba

def extract_key_entities(text: str) -> list:
    """
    提取文本中的关键实体（人名、地名、机构名等）
    
    Args:
        text: 输入文本
        
    Returns:
        关键实体列表
    """
    # 简单的实体识别模式
    entity_patterns = [
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',  # 英文人名/地名
        r'[\u4e00-\u9fff]{2,4}(?:公司|集团|大学|学院|医院|银行|政府|部门)',  # 机构名
        r'[\u4e00-\u9fff]{2,3}(?:市|省|县|区|镇|村)',  # 地名
        r'[\u4e00-\u9fff]{2,4}(?:先生|女士|教授|博士|主任|经理|总裁)',  # 人名+职位
    ]
    
    entities = []
    for pattern in entity_patterns:
        matches = re.findall(pattern, text)
        entities.extend(matches)
    
    # 使用jieba提取可能的人名
    words = jieba.cut(text)
    for word in words:
        if len(word) >= 2 and word.isalpha() and any('\u4e00' <= c <= '\u9fff' for c in word):
            # 简单判断是否可能是人名（2-4个汉字）
            if 2 <= len(word) <= 4:
                entities.append(word)
    
    # 去重并返回前5个最重要的实体
    unique_entities = list(dict.fromkeys(entities))[:5]
    return unique_entities

def generate_topic_summary(text: str, llm_client, max_length=50) -> str:
    """
    生成主题摘要，确保包含核心实体
    
    Args:
        text: 输入文本
        llm_client: LLM客户端
        max_length: 最大长度
        
    Returns:
        主题摘要
    """
    # 提取关键实体
    key_entities = extract_key_entities(text)
    
    # 构建包含实体信息的提示
    entity_info = ""
    if key_entities:
        entity_info = f"\n重要实体：{', '.join(key_entities)}"
    
    prompt = f"""
请为以下段落内容生成一个简洁明了的主题名称，不超过20个字。
请确保在主题名称中包含重要的人名、地名、机构名等关键实体。

内容：
{text[:500]}...{entity_info}

主题名称：
""".strip()
    
    try:
        response = llm_client.generate(prompt)
        summary = response.strip()[:max_length]
        
        # 验证摘要是否包含关键实体，如果没有则补充
        if key_entities and not any(entity in summary for entity in key_entities):
            # 如果摘要中没有包含任何关键实体，尝试添加最重要的实体
            main_entity = key_entities[0]
            if len(summary) + len(main_entity) + 3 <= max_length:
                summary = f"{main_entity}相关：{summary}"
            else:
                summary = f"{main_entity}：{summary[:max_length-len(main_entity)-1]}"
        
        return summary or text[:max_length]
        
    except (ValueError, ConnectionError, TimeoutError) as e:
        # 当LLM生成失败时，使用文本的第一句话和关键实体作为主题
        first_sentence = text.strip().split('。')[0][:30]
        if key_entities:
            entity_part = key_entities[0]
            fallback_summary = f"{entity_part}：{first_sentence}"
            return fallback_summary[:max_length]
        return first_sentence[:max_length] or "未知主题"
