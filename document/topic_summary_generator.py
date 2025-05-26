def generate_topic_summary(text: str, llm_client, max_length=50) -> str:
    prompt = f"""
请为以下段落内容生成一个简洁明了的主题名称，不超过20个字：
{text}
主题名称：
""".strip()
    try:
        response = llm_client.generate(prompt)
        return response.strip()[:max_length] or text[:max_length]
    except (ValueError, ConnectionError, TimeoutError) as e:
        # 当LLM生成失败时，使用文本的第一句话作为主题
        return text.strip().split('。')[0][:max_length] or "未知主题"
