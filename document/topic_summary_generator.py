def generate_topic_summary(text: str, llm_client, max_length=50) -> str:
    prompt = (
        "请为以下段落内容生成一个简洁明了的主题名称，不超过20个字：\n"
        f"{text}\n"
        "主题名称："
    )
    try:
        result = llm_client.generate(prompt)
        return result.strip()[:max_length]
    except Exception as e:
        return "未知主题"
