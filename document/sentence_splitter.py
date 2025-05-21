import re

def split_into_sentences(text: str) -> list[str]:
    """
    将一段文本按中英文标点分句，返回句子列表。
    """
    # 先统一换行空格，避免中断
    text = text.strip().replace("\n", " ").replace("\r", " ")

    # 中文分句
    cn_delimiters = r'(?<=[。！？；])'
    en_delimiters = r'(?<=[.!?;])'
    pattern = f'{cn_delimiters}|{en_delimiters}'

    # 分句（保留标点）
    sentences = re.split(pattern, text)
    # 清理空句与前后空格
    return [s.strip() for s in sentences if s.strip()]