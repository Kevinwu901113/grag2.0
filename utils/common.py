#!/usr/bin/env python3
"""
通用工具函数
整合项目中常用的辅助函数
"""

import os
import re
import jieba
from typing import List, Dict, Any, Set
from collections import defaultdict


def chunk_iterator(chunks: List[Dict[str, Any]], batch_size: int = 100):
    """
    文档块批量迭代器
    
    Args:
        chunks: 文档块列表
        batch_size: 批次大小
        
    Yields:
        批次文档块
    """
    for i in range(0, len(chunks), batch_size):
        yield chunks[i:i + batch_size]


def normalize_text(text: str) -> str:
    """
    文本归一化预处理，改进中英文混合处理
    
    Args:
        text: 输入文本
        
    Returns:
        归一化后的文本
    """
    if not text:
        return ""
    
    # 去除首尾空白
    text = text.strip()
    
    # 中英文之间添加空格（如果没有的话）
    text = re.sub(r'([\u4e00-\u9fff])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])([\u4e00-\u9fff])', r'\1 \2', text)
    
    # 统一空格（将多个连续空格替换为单个空格）
    text = re.sub(r'\s+', ' ', text)
    
    # 去除特殊控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    return text


def validate_embedding_dimension(embedding: List[float], expected_dim: int = None) -> bool:
    """
    验证嵌入向量维度
    
    Args:
        embedding: 嵌入向量
        expected_dim: 期望的维度，如果为None则不检查
        
    Returns:
        是否通过验证
    """
    if not embedding or not isinstance(embedding, list):
        return False
    
    if expected_dim is not None and len(embedding) != expected_dim:
        return False
    
    # 检查是否包含有效的浮点数
    try:
        for val in embedding:
            if not isinstance(val, (int, float)) or not (-1e10 < val < 1e10):
                return False
    except (TypeError, ValueError):
        return False
    
    return True


def normalize_scores(candidates: List[Dict[str, Any]], score_key: str = 'similarity', method: str = 'minmax') -> List[Dict[str, Any]]:
    """
    统一归一化不同检索方法的得分
    
    Args:
        candidates: 候选结果列表
        score_key: 得分字段名
        method: 归一化方法，支持 'minmax', 'zscore'
        
    Returns:
        归一化后的候选结果列表
    """
    if not candidates:
        return candidates
    
    scores = [c.get(score_key, 0.0) for c in candidates]
    
    if method == 'minmax':
        min_score, max_score = min(scores), max(scores)
        if max_score > min_score:
            for i, c in enumerate(candidates):
                c['normalized_similarity'] = (scores[i] - min_score) / (max_score - min_score)
        else:
            for c in candidates:
                c['normalized_similarity'] = 1.0
    elif method == 'zscore':
        import numpy as np
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        if std_score > 0:
            for i, c in enumerate(candidates):
                c['normalized_similarity'] = (scores[i] - mean_score) / std_score
        else:
            for c in candidates:
                c['normalized_similarity'] = 0.0
    
    return candidates


def improved_tokenize(text: str) -> List[str]:
    """
    改进的分词函数，更好处理中英文混合
    
    Args:
        text: 输入文本
        
    Returns:
        分词结果列表
    """
    # 先进行文本归一化
    text = normalize_text(text.lower())
    
    # 使用jieba分词
    tokens = list(jieba.cut(text))
    
    # 过滤停用词和单字符
    stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    return [token for token in tokens if len(token) > 1 and token not in stop_words]


def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """
    从文本中提取关键词
    
    Args:
        text: 输入文本
        top_k: 返回前k个关键词
        
    Returns:
        关键词列表
    """
    # 使用改进的分词函数
    words = improved_tokenize(text)
    word_freq = defaultdict(int)
    
    for word in words:
        word_freq[word] += 1
    
    # 按频率排序并返回前k个
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:top_k]]


def clean_text(text: str) -> str:
    """
    清理文本，移除多余的空白字符
    
    Args:
        text: 输入文本
        
    Returns:
        清理后的文本
    """
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    验证配置文件是否包含必需的键
    
    Args:
        config: 配置字典
        required_keys: 必需的键列表
        
    Returns:
        是否验证通过
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        print(f"❌ 配置文件缺少必需的键: {missing_keys}")
        return False
    return True


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误
    
    Args:
        a: 被除数
        b: 除数
        default: 除零时的默认值
        
    Returns:
        除法结果或默认值
    """
    return a / b if b != 0 else default


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个字典
    
    Args:
        *dicts: 要合并的字典
        
    Returns:
        合并后的字典
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def get_file_size(file_path: str) -> int:
    """
    获取文件大小（字节）
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件大小，如果文件不存在返回0
    """
    try:
        return os.path.getsize(file_path)
    except (OSError, FileNotFoundError):
        return 0


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小显示
    
    Args:
        size_bytes: 文件大小（字节）
        
    Returns:
        格式化的大小字符串
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断文本到指定长度
    
    Args:
        text: 输入文本
        max_length: 最大长度
        suffix: 截断后缀
        
    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def deduplicate_list(items: List[Any], key_func=None) -> List[Any]:
    """
    列表去重，保持顺序
    
    Args:
        items: 输入列表
        key_func: 用于提取比较键的函数
        
    Returns:
        去重后的列表
    """
    seen = set()
    result = []
    
    for item in items:
        key = key_func(item) if key_func else item
        if key not in seen:
            seen.add(key)
            result.append(item)
    
    return result