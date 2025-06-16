import os
import re
import json
from typing import List
from docx import Document
from llm.llm import LLMClient  
from document.topic_pool_manager import TopicPoolManager
from document.redundancy_buffer import RedundancyBuffer  # ✅ 新增

def read_docx(file_path: str) -> List[str]:
    doc = Document(file_path)
    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]

def read_json(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            return [item.get("content", "") for item in data if isinstance(item, dict) and "content" in item]
        return []

def read_jsonl(file_path: str) -> List[str]:
    """读取JSONL文件内容"""
    paragraphs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        # 支持多种常见的内容字段名
                        content = (
                            data.get("content") or 
                            data.get("text") or 
                            data.get("body") or 
                            data.get("message") or
                            data.get("question") or
                            data.get("answer")
                        )
                        if content and isinstance(content, str):
                            paragraphs.append(content.strip())
                        
                        # 处理paragraphs字段（如musique数据集）
                        if "paragraphs" in data and isinstance(data["paragraphs"], list):
                            for para in data["paragraphs"]:
                                if isinstance(para, dict) and "paragraph_text" in para:
                                    para_text = para["paragraph_text"]
                                    if para_text and isinstance(para_text, str):
                                        paragraphs.append(para_text.strip())
                except json.JSONDecodeError:
                    continue
    return paragraphs

def split_into_sentences(text: str) -> List[str]:
    text = text.strip().replace("\n", " ").replace("\r", " ")
    pattern = r'(?<=[。！？；.!?;])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def split_into_chunks_with_overlap(text: str, chunk_size: int = 800, chunk_overlap: int = 300) -> List[str]:
    """
    改进的分块策略：支持段落合并和chunk_size限制，引入chunk_overlap
    
    Args:
        text: 输入文本
        chunk_size: 每个块的最大字符数
        chunk_overlap: 块之间的重叠字符数
        
    Returns:
        分块后的文本列表
    """
    # 调用智能分块函数
    return intelligent_chunk_splitting(text, chunk_size, chunk_overlap)

def intelligent_chunk_splitting(text: str, chunk_size: int = 800, chunk_overlap: int = 300) -> List[str]:
    """
    智能分块策略：
    1. 优先按段落切分
    2. 对长度异常的段落，按句子重组再合并
    3. 确保重要内容不被块边界切分丢失
    4. 使生成的chunk在语义上更集中
    
    Args:
        text: 输入文本
        chunk_size: 每个块的最大字符数
        chunk_overlap: 块之间的重叠字符数
        
    Returns:
        分块后的文本列表
    """
    # 检测文本中的标题和段落结构
    # 标题模式：# 标题、## 二级标题、数字编号等
    title_pattern = r'(^|\n)(#+\s+|\d+\.\s+|第[一二三四五六七八九十]+[章节]\s+)'
    has_structure = bool(re.search(title_pattern, text))
    
    # 首先按段落分割
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    # 如果文本有明显的结构，优先保持结构完整性
    if has_structure:
        for para in paragraphs:
            # 检测是否为标题行
            is_title = bool(re.match(r'^(#+\s+|\d+\.\s+|第[一二三四五六七八九十]+[章节]\s+)', para))
            
            # 如果是标题且当前块不为空，先保存当前块再开始新块
            if is_title and current_chunk:
                chunks.append(current_chunk)
                current_chunk = para
            # 如果当前段落加上现有块不超过限制，则合并
            elif len(current_chunk) + len(para) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n" + para
                else:
                    current_chunk = para
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 如果单个段落超过chunk_size，需要进一步分割
                if len(para) > chunk_size:
                    # 处理长段落
                    long_para_chunks = split_long_paragraph(para, chunk_size)
                    chunks.extend(long_para_chunks[:-1])
                    current_chunk = long_para_chunks[-1]
                else:
                    current_chunk = para
    else:
        # 无明显结构，按常规方式处理
        for para in paragraphs:
            # 如果当前段落加上现有块不超过限制，则合并
            if len(current_chunk) + len(para) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n" + para
                else:
                    current_chunk = para
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 如果单个段落超过chunk_size，需要进一步分割
                if len(para) > chunk_size:
                    # 处理长段落
                    long_para_chunks = split_long_paragraph(para, chunk_size)
                    chunks.extend(long_para_chunks[:-1])
                    current_chunk = long_para_chunks[-1]
                else:
                    current_chunk = para
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
    
    # 应用重叠策略
    return apply_overlap(chunks, chunk_overlap)

def split_long_paragraph(paragraph: str, chunk_size: int) -> List[str]:
    """
    将长段落按句子分割成多个块
    
    Args:
        paragraph: 长段落文本
        chunk_size: 每个块的最大字符数
        
    Returns:
        分割后的块列表
    """
    # 按句子分割长段落
    sentences = split_into_sentences(paragraph)
    chunks = []
    temp_chunk = ""
    
    for sentence in sentences:
        # 处理超长句子
        if len(sentence) > chunk_size:
            # 如果当前临时块不为空，先保存
            if temp_chunk:
                chunks.append(temp_chunk)
                temp_chunk = ""
            
            # 将超长句子按字符分割
            for i in range(0, len(sentence), chunk_size):
                sub_sentence = sentence[i:i+chunk_size]
                chunks.append(sub_sentence)
        else:
            # 正常句子处理
            if len(temp_chunk) + len(sentence) + 1 <= chunk_size:
                if temp_chunk:
                    temp_chunk += " " + sentence
                else:
                    temp_chunk = sentence
            else:
                if temp_chunk:
                    chunks.append(temp_chunk)
                temp_chunk = sentence
    
    # 添加最后一个临时块
    if temp_chunk:
        chunks.append(temp_chunk)
    
    return chunks

def apply_overlap(chunks: List[str], chunk_overlap: int) -> List[str]:
    """
    对分块应用重叠策略
    
    Args:
        chunks: 原始分块列表
        chunk_overlap: 块之间的重叠字符数
        
    Returns:
        应用重叠后的分块列表
    """
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks
    
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped_chunks.append(chunk)
        else:
            # 从前一个块的末尾取overlap字符
            prev_chunk = chunks[i-1]
            overlap_text = prev_chunk[-chunk_overlap:] if len(prev_chunk) > chunk_overlap else prev_chunk
            overlapped_chunk = overlap_text + "\n" + chunk
            overlapped_chunks.append(overlapped_chunk)
    
    return overlapped_chunks

def run_document_processing(config: dict, work_dir: str, logger):
    input_dir = config["document"]["input_dir"]
    allowed_types = config["document"].get("allowed_types", [".docx", ".json", ".jsonl"])
    sim_threshold = config["document"].get("similarity_threshold", 0.80)
    redundancy_threshold = config["document"].get("redundancy_threshold", 0.95)  # ✅ 可配置
    
    # 从配置中读取分块参数，默认值调整为更小的chunk_size和更大的overlap
    chunk_size = config["document"].get("chunk_size", 800)
    chunk_overlap = config["document"].get("chunk_overlap", 300)
    
    logger.info(f"使用分块参数: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    llm_client = LLMClient(config)

    redundancy_filter = RedundancyBuffer(threshold=redundancy_threshold)
    
    # 传递配置给TopicPoolManager，统一使用LLMClient进行嵌入
    topic_manager = TopicPoolManager(
        model_name=None,  # 不再需要指定模型名称，由LLMClient统一管理
        similarity_threshold=sim_threshold, 
        redundancy_filter=redundancy_filter,
        config=config
    )

    chunk_id = 0

    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[-1].lower()
        file_path = os.path.join(input_dir, filename)
        if ext not in allowed_types:
            continue

        try:
            logger.info(f"正在处理文件: {filename}")
            if ext == ".docx":
                paragraphs = read_docx(file_path)
            elif ext == ".json":
                paragraphs = read_json(file_path)
            elif ext == ".jsonl":
                paragraphs = read_jsonl(file_path)
            else:
                continue

            # 将所有段落合并为完整文本
            full_text = "\n".join(paragraphs)
            
            # 使用改进的分块策略
            chunks = split_into_chunks_with_overlap(full_text, chunk_size, chunk_overlap)
            
            logger.info(f"文件 {filename} 分割为 {len(chunks)} 个块")
            
            for chunk_text in chunks:
                meta = {
                    "chunk_id": f"chunk_{chunk_id}",
                    "source": filename
                }
                topic_manager.add_sentence(chunk_text, meta)
                chunk_id += 1

        except (ValueError, IOError, UnicodeDecodeError, KeyError) as e:
            logger.error(f"处理文件 {filename} 时出错: {str(e)}")

    # 输出主题聚合块
    out_chunks = topic_manager.get_all_topics(llm_client=llm_client)
    out_path = os.path.join(work_dir, "chunks.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"共生成 {len(out_chunks)} 个主题块，保存至 {out_path}")

    # ✅ 输出冗余句日志
    redundant_log_path = os.path.join(work_dir, "redundant_sentences.json")
    with open(redundant_log_path, 'w', encoding='utf-8') as f:
        json.dump(redundancy_filter.get_redundant_log(), f, ensure_ascii=False, indent=2)
    logger.info(f"冗余句共计 {len(redundancy_filter.get_redundant_log())} 条，已记录至 {redundant_log_path}")
