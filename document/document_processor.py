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

def split_into_sentences(text: str) -> List[str]:
    text = text.strip().replace("\n", " ").replace("\r", " ")
    pattern = r'(?<=[。！？；.!?;])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def split_into_chunks_with_overlap(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    改进的分块策略：支持段落合并和chunk_size限制，引入chunk_overlap
    
    Args:
        text: 输入文本
        chunk_size: 每个块的最大字符数
        chunk_overlap: 块之间的重叠字符数
        
    Returns:
        分块后的文本列表
    """
    # 首先按段落分割
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
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
                # 按句子分割长段落
                sentences = split_into_sentences(para)
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) + 1 <= chunk_size:
                        if temp_chunk:
                            temp_chunk += " " + sentence
                        else:
                            temp_chunk = sentence
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = sentence
                
                if temp_chunk:
                    current_chunk = temp_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = para
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
    
    # 应用重叠策略
    if chunk_overlap > 0 and len(chunks) > 1:
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
    
    return chunks

def run_document_processing(config: dict, work_dir: str, logger):
    input_dir = config["document"]["input_dir"]
    allowed_types = config["document"].get("allowed_types", [".docx", ".json"])
    sim_threshold = config["document"].get("similarity_threshold", 0.80)
    redundancy_threshold = config["document"].get("redundancy_threshold", 0.95)  # ✅ 可配置
    
    # 从配置中读取分块参数
    chunk_size = config["document"].get("chunk_size", 1000)
    chunk_overlap = config["document"].get("chunk_overlap", 200)
    
    llm_client = LLMClient(config)

    redundancy_filter = RedundancyBuffer(threshold=redundancy_threshold)
    
    # 传递配置给TopicPoolManager
    embedding_config = config.get("embedding", {})
    model_mapping = {
        "bge-m3": "BAAI/bge-m3",
        "text2vec": "shibing624/text2vec-base-chinese",
        "text-embedding-ada-002": "all-MiniLM-L6-v2",
        "all-MiniLM-L6-v2": "all-MiniLM-L6-v2"
    }
    model_name = embedding_config.get("model_name", "bge-m3")
    actual_model_name = model_mapping.get(model_name, model_name)
    
    topic_manager = TopicPoolManager(
        model_name=actual_model_name, 
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
