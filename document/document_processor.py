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

def run_document_processing(config: dict, work_dir: str, logger):
    input_dir = config["document"]["input_dir"]
    allowed_types = config["document"].get("allowed_types", [".docx", ".json"])
    sim_threshold = config["document"].get("similarity_threshold", 0.80)
    redundancy_threshold = config["document"].get("redundancy_threshold", 0.95)  # ✅ 可配置
    llm_client = LLMClient(config["llm"])

    redundancy_filter = RedundancyBuffer(threshold=redundancy_threshold)
    topic_manager = TopicPoolManager(similarity_threshold=sim_threshold, redundancy_filter=redundancy_filter)  # ✅ 注入

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

            for para in paragraphs:
                sentences = split_into_sentences(para)
                for sentence in sentences:
                    meta = {
                        "chunk_id": f"chunk_{chunk_id}",
                        "source": filename
                    }
                    topic_manager.add_sentence(sentence, meta)
                    chunk_id += 1

        except (ValueError, IOError, UnicodeDecodeError, KeyError) as e:
            logger.error(f"处理文件 {filename} 时出错: {str(e)}")

    # 输出主题聚合块
    llm_client = LLMClient(config["llm"])
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
