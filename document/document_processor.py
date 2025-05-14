import os
import json
from typing import List
from docx import Document

def read_docx(file_path: str) -> List[str]:
    doc = Document(file_path)
    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]

def read_json(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list):
            return [item.get("content", "") for item in data if isinstance(item, dict) and "content" in item]
        return []

def split_into_chunks(paragraphs: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current += para + "\n"
        else:
            chunks.append(current.strip())
            current = para[-chunk_overlap:] if chunk_overlap > 0 else ""
    if current:
        chunks.append(current.strip())
    return chunks

def run_document_processing(config: dict, work_dir: str, logger):
    input_dir = config["document"]["input_dir"]
    chunk_size = config["document"].get("chunk_size", 1000)
    chunk_overlap = config["document"].get("chunk_overlap", 200)
    allowed_types = config["document"].get("allowed_types", [".docx", ".json"])

    chunks = []
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

            chunk_texts = split_into_chunks(paragraphs, chunk_size, chunk_overlap)
            for i, text in enumerate(chunk_texts):
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": text,
                    "source": filename,
                    "doc_index": i
                })
                chunk_id += 1

        except Exception as e:
            logger.error(f"处理文件 {filename} 时出错: {str(e)}")

    out_path = os.path.join(work_dir, "chunks.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"共生成 {len(chunks)} 个文本块，保存至 {out_path}")
