import os
import json
import numpy as np
import faiss
from llm.llm import LLMClient


def run_vector_indexer(config: dict, work_dir: str, logger):
    chunk_path = os.path.join(work_dir, "chunks.json")
    if not os.path.exists(chunk_path):
        logger.error("未找到 chunks.json，无法构建向量索引。请先运行文档处理模块。")
        return

    with open(chunk_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    llm = LLMClient(config)
    texts = [c['text'] for c in chunks]
    print(f"[DEBUG] 待嵌入文本条数: {len(texts)}")
    ids = [c['id'] for c in chunks]

    try:
        embeddings = llm.embed(texts)
    except Exception as e:
        logger.error(f"生成嵌入时出错: {e}")
        return

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))

    faiss_path = os.path.join(work_dir, "vector.index")
    mapping_path = os.path.join(work_dir, "embedding_map.json")
    faiss.write_index(index, faiss_path)

    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({i: ids[i] for i in range(len(ids))}, f, ensure_ascii=False, indent=2)

    logger.info(f"向量索引构建完成，共嵌入 {len(embeddings)} 个chunk，维度: {dim}")
    logger.info(f"索引文件保存至 {faiss_path}，映射关系保存至 {mapping_path}")
