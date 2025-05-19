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
    ids = [c['id'] for c in chunks]
    logger.info(f"开始生成嵌入，共 {len(texts)} 条文本...")

    try:
        embeddings = llm.embed(texts)
    except Exception as e:
        logger.error(f"生成嵌入时出错: {e}")
        return

    dim = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype('float32')

    vec_config = config.get("vector", {})
    index_type = vec_config.get("index_type", "Flat").upper()
    nlist = vec_config.get("nlist", 100)
    nprobe = vec_config.get("nprobe", 10)
    efSearch = vec_config.get("ef_search", 32)

    if index_type == "IVF":
        logger.info(f"构建 IVF 索引（nlist={nlist}）...")
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        index.train(embeddings_np)
        index.nprobe = nprobe
    elif index_type == "HNSW":
        logger.info(f"构建 HNSW 索引（efSearch={efSearch}）...")
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efSearch = efSearch
    else:
        logger.info(f"构建 Flat 索引...")
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings_np)

    faiss_path = os.path.join(work_dir, "vector.index")
    mapping_path = os.path.join(work_dir, "embedding_map.json")
    faiss.write_index(index, faiss_path)

    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump({i: ids[i] for i in range(len(ids))}, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 向量索引构建完成：{index_type}，维度 {dim}，共 {len(embeddings)} 条")
    logger.info(f"索引文件保存至 {faiss_path}，映射关系保存至 {mapping_path}")
