import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any
from llm.llm import LLMClient
from utils.logger import setup_logger
from utils.model_cache import model_cache

def run_vector_indexer(config: dict, work_dir: str, logger=None):
    """
    优化版向量索引器，实现批处理机制
    
    Args:
        config: 配置信息
        work_dir: 工作目录
        logger: 日志对象，如果为None则创建新的日志对象
    """
    if logger is None:
        logger = setup_logger(work_dir)
    
    chunk_path = os.path.join(work_dir, "chunks.json")
    if not os.path.exists(chunk_path):
        logger.error("未找到 chunks.json，无法构建向量索引。请先运行文档处理模块。")
        return

    with open(chunk_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # 获取嵌入配置
    embedding_config = config.get("embedding", {})
    provider = embedding_config.get("provider", "ollama")
    model_name = embedding_config.get("model_name", "bge-m3")
    
    # 初始化LLM客户端
    llm = LLMClient(config)
    texts = [c['text'] for c in chunks]
    ids = [c['id'] for c in chunks]
    
    # 获取批处理大小，默认为100
    batch_size = config.get("vector", {}).get("batch_size", 100)
    logger.info(f"开始生成嵌入，使用{provider}的{model_name}模型，共{len(texts)}条文本，批处理大小: {batch_size}")

    # 使用批处理生成嵌入
    all_embeddings = []
    from utils.logger import get_progress_bar
    
    # 创建进度条
    pbar = get_progress_bar(total=len(texts), desc="生成嵌入")
    
    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = llm.embed(batch_texts)
            all_embeddings.extend(batch_embeddings)
            pbar.update(len(batch_texts))
    except (ValueError, ConnectionError, TimeoutError) as e:
        logger.error(f"生成嵌入时出错: {e}")
        pbar.close()
        return
    
    pbar.close()
    
    if not all_embeddings:
        logger.error("生成的嵌入为空，无法构建索引")
        return
        
    dim = len(all_embeddings[0])
    embeddings_np = np.array(all_embeddings).astype('float32')

    vec_config = config.get("vector", {})
    index_type = vec_config.get("index_type", "Flat").upper()
    nlist = vec_config.get("nlist", 100)
    nprobe = vec_config.get("nprobe", 10)
    efSearch = vec_config.get("ef_search", 32)

    if index_type == "IVF":
        # 确保nlist不超过训练数据数量
        num_samples = len(embeddings_np)
        adjusted_nlist = min(nlist, num_samples)
        if adjusted_nlist != nlist:
            logger.warning(f"训练数据数量({num_samples})少于配置的nlist({nlist})，自动调整为{adjusted_nlist}")
        
        logger.info(f"构建 IVF 索引（nlist={adjusted_nlist}）...")
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, adjusted_nlist, faiss.METRIC_L2)
        index.train(embeddings_np)
        index.nprobe = min(nprobe, adjusted_nlist)  # nprobe也不能超过nlist
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

    logger.info(f"✅ 向量索引构建完成：{index_type}，维度 {dim}，共 {len(all_embeddings)} 条")
    logger.info(f"索引文件保存至 {faiss_path}，映射关系保存至 {mapping_path}")