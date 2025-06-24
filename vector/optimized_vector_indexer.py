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
    
    from utils.io import load_chunks, save_vector_index, file_exists
    
    # 检查是否存在文档块文件
    enhanced_chunk_path = os.path.join(work_dir, "enhanced_chunks.json")
    chunk_path = os.path.join(work_dir, "chunks.json")
    if not file_exists(enhanced_chunk_path) and not file_exists(chunk_path):
        logger.error("未找到 chunks.json 或 enhanced_chunks.json，无法构建向量索引。请先运行文档处理模块。")
        return

    chunks = load_chunks(work_dir)

    # 获取嵌入配置
    embedding_config = config.get("embedding", {})
    provider = embedding_config.get("provider", "ollama")
    model_name = embedding_config.get("model_name", "bge-m3")
    
    # 初始化LLM客户端
    llm = LLMClient(config)
    
    # 处理不同数据格式的兼容性
    texts = []
    ids = []
    for c in chunks:
        if 'text' in c:
            texts.append(c['text'])
        elif 'sentences' in c:
            # 处理static_chunk_processor生成的格式
            text = "\n".join(c['sentences']) if isinstance(c['sentences'], list) else str(c['sentences'])
            texts.append(text)
        else:
            logger.warning(f"块 {c.get('id', 'unknown')} 缺少文本内容，跳过处理")
            continue
        ids.append(c['id'])
    
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
            # 启用文本预处理和维度验证
            batch_embeddings = llm.embed(batch_texts, normalize_text=True, validate_dim=True)
            
            # 验证批次嵌入
            if not batch_embeddings:
                logger.warning(f"批次 {i//batch_size + 1} 嵌入生成失败，跳过")
                pbar.update(len(batch_texts))
                continue
                
            # 检查嵌入维度一致性
            if all_embeddings and len(batch_embeddings[0]) != len(all_embeddings[0]):
                logger.error(f"批次 {i//batch_size + 1} 嵌入维度不一致: 期望 {len(all_embeddings[0])}, 实际 {len(batch_embeddings[0])}")
                pbar.close()
                return
                
            all_embeddings.extend(batch_embeddings)
            pbar.update(len(batch_texts))
            
    except (ValueError, ConnectionError, TimeoutError, RuntimeError) as e:
        logger.error(f"生成嵌入时出错: {e}")
        pbar.close()
        return
    except Exception as e:
        logger.error(f"生成嵌入时发生未知错误: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        pbar.close()
        return
    
    pbar.close()
    
    if not all_embeddings:
        logger.error("生成的嵌入为空，无法构建索引")
        return
        
    dim = len(all_embeddings[0])
    logger.info(f"嵌入维度: {dim}, 有效嵌入数量: {len(all_embeddings)}")
    
    # 验证所有嵌入向量
    from utils.common import validate_embedding_dimension
    invalid_count = 0
    for i, embedding in enumerate(all_embeddings):
        if not validate_embedding_dimension(embedding, dim):
            logger.warning(f"第 {i} 个嵌入向量无效")
            invalid_count += 1
    
    if invalid_count > 0:
        logger.warning(f"发现 {invalid_count} 个无效嵌入向量")
        if invalid_count > len(all_embeddings) * 0.1:  # 如果超过10%的向量无效
            logger.error("无效嵌入向量过多，停止构建索引")
            return
    
    embeddings_np = np.array(all_embeddings).astype('float32')
    
    # 检查数组中的无效值
    if np.any(np.isnan(embeddings_np)) or np.any(np.isinf(embeddings_np)):
        logger.error("嵌入数组包含NaN或无穷大值，无法构建索引")
        return

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
        quantizer = faiss.IndexFlatIP(dim)  # 使用内积索引
        index = faiss.IndexIVFFlat(quantizer, dim, adjusted_nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings_np)
        index.nprobe = min(nprobe, adjusted_nlist)  # nprobe也不能超过nlist
    elif index_type == "HNSW":
        logger.info(f"构建 HNSW 索引（efSearch={efSearch}）...")
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = efSearch
    else:
        logger.info(f"构建 Flat 索引...")
        index = faiss.IndexFlatIP(dim)  # 使用内积索引

    # 归一化向量以确保内积等于余弦相似度
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)

    # 使用统一的保存函数
    id_map = {i: ids[i] for i in range(len(ids))}
    save_vector_index(index, id_map, work_dir)

    logger.info(f"✅ 向量索引构建完成：{index_type}，维度 {dim}，共 {len(all_embeddings)} 条")
    logger.info(f"索引文件已保存至工作目录 {work_dir}")