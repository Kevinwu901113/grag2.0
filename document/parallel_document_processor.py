import os
import json
import multiprocessing as mp
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from document.document_processor import read_docx, read_json, read_jsonl, split_into_sentences
from document.topic_pool_manager import TopicPoolManager
from document.redundancy_buffer import RedundancyBuffer
from llm.llm import LLMClient
from utils.io import save_json

def run_parallel_document_processing(config: dict, work_dir: str, logger):
    """
    并行文档处理主函数，支持多文档并行处理和向量化优化
    
    Args:
        config: 配置字典
        work_dir: 工作目录
        logger: 日志记录器
    """
    input_dir = config["document"]["input_dir"]
    allowed_types = config["document"].get("allowed_types", [".docx", ".json", ".jsonl"])
    
    # 并行处理参数
    enable_parallel = config.get("topic_pool", {}).get("enable_parallel", True)
    max_workers = config.get("topic_pool", {}).get("max_workers", min(4, mp.cpu_count()))
    
    # 分块参数
    chunk_size = config["document"].get("chunk_size", 800)
    chunk_overlap = config["document"].get("chunk_overlap", 300)
    
    logger.info(f"并行处理配置: enable_parallel={enable_parallel}, max_workers={max_workers}")
    logger.info(f"分块参数: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    # 读取所有文档并分块
    documents = []
    total_chunks = 0
    
    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[-1].lower()
        if ext not in allowed_types:
            continue
            
        file_path = os.path.join(input_dir, filename)
        try:
            logger.info(f"读取文件: {filename}")
            
            # 读取文档内容
            if ext == ".docx":
                paragraphs = read_docx(file_path)
            elif ext == ".json":
                paragraphs = read_json(file_path)
            elif ext == ".jsonl":
                paragraphs = read_jsonl(file_path)
            else:
                continue
            
            # 合并段落并分块
            full_text = "\n".join(paragraphs)
            from document.document_processor import split_into_chunks_with_overlap
            chunks = split_into_chunks_with_overlap(full_text, chunk_size, chunk_overlap)
            
            logger.info(f"文件 {filename} 分割为 {len(chunks)} 个块")
            documents.append((filename, chunks))
            total_chunks += len(chunks)
            
        except Exception as e:
            logger.error(f"读取文件 {filename} 时出错: {str(e)}")
    
    if not documents:
        logger.warning("没有找到有效的文档文件")
        return
    
    logger.info(f"开始并行处理文档，预计处理 {total_chunks} 个文档块")
    
    # 根据配置选择处理策略
    if enable_parallel and len(documents) > 1:
        logger.info(f"使用并行处理模式处理 {len(documents)} 个文档")
        topic_manager = TopicPoolManager.process_documents_parallel(
            documents, config, max_workers
        )
    else:
        logger.info(f"使用顺序处理模式处理 {len(documents)} 个文档")
        topic_manager = TopicPoolManager._process_documents_sequential(documents, config)
    
    # 生成输出
    llm_client = LLMClient(config)
    out_chunks = topic_manager.get_all_topics(llm_client=llm_client)
    
    # 保存结果
    out_path = os.path.join(work_dir, "chunks_parallel.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out_chunks, f, ensure_ascii=False, indent=2)
    
    logger.info(f"并行处理完成，共生成 {len(out_chunks)} 个主题块，保存至 {out_path}")
    
    # 输出性能统计
    total_sentences = sum(len(chunks) for _, chunks in documents)
    logger.info(f"处理统计: 文档数={len(documents)}, 总句子数={total_sentences}, 主题数={len(out_chunks)}")

def run_vectorized_batch_processing(sentences: List[str], config: dict, logger, 
                                  batch_size: int = None) -> TopicPoolManager:
    """
    向量化批量处理句子，专门用于大规模句子处理
    
    Args:
        sentences: 句子列表
        config: 配置字典
        logger: 日志记录器
        batch_size: 批处理大小
        
    Returns:
        TopicPoolManager实例
    """
    if batch_size is None:
        batch_size = config.get("topic_pool", {}).get("batch_size", 32)
    
    logger.info(f"开始向量化批量处理 {len(sentences)} 个句子，批大小={batch_size}")
    
    # 创建主题池管理器
    topic_manager = TopicPoolManager(config=config)
    
    # 生成元数据
    metas = [{"sentence_id": i, "source": "batch_input"} for i in range(len(sentences))]
    
    # 使用带进度条的批量添加方法
    with tqdm(total=len(sentences), desc="句子处理进度", unit="句") as pbar:
        topic_manager.add_sentences_batch_with_progress(sentences, metas, batch_size, pbar)
    
    logger.info(f"向量化批量处理完成，生成 {len(topic_manager.topics)} 个主题")
    
    return topic_manager

def benchmark_processing_methods(sentences: List[str], config: dict, logger):
    """
    对比不同处理方法的性能
    
    Args:
        sentences: 测试句子列表
        config: 配置字典
        logger: 日志记录器
    """
    import time
    
    logger.info(f"开始性能对比测试，句子数量: {len(sentences)}")
    
    # 测试1: 传统顺序处理
    start_time = time.time()
    config_sequential = config.copy()
    config_sequential["topic_pool"] = config_sequential.get("topic_pool", {})
    config_sequential["topic_pool"]["enable_parallel"] = False
    
    topic_manager_seq = TopicPoolManager(config=config_sequential)
    for i, sentence in enumerate(sentences):
        meta = {"sentence_id": i, "source": "benchmark"}
        topic_manager_seq.add_sentence(sentence, meta)
    
    sequential_time = time.time() - start_time
    logger.info(f"顺序处理耗时: {sequential_time:.2f}秒, 主题数: {len(topic_manager_seq.topics)}")
    
    # 测试2: 向量化批量处理
    start_time = time.time()
    topic_manager_batch = run_vectorized_batch_processing(sentences, config, logger)
    batch_time = time.time() - start_time
    logger.info(f"向量化批量处理耗时: {batch_time:.2f}秒, 主题数: {len(topic_manager_batch.topics)}")
    
    # 测试3: 并行处理（如果启用）
    if config.get("topic_pool", {}).get("enable_parallel", True) and len(sentences) > 100:
        start_time = time.time()
        # 将句子分成多个"文档"进行并行处理
        chunk_size = len(sentences) // 4
        documents = []
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i+chunk_size]
            documents.append((f"chunk_{i//chunk_size}", chunk))
        
        topic_manager_parallel = TopicPoolManager.process_documents_parallel(
            documents, config
        )
        parallel_time = time.time() - start_time
        logger.info(f"并行处理耗时: {parallel_time:.2f}秒, 主题数: {len(topic_manager_parallel.topics)}")
        
        # 性能提升统计
        speedup_batch = sequential_time / batch_time if batch_time > 0 else 0
        speedup_parallel = sequential_time / parallel_time if parallel_time > 0 else 0
        
        logger.info(f"性能提升 - 批量处理: {speedup_batch:.2f}x, 并行处理: {speedup_parallel:.2f}x")
    else:
        speedup_batch = sequential_time / batch_time if batch_time > 0 else 0
        logger.info(f"性能提升 - 批量处理: {speedup_batch:.2f}x")

if __name__ == "__main__":
    # 示例用法
    import yaml
    from utils.logger import setup_logger
    
    # 加载配置
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置日志
    logger = setup_logger("parallel_processor")
    
    # 运行并行文档处理
    work_dir = "./output"
    os.makedirs(work_dir, exist_ok=True)
    
    run_parallel_document_processing(config, work_dir, logger)