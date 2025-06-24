import os
import json
import sys
from typing import List, Dict
from tqdm import tqdm
from docx import Document
from llm.llm import LLMClient
# 移除聚类功能，保留主题池概念
from document.redundancy_buffer import RedundancyBuffer, EnhancedRedundancyBuffer
from document.sentence_splitter import split_into_sentences
from utils.config_manager import ConfigManager
from redundancy.redundancy_filter_factory import RedundancyFilterFactory, create_redundancy_filter
from utils.performance_monitor import performance_monitor

def read_docx(file_path: str) -> List[str]:
    """读取DOCX文件内容"""
    doc = Document(file_path)
    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]

def read_json(file_path: str) -> List[str]:
    """读取JSON文件内容"""
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

def read_txt(file_path: str) -> List[str]:
    """读取TXT文件内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 按段落分割
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        return paragraphs if paragraphs else [content]

class EnhancedDocumentProcessor:
    """
    增强的文档处理器：集成高级聚类功能
    支持句子级分解和多种聚类算法
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.llm_client = LLMClient(config)
        
        # 使用新的配置管理器
        if isinstance(config, dict) and 'config_path' in config:
            self.config_manager = ConfigManager(config['config_path'])
        else:
            # 兼容旧的配置方式
            self.config_manager = None
        
        # 冗余过滤器配置 - 使用新的工厂模式
        try:
            if self.config_manager:
                # 使用新的配置管理器和工厂模式
                self.redundancy_filter = RedundancyFilterFactory.create_from_config_manager(self.config_manager)
            else:
                # 兼容旧的配置方式
                redundancy_config = config.get("redundancy", config.get("redundancy_filter", {}))
                
                # 检查是否使用新的冗余配置结构
                if 'method' in redundancy_config:
                    # 新的配置结构
                    self.redundancy_filter = RedundancyFilterFactory.create_filter(redundancy_config)
                else:
                    # 旧的配置结构，使用传统方式
                    enable_enhanced_filter = redundancy_config.get("enable_enhanced_filter", False)
                    
                    if enable_enhanced_filter:
                        # 使用增强冗余过滤器
                        self.redundancy_filter = EnhancedRedundancyBuffer(redundancy_config)
                    else:
                        # 使用传统冗余过滤器
                        redundancy_threshold = config["document"].get("redundancy_threshold", 0.95)
                        redundancy_config = {
                            'threshold': redundancy_threshold,
                            'enable_logging': True,
                            'enable_progress': True
                        }
                        self.redundancy_filter = RedundancyBuffer(config=redundancy_config)
        except Exception as e:
            # 如果新方式失败，回退到传统方式
            print(f"警告：使用新的冗余过滤器失败，回退到传统方式: {e}")
            redundancy_threshold = config["document"].get("redundancy_threshold", 0.95)
            redundancy_config = {
                'threshold': redundancy_threshold,
                'enable_logging': True,
                'enable_progress': True
            }
            self.redundancy_filter = RedundancyBuffer(config=redundancy_config)
        
        # 处理模式配置 - 移除高级聚类，仅支持传统模式
        self.processing_mode = "traditional"
        
    @performance_monitor()
    def process_documents(self, input_dir: str, work_dir: str, logger) -> Dict:
        """
        处理文档目录，返回处理结果
        
        Args:
            input_dir: 输入文档目录
            work_dir: 工作目录
            logger: 日志记录器
            
        Returns:
            处理结果字典
        """
        allowed_types = self.config["document"].get("allowed_types", [".docx", ".json", ".jsonl", ".txt"])
        
        logger.info(f"使用处理模式: {self.processing_mode}")
        
        # 读取所有文档
        documents = self._load_documents(input_dir, allowed_types, logger)
        
        if not documents:
            logger.warning("未找到有效文档")
            return {"topics": [], "stats": {}}
        
        logger.info(f"共加载 {len(documents)} 个文档")
        
        # 应用冗余过滤
        filtered_documents = self._apply_redundancy_filter(documents, logger)
        
        # 使用传统主题池处理方式
        topics = self._process_with_traditional_method(filtered_documents, logger)
        
        # 保存结果
        logger.info("开始保存处理结果...")
        results = self._save_results(topics, work_dir, logger)
        logger.info("结果保存完成")
        
        return results
    
    @performance_monitor()
    def _load_documents(self, input_dir: str, allowed_types: List[str], logger) -> List[Dict]:
        """
        加载文档目录中的所有文档
        
        Args:
            input_dir: 输入目录
            allowed_types: 允许的文件类型
            logger: 日志记录器
            
        Returns:
            文档列表
        """
        documents = []
        doc_id = 0
        
        for filename in os.listdir(input_dir):
            ext = os.path.splitext(filename)[-1].lower()
            if ext not in allowed_types:
                continue
                
            file_path = os.path.join(input_dir, filename)
            
            try:
                logger.info(f"正在加载文件: {filename}")
                
                if ext == ".docx":
                    paragraphs = read_docx(file_path)
                elif ext == ".json":
                    paragraphs = read_json(file_path)
                elif ext == ".jsonl":
                    paragraphs = read_jsonl(file_path)
                elif ext == ".txt":
                    paragraphs = read_txt(file_path)
                else:
                    continue
                
                # 将段落合并为文档
                full_text = "\n".join(paragraphs)
                
                if full_text.strip():
                    documents.append({
                        "id": f"doc_{doc_id}",
                        "text": full_text,
                        "meta": {
                            "source": filename,
                            "file_path": file_path,
                            "paragraph_count": len(paragraphs)
                        }
                    })
                    doc_id += 1
                    
            except Exception as e:
                logger.error(f"处理文件 {filename} 时出错: {str(e)}")
                
        return documents
    
    @performance_monitor()
    def _apply_redundancy_filter(self, documents: List[Dict], logger) -> List[Dict]:
        """
        应用冗余过滤
        
        Args:
            documents: 文档列表
            logger: 日志记录器
            
        Returns:
            过滤后的文档列表
        """
        if not self.config["document"].get("enable_redundancy_filter", True):
            return documents
            
        filtered_documents = []
        
        for doc in documents:
            text = doc["text"]
            
            # 对于句子级处理，需要检查每个句子
            if self.config["document"].get("sentence_level", True):
                sentences = split_into_sentences(text)
                filtered_sentences = []
                
                min_sentence_length = self.config["document"].get("min_sentence_length", 10)
                # 过滤有效句子
                valid_sentences = [(i, sentence) for i, sentence in enumerate(sentences) 
                                 if len(sentence.strip()) >= min_sentence_length]
                
                if valid_sentences:
                    # 优化的批量生成嵌入向量 - 支持HuggingFace批量处理
                    batch_size = self.config["document"].get("embedding_batch_size", 64)
                    sentence_texts = [sentence for _, sentence in valid_sentences]
                    
                    try:
                        # 一次性批量处理所有句子，提高HuggingFace模型效率
                        all_embeddings = self.llm_client.embed(sentence_texts)
                        logger.debug(f"批量嵌入成功: {len(all_embeddings)} 个句子")
                    except Exception as e:
                        logger.warning(f"批量嵌入失败，回退到分批处理: {e}")
                        # 回退到分批处理
                        all_embeddings = []
                        for i in range(0, len(sentence_texts), batch_size):
                            batch_texts = sentence_texts[i:i+batch_size]
                            try:
                                batch_embeddings = self.llm_client.embed(batch_texts)
                                all_embeddings.extend(batch_embeddings)
                            except Exception as batch_e:
                                logger.warning(f"分批嵌入失败，回退到单句处理: {batch_e}")
                                # 最后回退到单句处理
                                for sentence in batch_texts:
                                    try:
                                        embedding = self.llm_client.embed([sentence])[0]
                                        all_embeddings.append(embedding)
                                    except Exception:
                                        # 使用零向量作为占位符
                                        embedding_dim = self.config.get("embedding", {}).get("dimension", 768)
                                        all_embeddings.append([0.0] * embedding_dim)
                    
                    # 逐句进行冗余检测
                    for (original_idx, sentence), embedding in zip(valid_sentences, all_embeddings):
                        # 检查是否使用增强冗余过滤器
                        if hasattr(self.redundancy_filter, 'is_redundant_enhanced'):
                            # 获取上下文
                            context_before = sentences[original_idx-1] if original_idx > 0 else ""
                            context_after = sentences[original_idx+1] if original_idx < len(sentences)-1 else ""
                            is_redundant = self.redundancy_filter.is_redundant_enhanced(
                                sentence, embedding, context_before, context_after
                            )
                        else:
                            is_redundant = self.redundancy_filter.is_redundant(sentence, embedding)
                            
                        if not is_redundant:
                            filtered_sentences.append(sentence)
                
                if filtered_sentences:
                    filtered_doc = doc.copy()
                    filtered_doc["text"] = "\n".join(filtered_sentences)
                    filtered_doc["meta"]["original_sentence_count"] = len(sentences)
                    filtered_doc["meta"]["filtered_sentence_count"] = len(filtered_sentences)
                    filtered_documents.append(filtered_doc)
            else:
                # 文档级冗余检测
                embedding = self.llm_client.embed([text])[0]
                
                # 检查是否使用增强冗余过滤器
                if hasattr(self.redundancy_filter, 'is_redundant_enhanced'):
                    is_redundant = self.redundancy_filter.is_redundant_enhanced(text, embedding)
                else:
                    is_redundant = self.redundancy_filter.is_redundant(text, embedding)
                    
                if not is_redundant:
                    filtered_documents.append(doc)
        
        logger.info(f"冗余过滤后保留 {len(filtered_documents)} 个文档")
        return filtered_documents
    
    # 移除高级聚类处理方法
    
    @performance_monitor()
    def _process_with_traditional_method(self, documents: List[Dict], logger) -> List[Dict]:
        """
        使用优化的传统主题池方法处理文档
        
        Args:
            documents: 文档列表
            logger: 日志记录器
            
        Returns:
            主题列表
        """
        logger.info("使用优化的传统处理方法...")
        
        # 导入传统的主题池管理器
        from document.topic_pool_manager import TopicPoolManager
        
        sim_threshold = self.config["document"].get("similarity_threshold", 0.80)
        
        # 初始化优化的主题池管理器
        topic_manager = TopicPoolManager(
            similarity_threshold=sim_threshold,
            redundancy_filter=self.redundancy_filter,
            config=self.config
        )
        
        # 批量嵌入配置
        batch_size = self.config["document"].get("embedding_batch_size", 32)
        min_sentence_length = self.config["document"].get("min_sentence_length", 10)
        enable_parallel_docs = self.config.get("topic_pool", {}).get("enable_parallel_document_processing", False)
        
        total_sentences = 0
        
        if enable_parallel_docs and len(documents) > 1:
            # 并行处理多个文档（实验性功能）
            logger.info(f"启用并行文档处理，共 {len(documents)} 个文档")
            total_sentences = self._process_documents_parallel(documents, topic_manager, logger)
        else:
            # 顺序处理文档
            # 首先计算总的处理单元数量（句子或文档）
            total_processing_units = 0
            for doc in documents:
                if self.config["document"].get("sentence_level_traditional", False):
                    sentences = split_into_sentences(doc["text"])
                    valid_sentences = [s for s in sentences if len(s.strip()) >= min_sentence_length]
                    total_processing_units += len(valid_sentences)
                else:
                    total_processing_units += 1
            
            logger.info(f"开始处理文档，预计处理 {total_processing_units} 个文本块")
            
            # 使用日志系统显示进度信息
            logger.info(f"开始处理 {total_processing_units} 个文本块...")
            processed_units = 0
            last_logged_percentage = -1
            
            # 使用日志显示进度
            def update_progress(increment=1):
                nonlocal processed_units, last_logged_percentage
                processed_units += increment
                percentage = int((processed_units / total_processing_units) * 100)
                # 每10%记录一次进度，避免日志过多
                if percentage >= last_logged_percentage + 10 or processed_units == total_processing_units:
                    logger.info(f"📊 处理进度: {processed_units}/{total_processing_units} ({percentage}%)")
                    last_logged_percentage = percentage
            
            # 创建进度条对象用于兼容现有代码
            class LogProgressBar:
                def update(self, n=1):
                    update_progress(n)
                def refresh(self):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    logger.info(f"✅ 处理完成: {processed_units}/{total_processing_units} (100%)")
            
            with LogProgressBar() as pbar:
                for doc_idx, doc in enumerate(documents):
                    logger.info(f"处理文档 {doc_idx + 1}/{len(documents)}: {doc['meta'].get('source', 'Unknown')}")
                    
                    text = doc["text"]
                    meta = doc["meta"]
                    
                    # 根据配置决定是否进行句子级分解
                    if self.config["document"].get("sentence_level_traditional", False):
                        sentences = split_into_sentences(text)
                        # 过滤有效句子
                        valid_sentences = [s for s in sentences if len(s.strip()) >= min_sentence_length]
                        
                        if valid_sentences:
                            # 为每个句子准备相同的元数据
                            sentence_metas = [meta] * len(valid_sentences)
                            # 使用批量处理接口，并传递进度条回调
                            topic_manager.add_sentences_batch_with_progress(
                                valid_sentences, sentence_metas, batch_size, pbar
                            )
                            total_sentences += len(valid_sentences)
                            # 强制刷新进度条显示
                            pbar.refresh()
                    else:
                        # 文档级处理，仍使用单句接口
                        topic_manager.add_sentence(text, meta)
                        total_sentences += 1
                        pbar.update(1)
                        # 强制刷新进度条显示
                        pbar.refresh()
        
        logger.info(f"优化传统方法处理完成，共处理 {total_sentences} 个句子")
        
        # 获取主题池统计信息
        if hasattr(topic_manager, 'get_topic_pool_stats'):
            stats = topic_manager.get_topic_pool_stats()
            logger.info(f"主题池统计: {stats}")
        
        # 优化主题池
        if hasattr(topic_manager, 'optimize_topic_pool'):
            logger.info("开始优化主题池...")
            topic_manager.optimize_topic_pool()
            logger.info("主题池优化完成")
        
        # 获取主题（不生成摘要以避免大量LLM调用导致hang）
        logger.info("正在获取主题列表...")
        topics = topic_manager.get_all_topics(llm_client=None)
        logger.info(f"主题列表获取完成，共 {len(topics)} 个主题")
        
        logger.info(f"传统方法处理完成，生成 {len(topics)} 个主题")
        return topics
        
    def _process_documents_parallel(self, documents: List[Dict], topic_manager, logger) -> int:
        """
        并行处理多个文档（实验性功能）
        
        Args:
            documents: 文档列表
            topic_manager: 主题池管理器
            logger: 日志记录器
            
        Returns:
            处理的句子总数
        """
        import concurrent.futures
        from document.sentence_splitter import split_into_sentences
        
        min_sentence_length = self.config["document"].get("min_sentence_length", 10)
        batch_size = self.config["document"].get("embedding_batch_size", 32)
        max_workers = self.config.get("topic_pool", {}).get("max_workers", 4)
        
        def process_single_document(doc_data):
            doc_idx, doc = doc_data
            text = doc["text"]
            meta = doc["meta"]
            
            if self.config["document"].get("sentence_level_traditional", False):
                sentences = split_into_sentences(text)
                valid_sentences = [s for s in sentences if len(s.strip()) >= min_sentence_length]
                return valid_sentences, [meta] * len(valid_sentences)
            else:
                return [text], [meta]
        
        total_sentences = 0
        
        # 并行处理文档
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_doc = {executor.submit(process_single_document, (idx, doc)): idx 
                           for idx, doc in enumerate(documents)}
            
            for future in concurrent.futures.as_completed(future_to_doc):
                doc_idx = future_to_doc[future]
                try:
                    sentences, metas = future.result()
                    if sentences:
                        topic_manager.add_sentences_batch(sentences, metas, batch_size)
                        total_sentences += len(sentences)
                        logger.info(f"并行处理完成文档 {doc_idx + 1}，句子数: {len(sentences)}")
                except Exception as exc:
                    logger.error(f"文档 {doc_idx + 1} 处理出错: {exc}")
        
        return total_sentences
    
    def _save_results(self, topics: List[Dict], work_dir: str, logger) -> Dict:
        """
        保存处理结果
        
        Args:
            topics: 主题列表
            work_dir: 工作目录
            logger: 日志记录器
            
        Returns:
            结果统计信息
        """
        # 保存主题块
        chunks_path = os.path.join(work_dir, "enhanced_chunks.json")
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(topics, f, ensure_ascii=False, indent=2)
        logger.info(f"主题块已保存至: {chunks_path}")
        
        # 保存冗余句日志
        redundant_log_path = os.path.join(work_dir, "redundant_sentences.json")
        redundant_log = self.redundancy_filter.get_redundant_log()
        with open(redundant_log_path, 'w', encoding='utf-8') as f:
            json.dump(redundant_log, f, ensure_ascii=False, indent=2)
        logger.info(f"冗余句日志已保存至: {redundant_log_path}")
        
        # 生成统计信息
        stats = {
            "total_topics": len(topics),
            "total_redundant_sentences": len(redundant_log),
            "processing_mode": self.processing_mode,
            "clustering_method": "traditional_topic_pool",
            "topics_by_size": self._analyze_topic_sizes(topics),
            "average_topic_length": sum(len(topic["text"]) for topic in topics) / len(topics) if topics else 0
        }
        
        # 保存统计信息
        stats_path = os.path.join(work_dir, "processing_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"统计信息已保存至: {stats_path}")
        
        return {
            "topics": topics,
            "stats": stats,
            "paths": {
                "chunks": chunks_path,
                "redundant_log": redundant_log_path,
                "stats": stats_path
            }
        }
    
    def _analyze_topic_sizes(self, topics: List[Dict]) -> Dict:
        """
        分析主题大小分布
        
        Args:
            topics: 主题列表
            
        Returns:
            大小分布统计
        """
        if not topics:
            return {}
            
        sizes = [len(topic["text"]) for topic in topics]
        sentence_counts = [topic.get("sentence_count", 0) for topic in topics]
        
        return {
            "min_length": min(sizes),
            "max_length": max(sizes),
            "avg_length": sum(sizes) / len(sizes),
            "min_sentences": min(sentence_counts) if sentence_counts else 0,
            "max_sentences": max(sentence_counts) if sentence_counts else 0,
            "avg_sentences": sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0
        }

def run_enhanced_document_processing(config: dict, work_dir: str, logger):
    """
    运行增强的文档处理流程
    
    根据配置中的 document_processing.strategy 选择处理策略：
    - "clustered": 使用新的静态批量聚类处理
    - "incremental": 使用原有的主题池增量处理（默认）
    
    Args:
        config: 配置字典
        work_dir: 工作目录
        logger: 日志记录器
        
    Returns:
        处理结果
    """
    # 检查处理策略
    strategy = config.get("document_processing", {}).get("strategy", "incremental")
    
    if strategy == "clustered":
        # 使用新的静态批量聚类处理
        logger.info("使用静态批量聚类处理策略")
        from .static_chunk_processor import run_static_chunk_processing
        return run_static_chunk_processing(config, work_dir, logger)
    else:
        # 使用原有的主题池增量处理
        logger.info("使用传统主题池增量处理策略")
        input_dir = config["document"]["input_dir"]
        
        # 创建增强文档处理器
        processor = EnhancedDocumentProcessor(config)
        
        # 执行处理
        results = processor.process_documents(input_dir, work_dir, logger)
        
        logger.info(f"增强文档处理完成: {results['stats']}")
        
        return results