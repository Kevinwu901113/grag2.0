import os
import json
from typing import List, Dict
from docx import Document
from llm.llm import LLMClient
# 移除聚类功能，保留主题池概念
from document.redundancy_buffer import RedundancyBuffer, EnhancedRedundancyBuffer
from document.sentence_splitter import split_into_sentences

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
        
        # 移除聚类功能，仅保留传统主题池处理
        
        # 冗余过滤器配置
        redundancy_config = config.get("redundancy_filter", {})
        enable_enhanced_filter = redundancy_config.get("enable_enhanced_filter", False)
        
        if enable_enhanced_filter:
            # 使用增强冗余过滤器
            self.redundancy_filter = EnhancedRedundancyBuffer(
                base_threshold=redundancy_config.get("base_threshold", 0.95),
                enable_dynamic_threshold=redundancy_config.get("enable_dynamic_threshold", True),
                context_window=redundancy_config.get("context_window", 100),
                length_factor_weight=redundancy_config.get("length_factor_weight", 0.3),
                semantic_weight=redundancy_config.get("semantic_weight", 0.8),
                context_weight=redundancy_config.get("context_weight", 0.2)
            )
        else:
            # 使用传统冗余过滤器
            redundancy_threshold = config["document"].get("redundancy_threshold", 0.95)
            self.redundancy_filter = RedundancyBuffer(threshold=redundancy_threshold)
        
        # 处理模式配置 - 移除高级聚类，仅支持传统模式
        self.processing_mode = "traditional"
        
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
        results = self._save_results(topics, work_dir, logger)
        
        return results
    
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
                for i, sentence in enumerate(sentences):
                    if len(sentence.strip()) >= min_sentence_length:
                        # 生成嵌入向量进行冗余检测
                        embedding = self.llm_client.embed([sentence])[0]
                        
                        # 检查是否使用增强冗余过滤器
                        if hasattr(self.redundancy_filter, 'is_redundant_enhanced'):
                            # 获取上下文
                            context_before = sentences[i-1] if i > 0 else ""
                            context_after = sentences[i+1] if i < len(sentences)-1 else ""
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
    
    def _process_with_traditional_method(self, documents: List[Dict], logger) -> List[Dict]:
        """
        使用传统方法处理文档（兼容原有逻辑）
        
        Args:
            documents: 文档列表
            logger: 日志记录器
            
        Returns:
            主题列表
        """
        logger.info("使用传统处理方法...")
        
        # 导入传统的主题池管理器
        from document.topic_pool_manager import TopicPoolManager
        
        sim_threshold = self.config["document"].get("similarity_threshold", 0.80)
        
        topic_manager = TopicPoolManager(
            similarity_threshold=sim_threshold,
            redundancy_filter=self.redundancy_filter,
            config=self.config
        )
        
        # 处理每个文档
        for doc in documents:
            text = doc["text"]
            meta = doc["meta"]
            
            # 根据配置决定是否进行句子级分解
            if self.config["document"].get("sentence_level_traditional", False):
                sentences = split_into_sentences(text)
                for sentence in sentences:
                    if len(sentence.strip()) >= 10:
                        topic_manager.add_sentence(sentence, meta)
            else:
                topic_manager.add_sentence(text, meta)
        
        # 获取主题
        topics = topic_manager.get_all_topics(llm_client=self.llm_client)
        
        logger.info(f"传统方法处理完成，生成 {len(topics)} 个主题")
        return topics
    
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
    
    Args:
        config: 配置字典
        work_dir: 工作目录
        logger: 日志记录器
        
    Returns:
        处理结果
    """
    input_dir = config["document"]["input_dir"]
    
    # 创建增强文档处理器
    processor = EnhancedDocumentProcessor(config)
    
    # 执行处理
    results = processor.process_documents(input_dir, work_dir, logger)
    
    logger.info(f"增强文档处理完成: {results['stats']}")
    
    return results