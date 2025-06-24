import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import numpy as np
import faiss
import hashlib
# 移除聚类依赖，保留主题池概念
from llm.llm import LLMClient
from document.topic_summary_generator import generate_topic_summary

class TopicPoolManager:
    def __init__(self, model_name=None, similarity_threshold=0.65, redundancy_filter=None, config=None):
        self.topics: List[Dict] = []
        self.similarity_threshold = similarity_threshold
        self.config = config or {}
        # 添加主题大小限制 - 从配置文件读取
        self.max_sentences_per_topic = config.get("topic_pool", {}).get("max_sentences_per_topic", 30)
        self.max_chars_per_topic = config.get("topic_pool", {}).get("max_chars_per_topic", 3000)
        
        # 统一使用LLMClient进行嵌入向量生成
        self.llm_client = LLMClient(self.config)
        
        self.topic_id_counter = 0
        self.redundancy_filter = redundancy_filter  # ✅ 新增
        
        # 主题分裂参数
        self.coherence_threshold = config.get("topic_pool", {}).get("coherence_threshold", 0.6)
        self.min_topic_size = config.get("topic_pool", {}).get("min_topic_size", 3)
        
        # 并行处理参数
        self.enable_parallel = config.get("topic_pool", {}).get("enable_parallel", True)
        self.max_workers = config.get("topic_pool", {}).get("max_workers", min(4, mp.cpu_count()))
        self.batch_size = config.get("topic_pool", {}).get("batch_size", 32)
        
        # 主题合并和清理参数
        self.enable_topic_merging = config.get("topic_pool", {}).get("enable_topic_merging", True)
        self.topic_merge_threshold = config.get("topic_pool", {}).get("topic_merge_threshold", 0.8)
        self.enable_document_isolation = config.get("topic_pool", {}).get("enable_document_isolation", False)
        self.max_topic_pool_size = config.get("topic_pool", {}).get("max_topic_pool_size", 1000)
        self.merge_check_interval = config.get("topic_pool", {}).get("merge_check_interval", 50)  # 每处理N个句子检查一次合并
        
        # 线程安全锁
        self._lock = threading.Lock()
        
        # 向量化计算缓存
        self._topic_centers_cache = None
        self._cache_valid = False
        
        # FAISS索引相关配置 - 优化配置
        self.enable_faiss_index = config.get("topic_pool", {}).get("enable_faiss_index", True)
        self.faiss_index_type = config.get("topic_pool", {}).get("faiss_index_type", "HNSW")  # HNSW, IVF, Flat
        self.faiss_nlist = config.get("topic_pool", {}).get("faiss_nlist", 100)
        self.faiss_ef_search = config.get("topic_pool", {}).get("faiss_ef_search", 64)  # 提高搜索精度
        self.faiss_search_k = config.get("topic_pool", {}).get("faiss_search_k", 10)  # 增加候选数量
        self.faiss_rebuild_threshold = config.get("topic_pool", {}).get("faiss_rebuild_threshold", 100)  # 新增：重建阈值
        
        # FAISS索引对象
        self._faiss_index = None
        self._faiss_id_map = []  # 索引ID到主题索引的映射
        self._embedding_dim = None
        self._faiss_needs_rebuild = False  # 新增：标记是否需要重建索引
        
        # 嵌入缓存机制 - 新增
        self.enable_embedding_cache = config.get("topic_pool", {}).get("enable_embedding_cache", True)
        self.embedding_cache_size = config.get("topic_pool", {}).get("embedding_cache_size", 5000)
        self._embedding_cache = {} if self.enable_embedding_cache else None
        
        # 统计信息
        self._sentences_processed = 0
        self._last_merge_check = 0
        self._current_document = None  # 当前处理的文档名
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_text_hash(self, text: str) -> str:
        """生成文本的哈希值用于缓存"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _manage_embedding_cache_size(self):
        """管理嵌入缓存大小，使用LRU策略"""
        if self._embedding_cache and len(self._embedding_cache) > self.embedding_cache_size:
            # 简单的FIFO策略，删除最旧的条目
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]

    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入向量，支持缓存机制"""
        # 检查缓存
        if self.enable_embedding_cache and self._embedding_cache is not None:
            text_hash = self._get_text_hash(text)
            if text_hash in self._embedding_cache:
                self._cache_hits += 1
                return self._embedding_cache[text_hash]
            else:
                self._cache_misses += 1
        
        # 缓存未命中，计算嵌入
        embedding = self.llm_client.embed([text])[0]
        embedding_array = np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding
        
        # 添加到缓存
        if self.enable_embedding_cache and self._embedding_cache is not None:
            text_hash = self._get_text_hash(text)
            self._embedding_cache[text_hash] = embedding_array
            self._manage_embedding_cache_size()
        
        # 初始化嵌入维度
        if self._embedding_dim is None:
            self._embedding_dim = len(embedding_array)
            if self.enable_faiss_index:
                self._init_faiss_index()
        
        return embedding_array
    
    def _init_faiss_index(self):
        """初始化FAISS索引 - 优化版本"""
        if not self.enable_faiss_index or self._embedding_dim is None:
            return
        
        try:
            if self.faiss_index_type.upper() == "HNSW":
                # 优化HNSW参数
                self._faiss_index = faiss.IndexHNSWFlat(self._embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
                self._faiss_index.hnsw.efSearch = self.faiss_ef_search
                self._faiss_index.hnsw.efConstruction = max(200, self.faiss_ef_search * 2)  # 提高构建质量
            elif self.faiss_index_type.upper() == "IVF":
                quantizer = faiss.IndexFlatIP(self._embedding_dim)
                self._faiss_index = faiss.IndexIVFFlat(quantizer, self._embedding_dim, 
                                                      self.faiss_nlist, faiss.METRIC_INNER_PRODUCT)
                self._faiss_index.nprobe = min(10, self.faiss_nlist // 4)  # 设置搜索探测数
            else:  # Flat
                self._faiss_index = faiss.IndexFlatIP(self._embedding_dim)
            
            self._faiss_id_map = []
            self._faiss_needs_rebuild = False
            print(f"[TopicPoolManager] FAISS索引初始化成功: {self.faiss_index_type}, 维度: {self._embedding_dim}")
        except Exception as e:
            print(f"FAISS索引初始化失败: {e}，回退到传统方法")
            self.enable_faiss_index = False
    
    def _add_to_faiss_index(self, topic_idx: int, embedding: np.ndarray):
        """将主题中心向量添加到FAISS索引 - 优化版本"""
        if not self.enable_faiss_index or self._faiss_index is None:
            return
        
        try:
            # 归一化向量以确保内积等于余弦相似度
            embedding_norm = embedding.copy().astype('float32')
            faiss.normalize_L2(embedding_norm.reshape(1, -1))
            
            # 如果是IVF索引且未训练，需要先训练
            if (self.faiss_index_type.upper() == "IVF" and 
                hasattr(self._faiss_index, 'is_trained') and 
                not self._faiss_index.is_trained):
                # 收集现有的所有主题中心向量进行训练
                if len(self.topics) >= self.faiss_nlist:
                    training_vectors = np.vstack([topic["center"] for topic in self.topics]).astype('float32')
                    faiss.normalize_L2(training_vectors)
                    self._faiss_index.train(training_vectors)
            
            self._faiss_index.add(embedding_norm.reshape(1, -1))
            self._faiss_id_map.append(topic_idx)
            
            # 检查是否需要重建索引（当主题数量增长较多时）
            if len(self._faiss_id_map) % self.faiss_rebuild_threshold == 0:
                self._faiss_needs_rebuild = True
                
        except Exception as e:
            print(f"添加到FAISS索引失败: {e}")
    
    def _rebuild_faiss_index(self):
        """重建FAISS索引（在主题合并或删除后使用） - 优化版本"""
        if not self.enable_faiss_index or not self.topics:
            return
        
        try:
            # 重新初始化索引
            self._init_faiss_index()
            
            # 添加所有现有主题的中心向量
            if self._faiss_index is not None:
                centers = np.vstack([topic["center"] for topic in self.topics]).astype('float32')
                faiss.normalize_L2(centers)
                
                # 如果是IVF索引，先训练
                if (self.faiss_index_type.upper() == "IVF" and 
                    hasattr(self._faiss_index, 'is_trained') and 
                    not self._faiss_index.is_trained and 
                    len(centers) >= self.faiss_nlist):
                    self._faiss_index.train(centers)
                
                self._faiss_index.add(centers)
                self._faiss_id_map = list(range(len(self.topics)))
                self._faiss_needs_rebuild = False
                print(f"[TopicPoolManager] FAISS索引重建完成，包含 {len(self.topics)} 个主题")
        except Exception as e:
            print(f"重建FAISS索引失败: {e}，禁用FAISS索引")
            self.enable_faiss_index = False
    
    def _search_similar_topics_faiss(self, embedding: np.ndarray, k: int = None) -> List[Tuple[int, float]]:
        """使用FAISS索引搜索最相似的主题 - 优化版本
        
        Args:
            embedding: 查询向量
            k: 返回的候选数量
            
        Returns:
            [(topic_idx, similarity_score), ...] 按相似度降序排列
        """
        if not self.enable_faiss_index or self._faiss_index is None or len(self._faiss_id_map) == 0:
            return []
        
        # 检查是否需要重建索引
        if self._faiss_needs_rebuild:
            self._rebuild_faiss_index()
        
        if k is None:
            k = min(self.faiss_search_k, len(self._faiss_id_map))
        
        try:
            # 归一化查询向量
            query_vector = embedding.copy().astype('float32')
            faiss.normalize_L2(query_vector.reshape(1, -1))
            
            # 搜索最近邻
            distances, indices = self._faiss_index.search(query_vector.reshape(1, -1), k)
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS返回-1表示无效结果
                    break
                
                if idx < len(self._faiss_id_map):
                    topic_idx = self._faiss_id_map[idx]
                    # 将内积距离转换为余弦相似度（因为向量已归一化）
                    similarity = float(dist)  # 归一化向量的内积就是余弦相似度
                    results.append((topic_idx, similarity))
            
            return results
        except Exception as e:
            print(f"FAISS搜索失败: {e}")
            return []

    def _update_topic_centers_cache(self):
        """更新主题中心向量缓存，用于向量化计算"""
        if not self.topics:
            self._topic_centers_cache = None
            self._cache_valid = False
            return
        
        # 将所有主题中心堆叠成矩阵
        centers = [topic["center"] for topic in self.topics]
        self._topic_centers_cache = np.vstack(centers)
        self._cache_valid = True
    
    def _vectorized_similarity_calculation(self, embedding: np.ndarray, topic_centers: List[np.ndarray] = None, valid_topic_indices: List[int] = None) -> np.ndarray:
        """
        向量化计算相似度，替代逐个计算 - 优化版本
        
        Args:
            embedding: 查询向量
            topic_centers: 主题中心向量列表（可选）
            valid_topic_indices: 有效主题索引列表（可选）
            
        Returns:
            相似度数组
        """
        if topic_centers is not None:
            # 使用提供的主题中心向量
            if not topic_centers:
                return np.array([])
            centers_matrix = np.vstack(topic_centers)
        else:
            # 使用现有主题的中心向量
            if not self.topics:
                return np.array([])
            
            if valid_topic_indices:
                centers = [self.topics[i]['center'] for i in valid_topic_indices if i in self.topics]
            else:
                centers = [topic['center'] for topic in self.topics.values()]
            
            if not centers:
                return np.array([])
            
            centers_matrix = np.vstack(centers)
        
        # 确保向量已归一化
        embedding_norm = embedding / np.linalg.norm(embedding)
        centers_norm = centers_matrix / np.linalg.norm(centers_matrix, axis=1, keepdims=True)
        
        # 批量计算余弦相似度
        similarities = np.dot(centers_norm, embedding_norm)
        
        return similarities
    
    def _find_best_topic_vectorized(self, embedding: np.ndarray, similarity_threshold: float = None) -> Tuple[int, float]:
        """
        使用向量化计算找到最佳主题 - 新增优化方法
        
        Args:
            embedding: 句子嵌入向量
            similarity_threshold: 相似度阈值
            
        Returns:
            (主题ID, 相似度分数)，如果没有合适主题则返回(-1, 0.0)
        """
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
        
        if not self.topics:
            return -1, 0.0
        
        # 获取所有主题中心向量
        topic_centers = []
        topic_ids = []
        
        for topic_id, topic in self.topics.items():
            # 检查主题容量
            if (len(topic['sentences']) < self.max_sentences_per_topic and 
                sum(len(s) for s in topic['sentences']) < self.max_chars_per_topic):
                topic_centers.append(topic['center'])
                topic_ids.append(topic_id)
        
        if not topic_centers:
            return -1, 0.0
        
        # 向量化计算相似度
        similarities = self._vectorized_similarity_calculation(embedding, topic_centers)
        
        # 找到最高相似度
        max_idx = np.argmax(similarities)
        max_similarity = similarities[max_idx]
        
        if max_similarity >= similarity_threshold:
            return topic_ids[max_idx], max_similarity
        
        return -1, 0.0

    def _vectorized_similarity_calculation(self, embedding: np.ndarray, valid_topic_indices: List[int] = None) -> np.ndarray:
        """向量化计算新嵌入与所有主题中心的相似度
        
        Args:
            embedding: 新句子的嵌入向量
            valid_topic_indices: 有效主题的索引列表，如果为None则计算所有主题
            
        Returns:
            相似度数组
        """
        if not self._cache_valid or self._topic_centers_cache is None:
            self._update_topic_centers_cache()
        
        if self._topic_centers_cache is None:
            return np.array([])
        
        # 确保embedding是二维数组
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        if valid_topic_indices is not None:
            # 只计算有效主题的相似度
            valid_centers = self._topic_centers_cache[valid_topic_indices]
            similarities = cosine_similarity(embedding, valid_centers)[0]
        else:
            # 计算与所有主题的相似度
            similarities = cosine_similarity(embedding, self._topic_centers_cache)[0]
        
        return similarities

    def add_sentence(self, sentence: str, meta: Dict = None):
        sent_emb = self._get_embedding(sentence)
        self.add_sentence_with_embedding(sentence, sent_emb, meta)

    def add_sentence_with_embedding(self, sentence: str, embedding: np.ndarray, meta: Dict = None):
        """
        使用预计算的嵌入向量添加句子到主题池（线程安全版本）
        
        Args:
            sentence: 句子文本
            embedding: 预计算的嵌入向量
            meta: 元数据
        """
        with self._lock:
            self._add_sentence_with_embedding_unsafe(sentence, embedding, meta)
    
    def _add_sentence_with_embedding_unsafe(self, sentence: str, embedding: np.ndarray, meta: Dict = None):
        """
        内部方法：不加锁的添加句子实现，优先使用FAISS索引进行主题匹配
        """
        # 确保embedding是numpy数组
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # 注意：冗余检测现在在批量处理中进行，单句添加时跳过冗余检测以避免重复处理
        # 如果需要单句冗余检测，可以在这里恢复相关代码

        if not self.topics:
            self._create_topic_unsafe(sentence, embedding, meta)
            return

        # 使用FAISS索引进行快速主题匹配
        if self.enable_faiss_index and self._faiss_index is not None:
            similar_topics = self._search_similar_topics_faiss(embedding)
            
            # 检查FAISS返回的候选主题
            for topic_idx, similarity in similar_topics:
                if topic_idx >= len(self.topics):  # 防止索引越界
                    continue
                    
                topic = self.topics[topic_idx]
                
                # 检查主题容量和相似度阈值
                if (similarity >= self.similarity_threshold and
                    len(topic["sentences"]) < self.max_sentences_per_topic and 
                    sum(len(s) for s in topic["sentences"]) < self.max_chars_per_topic):
                    
                    # 检查添加后是否会超过限制
                    if (len(topic["sentences"]) + 1 <= self.max_sentences_per_topic and 
                        sum(len(s) for s in topic["sentences"]) + len(sentence) <= self.max_chars_per_topic):
                        self._add_to_topic_unsafe(topic_idx, sentence, embedding, meta)
                        return
            
            # 如果FAISS搜索的候选都不合适，创建新主题
            self._create_topic_unsafe(sentence, embedding, meta)
            return
        
        # 回退到传统的向量化计算方法（当FAISS不可用时）
        # 过滤掉已经达到大小限制的主题
        valid_indices = []
        for i, topic in enumerate(self.topics):
            if (len(topic["sentences"]) < self.max_sentences_per_topic and 
                sum(len(s) for s in topic["sentences"]) < self.max_chars_per_topic):
                valid_indices.append(i)
        
        if not valid_indices:
            # 所有主题都已满，创建新主题
            self._create_topic_unsafe(sentence, embedding, meta)
            return

        # 使用向量化计算相似度
        similarities = self._vectorized_similarity_calculation(embedding, valid_indices)
        
        if len(similarities) == 0:
            self._create_topic_unsafe(sentence, embedding, meta)
            return
        
        max_idx_in_valid = int(np.argmax(similarities))
        max_sim = similarities[max_idx_in_valid]
        actual_topic_idx = valid_indices[max_idx_in_valid]

        if max_sim >= self.similarity_threshold:
            # 检查添加后是否会超过限制
            topic = self.topics[actual_topic_idx]
            if (len(topic["sentences"]) + 1 <= self.max_sentences_per_topic and 
                sum(len(s) for s in topic["sentences"]) + len(sentence) <= self.max_chars_per_topic):
                self._add_to_topic_unsafe(actual_topic_idx, sentence, embedding, meta)
            else:
                # 会超过限制，创建新主题
                self._create_topic_unsafe(sentence, embedding, meta)
        else:
            self._create_topic_unsafe(sentence, embedding, meta)

    def add_sentences_batch(self, sentences: List[str], metas: List[Dict] = None, batch_size: int = None, document_name: str = None):
        """
        批量添加句子到主题池，使用批量嵌入和并行处理优化性能 - 优化版本
        
        Args:
            sentences: 句子列表
            metas: 元数据列表，长度应与sentences相同
            batch_size: 批量嵌入的大小，如果为None则使用配置中的默认值
            document_name: 文档名称，用于文档隔离处理
        """
        if not sentences:
            return
            
        if metas is None:
            metas = [None] * len(sentences)
        elif len(metas) != len(sentences):
            raise ValueError("metas长度必须与sentences相同")
        
        if batch_size is None:
            batch_size = self.batch_size
        
        # 文档隔离处理：如果启用且文档发生变化，清空主题池
        if self.enable_document_isolation and document_name:
            if self._current_document and self._current_document != document_name:
                self.clear_topic_pool()
            self._current_document = document_name
        
        # 优化的批量嵌入处理
        all_embeddings = self._batch_embed_with_cache(sentences, batch_size)
        
        # 批量冗余过滤（如果启用）
        if self.redundancy_filter:
            # 定期优化冗余过滤器缓冲区
            if hasattr(self.redundancy_filter, 'optimize_buffer') and self._sentences_processed % 1000 == 0:
                self.redundancy_filter.optimize_buffer()
            
            filtered_sentences, filtered_embeddings, filtered_metas = self._batch_redundancy_filter(
                sentences, all_embeddings, metas
            )
        else:
            filtered_sentences, filtered_embeddings, filtered_metas = sentences, all_embeddings, metas
        
        # 根据配置决定是否使用并行处理
        if self.enable_parallel and len(filtered_sentences) > 50:  # 只有足够多的句子才使用并行
            self._add_sentences_parallel_optimized(filtered_sentences, filtered_embeddings, filtered_metas)
        else:
            # 逐个添加到主题池（使用预计算的嵌入向量）
            for sentence, embedding, meta in zip(filtered_sentences, filtered_embeddings, filtered_metas):
                if sentence.strip():  # 跳过空句子
                    self.add_sentence_with_embedding(sentence, np.array(embedding), meta)
                    self._sentences_processed += 1
                    
                    # 定期检查主题合并
                    if (self.enable_topic_merging and 
                        self._sentences_processed - self._last_merge_check >= self.merge_check_interval):
                        self._check_and_merge_topics()
                        self._last_merge_check = self._sentences_processed
    
    def add_sentences_batch_with_progress(self, sentences: List[str], metas: List[Dict] = None, 
                                        batch_size: int = None, progress_bar=None, document_name: str = None):
        """
        批量添加句子到主题池，支持进度条更新
        
        Args:
            sentences: 句子列表
            metas: 元数据列表，长度应与sentences相同
            batch_size: 批量嵌入的大小，如果为None则使用配置中的默认值
            progress_bar: tqdm进度条对象
            document_name: 文档名称，用于文档隔离处理
        """
        if not sentences:
            return
            
        if metas is None:
            metas = [None] * len(sentences)
        elif len(metas) != len(sentences):
            raise ValueError("metas长度必须与sentences相同")
        
        if batch_size is None:
            batch_size = self.batch_size
        
        # 文档隔离处理：如果启用且文档发生变化，清空主题池
        if self.enable_document_isolation and document_name:
            if self._current_document and self._current_document != document_name:
                self.clear_topic_pool()
            self._current_document = document_name
        
        # 优化的批量嵌入处理
        all_embeddings = self._batch_embed_with_cache(sentences, batch_size)
        
        # 根据配置决定是否使用并行处理
        if self.enable_parallel and len(sentences) > 50:  # 只有足够多的句子才使用并行
            self._add_sentences_parallel_with_progress(sentences, all_embeddings, metas, progress_bar)
        else:
            # 逐个添加到主题池（使用预计算的嵌入向量）
            for sentence, embedding, meta in zip(sentences, all_embeddings, metas):
                if sentence.strip():  # 跳过空句子
                    self.add_sentence_with_embedding(sentence, np.array(embedding), meta)
                    self._sentences_processed += 1
                    
                    # 更新进度条
                    if progress_bar is not None:
                        progress_bar.update(1)
                    
                    # 定期检查主题合并
                    if (self.enable_topic_merging and 
                        self._sentences_processed - self._last_merge_check >= self.merge_check_interval):
                        self._check_and_merge_topics()
                        self._last_merge_check = self._sentences_processed
    
    def _batch_embed_with_cache(self, sentences: List[str], batch_size: int) -> List[np.ndarray]:
        """
        批量生成嵌入向量，支持缓存机制 - 优化HuggingFace模型批量处理
        
        Args:
            sentences: 句子列表
            batch_size: 批量大小
            
        Returns:
            嵌入向量列表
        """
        all_embeddings = []
        cache_hits = 0
        cache_misses = 0
        
        # 检查缓存并收集需要计算的句子
        sentences_to_embed = []
        sentence_indices = []
        cached_embeddings = {}
        
        if self.enable_embedding_cache and self._embedding_cache is not None:
            for i, sentence in enumerate(sentences):
                text_hash = self._get_text_hash(sentence)
                if text_hash in self._embedding_cache:
                    cached_embeddings[i] = self._embedding_cache[text_hash]
                    cache_hits += 1
                else:
                    sentences_to_embed.append(sentence)
                    sentence_indices.append(i)
                    cache_misses += 1
        else:
            sentences_to_embed = sentences
            sentence_indices = list(range(len(sentences)))
        
        # 批量计算未缓存的嵌入
        computed_embeddings = {}
        if sentences_to_embed:
            # 优化HuggingFace模型：首先尝试一次性批量处理所有句子
            try:
                print(f"[TopicPoolManager] 尝试一次性批量处理 {len(sentences_to_embed)} 个句子")
                batch_embeddings = self.llm_client.embed(sentences_to_embed)
                
                # 将计算结果添加到缓存
                for j, (sentence, embedding) in enumerate(zip(sentences_to_embed, batch_embeddings)):
                    original_idx = sentence_indices[j]
                    embedding_array = np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding
                    computed_embeddings[original_idx] = embedding_array
                    
                    # 添加到缓存
                    if self.enable_embedding_cache and self._embedding_cache is not None:
                        text_hash = self._get_text_hash(sentence)
                        self._embedding_cache[text_hash] = embedding_array
                        self._manage_embedding_cache_size()
                        
                print(f"[TopicPoolManager] 一次性批量处理成功")
                
            except Exception as e:
                print(f"[TopicPoolManager] 一次性批量处理失败，回退到分批处理: {e}")
                computed_embeddings = self._process_embeddings_in_batches(sentences_to_embed, sentence_indices, batch_size)
        
        # 合并缓存和计算的结果，保持原始顺序
        for i in range(len(sentences)):
            if i in cached_embeddings:
                all_embeddings.append(cached_embeddings[i])
            else:
                all_embeddings.append(computed_embeddings[i])
        
        # 更新统计信息
        self._cache_hits += cache_hits
        self._cache_misses += cache_misses
        
        if cache_hits > 0 or cache_misses > 0:
            print(f"[TopicPoolManager] 批量嵌入完成: 缓存命中 {cache_hits}, 新计算 {cache_misses}")
        
        return all_embeddings
    
    def _process_embeddings_in_batches(self, sentences_to_embed: List[str], sentence_indices: List[int], batch_size: int) -> dict:
        """
        分批处理嵌入计算的私有方法
        
        Args:
            sentences_to_embed: 需要计算嵌入的句子列表
            sentence_indices: 句子在原始列表中的索引
            batch_size: 批量大小
            
        Returns:
            计算得到的嵌入字典 {原始索引: 嵌入向量}
        """
        computed_embeddings = {}
        
        for i in range(0, len(sentences_to_embed), batch_size):
            batch_texts = sentences_to_embed[i:i+batch_size]
            batch_embeddings = self.llm_client.embed(batch_texts)
            
            # 将计算结果添加到缓存
            for j, (sentence, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                original_idx = sentence_indices[i + j]
                embedding_array = np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding
                computed_embeddings[original_idx] = embedding_array
                
                # 添加到缓存
                if self.enable_embedding_cache and self._embedding_cache is not None:
                    text_hash = self._get_text_hash(sentence)
                    self._embedding_cache[text_hash] = embedding_array
                    self._manage_embedding_cache_size()
        
        return computed_embeddings
    
    def _batch_redundancy_filter(self, sentences: List[str], embeddings: List, metas: List[Dict]) -> Tuple[List[str], List, List[Dict]]:
        """
        批量冗余过滤，移除冗余句子
        
        Args:
            sentences: 句子列表
            embeddings: 嵌入向量列表
            metas: 元数据列表
            
        Returns:
            (过滤后的句子列表, 过滤后的嵌入向量列表, 过滤后的元数据列表)
        """
        if not self.redundancy_filter or not sentences:
            return sentences, embeddings, metas
        
        # 转换为numpy数组
        embeddings_array = [np.array(emb) if not isinstance(emb, np.ndarray) else emb for emb in embeddings]
        
        try:
            # 使用批量冗余检测
            if hasattr(self.redundancy_filter, 'is_redundant_enhanced_batch'):
                redundant_flags = self.redundancy_filter.is_redundant_enhanced_batch(sentences, embeddings_array)
            elif hasattr(self.redundancy_filter, 'is_redundant_batch'):
                redundant_flags = self.redundancy_filter.is_redundant_batch(sentences, embeddings_array)
            else:
                # 回退到逐个检测
                redundant_flags = []
                for sentence, embedding in zip(sentences, embeddings_array):
                    if hasattr(self.redundancy_filter, 'is_redundant_enhanced'):
                        is_redundant = self.redundancy_filter.is_redundant_enhanced(sentence, embedding)
                    else:
                        is_redundant = self.redundancy_filter.is_redundant(sentence, embedding)
                    redundant_flags.append(is_redundant)
            
            # 过滤掉冗余的句子
            filtered_sentences = []
            filtered_embeddings = []
            filtered_metas = []
            
            redundant_count = 0
            for i, (sentence, embedding, meta, is_redundant) in enumerate(zip(sentences, embeddings, metas, redundant_flags)):
                if not is_redundant:
                    filtered_sentences.append(sentence)
                    filtered_embeddings.append(embedding)
                    filtered_metas.append(meta)
                else:
                    redundant_count += 1
            
            if redundant_count > 0:
                print(f"[TopicPoolManager] 批量冗余过滤: 输入 {len(sentences)} 句子, 过滤掉 {redundant_count} 个冗余句子, 保留 {len(filtered_sentences)} 句子")
            
            return filtered_sentences, filtered_embeddings, filtered_metas
            
        except Exception as e:
            print(f"[TopicPoolManager] 批量冗余过滤失败，回退到原始数据: {e}")
            return sentences, embeddings, metas
    
    def _add_sentences_parallel_optimized(self, sentences: List[str], embeddings: List, metas: List[Dict]):
        """
        优化的并行添加句子到主题池 - 改进版本
        
        Args:
            sentences: 句子列表
            embeddings: 预计算的嵌入向量列表
            metas: 元数据列表
        """
        # 将句子分批处理，每批内部顺序执行，批间可以并行
        chunk_size = max(20, len(sentences) // self.max_workers)  # 增加块大小以减少锁竞争
        chunks = []
        
        for i in range(0, len(sentences), chunk_size):
            chunk_sentences = sentences[i:i+chunk_size]
            chunk_embeddings = embeddings[i:i+chunk_size]
            chunk_metas = metas[i:i+chunk_size]
            chunks.append((chunk_sentences, chunk_embeddings, chunk_metas))
        
        # 使用线程池并行处理各批次
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for chunk_sentences, chunk_embeddings, chunk_metas in chunks:
                future = executor.submit(self._process_sentence_chunk_optimized, 
                                       chunk_sentences, chunk_embeddings, chunk_metas)
                futures.append(future)
            
            # 等待所有任务完成
            for future in futures:
                future.result()
    
    def _add_sentences_parallel_with_progress(self, sentences: List[str], embeddings: List, 
                                            metas: List[Dict], progress_bar=None):
        """
        优化的并行添加句子到主题池，支持进度条更新
        
        Args:
            sentences: 句子列表
            embeddings: 预计算的嵌入向量列表
            metas: 元数据列表
            progress_bar: tqdm进度条对象
        """
        # 将句子分批处理，每批内部顺序执行，批间可以并行
        chunk_size = max(20, len(sentences) // self.max_workers)  # 增加块大小以减少锁竞争
        chunks = []
        
        for i in range(0, len(sentences), chunk_size):
            chunk_sentences = sentences[i:i+chunk_size]
            chunk_embeddings = embeddings[i:i+chunk_size]
            chunk_metas = metas[i:i+chunk_size]
            chunks.append((chunk_sentences, chunk_embeddings, chunk_metas))
        
        # 使用线程池并行处理各批次
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for chunk_sentences, chunk_embeddings, chunk_metas in chunks:
                future = executor.submit(self._process_sentence_chunk_with_progress, 
                                       chunk_sentences, chunk_embeddings, chunk_metas, progress_bar)
                futures.append(future)
            
            # 等待所有任务完成
            for future in futures:
                future.result()
    
    def _process_sentence_chunk_optimized(self, sentences: List[str], embeddings: List, metas: List[Dict]):
        """
        优化的处理一个句子批次 - 改进版本
        
        Args:
            sentences: 句子列表
            embeddings: 嵌入向量列表
            metas: 元数据列表
        """
        for sentence, embedding, meta in zip(sentences, embeddings, metas):
            if sentence.strip():  # 跳过空句子
                self.add_sentence_with_embedding(sentence, np.array(embedding), meta)
                
                # 在并行处理中减少合并检查频率以避免锁竞争
                with self._lock:
                    self._sentences_processed += 1
    
    def _process_sentence_chunk_with_progress(self, sentences: List[str], embeddings: List, 
                                            metas: List[Dict], progress_bar=None):
        """
        处理一个句子批次，支持进度条更新
        
        Args:
            sentences: 句子列表
            embeddings: 嵌入向量列表
            metas: 元数据列表
            progress_bar: tqdm进度条对象
        """
        for sentence, embedding, meta in zip(sentences, embeddings, metas):
            if sentence.strip():  # 跳过空句子
                self.add_sentence_with_embedding(sentence, np.array(embedding), meta)
                
                # 在并行处理中减少合并检查频率以避免锁竞争
                with self._lock:
                    self._sentences_processed += 1
                    
                    # 更新进度条（需要在锁内更新以避免竞争）
                    if progress_bar is not None:
                        progress_bar.update(1)

    def _create_topic_unsafe(self, sentence: str, embedding: np.ndarray, meta: Dict):
        """创建新主题（不加锁版本）"""
        topic = {
            "id": f"topic_{self.topic_id_counter}",
            "sentences": [sentence],
            "meta": [meta] if meta else [],
            "center": embedding
        }
        topic_idx = len(self.topics)  # 新主题的索引
        self.topics.append(topic)
        self.topic_id_counter += 1
        
        # 将新主题添加到FAISS索引
        self._add_to_faiss_index(topic_idx, embedding)
        
        # 标记缓存失效
        self._cache_valid = False

    def _add_to_topic_unsafe(self, idx: int, sentence: str, embedding: np.ndarray, meta: Dict):
        """添加句子到主题（不加锁版本）"""
        topic = self.topics[idx]
        topic["sentences"].append(sentence)
        topic["meta"].append(meta)
        
        # 改进的中心向量更新策略：使用加权平均
        current_count = len(topic["sentences"]) - 1  # 之前的句子数
        if current_count == 0:
            topic["center"] = embedding
        else:
            # 使用加权平均，给新句子较小的权重以保持稳定性
            weight = min(0.3, 1.0 / (current_count + 1))  # 动态权重
            topic["center"] = (1 - weight) * topic["center"] + weight * embedding
        
        # 标记缓存失效
        self._cache_valid = False
        
        # 检查是否需要分裂主题
        self._check_and_split_topic_unsafe(idx)

    def _check_and_split_topic_unsafe(self, idx: int):
        """检查主题是否需要分裂，使用智能分裂策略而非简单均匀分裂（不加锁版本）"""
        topic = self.topics[idx]
        
        # 如果句子数超过80%的限制，考虑分裂
        if len(topic["sentences"]) > self.max_sentences_per_topic * 0.8:
            sentences = topic["sentences"]
            metas = topic["meta"]
            
            # 计算当前主题的内聚度
            embeddings = [self._get_embedding(s) for s in sentences]
            embeddings_array = np.array(embeddings)
            coherence_score = self._calculate_topic_coherence(embeddings_array)
            
            # 如果内聚度较低，使用智能分裂策略
            if coherence_score < self.coherence_threshold and len(sentences) >= 4:
                split_result = self._intelligent_split_topic(sentences, metas, embeddings_array)
                if split_result:
                    # 更新原主题
                    topic["sentences"] = split_result[0]["sentences"]
                    topic["meta"] = split_result[0]["meta"]
                    topic["center"] = split_result[0]["center"]
                    topic["coherence_score"] = split_result[0]["coherence_score"]
                    
                    # 添加新主题
                    new_topic = split_result[1]
                    new_topic["id"] = f"topic_{self.topic_id_counter}"
                    self.topics.append(new_topic)
                    self.topic_id_counter += 1
                    # 标记缓存失效
                    self._cache_valid = False
                    return
            
            # 如果智能分裂失败或不适用，使用传统的中点分裂
            self._simple_split_topic_unsafe(topic, sentences, metas, embeddings)
    
    def _intelligent_split_topic(self, sentences: List[str], metas: List[Dict], embeddings: np.ndarray) -> Optional[List[Dict]]:
        """
        基于简单分裂的主题分裂策略（移除聚类依赖）
        
        Args:
            sentences: 句子列表
            metas: 元数据列表
            embeddings: 嵌入向量数组
            
        Returns:
            分裂后的两个主题数据，如果分裂失败返回None
        """
        try:
            # 简单的中点分裂策略
            mid_point = len(sentences) // 2
            
            # 确保两个组都有足够的句子
            if mid_point < self.min_topic_size or (len(sentences) - mid_point) < self.min_topic_size:
                return None
            
            # 创建两个分裂后的主题数据
            split_topics = []
            for start, end in [(0, mid_point), (mid_point, len(sentences))]:
                sub_sentences = sentences[start:end]
                sub_metas = metas[start:end]
                sub_embeddings = embeddings[start:end]
                
                # 计算新的中心向量和内聚度
                sub_center = np.mean(sub_embeddings, axis=0)
                sub_coherence = self._calculate_topic_coherence(sub_embeddings)
                
                split_topics.append({
                    "sentences": sub_sentences,
                    "meta": sub_metas,
                    "center": sub_center,
                    "coherence_score": sub_coherence
                })
            
            return split_topics
            
        except Exception:
            return None
    
    def _simple_split_topic_unsafe(self, topic: Dict, sentences: List[str], metas: List[Dict], embeddings: List[np.ndarray]):
        """
        传统的简单中点分裂策略（作为备选方案，不加锁版本）
        
        Args:
            topic: 原主题
            sentences: 句子列表
            metas: 元数据列表
            embeddings: 嵌入向量列表
        """
        # 将句子分成两组
        mid_point = len(sentences) // 2
        
        # 更新原主题
        topic["sentences"] = sentences[:mid_point]
        topic["meta"] = metas[:mid_point]
        
        # 重新计算原主题的中心
        if topic["sentences"]:
            topic_embeddings = embeddings[:mid_point]
            topic["center"] = np.mean(topic_embeddings, axis=0)
            topic["coherence_score"] = self._calculate_topic_coherence(np.array(topic_embeddings))
        
        # 创建新主题
        if len(sentences) > mid_point:
            new_sentences = sentences[mid_point:]
            new_metas = metas[mid_point:]
            new_embeddings = embeddings[mid_point:]
            new_center = np.mean(new_embeddings, axis=0)
            new_coherence = self._calculate_topic_coherence(np.array(new_embeddings))
            
            new_topic = {
                "id": f"topic_{self.topic_id_counter}",
                "sentences": new_sentences,
                "meta": new_metas,
                "center": new_center,
                "coherence_score": new_coherence
            }
            self.topics.append(new_topic)
            self.topic_id_counter += 1
            # 标记缓存失效
            self._cache_valid = False
    
    def _calculate_topic_coherence(self, embeddings: np.ndarray) -> float:
        """
        计算主题内各句间平均相似度作为内聚度指标
        
        Args:
            embeddings: 主题内句子的嵌入向量矩阵
            
        Returns:
            主题内聚度分数 (0-1之间)
        """
        if len(embeddings) < 2:
            return 1.0
        
        # 计算所有句子对之间的余弦相似度
        similarity_matrix = cosine_similarity(embeddings)
        
        # 排除对角线元素（自相似度），计算平均相似度
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        similarities = similarity_matrix[mask]
        
        return float(np.mean(similarities))
    
    @staticmethod
    def process_documents_parallel(documents: List[Tuple[str, List[str]]], config: dict, 
                                 max_workers: int = None) -> 'TopicPoolManager':
        """
        并行处理多个文档，每个文档独立处理后合并主题池
        
        Args:
            documents: 文档列表，每个元素为(文档名, 句子列表)的元组
            config: 配置字典
            max_workers: 最大工作线程数
            
        Returns:
            合并后的TopicPoolManager实例
        """
        if max_workers is None:
            max_workers = min(4, mp.cpu_count())
        
        if len(documents) <= 1 or not config.get("topic_pool", {}).get("enable_parallel", True):
            # 单文档或禁用并行时，使用顺序处理
            return TopicPoolManager._process_documents_sequential(documents, config)
        
        # 并行处理文档
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for doc_name, sentences in documents:
                future = executor.submit(TopicPoolManager._process_single_document, 
                                       doc_name, sentences, config)
                futures.append(future)
            
            # 收集结果
            document_results = []
            for future in futures:
                result = future.result()
                if result:
                    document_results.append(result)
        
        # 合并所有文档的主题池
        return TopicPoolManager._merge_topic_pools(document_results, config)
    
    @staticmethod
    def _process_single_document(doc_name: str, sentences: List[str], config: dict) -> Dict:
        """
        处理单个文档
        
        Args:
            doc_name: 文档名称
            sentences: 句子列表
            config: 配置字典
            
        Returns:
            包含主题信息的字典
        """
        try:
            # 为每个进程创建独立的TopicPoolManager
            topic_manager = TopicPoolManager(config=config)
            
            # 批量添加句子，传入文档名称用于隔离处理
            metas = [{"source": doc_name, "sentence_id": i} for i in range(len(sentences))]
            topic_manager.add_sentences_batch(sentences, metas, document_name=doc_name)
            
            # 处理完成后进行最终优化
            topic_manager.optimize_topic_pool()
            
            # 返回主题数据
            return {
                "doc_name": doc_name,
                "topics": topic_manager.topics,
                "topic_id_counter": topic_manager.topic_id_counter,
                "stats": topic_manager.get_topic_pool_stats()
            }
        except Exception as e:
            print(f"处理文档 {doc_name} 时出错: {e}")
            return None
    
    @staticmethod
    def _process_documents_sequential(documents: List[Tuple[str, List[str]]], config: dict) -> 'TopicPoolManager':
        """
        顺序处理多个文档
        
        Args:
            documents: 文档列表
            config: 配置字典
            
        Returns:
            TopicPoolManager实例
        """
        topic_manager = TopicPoolManager(config=config)
        
        for doc_name, sentences in documents:
            metas = [{"source": doc_name, "sentence_id": i} for i in range(len(sentences))]
            topic_manager.add_sentences_batch(sentences, metas)
        
        return topic_manager
    
    @staticmethod
    def _merge_topic_pools(document_results: List[Dict], config: dict) -> 'TopicPoolManager':
        """
        合并多个文档的主题池
        
        Args:
            document_results: 文档处理结果列表
            config: 配置字典
            
        Returns:
            合并后的TopicPoolManager实例
        """
        merged_manager = TopicPoolManager(config=config)
        
        # 简单合并策略：将所有主题直接添加到新的管理器中
        for result in document_results:
            if result and "topics" in result:
                for topic in result["topics"]:
                    # 重新分配主题ID以避免冲突
                    topic["id"] = f"topic_{merged_manager.topic_id_counter}"
                    merged_manager.topics.append(topic)
                    merged_manager.topic_id_counter += 1
        
        # 可选：对合并后的主题进行二次聚类以减少冗余
        if config.get("topic_pool", {}).get("enable_post_merge_clustering", False):
            merged_manager._post_merge_clustering()
        
        return merged_manager
    
    def _post_merge_clustering(self):
        """
        合并后的二次聚类，用于减少相似主题
        """
        if len(self.topics) <= 1:
            return
        
        # 计算所有主题中心之间的相似度矩阵
        centers = np.vstack([topic["center"] for topic in self.topics])
        similarity_matrix = cosine_similarity(centers)
        
        # 找出高度相似的主题对并合并
        merge_threshold = self.config.get("topic_pool", {}).get("merge_threshold", 0.85)
        merged_indices = set()
        
        for i in range(len(self.topics)):
            if i in merged_indices:
                continue
                
            for j in range(i + 1, len(self.topics)):
                if j in merged_indices:
                    continue
                    
                if similarity_matrix[i][j] >= merge_threshold:
                    # 合并主题j到主题i
                    self.topics[i]["sentences"].extend(self.topics[j]["sentences"])
                    self.topics[i]["meta"].extend(self.topics[j]["meta"])
                    
                    # 重新计算中心向量
                    all_embeddings = [self._get_embedding(s) for s in self.topics[i]["sentences"]]
                    self.topics[i]["center"] = np.mean(all_embeddings, axis=0)
                    
                    merged_indices.add(j)
        
        # 移除被合并的主题
        self.topics = [topic for i, topic in enumerate(self.topics) if i not in merged_indices]
        
        # 重置缓存
        self._cache_valid = False
    
    def clear_topic_pool(self):
        """
        清空主题池，用于文档隔离处理
        """
        with self._lock:
            self.topics.clear()
            self.topic_id_counter = 0
            self._topic_centers_cache = None
            self._cache_valid = False
            self._sentences_processed = 0
            self._last_merge_check = 0
            
            # 清空FAISS索引
            self._faiss_index = None
            self._faiss_id_map = []
            if self.enable_faiss_index and self._embedding_dim is not None:
                self._init_faiss_index()
    
    def save_and_reset_topic_pool(self) -> List[Dict]:
        """
        保存当前主题池并重置，用于文档隔离处理
        
        Returns:
            保存的主题列表
        """
        with self._lock:
            saved_topics = self.topics.copy()
            self.clear_topic_pool()
            return saved_topics
    
    def _check_and_merge_topics(self):
        """
        检查并合并相似的主题，使用向量化计算优化性能
        """
        if len(self.topics) < 2 or not self.enable_topic_merging:
            return
        
        with self._lock:
            self._merge_similar_topics_unsafe()
    
    def _merge_similar_topics_unsafe(self):
        """
        合并相似主题的内部实现（不加锁版本）
        使用向量化计算和ANN索引优化性能
        """
        if len(self.topics) < 2:
            return
        
        # 更新主题中心缓存
        self._update_topic_centers_cache()
        
        if self._topic_centers_cache is None:
            return
        
        # 计算所有主题中心之间的相似度矩阵
        similarity_matrix = cosine_similarity(self._topic_centers_cache)
        
        # 找出需要合并的主题对
        merge_pairs = []
        merged_indices = set()
        
        for i in range(len(self.topics)):
            if i in merged_indices:
                continue
            
            for j in range(i + 1, len(self.topics)):
                if j in merged_indices:
                    continue
                
                if similarity_matrix[i][j] >= self.topic_merge_threshold:
                    merge_pairs.append((i, j))
                    merged_indices.add(j)
        
        # 执行合并操作
        if merge_pairs:
            self._execute_topic_merges_unsafe(merge_pairs, merged_indices)
    
    def _execute_topic_merges_unsafe(self, merge_pairs: List[Tuple[int, int]], merged_indices: set):
        """
        执行主题合并操作（不加锁版本）
        
        Args:
            merge_pairs: 需要合并的主题对列表
            merged_indices: 被合并的主题索引集合
        """
        # 执行合并
        for i, j in merge_pairs:
            if i < len(self.topics) and j < len(self.topics):
                # 合并主题j到主题i
                self.topics[i]["sentences"].extend(self.topics[j]["sentences"])
                self.topics[i]["meta"].extend(self.topics[j]["meta"])
                
                # 重新计算中心向量（使用加权平均）
                all_embeddings = [self._get_embedding(s) for s in self.topics[i]["sentences"]]
                self.topics[i]["center"] = np.mean(all_embeddings, axis=0)
                
                # 更新内聚度分数
                if len(all_embeddings) > 1:
                    self.topics[i]["coherence_score"] = self._calculate_topic_coherence(np.array(all_embeddings))
        
        # 移除被合并的主题（从后往前删除以避免索引问题）
        for idx in sorted(merged_indices, reverse=True):
            if idx < len(self.topics):
                del self.topics[idx]
        
        # 重建FAISS索引（因为主题索引发生了变化）
        self._rebuild_faiss_index()
        
        # 重置缓存
        self._cache_valid = False
    
    def force_merge_check(self):
        """
        强制执行主题合并检查，用于手动触发合并
        """
        if self.enable_topic_merging:
            self._check_and_merge_topics()
    
    def get_topic_pool_stats(self) -> Dict:
        """
        获取主题池统计信息
        
        Returns:
            包含统计信息的字典
        """
        with self._lock:
            total_sentences = sum(len(topic["sentences"]) for topic in self.topics)
            total_chars = sum(sum(len(s) for s in topic["sentences"]) for topic in self.topics)
            avg_sentences_per_topic = total_sentences / len(self.topics) if self.topics else 0
            avg_chars_per_topic = total_chars / len(self.topics) if self.topics else 0
            
            # FAISS索引统计信息
            faiss_stats = {
                "faiss_enabled": self.enable_faiss_index,
                "faiss_index_type": self.faiss_index_type if self.enable_faiss_index else None,
                "faiss_index_size": len(self._faiss_id_map) if self._faiss_index is not None else 0,
                "faiss_embedding_dim": self._embedding_dim
            }
            
            return {
                "total_topics": len(self.topics),
                "total_sentences": total_sentences,
                "total_chars": total_chars,
                "avg_sentences_per_topic": avg_sentences_per_topic,
                "avg_chars_per_topic": avg_chars_per_topic,
                "sentences_processed": self._sentences_processed,
                "current_document": self._current_document,
                "cache_valid": self._cache_valid,
                **faiss_stats
            }
    
    def optimize_topic_pool(self):
        """
        优化主题池：执行合并、清理和重组操作
        """
        with self._lock:
            # 1. 强制合并相似主题
            if self.enable_topic_merging:
                self._merge_similar_topics_unsafe()
            
            # 2. 清理过小的主题（合并到最相似的主题中）
            self._cleanup_small_topics_unsafe()
            
            # 3. 如果主题池过大，保留最有代表性的主题
            if len(self.topics) > self.max_topic_pool_size:
                self._trim_topic_pool_unsafe()
    
    def _cleanup_small_topics_unsafe(self):
        """
        清理过小的主题，将其合并到最相似的大主题中（不加锁版本）
        """
        if len(self.topics) < 2:
            return
        
        small_topics = []
        large_topics = []
        
        for i, topic in enumerate(self.topics):
            if len(topic["sentences"]) < self.min_topic_size:
                small_topics.append(i)
            else:
                large_topics.append(i)
        
        if not small_topics or not large_topics:
            return
        
        # 为每个小主题找到最相似的大主题
        for small_idx in small_topics:
            best_large_idx = None
            best_similarity = -1
            
            for large_idx in large_topics:
                similarity = cosine_similarity(
                    self.topics[small_idx]["center"].reshape(1, -1),
                    self.topics[large_idx]["center"].reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_large_idx = large_idx
            
            # 合并到最相似的大主题
            if best_large_idx is not None:
                self.topics[best_large_idx]["sentences"].extend(self.topics[small_idx]["sentences"])
                self.topics[best_large_idx]["meta"].extend(self.topics[small_idx]["meta"])
                
                # 重新计算中心向量
                all_embeddings = [self._get_embedding(s) for s in self.topics[best_large_idx]["sentences"]]
                self.topics[best_large_idx]["center"] = np.mean(all_embeddings, axis=0)
        
        # 移除小主题（从后往前删除）
        for idx in sorted(small_topics, reverse=True):
            del self.topics[idx]
        
        # 重建FAISS索引
        self._rebuild_faiss_index()
        
        self._cache_valid = False
    
    def _trim_topic_pool_unsafe(self):
        """
        修剪主题池，保留最有代表性的主题（不加锁版本）
        """
        if len(self.topics) <= self.max_topic_pool_size:
            return
        
        # 计算每个主题的重要性分数（基于句子数量和内聚度）
        topic_scores = []
        for i, topic in enumerate(self.topics):
            sentence_count = len(topic["sentences"])
            coherence = topic.get("coherence_score", 0.5)
            # 重要性分数 = 句子数量 * 内聚度
            score = sentence_count * coherence
            topic_scores.append((score, i))
        
        # 按分数排序，保留前max_topic_pool_size个主题
        topic_scores.sort(reverse=True)
        keep_indices = [idx for _, idx in topic_scores[:self.max_topic_pool_size]]
        
        # 保留选中的主题
        self.topics = [self.topics[i] for i in sorted(keep_indices)]
        
        # 重建FAISS索引
        self._rebuild_faiss_index()
        
        self._cache_valid = False

    def get_all_topics(self, llm_client=None) -> List[Dict]:
        result = []
        for topic in self.topics:
            text = "\n".join(topic["sentences"])
            sources = list({m['source'] for m in topic["meta"] if m and 'source' in m})
            summary = generate_topic_summary(text, llm_client) if llm_client else topic["id"]
            result.append({
                "id": topic["id"],
                "text": text,
                "source": ",".join(sources),
                "summary": summary,
                "title": summary[:50] + "..." if len(summary) > 50 else summary  # 添加标题字段
            })
        return result
    
    def get_topic_pool_info(self) -> Dict[str, Any]:
        """
        获取主题池的详细信息
        
        Returns:
            包含主题池统计信息的字典
        """
        total_sentences = sum(len(topic["sentences"]) for topic in self.topics)
        total_chars = sum(sum(len(s) for s in topic["sentences"]) for topic in self.topics)
        
        return {
            "total_topics": len(self.topics),
            "total_sentences": total_sentences,
            "total_chars": total_chars,
            "avg_sentences_per_topic": total_sentences / len(self.topics) if self.topics else 0,
            "avg_chars_per_topic": total_chars / len(self.topics) if self.topics else 0,
            "max_sentences_per_topic": self.max_sentences_per_topic,
            "max_chars_per_topic": self.max_chars_per_topic,
            "similarity_threshold": self.similarity_threshold,
            "embedding_model": self.llm_client.model_name if hasattr(self.llm_client, 'model_name') else "unknown",
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            "sentences_processed": self._sentences_processed,
            "batch_processing_enabled": True,
            "parallel_processing_enabled": self.enable_parallel,
            "redundancy_filtering_enabled": self.redundancy_filter is not None
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        获取性能指标
        
        Returns:
            包含性能统计的字典
        """
        metrics = {
            "embedding_cache_size": len(self._embedding_cache) if self._embedding_cache else 0,
            "embedding_cache_hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
            "total_embeddings_computed": self._cache_misses,
            "total_embeddings_cached": self._cache_hits,
            "sentences_processed": self._sentences_processed,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers if self.enable_parallel else 1
        }
        
        # 添加冗余过滤器的性能指标
        if self.redundancy_filter and hasattr(self.redundancy_filter, 'get_statistics'):
            redundancy_stats = self.redundancy_filter.get_statistics()
            metrics["redundancy_filter_stats"] = redundancy_stats
            
            if hasattr(self.redundancy_filter, 'get_performance_metrics'):
                redundancy_perf = self.redundancy_filter.get_performance_metrics()
                metrics["redundancy_filter_performance"] = redundancy_perf
        
        return metrics
    
    def optimize_performance(self):
        """
        执行性能优化操作
        """
        optimizations_applied = []
        
        # 优化嵌入缓存
        if self._embedding_cache and len(self._embedding_cache) > self.max_cache_size:
            # 清理最旧的缓存项
            cache_items = list(self._embedding_cache.items())
            keep_count = self.max_cache_size // 2
            self._embedding_cache = dict(cache_items[-keep_count:])
            optimizations_applied.append(f"嵌入缓存优化: 保留 {keep_count} 项")
        
        # 优化冗余过滤器
        if self.redundancy_filter and hasattr(self.redundancy_filter, 'optimize_buffer'):
            self.redundancy_filter.optimize_buffer()
            optimizations_applied.append("冗余过滤器缓冲区优化")
        
        # 检查主题合并
        if self.enable_topic_merging:
            merged_count = self._check_and_merge_topics()
            if merged_count > 0:
                optimizations_applied.append(f"主题合并: {merged_count} 个主题")
        
        if optimizations_applied:
            print(f"[TopicPoolManager] 性能优化完成: {', '.join(optimizations_applied)}")
        
        return optimizations_applied
