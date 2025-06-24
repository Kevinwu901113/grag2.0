import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import deque
import gc
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from loguru import logger

class UltraOptimizedRedundancyBuffer:
    """
    超高效批处理冗余过滤器
    
    优化特性：
    1. 向量化批量计算
    2. 内存池管理
    3. 多线程并行处理
    4. 智能缓存策略
    5. 动态批量大小调整
    6. 内存使用优化
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # 基础配置
        self.base_threshold = config.get("base_threshold", 0.95)
        self.enable_dynamic_threshold = config.get("enable_dynamic_threshold", True)
        self.max_buffer_size = config.get("max_buffer_size", 5000)
        self.min_batch_size = config.get("min_batch_size", 50)
        self.max_batch_size = config.get("max_batch_size", 500)
        self.adaptive_batch_size = config.get("adaptive_batch_size", True)
        
        # 并行处理配置
        self.enable_parallel = config.get("enable_parallel", True)
        self.max_workers = min(config.get('max_workers', cpu_count()), 8)
        self.parallel_threshold = config.get("parallel_threshold", 100)  # 超过此数量才启用并行
        
        # 内存优化配置
        self.enable_memory_pool = config.get("enable_memory_pool", True)
        self.memory_cleanup_interval = config.get("memory_cleanup_interval", 1000)
        self.similarity_cache_size = config.get("similarity_cache_size", 10000)
        
        # 日志和进度条配置
        self.enable_logging = config.get("enable_logging", True)
        self.enable_progress = config.get("enable_progress", True)
        self.log_interval = config.get("log_interval", 100)
        
        # 存储数据
        self.embeddings = []  # 使用numpy数组存储以提高效率
        self.sentences = []
        self.redundant = []
        
        # 性能优化相关
        self.similarity_history = deque(maxlen=1000)  # 使用deque提高性能
        self.processing_times = deque(maxlen=100)
        self.current_batch_size = self.min_batch_size
        self.processed_count = 0
        
        # 缓存和内存池
        self.similarity_cache = {}
        self.embedding_matrix = None  # 预分配的嵌入矩阵
        self.last_cleanup = 0
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 初始化日志
        if self.enable_logging:
            logger.info(f"初始化超高效冗余过滤器 - 基础阈值: {self.base_threshold}, "
                       f"缓冲区大小: {self.max_buffer_size}, 并行处理: {self.enable_parallel}, "
                       f"最大工作线程: {self.max_workers}, 内存池: {self.enable_memory_pool}")
        
    def _update_embedding_matrix(self):
        """更新嵌入矩阵以支持向量化计算"""
        if self.embeddings:
            self.embedding_matrix = np.vstack(self.embeddings)
        else:
            self.embedding_matrix = None
    
    def _adaptive_batch_size_adjustment(self, processing_time: float, batch_size: int):
        """根据处理时间动态调整批量大小"""
        if not self.adaptive_batch_size:
            return
            
        self.processing_times.append(processing_time)
        
        if len(self.processing_times) < 5:
            return
            
        # 计算最近几次的平均处理时间
        recent_avg_time = np.mean(list(self.processing_times)[-5:])
        
        # 目标处理时间：每批次0.1-0.5秒
        target_time = 0.3
        
        if recent_avg_time > target_time * 1.5 and self.current_batch_size > self.min_batch_size:
            # 处理太慢，减小批量
            self.current_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.8))
        elif recent_avg_time < target_time * 0.5 and self.current_batch_size < self.max_batch_size:
            # 处理很快，增大批量
            self.current_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
    
    def _vectorized_similarity_computation(self, new_embeddings: np.ndarray) -> np.ndarray:
        """向量化相似度计算"""
        if self.embedding_matrix is None or len(self.embedding_matrix) == 0:
            return np.array([])
            
        # 使用批量余弦相似度计算
        similarities = cosine_similarity(new_embeddings, self.embedding_matrix)
        return similarities
    
    def _parallel_redundancy_check(self, sentences: List[str], embeddings: List[np.ndarray]) -> List[bool]:
        """并行冗余检测"""
        if len(sentences) < self.parallel_threshold or not self.enable_parallel:
            return self._sequential_redundancy_check(sentences, embeddings)
        
        results = [False] * len(sentences)
        chunk_size = max(10, len(sentences) // self.max_workers)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i in range(0, len(sentences), chunk_size):
                chunk_sentences = sentences[i:i+chunk_size]
                chunk_embeddings = embeddings[i:i+chunk_size]
                chunk_start_idx = i
                
                future = executor.submit(
                    self._process_chunk, 
                    chunk_sentences, 
                    chunk_embeddings, 
                    chunk_start_idx
                )
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                chunk_results, start_idx = future.result()
                for j, result in enumerate(chunk_results):
                    results[start_idx + j] = result
        
        return results
    
    def _process_chunk(self, sentences: List[str], embeddings: List[np.ndarray], start_idx: int) -> Tuple[List[bool], int]:
        """处理单个数据块"""
        chunk_results = []
        
        with self.lock:
            current_embedding_matrix = self.embedding_matrix.copy() if self.embedding_matrix is not None else None
            current_threshold = self.base_threshold
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            if current_embedding_matrix is None or len(current_embedding_matrix) == 0:
                chunk_results.append(False)
                continue
            
            # 计算相似度
            similarities = cosine_similarity([embedding], current_embedding_matrix)[0]
            max_similarity = np.max(similarities)
            
            # 动态阈值调整
            adjusted_threshold = self._get_adjusted_threshold(sentence, max_similarity)
            
            is_redundant = max_similarity >= adjusted_threshold
            chunk_results.append(is_redundant)
            
            # 如果不冗余，需要添加到缓冲区（这部分需要同步）
            if not is_redundant:
                with self.lock:
                    self._add_sentence_unsafe(sentence, embedding)
                    self._update_embedding_matrix()
                    current_embedding_matrix = self.embedding_matrix.copy()
        
        return chunk_results, start_idx
    
    def _sequential_redundancy_check(self, sentences: List[str], embeddings: List[np.ndarray]) -> List[bool]:
        """顺序冗余检测"""
        results = []
        
        # 转换为numpy数组以提高效率
        if embeddings:
            embeddings_array = np.vstack(embeddings)
        else:
            return []
        
        # 批量计算所有相似度
        if self.embedding_matrix is not None and len(self.embedding_matrix) > 0:
            all_similarities = self._vectorized_similarity_computation(embeddings_array)
        else:
            all_similarities = np.array([])
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            if self.embedding_matrix is None or len(self.embedding_matrix) == 0:
                # 缓冲区为空，直接添加
                self._add_sentence_unsafe(sentence, embedding)
                results.append(False)
            else:
                # 获取当前句子的相似度
                similarities = all_similarities[i] if len(all_similarities) > 0 else np.array([])
                
                if len(similarities) == 0:
                    self._add_sentence_unsafe(sentence, embedding)
                    results.append(False)
                    continue
                
                max_similarity = np.max(similarities)
                max_idx = np.argmax(similarities)
                
                # 动态阈值调整
                adjusted_threshold = self._get_adjusted_threshold(sentence, max_similarity)
                
                # 记录相似度历史
                self.similarity_history.append(max_similarity)
                
                if max_similarity >= adjusted_threshold:
                    # 冗余
                    self._record_redundancy_unsafe(sentence, max_idx, max_similarity)
                    results.append(True)
                else:
                    # 非冗余，添加到缓冲区
                    self._add_sentence_unsafe(sentence, embedding)
                    results.append(False)
                
                # 重新计算嵌入矩阵（仅在添加新句子时）
                if not results[-1]:  # 如果不是冗余
                    self._update_embedding_matrix()
                    # 重新计算后续句子的相似度矩阵
                    if i < len(sentences) - 1:
                        remaining_embeddings = embeddings_array[i+1:]
                        all_similarities = self._vectorized_similarity_computation(remaining_embeddings)
                        # 调整索引
                        all_similarities = np.vstack([np.zeros((i+1, all_similarities.shape[1])), all_similarities]) if len(all_similarities) > 0 else np.array([])
        
        return results
    
    def _get_adjusted_threshold(self, sentence: str, current_similarity: float) -> float:
        """获取调整后的阈值"""
        if not self.enable_dynamic_threshold:
            return self.base_threshold
        
        # 基于句子长度的调整
        length_factor = min(1.0, len(sentence) / 50)
        length_adjustment = (1.0 - length_factor) * 0.02
        
        # 基于历史相似度的调整
        history_adjustment = 0.0
        if len(self.similarity_history) > 10:
            recent_avg = np.mean(list(self.similarity_history)[-10:])
            if recent_avg > 0.8:
                history_adjustment = 0.01
            elif recent_avg < 0.5:
                history_adjustment = -0.01
        
        adjusted = self.base_threshold + length_adjustment + history_adjustment
        return max(0.8, min(0.99, adjusted))
    
    def _add_sentence_unsafe(self, sentence: str, embedding: np.ndarray):
        """添加句子（非线程安全版本）"""
        self.sentences.append(sentence)
        self.embeddings.append(embedding)
        
        # 缓冲区大小控制
        if len(self.embeddings) > self.max_buffer_size:
            self._trim_buffer_unsafe()
    
    def _record_redundancy_unsafe(self, sentence: str, matched_idx: int, similarity: float):
        """记录冗余（非线程安全版本）"""
        self.redundant.append({
            "duplicate": sentence,
            "matched_to": self.sentences[matched_idx] if matched_idx < len(self.sentences) else "unknown",
            "score": float(similarity),
            "matched_index": matched_idx
        })
    
    def _trim_buffer_unsafe(self):
        """修剪缓冲区（非线程安全版本）"""
        keep_size = self.max_buffer_size // 2
        self.sentences = self.sentences[-keep_size:]
        self.embeddings = self.embeddings[-keep_size:]
        self._update_embedding_matrix()
    
    def _memory_cleanup(self):
        """内存清理"""
        if self.processed_count - self.last_cleanup > self.memory_cleanup_interval:
            # 清理相似度缓存
            if len(self.similarity_cache) > self.similarity_cache_size:
                # 保留最近使用的缓存项
                cache_items = list(self.similarity_cache.items())
                self.similarity_cache = dict(cache_items[-self.similarity_cache_size//2:])
            
            # 强制垃圾回收
            gc.collect()
            
            self.last_cleanup = self.processed_count
    
    def is_redundant_ultra_batch(self, sentences: List[str], embeddings: List[np.ndarray], 
                                desc: str = "超高效批量冗余过滤") -> List[bool]:
        """超高效批量冗余检测"""
        if not sentences:
            return []
        
        start_time = time.time()
        
        # 初始化进度条
        progress_bar = None
        if self.enable_progress:
            progress_bar = tqdm(
                total=len(sentences),
                desc=desc,
                unit="句",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )
            
        if self.enable_logging:
            logger.info(f"开始超高效批量冗余过滤，总计 {len(sentences)} 个句子，"
                       f"并行处理: {self.enable_parallel}, 最大工作线程: {self.max_workers}")
        
        try:
            with self.lock:
                # 使用当前最优批量大小
                batch_size = self.current_batch_size
                all_results = []
                
                # 分批处理
                for i in range(0, len(sentences), batch_size):
                    batch_sentences = sentences[i:i+batch_size]
                    batch_embeddings = embeddings[i:i+batch_size]
                    
                    # 选择处理策略
                    if len(batch_sentences) >= self.parallel_threshold and self.enable_parallel:
                        batch_results = self._parallel_redundancy_check(batch_sentences, batch_embeddings)
                        if self.enable_logging and i == 0:
                            logger.debug(f"使用并行处理模式，批量大小: {batch_size}")
                    else:
                        batch_results = self._sequential_redundancy_check(batch_sentences, batch_embeddings)
                        if self.enable_logging and i == 0:
                            logger.debug(f"使用顺序处理模式，批量大小: {batch_size}")
                    
                    all_results.extend(batch_results)
                    self.processed_count += len(batch_sentences)
                    
                    # 更新进度条
                    if progress_bar:
                        progress_bar.update(len(batch_sentences))
                        redundant_count = sum(all_results)
                        progress_bar.set_postfix({
                            '冗余': f'{redundant_count}/{len(all_results)}',
                            '冗余率': f'{redundant_count/len(all_results):.1%}' if all_results else '0.0%',
                            '批量': f'{batch_size}'
                        })
                    
                    # 定期日志输出
                    if self.enable_logging and (i // batch_size + 1) % 10 == 0:
                        processed_so_far = len(all_results)
                        redundant_so_far = sum(all_results)
                        elapsed = time.time() - start_time
                        speed = processed_so_far / elapsed if elapsed > 0 else 0
                        remaining_count = len(sentences) - processed_so_far
                        logger.info(f"已处理 {processed_so_far}/{len(sentences)} 个句子，"
                                   f"冗余 {redundant_so_far} 个 (冗余率: {redundant_so_far/processed_so_far:.1%})，"
                                   f"剩余 {remaining_count} 个句子待处理，处理速度: {speed:.1f} 句/秒")
                
                # 性能调优
                processing_time = time.time() - start_time
                self._adaptive_batch_size_adjustment(processing_time, len(sentences))
                
                # 内存清理
                self._memory_cleanup()
                
        finally:
            # 关闭进度条
            if progress_bar:
                progress_bar.close()
        
        # 最终统计日志
        if self.enable_logging:
            processing_time = time.time() - start_time
            redundant_count = sum(all_results)
            speed = len(sentences) / processing_time if processing_time > 0 else 0
            processed_count = len(sentences) - redundant_count
            logger.info(f"超高效批量冗余过滤完成！总计 {len(sentences)} 个句子，"
                       f"处理 {processed_count} 个句子，检测到 {redundant_count} 个冗余句子 "
                       f"(冗余率: {redundant_count/len(sentences):.1%})，"
                       f"平均处理速度: {speed:.1f} 句/秒，总耗时: {processing_time:.2f}秒")
        
        return all_results
    
    def get_ultra_statistics(self) -> Dict[str, Any]:
        """获取超详细统计信息"""
        with self.lock:
            total_processed = len(self.sentences) + len(self.redundant)
            
            return {
                "total_sentences": len(self.sentences),
                "redundant_count": len(self.redundant),
                "total_processed": total_processed,
                "redundancy_rate": len(self.redundant) / max(1, total_processed),
                "avg_similarity": np.mean(list(self.similarity_history)) if self.similarity_history else 0.0,
                "max_similarity": np.max(list(self.similarity_history)) if self.similarity_history else 0.0,
                "min_similarity": np.min(list(self.similarity_history)) if self.similarity_history else 0.0,
                "current_threshold": self.base_threshold,
                "current_batch_size": self.current_batch_size,
                "buffer_size": len(self.embeddings),
                "avg_processing_time": np.mean(list(self.processing_times)) if self.processing_times else 0.0,
                "cache_size": len(self.similarity_cache),
                "parallel_enabled": self.enable_parallel,
                "max_workers": self.max_workers,
                "memory_cleanup_count": self.processed_count // self.memory_cleanup_interval,
                "embedding_matrix_shape": self.embedding_matrix.shape if self.embedding_matrix is not None else (0, 0)
            }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """执行性能优化"""
        with self.lock:
            optimization_results = {
                "buffer_trimmed": False,
                "cache_cleared": False,
                "batch_size_adjusted": False,
                "memory_cleaned": False
            }
            
            # 1. 缓冲区优化
            if len(self.embeddings) > self.max_buffer_size * 0.8:
                old_size = len(self.embeddings)
                self._trim_buffer_unsafe()
                optimization_results["buffer_trimmed"] = True
                optimization_results["buffer_size_change"] = f"{old_size} -> {len(self.embeddings)}"
            
            # 2. 缓存清理
            if len(self.similarity_cache) > self.similarity_cache_size * 0.8:
                old_cache_size = len(self.similarity_cache)
                self.similarity_cache.clear()
                optimization_results["cache_cleared"] = True
                optimization_results["cache_size_change"] = f"{old_cache_size} -> 0"
            
            # 3. 批量大小调整
            if len(self.processing_times) >= 5:
                old_batch_size = self.current_batch_size
                avg_time = np.mean(list(self.processing_times))
                if avg_time > 0.5:  # 太慢
                    self.current_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.7))
                elif avg_time < 0.1:  # 太快
                    self.current_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.3))
                
                if old_batch_size != self.current_batch_size:
                    optimization_results["batch_size_adjusted"] = True
                    optimization_results["batch_size_change"] = f"{old_batch_size} -> {self.current_batch_size}"
            
            # 4. 强制内存清理
            gc.collect()
            optimization_results["memory_cleaned"] = True
            
            return optimization_results
    
    def get_redundant_log(self):
        """获取冗余记录"""
        with self.lock:
            return self.redundant.copy()
    
    def clear_all(self):
        """清空所有数据"""
        with self.lock:
            self.embeddings.clear()
            self.sentences.clear()
            self.redundant.clear()
            self.similarity_history.clear()
            self.processing_times.clear()
            self.similarity_cache.clear()
            self.embedding_matrix = None
            self.processed_count = 0
            self.last_cleanup = 0
            gc.collect()