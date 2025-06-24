import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
from loguru import logger
import time

class RedundancyBuffer:
    def __init__(self, config=None):
        self.embeddings = []  # 已接收句子的嵌入列表
        self.sentences = []   # 已接收的句子
        
        # 处理配置参数
        if config is None:
            config = {}
        
        if isinstance(config, dict):
            self.threshold = config.get('threshold', 0.95)
            self.enable_logging = config.get('enable_logging', True)
            self.enable_progress = config.get('enable_progress', True)
            self.log_interval = config.get('log_interval', 100)  # 每处理多少句子记录一次日志
        else:
            # 兼容旧的直接传递threshold的方式
            self.threshold = config
            self.enable_logging = True
            self.enable_progress = True
            self.log_interval = 100
            
        self.redundant = []   # 存冗余记录
        self.similarity_history = []  # 相似度历史记录
        
        # 统计信息
        self.processed_count = 0
        self.redundant_count = 0
        self.start_time = None
        
        # 进度条
        self.progress_bar = None
        
        if self.enable_logging:
            logger.info(f"初始化冗余过滤器，阈值: {self.threshold}")
        
    def is_redundant(self, sentence: str, embedding: np.ndarray) -> bool:
        # 初始化计时
        if self.start_time is None:
            self.start_time = time.time()
            
        self.processed_count += 1
        
        if not self.embeddings:
            self.embeddings.append(embedding)
            self.sentences.append(sentence)
            
            if self.enable_logging:
                logger.debug(f"添加第一个句子: {sentence[:50]}...")
            
            # 更新进度条
            if self.progress_bar:
                self.progress_bar.update(1)
                
            return False

        sims = cosine_similarity([embedding], self.embeddings)[0]
        max_sim = np.max(sims)
        
        # 记录相似度到历史记录中
        if not hasattr(self, 'similarity_history'):
            self.similarity_history = []
        self.similarity_history.append(max_sim)
        
        if max_sim >= self.threshold:
            # 冗余，记录来源
            idx = int(np.argmax(sims))
            self.redundant.append({
                "duplicate": sentence,
                "matched_to": self.sentences[idx],
                "score": float(max_sim)
            })
            
            self.redundant_count += 1
            
            if self.enable_logging:
                logger.debug(f"检测到冗余句子 (相似度: {max_sim:.3f}): {sentence[:50]}...")
                
            # 更新进度条
            if self.progress_bar:
                self.progress_bar.update(1)
                self.progress_bar.set_postfix({
                    '冗余': f'{self.redundant_count}/{self.processed_count}',
                    '冗余率': f'{self.redundant_count/self.processed_count:.1%}'
                })
                
            return True
        else:
            self.embeddings.append(embedding)
            self.sentences.append(sentence)
            
            if self.enable_logging and self.processed_count % self.log_interval == 0:
                elapsed_time = time.time() - self.start_time
                speed = self.processed_count / elapsed_time if elapsed_time > 0 else 0
                remaining = getattr(self, 'total_sentences', 0) - self.processed_count
                logger.info(f"已处理 {self.processed_count} 个句子，检测到 {self.redundant_count} 个冗余句子，"
                           f"剩余 {remaining} 个句子待处理 (冗余率: {self.redundant_count/self.processed_count:.1%})，"
                           f"处理速度: {speed:.1f} 句/秒")
            
            # 更新进度条
            if self.progress_bar:
                self.progress_bar.update(1)
                if self.processed_count % 10 == 0:  # 每10个句子更新一次后缀
                    self.progress_bar.set_postfix({
                        '冗余': f'{self.redundant_count}/{self.processed_count}',
                        '冗余率': f'{self.redundant_count/self.processed_count:.1%}'
                    })
                
            return False

    def is_redundant_batch(self, sentences: List[str], embeddings: List[np.ndarray]) -> List[bool]:
        """
        批量冗余检测，优化性能
        注意：为了保持与逐个处理的一致性，需要按顺序处理每个句子
        
        Args:
            sentences: 句子列表
            embeddings: 对应的嵌入向量列表
            
        Returns:
            布尔列表，表示每个句子是否冗余
        """
        if not sentences:
            return []
            
        if len(sentences) != len(embeddings):
            raise ValueError("sentences和embeddings长度必须相同")
        
        results = []
        
        # 逐个处理每个句子，但可以优化相似度计算
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            if not self.embeddings:
                # 缓冲区为空，直接添加
                self.embeddings.append(embedding)
                self.sentences.append(sentence)
                results.append(False)
            else:
                # 计算与已存储句子的相似度
                stored_embeddings_array = np.array(self.embeddings)
                embedding_2d = embedding.reshape(1, -1)
                
                # 计算相似度
                similarities = cosine_similarity(embedding_2d, stored_embeddings_array)[0]
                max_sim = np.max(similarities)
                
                if max_sim >= self.threshold:
                    # 冗余，记录来源
                    idx = int(np.argmax(similarities))
                    self.redundant.append({
                        "duplicate": sentence,
                        "matched_to": self.sentences[idx],
                        "score": float(max_sim)
                    })
                    results.append(True)
                else:
                    # 不冗余，添加到缓冲区
                    self.embeddings.append(embedding)
                    self.sentences.append(sentence)
                    results.append(False)
        
        return results
    
    def get_statistics(self):
        """
        获取冗余过滤器的统计信息
        
        Returns:
            dict: 包含统计信息的字典
        """
        if not hasattr(self, 'similarity_history'):
            self.similarity_history = []
            
        total_checks = len(self.similarity_history)
        if total_checks == 0:
            return {
                'total_checks': 0,
                'avg_similarity': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 0.0,
                'buffer_size': len(self.embeddings),
                'processed_count': self.processed_count,
                'redundant_count': self.redundant_count,
                'redundancy_rate': 0.0,
                'processing_speed': 0.0
            }
            
        return {
            'total_checks': total_checks,
            'avg_similarity': sum(self.similarity_history) / total_checks,
            'max_similarity': max(self.similarity_history),
            'min_similarity': min(self.similarity_history),
            'buffer_size': len(self.embeddings),
            'processed_count': self.processed_count,
            'redundant_count': self.redundant_count,
            'redundancy_rate': self.redundant_count / self.processed_count if self.processed_count > 0 else 0.0,
            'processing_speed': self.processed_count / (time.time() - self.start_time) if self.start_time else 0.0
        }
    
    def init_progress_bar(self, total: int, desc: str = "冗余过滤进度"):
        """初始化进度条"""
        if self.enable_progress:
            self.progress_bar = tqdm(
                total=total,
                desc=desc,
                unit="句",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )
            if self.enable_logging:
                logger.info(f"开始冗余过滤，总计 {total} 个句子")
    
    def close_progress_bar(self):
        """关闭进度条并输出最终统计"""
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None
            
        if self.enable_logging and self.processed_count > 0:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            speed = self.processed_count / elapsed_time if elapsed_time > 0 else 0
            logger.info(f"冗余过滤完成！总计处理 {self.processed_count} 个句子，"
                       f"检测到 {self.redundant_count} 个冗余句子 "
                       f"(冗余率: {self.redundant_count/self.processed_count:.1%})，"
                       f"平均处理速度: {speed:.1f} 句/秒，总耗时: {elapsed_time:.1f}秒")
    
    def process_batch_with_progress(self, sentences: List[str], embeddings: List[np.ndarray],
                                   desc: str = "批量冗余过滤") -> List[bool]:
        """带进度条的批量处理方法"""
        results = []
        
        # 设置总句子数，用于显示剩余数量
        self._total_sentences = len(sentences)
        
        # 设置总句子数
        self.total_sentences = len(sentences)
        
        if self.enable_logging:
            logger.info(f"开始批量冗余过滤，总计 {len(sentences)} 个句子")
        
        # 初始化进度条
        self.init_progress_bar(len(sentences), desc)
        
        try:
            for sentence, embedding in zip(sentences, embeddings):
                is_redundant = self.is_redundant(sentence, embedding)
                results.append(is_redundant)
                
        finally:
            # 确保进度条被正确关闭
            self.close_progress_bar()
            
            # 记录完成日志
            if self.enable_logging:
                redundant_count = sum(results)
                processed_count = len(sentences) - redundant_count
                logger.info(f"批量冗余过滤完成！总计 {len(sentences)} 个句子，"
                           f"处理 {processed_count} 个句子，检测到 {redundant_count} 个冗余句子 "
                           f"(冗余率: {redundant_count/len(sentences):.1%})")
            
            # 清除总句子数标记
            if hasattr(self, '_total_sentences'):
                delattr(self, '_total_sentences')
            
        return results
    
    def get_redundant_log(self):
        return self.redundant
        
class EnhancedRedundancyBuffer:
    """
    增强的冗余过滤器，支持局部上下文比对和动态阈值
    """
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # 基础配置
        self.base_threshold = config.get("base_threshold", 0.95)
        self.enable_dynamic_threshold = config.get("enable_dynamic_threshold", True)
        self.context_window = config.get("context_window", 2)
        self.length_factor_weight = config.get("length_factor_weight", 0.3)
        self.semantic_weight = config.get("semantic_weight", 0.8)
        self.context_weight = config.get("context_weight", 0.2)
        self.enable_logging = config.get("enable_logging", True)
        self.enable_progress = config.get("enable_progress", True)
        self.log_interval = config.get("log_interval", 100)
        
        # 存储数据
        self.embeddings = []  # 句子嵌入
        self.sentences = []   # 句子文本
        self.contexts = []    # 上下文信息
        self.redundant = []   # 冗余记录
        
        # 动态阈值相关
        self.similarity_history = []  # 历史相似度记录
        
        # 批处理配置
        self.batch_size = config.get("batch_size", 100)  # 批处理大小
        
        if self.enable_logging:
            logger.info(f"初始化增强冗余过滤器 - 基础阈值: {self.base_threshold}, "
                       f"上下文窗口: {self.context_window}, 自适应阈值: {self.enable_dynamic_threshold}")

    def is_redundant_enhanced(self, sentence: str, embedding: np.ndarray, 
                            context_before: str = "", context_after: str = "") -> bool:
        """
        增强的冗余检测，考虑局部上下文和动态阈值
        
        Args:
            sentence: 待检测的句子
            embedding: 句子的嵌入向量
            context_before: 前文上下文
            context_after: 后文上下文
            
        Returns:
            是否为冗余
        """
        if not self.embeddings:
            self._add_sentence(sentence, embedding, context_before, context_after)
            if self.enable_logging:
                logger.debug(f"添加第一个句子: {sentence[:50]}...")
            return False
            
        # 1. 计算语义相似度
        semantic_sims = cosine_similarity([embedding], self.embeddings)[0]
        
        # 2. 计算上下文相似度（如果提供上下文）
        context_sims = self._calculate_context_similarity(context_before, context_after)
        
        # 3. 动态阈值调整
        adjusted_threshold = self._adjust_threshold(sentence, semantic_sims)
        
        # 4. 综合判断
        max_sim_idx = np.argmax(semantic_sims)
        max_semantic_sim = semantic_sims[max_sim_idx]
        context_sim = context_sims[max_sim_idx] if context_sims else 0
        
        # 综合相似度：语义相似度为主，上下文差异可以降低冗余判断
        combined_sim = (self.semantic_weight * max_semantic_sim + 
                       self.context_weight * context_sim)
        
        # 记录相似度历史
        self.similarity_history.append(max_semantic_sim)
        if len(self.similarity_history) > 100:  # 保持历史记录在合理范围内
            self.similarity_history.pop(0)
        
        if combined_sim >= adjusted_threshold:
            self._record_redundancy(sentence, max_sim_idx, combined_sim, 
                                  semantic_sim=max_semantic_sim, context_sim=context_sim)
            if self.enable_logging:
                logger.debug(f"检测到冗余句子 (综合相似度: {combined_sim:.3f}): {sentence[:50]}...")
            return True
        else:
            self._add_sentence(sentence, embedding, context_before, context_after)
            if self.enable_logging and len(self.sentences) % self.log_interval == 0:
                processed_count = len(self.sentences)
                redundant_count = len(self.redundant)
                # 如果有总数信息，显示剩余数量
                if hasattr(self, '_total_sentences') and self._total_sentences > 0:
                    remaining_count = self._total_sentences - processed_count - redundant_count
                    logger.info(f"已处理 {processed_count} 个句子，检测到 {redundant_count} 个冗余句子，剩余 {remaining_count} 个句子待处理")
                else:
                    logger.info(f"已处理 {processed_count} 个句子，检测到 {redundant_count} 个冗余句子")
            return False
    
    def _add_sentence(self, sentence: str, embedding: np.ndarray, 
                     context_before: str = "", context_after: str = ""):
        """添加新句子到缓冲区"""
        self.embeddings.append(embedding)
        self.sentences.append(sentence)
        self.contexts.append({
            'before': context_before,
            'after': context_after
        })
    
    def _calculate_context_similarity(self, context_before: str, context_after: str) -> Optional[List[float]]:
        """
        计算上下文相似度
        
        Args:
            context_before: 前文上下文
            context_after: 后文上下文
            
        Returns:
            与历史上下文的相似度列表
        """
        if not context_before and not context_after:
            return None
            
        current_context = context_before + " " + context_after
        context_sims = []
        
        for stored_context in self.contexts:
            stored_context_text = stored_context['before'] + " " + stored_context['after']
            
            if not stored_context_text.strip():
                context_sims.append(0.0)
                continue
                
            # 简单的词汇重叠相似度
            current_words = set(current_context.split())
            stored_words = set(stored_context_text.split())
            
            if not current_words and not stored_words:
                sim = 1.0
            elif not current_words or not stored_words:
                sim = 0.0
            else:
                intersection = len(current_words & stored_words)
                union = len(current_words | stored_words)
                sim = intersection / union if union > 0 else 0.0
                
            context_sims.append(sim)
            
        return context_sims
    
    def _adjust_threshold(self, sentence: str, similarities: np.ndarray) -> float:
        """
        动态调整阈值
        
        Args:
            sentence: 当前句子
            similarities: 与历史句子的相似度数组
            
        Returns:
            调整后的阈值
        """
        if not self.enable_dynamic_threshold:
            return self.base_threshold
            
        # 1. 根据句子长度调整：短句子要求更高的相似度
        length_factor = min(1.0, len(sentence) / 50)  # 50字符为基准
        length_adjustment = (1.0 - length_factor) * self.length_factor_weight * 0.05
        
        # 2. 根据历史相似度分布调整
        history_adjustment = 0.0
        if len(self.similarity_history) > 5:
            avg_sim = np.mean(self.similarity_history[-10:])  # 最近10次的平均相似度
            if avg_sim > 0.8:  # 高相似度环境，稍微提高阈值
                history_adjustment = 0.01
            elif avg_sim < 0.5:  # 低相似度环境，稍微降低阈值
                history_adjustment = -0.01
                
        # 3. 根据当前相似度分布调整
        current_adjustment = 0.0
        if len(similarities) > 3:
            max_sim = np.max(similarities)
            avg_sim = np.mean(similarities)
            if max_sim - avg_sim > 0.3:  # 存在明显的高相似度峰值
                current_adjustment = -0.01  # 稍微降低阈值，避免误判
                
        final_threshold = (self.base_threshold + 
                          length_adjustment + 
                          history_adjustment + 
                          current_adjustment)
        
        # 确保阈值在合理范围内
        return max(0.8, min(0.99, final_threshold))
    
    def _record_redundancy(self, sentence: str, matched_idx: int, combined_sim: float,
                          semantic_sim: float = 0.0, context_sim: float = 0.0):
        """记录冗余信息"""
        self.redundant.append({
            "duplicate": sentence,
            "matched_to": self.sentences[matched_idx],
            "combined_score": float(combined_sim),
            "semantic_score": float(semantic_sim),
            "context_score": float(context_sim),
            "matched_index": matched_idx
        })
    
    def get_redundant_log(self):
        """获取冗余记录"""
        return self.redundant
    
    def is_redundant_enhanced_batch(self, sentences: List[str], embeddings: List[np.ndarray], 
                                   contexts_before: List[str] = None, contexts_after: List[str] = None,
                                   desc: str = "增强批量冗余过滤") -> List[bool]:
        """
        增强批量冗余检测，支持动态阈值和上下文窗口
        注意：为了保持与逐个处理的一致性，需要按顺序处理每个句子
        
        Args:
            sentences: 句子列表
            embeddings: 对应的嵌入向量列表
            contexts_before: 前文上下文列表
            contexts_after: 后文上下文列表
            desc: 进度条描述
            
        Returns:
            布尔列表，表示每个句子是否冗余
        """
        if not sentences:
            return []
            
        if len(sentences) != len(embeddings):
            raise ValueError("sentences和embeddings长度必须相同")
        
        # 处理上下文参数
        if contexts_before is None:
            contexts_before = [""] * len(sentences)
        if contexts_after is None:
            contexts_after = [""] * len(sentences)
            
        if len(contexts_before) != len(sentences) or len(contexts_after) != len(sentences):
            raise ValueError("上下文列表长度必须与句子列表相同")
        
        results = []
        batch_start_time = time.time()
        
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
            logger.info(f"开始增强批量冗余过滤，总计 {len(sentences)} 个句子")
        
        # 设置总句子数，用于显示剩余数量
        self._total_sentences = len(sentences)
        
        try:
            # 逐个处理每个句子，保持与单句处理的一致性
            for i, (sentence, embedding, ctx_before, ctx_after) in enumerate(
                zip(sentences, embeddings, contexts_before, contexts_after)):
                
                if not self.embeddings:
                    # 缓冲区为空，直接添加
                    self._add_sentence(sentence, embedding, ctx_before, ctx_after)
                    results.append(False)
                else:
                    # 计算与已存储句子的相似度
                    stored_embeddings_array = np.array(self.embeddings)
                    embedding_2d = embedding.reshape(1, -1)
                    
                    # 计算语义相似度
                    semantic_sims = cosine_similarity(embedding_2d, stored_embeddings_array)[0]
                    
                    # 计算上下文相似度
                    context_sims = self._calculate_context_similarity(ctx_before, ctx_after)
                    
                    # 动态阈值调整
                    adjusted_threshold = self._adjust_threshold(sentence, semantic_sims)
                    
                    # 综合判断
                    max_sim_idx = np.argmax(semantic_sims)
                    max_semantic_sim = semantic_sims[max_sim_idx]
                    context_sim = context_sims[max_sim_idx] if context_sims else 0
                    
                    # 综合相似度：语义相似度为主，上下文差异可以降低冗余判断
                    combined_sim = (self.semantic_weight * max_semantic_sim + 
                                   self.context_weight * context_sim)
                    
                    # 记录相似度历史
                    self.similarity_history.append(max_semantic_sim)
                    if len(self.similarity_history) > 100:  # 保持历史记录在合理范围内
                        self.similarity_history.pop(0)
                    
                    if combined_sim >= adjusted_threshold:
                        self._record_redundancy(sentence, max_sim_idx, combined_sim, 
                                              semantic_sim=max_semantic_sim, context_sim=context_sim)
                        results.append(True)
                    else:
                        self._add_sentence(sentence, embedding, ctx_before, ctx_after)
                        results.append(False)
                
                # 更新进度条
                if progress_bar:
                    progress_bar.update(1)
                    if i % 10 == 0:  # 每10个句子更新一次后缀
                        redundant_count = sum(results)
                        progress_bar.set_postfix({
                            '冗余': f'{redundant_count}/{len(results)}',
                            '冗余率': f'{redundant_count/len(results):.1%}' if results else '0.0%'
                        })
                        
        finally:
            # 关闭进度条
            if progress_bar:
                progress_bar.close()
        
        # 记录批处理时间
        batch_time = time.time() - batch_start_time
        redundant_count = sum(results)
        
        if self.enable_logging:
            speed = len(sentences) / batch_time if batch_time > 0 else 0
            processed_count = len(sentences) - redundant_count
            logger.info(f"增强批量冗余过滤完成！总计 {len(sentences)} 个句子，"
                       f"处理 {processed_count} 个句子，检测到 {redundant_count} 个冗余句子 "
                       f"(冗余率: {redundant_count/len(sentences):.1%})，"
                       f"处理速度: {speed:.1f} 句/秒，耗时: {batch_time:.2f}秒")
        
        # 清除总句子数标记
        if hasattr(self, '_total_sentences'):
            delattr(self, '_total_sentences')
        
        return results
    
    def process_batch_optimized(self, sentences: List[str], embeddings: List[np.ndarray], 
                               contexts_before: List[str] = None, contexts_after: List[str] = None) -> Tuple[List[bool], Dict[str, Any]]:
        """
        优化的批量处理，支持分块处理大批量数据
        
        Args:
            sentences: 句子列表
            embeddings: 对应的嵌入向量列表
            contexts_before: 前文上下文列表
            contexts_after: 后文上下文列表
            
        Returns:
            (冗余结果列表, 处理统计信息)
        """
        if not sentences:
            return [], {"processed": 0, "redundant": 0, "chunks": 0}
        
        import time
        start_time = time.time()
        
        all_results = []
        total_redundant = 0
        chunks_processed = 0
        
        # 动态调整批量大小
        adaptive_batch_size = self._get_adaptive_batch_size(len(sentences))
        
        # 分块处理以避免内存溢出
        for i in range(0, len(sentences), adaptive_batch_size):
            chunk_start_time = time.time()
            
            chunk_sentences = sentences[i:i+adaptive_batch_size]
            chunk_embeddings = embeddings[i:i+adaptive_batch_size]
            
            chunk_ctx_before = None
            chunk_ctx_after = None
            if contexts_before:
                chunk_ctx_before = contexts_before[i:i+adaptive_batch_size]
            if contexts_after:
                chunk_ctx_after = contexts_after[i:i+adaptive_batch_size]
            
            # 使用向量化处理当前块
            chunk_results = self._process_chunk_vectorized(
                chunk_sentences, chunk_embeddings, chunk_ctx_before, chunk_ctx_after
            )
            
            all_results.extend(chunk_results)
            total_redundant += sum(chunk_results)
            chunks_processed += 1
            
            # 记录处理时间用于性能调优
            chunk_time = time.time() - chunk_start_time
            if not hasattr(self, 'chunk_processing_times'):
                self.chunk_processing_times = []
            self.chunk_processing_times.append(chunk_time)
            if len(self.chunk_processing_times) > 20:
                self.chunk_processing_times.pop(0)
        
        total_time = time.time() - start_time
        
        stats = {
            "processed": len(sentences),
            "redundant": total_redundant,
            "chunks": chunks_processed,
            "redundancy_rate": total_redundant / len(sentences) if sentences else 0.0,
            "total_processing_time": total_time,
            "avg_chunk_time": total_time / chunks_processed if chunks_processed > 0 else 0.0,
            "sentences_per_second": len(sentences) / total_time if total_time > 0 else 0.0,
            "adaptive_batch_size": adaptive_batch_size
        }
        
        return all_results, stats
    
    def _get_adaptive_batch_size(self, total_sentences: int) -> int:
        """根据数据量和历史性能动态调整批量大小"""
        # 基础批量大小
        base_size = self.batch_size
        
        # 根据总数据量调整
        if total_sentences > 10000:
            base_size = min(500, base_size * 2)
        elif total_sentences < 100:
            base_size = max(10, base_size // 2)
        
        # 根据历史处理时间调整
        if hasattr(self, 'chunk_processing_times') and len(self.chunk_processing_times) > 3:
            avg_time = sum(self.chunk_processing_times[-3:]) / 3
            if avg_time > 0.5:  # 处理太慢，减小批量
                base_size = max(10, int(base_size * 0.7))
            elif avg_time < 0.1:  # 处理很快，增大批量
                base_size = min(1000, int(base_size * 1.3))
        
        return base_size
    
    def _process_chunk_vectorized(self, sentences: List[str], embeddings: List[np.ndarray],
                                 contexts_before: List[str] = None, contexts_after: List[str] = None) -> List[bool]:
        """向量化处理数据块"""
        if not sentences:
            return []
        
        results = []
        
        # 如果缓冲区为空，直接添加第一个句子
        if not self.embeddings:
            self._add_sentence(sentences[0], embeddings[0], 
                             contexts_before[0] if contexts_before else "",
                             contexts_after[0] if contexts_after else "")
            results.append(False)
            sentences = sentences[1:]
            embeddings = embeddings[1:]
            if contexts_before:
                contexts_before = contexts_before[1:]
            if contexts_after:
                contexts_after = contexts_after[1:]
        
        if not sentences:
            return results
        
        # 转换为numpy数组进行向量化计算
        try:
            new_embeddings_array = np.vstack(embeddings)
            stored_embeddings_array = np.vstack(self.embeddings)
            
            # 批量计算相似度矩阵
            similarity_matrix = cosine_similarity(new_embeddings_array, stored_embeddings_array)
        except Exception as e:
            # 如果向量化失败，回退到逐个处理
            return self._process_chunk_fallback(sentences, embeddings, contexts_before, contexts_after)
        
        # 处理每个句子
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            ctx_before = contexts_before[i] if contexts_before else ""
            ctx_after = contexts_after[i] if contexts_after else ""
            
            # 获取当前句子与所有已存储句子的相似度
            similarities = similarity_matrix[i]
            
            # 计算上下文相似度
            context_sims = self._calculate_context_similarity(ctx_before, ctx_after)
            
            # 动态阈值调整
            adjusted_threshold = self._adjust_threshold(sentence, similarities)
            
            # 综合判断
            max_sim_idx = np.argmax(similarities)
            max_semantic_sim = similarities[max_sim_idx]
            context_sim = context_sims[max_sim_idx] if context_sims else 0
            
            # 综合相似度
            combined_sim = (self.semantic_weight * max_semantic_sim + 
                           self.context_weight * context_sim)
            
            # 记录相似度历史
            self.similarity_history.append(max_semantic_sim)
            if len(self.similarity_history) > 100:
                self.similarity_history.pop(0)
            
            if combined_sim >= adjusted_threshold:
                self._record_redundancy(sentence, max_sim_idx, combined_sim, 
                                      semantic_sim=max_semantic_sim, context_sim=context_sim)
                results.append(True)
            else:
                self._add_sentence(sentence, embedding, ctx_before, ctx_after)
                results.append(False)
                
                # 更新存储的嵌入数组以供后续计算使用
                if i < len(sentences) - 1:
                    try:
                        stored_embeddings_array = np.vstack([stored_embeddings_array, embedding])
                        # 重新计算剩余句子的相似度
                        remaining_embeddings = new_embeddings_array[i+1:]
                        if len(remaining_embeddings) > 0:
                            new_similarities = cosine_similarity(remaining_embeddings, stored_embeddings_array)
                            # 更新相似度矩阵
                            similarity_matrix[i+1:] = new_similarities
                    except Exception:
                        # 如果更新失败，继续处理但不更新矩阵
                        pass
        
        return results
    
    def _process_chunk_fallback(self, sentences: List[str], embeddings: List[np.ndarray],
                               contexts_before: List[str] = None, contexts_after: List[str] = None,
                               progress_bar=None) -> List[bool]:
        """回退处理方法，逐个处理句子"""
        results = []
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            ctx_before = contexts_before[i] if contexts_before else ""
            ctx_after = contexts_after[i] if contexts_after else ""
            
            # 使用单句处理方法
            is_redundant = self.is_redundant_enhanced(sentence, embedding, ctx_before, ctx_after)
            results.append(is_redundant)
            
            # 更新进度条
            if progress_bar:
                progress_bar.update(1)
                if i % 10 == 0:  # 每10个句子更新一次后缀
                    redundant_count = sum(results)
                    progress_bar.set_postfix({
                        '冗余': f'{redundant_count}/{len(results)}',
                        '冗余率': f'{redundant_count/len(results):.1%}' if results else '0.0%'
                    })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取详细统计信息"""
        total_processed = len(self.sentences) + len(self.redundant)
        return {
            "total_sentences": len(self.sentences),
            "redundant_count": len(self.redundant),
            "total_processed": total_processed,
            "redundancy_rate": len(self.redundant) / max(1, total_processed),
            "avg_similarity": np.mean(self.similarity_history) if self.similarity_history else 0.0,
            "max_similarity": np.max(self.similarity_history) if self.similarity_history else 0.0,
            "min_similarity": np.min(self.similarity_history) if self.similarity_history else 0.0,
            "current_threshold": self.base_threshold,
            "batch_size": self.batch_size,
            "buffer_size": len(self.embeddings),
            "context_window_size": getattr(self, 'context_window', 0),
            "dynamic_threshold_enabled": getattr(self, 'enable_dynamic_threshold', False)
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        memory_usage = 0
        if self.embeddings:
            try:
                # 尝试计算内存使用量
                if hasattr(self.embeddings[0], 'nbytes'):
                    memory_usage = len(self.embeddings) * self.embeddings[0].nbytes
                else:
                    # 估算内存使用量
                    memory_usage = len(self.embeddings) * len(self.embeddings[0]) * 8  # 假设float64
            except (AttributeError, IndexError):
                memory_usage = 0
        
        return {
            "buffer_memory_usage": memory_usage,
            "similarity_computation_count": len(self.similarity_history),
            "avg_buffer_size_during_processing": np.mean([i for i in range(1, len(self.sentences) + 1)]) if self.sentences else 0,
            "redundancy_detection_efficiency": len(self.redundant) / max(1, len(self.similarity_history))
        }
    
    def optimize_buffer(self, max_buffer_size: int = 1000):
        """优化缓冲区大小以提高性能"""
        if len(self.embeddings) > max_buffer_size:
            # 保留最近的句子和嵌入
            keep_count = max_buffer_size // 2
            self.sentences = self.sentences[-keep_count:]
            self.embeddings = self.embeddings[-keep_count:]
            if hasattr(self, 'contexts'):
                self.contexts = self.contexts[-keep_count:]
            print(f"[RedundancyBuffer] 缓冲区优化: 保留最近 {keep_count} 个句子")
    
    def clear_statistics(self):
        """清空统计信息"""
        self.similarity_history = []
        self.redundant = []
        print("[RedundancyBuffer] 统计信息已清空")
    
    def clear(self):
        """清空缓冲区（兼容BaseRedundancyFilter接口）"""
        self.embeddings = []
        self.sentences = []
        self.redundant = []
        self.similarity_history = []
        self.processed_count = 0
        self.redundant_count = 0
        self.start_time = None
        if hasattr(self, 'contexts'):
            self.contexts = []
        if self.enable_logging:
            logger.info("RedundancyBuffer缓冲区已清空")
    
    # 兼容BaseRedundancyFilter接口的方法
    def is_duplicate(self, text: str) -> bool:
        """检查文本是否为重复（兼容BaseRedundancyFilter接口）
        
        注意：此方法需要embedding，这里抛出异常提示使用正确的方法
        
        Args:
            text: 待检查的文本
            
        Returns:
            True if duplicate, False otherwise
        """
        raise NotImplementedError(
            "RedundancyBuffer需要embedding向量，请使用is_redundant(sentence, embedding)方法"
        )
    
    def add_text(self, text: str) -> None:
        """添加文本到缓冲区（兼容BaseRedundancyFilter接口）
        
        注意：此方法需要embedding，这里抛出异常提示使用正确的方法
        
        Args:
            text: 要添加的文本
        """
        raise NotImplementedError(
            "RedundancyBuffer需要embedding向量，请直接调用is_redundant(sentence, embedding)方法"
        )
    
    def get_memory_usage(self) -> float:
        """获取内存使用量 (MB)
        
        Returns:
            内存使用量
        """
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # 转换为MB
        except Exception:
            # 如果无法获取内存信息，返回估算值
            memory_usage = 0
            if self.embeddings:
                try:
                    # 尝试计算内存使用量
                    if hasattr(self.embeddings[0], 'nbytes'):
                        memory_usage = len(self.embeddings) * self.embeddings[0].nbytes
                    else:
                        # 估算内存使用量
                        memory_usage = len(self.embeddings) * len(self.embeddings[0]) * 8  # 假设float64
                except (AttributeError, IndexError):
                    memory_usage = len(self.embeddings) * 768 * 8  # 假设768维向量
            return memory_usage / 1024 / 1024  # 转换为MB
