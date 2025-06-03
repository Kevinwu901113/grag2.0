import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional

class RedundancyBuffer:
    def __init__(self, threshold=0.95):
        self.embeddings = []  # 已接收句子的嵌入列表
        self.sentences = []   # 已接收的句子
        self.threshold = threshold
        self.redundant = []   # 存冗余记录
        
    def is_redundant(self, sentence: str, embedding: np.ndarray) -> bool:
        if not self.embeddings:
            self.embeddings.append(embedding)
            self.sentences.append(sentence)
            return False

        sims = cosine_similarity([embedding], self.embeddings)[0]
        max_sim = np.max(sims)
        if max_sim >= self.threshold:
            # 冗余，记录来源
            idx = int(np.argmax(sims))
            self.redundant.append({
                "duplicate": sentence,
                "matched_to": self.sentences[idx],
                "score": float(max_sim)
            })
            return True
        else:
            self.embeddings.append(embedding)
            self.sentences.append(sentence)
            return False

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
        
        # 存储数据
        self.embeddings = []  # 句子嵌入
        self.sentences = []   # 句子文本
        self.contexts = []    # 上下文信息
        self.redundant = []   # 冗余记录
        
        # 动态阈值相关
        self.similarity_history = []  # 历史相似度记录

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
            return True
        else:
            self._add_sentence(sentence, embedding, context_before, context_after)
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_sentences": len(self.sentences),
            "redundant_count": len(self.redundant),
            "redundancy_rate": len(self.redundant) / max(1, len(self.sentences) + len(self.redundant)),
            "avg_similarity": np.mean(self.similarity_history) if self.similarity_history else 0.0,
            "current_threshold": self.base_threshold
        }
