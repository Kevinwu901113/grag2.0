import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
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

    def _get_embedding(self, text: str) -> np.ndarray:
        # 统一使用LLMClient的embed接口
        embedding = self.llm_client.embed(text)[0]
        # 确保返回numpy数组
        return np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding

    def add_sentence(self, sentence: str, meta: Dict = None):
        sent_emb = self._get_embedding(sentence)

        # ✅ 冗余检测（可选跳过）
        if self.redundancy_filter and self.redundancy_filter.is_redundant(sentence, sent_emb):
            return  # 不再进入任何主题

        if not self.topics:
            self._create_topic(sentence, sent_emb, meta)
            return

        # 过滤掉已经达到大小限制的主题
        valid_topics = []
        for i, topic in enumerate(self.topics):
            if (len(topic["sentences"]) < self.max_sentences_per_topic and 
                sum(len(s) for s in topic["sentences"]) < self.max_chars_per_topic):
                valid_topics.append((i, topic))
        
        if not valid_topics:
            # 所有主题都已满，创建新主题
            self._create_topic(sentence, sent_emb, meta)
            return

        # 计算与有效主题的相似度
        similarities = []
        valid_indices = []
        for idx, topic in valid_topics:
            sim = cosine_similarity([sent_emb], [topic["center"]])[0][0]
            similarities.append(sim)
            valid_indices.append(idx)
        
        max_idx_in_valid = int(np.argmax(similarities))
        max_sim = similarities[max_idx_in_valid]
        actual_topic_idx = valid_indices[max_idx_in_valid]

        if max_sim >= self.similarity_threshold:
            # 检查添加后是否会超过限制
            topic = self.topics[actual_topic_idx]
            if (len(topic["sentences"]) + 1 <= self.max_sentences_per_topic and 
                sum(len(s) for s in topic["sentences"]) + len(sentence) <= self.max_chars_per_topic):
                self._add_to_topic(actual_topic_idx, sentence, sent_emb, meta)
            else:
                # 会超过限制，创建新主题
                self._create_topic(sentence, sent_emb, meta)
        else:
            self._create_topic(sentence, sent_emb, meta)

    def _create_topic(self, sentence: str, embedding: np.ndarray, meta: Dict):
        topic = {
            "id": f"topic_{self.topic_id_counter}",
            "sentences": [sentence],
            "meta": [meta] if meta else [],
            "center": embedding
        }
        self.topics.append(topic)
        self.topic_id_counter += 1

    def _add_to_topic(self, idx: int, sentence: str, embedding: np.ndarray, meta: Dict):
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
        
        # 检查是否需要分裂主题
        self._check_and_split_topic(idx)

    def _check_and_split_topic(self, idx: int):
        """检查主题是否需要分裂，使用智能分裂策略而非简单均匀分裂"""
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
                    return
            
            # 如果智能分裂失败或不适用，使用传统的中点分裂
            self._simple_split_topic(topic, sentences, metas, embeddings)
    
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
    
    def _simple_split_topic(self, topic: Dict, sentences: List[str], metas: List[Dict], embeddings: List[np.ndarray]):
        """
        传统的简单中点分裂策略（作为备选方案）
        
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
