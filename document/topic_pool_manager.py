import numpy as np
from typing import List, Dict
from document.topic_summary_generator import generate_topic_summary  # 我们新建此工具文件
from sklearn.metrics.pairwise import cosine_similarity
from utils.model_cache import model_cache
from llm.llm import LLMClient

# 条件导入SentenceTransformer，避免在只使用Ollama API时出错
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

class TopicPoolManager:
    def __init__(self, model_name="all-MiniLM-L6-v2", similarity_threshold=0.65, redundancy_filter=None, config=None):
        self.topics: List[Dict] = []
        self.similarity_threshold = similarity_threshold
        self.config = config or {}
        # 添加主题大小限制
        self.max_sentences_per_topic = 100  # 每个主题最大句子数
        self.max_chars_per_topic = 10000    # 每个主题最大字符数
        
        # 检查是否是BGE-M3模型
        self.use_ollama = False
        if model_name == "bge-m3" or model_name == "BAAI/bge-m3":
            # 如果是BGE-M3模型，使用Ollama API
            print(f"使用Ollama API调用BGE-M3模型进行嵌入")
            self.llm_client = LLMClient(self.config)
            self.use_ollama = True
            self.model = None
        else:
            # 只有在不使用Ollama API时才需要SentenceTransformer
            if not SENTENCE_TRANSFORMER_AVAILABLE:
                raise ImportError("SentenceTransformer库未安装，无法使用本地模型。请安装sentence-transformers或使用Ollama API。")
                
            # 使用模型缓存获取模型
            self.model = model_cache.get_sentence_transformer(model_name)
            if self.model is None:
                # 如果缓存中没有，则直接创建
                self.model = SentenceTransformer(model_name)
            
        self.topic_id_counter = 0
        self.redundancy_filter = redundancy_filter  # ✅ 新增

    def _get_embedding(self, text: str) -> np.ndarray:
        if self.use_ollama:
            # 使用Ollama API进行嵌入
            embedding = self.llm_client.embed([text])[0]
            # 确保返回numpy数组
            return np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding
        else:
            # 使用本地SentenceTransformer模型进行嵌入
            return self.model.encode(text)

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
        """检查主题是否需要分裂，如果句子数超过阈值则分裂"""
        topic = self.topics[idx]
        
        # 如果句子数超过80%的限制，考虑分裂
        if len(topic["sentences"]) > self.max_sentences_per_topic * 0.8:
            sentences = topic["sentences"]
            metas = topic["meta"]
            
            # 将句子分成两组
            mid_point = len(sentences) // 2
            
            # 更新原主题
            topic["sentences"] = sentences[:mid_point]
            topic["meta"] = metas[:mid_point]
            
            # 重新计算原主题的中心
            if topic["sentences"]:
                embeddings = [self._get_embedding(s) for s in topic["sentences"]]
                topic["center"] = np.mean(embeddings, axis=0)
            
            # 创建新主题
            if len(sentences) > mid_point:
                new_sentences = sentences[mid_point:]
                new_metas = metas[mid_point:]
                new_embeddings = [self._get_embedding(s) for s in new_sentences]
                new_center = np.mean(new_embeddings, axis=0)
                
                new_topic = {
                    "id": f"topic_{self.topic_id_counter}",
                    "sentences": new_sentences,
                    "meta": new_metas,
                    "center": new_center
                }
                self.topics.append(new_topic)
                self.topic_id_counter += 1

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
