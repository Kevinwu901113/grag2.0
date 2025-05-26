import numpy as np
from typing import List, Dict
from document.topic_summary_generator import generate_topic_summary  # 我们新建此工具文件
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TopicPoolManager:
    def __init__(self, model_name="all-MiniLM-L6-v2", similarity_threshold=0.80, redundancy_filter=None):
        self.topics: List[Dict] = []
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer(model_name)
        self.topic_id_counter = 0
        self.redundancy_filter = redundancy_filter  # ✅ 新增

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def add_sentence(self, sentence: str, meta: Dict = None):
        sent_emb = self._get_embedding(sentence)

        # ✅ 冗余检测（可选跳过）
        if self.redundancy_filter and self.redundancy_filter.is_redundant(sentence, sent_emb):
            return  # 不再进入任何主题

        if not self.topics:
            self._create_topic(sentence, sent_emb, meta)
            return

        similarities = [cosine_similarity([sent_emb], [topic["center"]])[0][0] for topic in self.topics]
        max_idx = int(np.argmax(similarities))
        max_sim = similarities[max_idx]

        if max_sim >= self.similarity_threshold:
            self._add_to_topic(max_idx, sentence, sent_emb, meta)
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
        # 更新embedding中心（简单均值）
        topic["center"] = np.mean([topic["center"], embedding], axis=0)

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
                "summary": summary
            })
        return result