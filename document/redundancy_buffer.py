import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
