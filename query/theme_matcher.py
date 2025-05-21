from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ThemeMatcher:
    def __init__(self, theme_summaries: list[dict], model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.summaries = theme_summaries
        self.embeddings = self.model.encode([t["summary"] for t in theme_summaries])

    def match(self, query: str, top_k=3) -> list[dict]:
        query_emb = self.model.encode(query)
        sims = cosine_similarity([query_emb], self.embeddings)[0]
        sorted_idx = sims.argsort()[::-1][:top_k]
        results = []
        for idx in sorted_idx:
            results.append({
                "summary": self.summaries[idx]["summary"],
                "node_id": self.summaries[idx]["id"],
                "score": float(sims[idx])
            })
        return results
