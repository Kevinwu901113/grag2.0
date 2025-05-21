import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.model_cache import model_cache
from utils.logger import setup_logger

logger = setup_logger(os.getcwd())

class ThemeMatcher:
    """
    主题匹配器，用于匹配查询与主题摘要
    优化版本：从配置中读取模型名称，使用模型缓存
    """
    def __init__(self, theme_summaries: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None):
        # 从配置中读取模型名称，如果没有则使用默认值
        self.config = config or {}
        model_name = self.config.get("matcher", {}).get("model_name", "all-MiniLM-L6-v2")
        
        # 使用模型缓存获取模型
        self.model = model_cache.get_sentence_transformer(model_name)
        if self.model is None:
            logger.warning(f"无法加载模型 {model_name}，尝试使用默认模型")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            
        self.summaries = theme_summaries
        logger.info(f"正在编码 {len(theme_summaries)} 个主题摘要...")
        
        # 使用进度条显示编码进度
        from utils.logger import get_progress_bar
        summaries_text = [t.get("summary", "") for t in theme_summaries]
        
        # 批量编码以提高效率
        batch_size = 32
        all_embeddings = []
        
        with get_progress_bar(total=len(summaries_text), desc="编码主题摘要") as pbar:
            for i in range(0, len(summaries_text), batch_size):
                batch = summaries_text[i:i+batch_size]
                batch_embeddings = self.model.encode(batch)
                all_embeddings.extend(batch_embeddings)
                pbar.update(len(batch))
        
        self.embeddings = all_embeddings
        logger.info(f"主题匹配器初始化完成，共 {len(self.embeddings)} 个主题摘要")

    def match(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        匹配查询与主题摘要
        
        Args:
            query: 查询文本
            top_k: 返回的最大匹配数量
            
        Returns:
            匹配结果列表，每个结果包含摘要、ID和得分
        """
        query_emb = self.model.encode(query)
        sims = cosine_similarity([query_emb], self.embeddings)[0]
        sorted_idx = sims.argsort()[::-1][:top_k]
        
        results = []
        for idx in sorted_idx:
            results.append({
                "summary": self.summaries[idx].get("summary", ""),
                "id": self.summaries[idx].get("id", ""),
                "score": float(sims[idx])
            })
            
        logger.debug(f"查询 '{query}' 匹配到 {len(results)} 个主题，最高分: {results[0]['score'] if results else 0}")
        return results