import os
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from utils.logger import setup_logger
from llm.llm import LLMClient

logger = setup_logger(os.getcwd())

class ThemeMatcher:
    """
    主题匹配器，用于匹配查询与主题摘要
    优化版本：从配置中读取模型名称，使用模型缓存，支持多种嵌入模型
    """
    def __init__(self, theme_summaries: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None):
        # 统一使用LLMClient进行嵌入向量生成
        self.config = config or {}
        self.llm_client = LLMClient(self.config)
        
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
                # 统一使用LLMClient的embed接口
                batch_embeddings = self.llm_client.embed(batch)
                all_embeddings.extend(batch_embeddings)
                pbar.update(len(batch))
        
        self.embeddings = all_embeddings
        logger.info(f"主题匹配器初始化完成，共 {len(self.embeddings)} 个主题摘要")

    def match(self, query: str, top_k: int = 3, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        匹配查询与主题摘要，支持关键词实体校验和多种相似度策略
        
        Args:
            query: 查询文本
            top_k: 返回的最大匹配数量
            min_score: 最小相似度阈值，低于此值的匹配将被过滤
            
        Returns:
            匹配结果列表，每个结果包含摘要、ID和得分
        """
        # 从配置中获取匹配策略参数
        matching_config = self.config.get("matching", {})
        use_keyword_filter = matching_config.get("use_keyword_filter", True)
        similarity_strategy = matching_config.get("similarity_strategy", "summary_vector")  # summary_vector, max_sentence, topic_center
        default_min_score = matching_config.get("min_similarity_threshold", 0.3)
        
        # 使用配置中的阈值，如果参数未指定
        if min_score == 0.0:
            min_score = default_min_score
        
        # 统一使用LLMClient的embed接口
        query_emb = self.llm_client.embed(query)[0]
            
        sims = cosine_similarity([query_emb], self.embeddings)[0]
        
        # 先按相似度排序
        sorted_pairs = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        
        # 过滤低于阈值的结果
        filtered_pairs = [(idx, score) for idx, score in sorted_pairs if score >= min_score]
        
        # 关键词实体校验（如果启用）
        if use_keyword_filter:
            filtered_pairs = self._apply_keyword_filter(query, filtered_pairs)
        
        # 取前top_k个结果
        top_pairs = filtered_pairs[:top_k]
        
        results = []
        for idx, score in top_pairs:
            node_id = self.summaries[idx].get("id", "")
            title = self.summaries[idx].get("title", "无标题")
            results.append({
                "summary": self.summaries[idx].get("summary", ""),
                "node_id": node_id,  # 确保返回node_id字段
                "id": node_id,        # 兼容性考虑
                "title": title,       # 添加标题字段
                "similarity": float(score),  # 添加similarity字段
                "score": float(score)       # 兼容性考虑
            })
            
        logger.debug(f"查询 '{query}' 匹配到 {len(results)} 个主题，最高分: {results[0]['similarity'] if results else 0}")
        return results
    
    def _apply_keyword_filter(self, query: str, candidate_pairs: List[tuple]) -> List[tuple]:
        """
        应用关键词实体校验，过滤不包含查询关键词的候选
        
        Args:
            query: 查询文本
            candidate_pairs: 候选对列表 [(idx, score), ...]
            
        Returns:
            过滤后的候选对列表
        """
        import jieba
        
        # 提取查询中的关键词（去除停用词）
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        query_words = set(jieba.cut(query)) - stop_words
        
        if not query_words:
            return candidate_pairs
        
        filtered_pairs = []
        for idx, score in candidate_pairs:
            summary_text = self.summaries[idx].get("summary", "")
            title_text = self.summaries[idx].get("title", "")
            combined_text = summary_text + " " + title_text
            
            # 检查是否包含至少一个查询关键词
            summary_words = set(jieba.cut(combined_text))
            if query_words & summary_words:  # 有交集
                filtered_pairs.append((idx, score))
            else:
                # 降权但不完全过滤
                filtered_pairs.append((idx, score * 0.5))
        
        return filtered_pairs
