import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
# import matplotlib.pyplot as plt  # 注释掉非必需的依赖
from llm.llm import LLMClient
from document.sentence_splitter import split_into_sentences

class AdvancedClusteringProcessor:
    """
    高级聚类处理器：实现句子级分解和多种聚类算法
    支持谱聚类、层次聚类、K-means、HDBSCAN等算法进行全局聚类
    增强功能：主题内聚度评估和智能分裂策略
    
    HDBSCAN算法特点：
    - 基于密度的聚类，能够发现任意形状的簇
    - 自动确定聚类数量，无需预先指定
    - 能够识别噪声点和离群值
    - 对参数相对不敏感，聚类质量更稳定
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.llm_client = LLMClient(config)
        
        # 聚类参数配置
        self.clustering_config = config.get("advanced_clustering", {})
        self.similarity_threshold = self.clustering_config.get("similarity_threshold", 0.7)
        self.min_cluster_size = self.clustering_config.get("min_cluster_size", 3)
        self.max_cluster_size = self.clustering_config.get("max_cluster_size", 50)
        self.clustering_method = self.clustering_config.get("method", "hierarchical")  # spectral, hierarchical, kmeans, hdbscan
        
        # HDBSCAN特定参数
        self.hdbscan_min_cluster_size = self.clustering_config.get("hdbscan_min_cluster_size", 5)
        self.hdbscan_min_samples = self.clustering_config.get("hdbscan_min_samples", None)  # 默认为None，使用min_cluster_size
        self.hdbscan_cluster_selection_epsilon = self.clustering_config.get("hdbscan_cluster_selection_epsilon", 0.0)
        self.hdbscan_metric = self.clustering_config.get("hdbscan_metric", "euclidean")  # euclidean, cosine等
        
        # 句子级处理参数
        self.sentence_level = self.clustering_config.get("sentence_level", True)
        self.min_sentence_length = self.clustering_config.get("min_sentence_length", 10)
        
        # 新增：主题内聚度评估参数
        self.coherence_threshold = self.clustering_config.get("coherence_threshold", 0.6)
        self.use_topic_summary_enhancement = self.clustering_config.get("use_topic_summary_enhancement", True)
        
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        处理文档列表，返回聚类后的主题块
        
        Args:
            documents: 文档列表，每个文档包含text和meta信息
            
        Returns:
            聚类后的主题块列表
        """
        # 第一步：句子级分解
        sentences_data = self._extract_sentences(documents)
        
        if len(sentences_data) < 2:
            return self._create_single_topic(sentences_data)
            
        # 第二步：生成嵌入向量
        embeddings = self._generate_embeddings(sentences_data)
        
        # 第三步：执行聚类
        cluster_labels = self._perform_clustering(embeddings, len(sentences_data))
        
        # 第四步：构建主题块
        topics = self._build_topics(sentences_data, cluster_labels, embeddings)
        
        # 第五步：评估和优化主题内聚度
        topics = self._evaluate_and_optimize_coherence(topics, sentences_data, embeddings)
        
        return topics
    
    def _extract_sentences(self, documents: List[Dict]) -> List[Dict]:
        """
        从文档中提取句子，支持句子级分解
        
        Args:
            documents: 文档列表
            
        Returns:
            句子数据列表，包含句子文本和元数据
        """
        sentences_data = []
        sentence_id = 0
        
        for doc in documents:
            text = doc.get("text", "")
            meta = doc.get("meta", {})
            
            if self.sentence_level:
                # 句子级分解
                sentences = split_into_sentences(text)
                for sentence in sentences:
                    if len(sentence.strip()) >= self.min_sentence_length:
                        sentences_data.append({
                            "id": f"sent_{sentence_id}",
                            "text": sentence.strip(),
                            "meta": {**meta, "original_doc": doc.get("id", "")}
                        })
                        sentence_id += 1
            else:
                # 保持原始文档块
                sentences_data.append({
                    "id": f"doc_{sentence_id}",
                    "text": text,
                    "meta": meta
                })
                sentence_id += 1
                
        return sentences_data
    
    def _generate_embeddings(self, sentences_data: List[Dict]) -> np.ndarray:
        """
        为句子生成嵌入向量
        
        Args:
            sentences_data: 句子数据列表
            
        Returns:
            嵌入向量矩阵
        """
        texts = [item["text"] for item in sentences_data]
        embeddings = []
        
        # 批量生成嵌入向量以提高效率
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.llm_client.embed(batch_texts)
            embeddings.extend(batch_embeddings)
            
        return np.array(embeddings)
    
    def _perform_clustering(self, embeddings: np.ndarray, n_samples: int) -> np.ndarray:
        """
        执行聚类算法
        
        Args:
            embeddings: 嵌入向量矩阵
            n_samples: 样本数量
            
        Returns:
            聚类标签数组
        """
        if self.clustering_method == "spectral":
            return self._spectral_clustering(embeddings, n_samples)
        elif self.clustering_method == "hierarchical":
            return self._hierarchical_clustering(embeddings)
        elif self.clustering_method == "kmeans":
            return self._kmeans_clustering(embeddings, n_samples)
        elif self.clustering_method == "hdbscan":
            return self._hdbscan_clustering(embeddings)
        else:
            raise ValueError(f"不支持的聚类方法: {self.clustering_method}")
    
    def _spectral_clustering(self, embeddings: np.ndarray, n_samples: int) -> np.ndarray:
        """
        谱聚类实现
        
        Args:
            embeddings: 嵌入向量矩阵
            n_samples: 样本数量
            
        Returns:
            聚类标签数组
        """
        # 动态确定聚类数量
        n_clusters = self._estimate_cluster_number(n_samples)
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(embeddings)
        
        # 应用相似度阈值，构建邻接矩阵
        adjacency_matrix = (similarity_matrix > self.similarity_threshold).astype(float)
        
        # 执行谱聚类
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        
        try:
            cluster_labels = spectral.fit_predict(adjacency_matrix)
        except ValueError:
            # 如果谱聚类失败，回退到层次聚类
            return self._hierarchical_clustering(embeddings)
            
        return cluster_labels
    
    def _hierarchical_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        层次聚类实现
        
        Args:
            embeddings: 嵌入向量矩阵
            
        Returns:
            聚类标签数组
        """
        # 计算距离矩阵（使用余弦距离）
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
        
        # 执行层次聚类
        linkage_matrix = linkage(squareform(distance_matrix), method='ward')
        
        # 动态确定聚类数量或使用距离阈值
        if self.clustering_config.get("use_distance_threshold", True):
            # 使用距离阈值
            distance_threshold = 1 - self.similarity_threshold
            cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance') - 1
        else:
            # 使用固定聚类数量
            n_clusters = self._estimate_cluster_number(len(embeddings))
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
            
        return cluster_labels
    
    def _kmeans_clustering(self, embeddings: np.ndarray, n_samples: int) -> np.ndarray:
        """
        K-means聚类实现
        
        Args:
            embeddings: 嵌入向量矩阵
            n_samples: 样本数量
            
        Returns:
            聚类标签数组
        """
        # 标准化嵌入向量
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(embeddings)
        
        # 动态确定聚类数量
        n_clusters = self._estimate_cluster_number(n_samples)
        
        # 执行K-means聚类
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        cluster_labels = kmeans.fit_predict(normalized_embeddings)
        
        return cluster_labels
    
    def _hdbscan_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        HDBSCAN聚类实现
        
        HDBSCAN是一种基于密度的聚类算法，具有以下优势：
        1. 自动确定聚类数量，无需预先指定
        2. 能够发现任意形状的簇
        3. 能够识别噪声点和离群值
        4. 对参数相对不敏感
        
        Args:
            embeddings: 嵌入向量矩阵
            
        Returns:
            聚类标签数组（-1表示噪声点）
        """
        if not HDBSCAN_AVAILABLE:
            print("警告：HDBSCAN库未安装，回退到层次聚类")
            return self._hierarchical_clustering(embeddings)
        
        # 如果使用余弦距离，需要先标准化
        if self.hdbscan_metric == "cosine":
            # 对于余弦距离，标准化嵌入向量
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            processed_embeddings = normalized_embeddings
        else:
            # 对于欧几里得距离，可选择是否标准化
            scaler = StandardScaler()
            processed_embeddings = scaler.fit_transform(embeddings)
        
        # 设置HDBSCAN参数
        min_samples = self.hdbscan_min_samples
        if min_samples is None:
            min_samples = self.hdbscan_min_cluster_size
        
        # 创建HDBSCAN聚类器
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=self.hdbscan_cluster_selection_epsilon,
            metric=self.hdbscan_metric,
            cluster_selection_method='eom',  # Excess of Mass
            algorithm='best'
        )
        
        try:
            cluster_labels = clusterer.fit_predict(processed_embeddings)
            
            # 处理噪声点：将噪声点(-1)分配到最近的聚类或创建单独的聚类
            cluster_labels = self._handle_noise_points(cluster_labels, processed_embeddings)
            
            # 如果没有找到任何聚类，回退到层次聚类
            if len(set(cluster_labels)) <= 1:
                print("HDBSCAN未找到有效聚类，回退到层次聚类")
                return self._hierarchical_clustering(embeddings)
            
            return cluster_labels
            
        except Exception as e:
            print(f"HDBSCAN聚类失败: {e}，回退到层次聚类")
            return self._hierarchical_clustering(embeddings)
    
    def _handle_noise_points(self, cluster_labels: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """
        处理HDBSCAN产生的噪声点
        
        Args:
            cluster_labels: 原始聚类标签（包含-1噪声点）
            embeddings: 嵌入向量矩阵
            
        Returns:
            处理后的聚类标签（无噪声点）
        """
        # 找到噪声点
        noise_mask = cluster_labels == -1
        noise_indices = np.where(noise_mask)[0]
        
        if len(noise_indices) == 0:
            return cluster_labels
        
        # 获取非噪声点的聚类中心
        unique_labels = np.unique(cluster_labels[~noise_mask])
        
        if len(unique_labels) == 0:
            # 如果所有点都是噪声，创建单一聚类
            return np.zeros_like(cluster_labels)
        
        # 计算每个聚类的中心
        cluster_centers = {}
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_centers[label] = np.mean(embeddings[cluster_mask], axis=0)
        
        # 为每个噪声点分配到最近的聚类
        for noise_idx in noise_indices:
            noise_embedding = embeddings[noise_idx]
            
            # 计算到各聚类中心的距离
            distances = {}
            for label, center in cluster_centers.items():
                distances[label] = np.linalg.norm(noise_embedding - center)
            
            # 分配到最近的聚类
            closest_cluster = min(distances.keys(), key=lambda x: distances[x])
            cluster_labels[noise_idx] = closest_cluster
        
        return cluster_labels
    
    def _estimate_cluster_number(self, n_samples: int) -> int:
        """
        动态估计聚类数量
        
        Args:
            n_samples: 样本数量
            
        Returns:
            估计的聚类数量
        """
        # 基于样本数量和配置参数估计聚类数量
        min_clusters = max(2, n_samples // self.max_cluster_size)
        max_clusters = min(n_samples // self.min_cluster_size, n_samples // 2)
        
        # 使用启发式规则
        estimated_clusters = int(np.sqrt(n_samples / 2))
        
        return max(min_clusters, min(estimated_clusters, max_clusters))
    
    def _build_topics(self, sentences_data: List[Dict], cluster_labels: np.ndarray, embeddings: np.ndarray) -> List[Dict]:
        """
        根据聚类结果构建主题块，增强版本包含内聚度评估
        
        Args:
            sentences_data: 句子数据列表
            cluster_labels: 聚类标签数组
            embeddings: 嵌入向量矩阵
            
        Returns:
            主题块列表
        """
        # 按聚类标签分组
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append((sentences_data[i], embeddings[i]))
        
        topics = []
        for cluster_id, cluster_data in clusters.items():
            # 过滤太小的聚类
            if len(cluster_data) < self.min_cluster_size:
                continue
            
            cluster_sentences = [item[0] for item in cluster_data]
            cluster_embeddings = np.array([item[1] for item in cluster_data])
            
            # 计算主题内聚度
            coherence_score = self._calculate_topic_coherence(cluster_embeddings)
            
            # 构建主题文本
            topic_text = "\n".join([sent["text"] for sent in cluster_sentences])
            
            # 收集源文件信息
            sources = list(set([sent["meta"].get("source", "") for sent in cluster_sentences]))
            sources = [s for s in sources if s]  # 过滤空字符串
            
            # 生成主题摘要
            summary = self._generate_topic_summary(topic_text)
            
            topic = {
                "id": f"advanced_topic_{cluster_id}",
                "text": topic_text,
                "source": ",".join(sources),
                "summary": summary,
                "title": summary[:50] + "..." if len(summary) > 50 else summary,
                "sentence_count": len(cluster_sentences),
                "clustering_method": self.clustering_method,
                "coherence_score": coherence_score,
                "embeddings": cluster_embeddings,
                "sentences_data": cluster_sentences
            }
            
            topics.append(topic)
        
        return topics
    
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
    
    def _evaluate_and_optimize_coherence(self, topics: List[Dict], sentences_data: List[Dict], embeddings: np.ndarray) -> List[Dict]:
        """
        评估主题内聚度并进行优化
        
        Args:
            topics: 初始主题列表
            sentences_data: 原始句子数据
            embeddings: 嵌入向量矩阵
            
        Returns:
            优化后的主题列表
        """
        optimized_topics = []
        
        for topic in topics:
            coherence_score = topic.get("coherence_score", 0.0)
            
            # 如果内聚度低于阈值且主题较大，考虑分裂
            if (coherence_score < self.coherence_threshold and 
                topic["sentence_count"] > self.max_cluster_size * 0.6):
                
                # 使用智能分裂策略
                split_topics = self._intelligent_topic_split(topic)
                optimized_topics.extend(split_topics)
            else:
                # 如果使用主题总结词增强一致性
                if self.use_topic_summary_enhancement:
                    topic = self._enhance_topic_with_summary(topic)
                optimized_topics.append(topic)
        
        return optimized_topics
    
    def _intelligent_topic_split(self, topic: Dict) -> List[Dict]:
        """
        智能主题分裂策略：基于二聚类而非简单均匀分裂
        
        Args:
            topic: 需要分裂的主题
            
        Returns:
            分裂后的主题列表
        """
        embeddings = topic["embeddings"]
        sentences_data = topic["sentences_data"]
        
        if len(embeddings) < 4:  # 太小无法分裂
            return [topic]
        
        try:
            # 使用K-means进行二聚类找到最佳分界点
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            split_labels = kmeans.fit_predict(embeddings)
            
            # 按聚类标签分组
            group_0_indices = np.where(split_labels == 0)[0]
            group_1_indices = np.where(split_labels == 1)[0]
            
            # 确保两个组都有足够的句子
            if len(group_0_indices) < self.min_cluster_size or len(group_1_indices) < self.min_cluster_size:
                return [topic]  # 无法有效分裂，返回原主题
            
            # 创建两个新主题
            split_topics = []
            for i, indices in enumerate([group_0_indices, group_1_indices]):
                sub_sentences = [sentences_data[idx] for idx in indices]
                sub_embeddings = embeddings[indices]
                
                # 计算子主题的内聚度
                sub_coherence = self._calculate_topic_coherence(sub_embeddings)
                
                # 构建子主题文本
                sub_text = "\n".join([sent["text"] for sent in sub_sentences])
                
                # 收集源文件信息
                sources = list(set([sent["meta"].get("source", "") for sent in sub_sentences]))
                sources = [s for s in sources if s]  # 过滤空字符串
                
                # 生成子主题摘要
                sub_summary = self._generate_topic_summary(sub_text)
                
                sub_topic = {
                    "id": f"{topic['id']}_split_{i}",
                    "text": sub_text,
                    "source": ",".join(sources),
                    "summary": sub_summary,
                    "title": sub_summary[:50] + "..." if len(sub_summary) > 50 else sub_summary,
                    "sentence_count": len(sub_sentences),
                    "clustering_method": f"{self.clustering_method}_split",
                    "coherence_score": sub_coherence,
                    "parent_topic_id": topic["id"]
                }
                
                split_topics.append(sub_topic)
            
            return split_topics
            
        except Exception as e:
            # 分裂失败，返回原主题
            return [topic]
    
    def _enhance_topic_with_summary(self, topic: Dict) -> Dict:
        """
        利用主题总结词增强主题一致性
        
        Args:
            topic: 原始主题
            
        Returns:
            增强后的主题
        """
        try:
            # 提取主题关键词
            keywords = self._extract_topic_keywords(topic["text"])
            
            # 将关键词融入主题摘要
            enhanced_summary = f"{topic['summary']} [关键词: {', '.join(keywords[:5])}]"
            
            topic["summary"] = enhanced_summary
            topic["keywords"] = keywords
            topic["enhanced"] = True
            
        except Exception:
            # 增强失败，保持原样
            pass
        
        return topic
    
    def _extract_topic_keywords(self, text: str) -> List[str]:
        """
        从主题文本中提取关键词
        
        Args:
            text: 主题文本
            
        Returns:
            关键词列表
        """
        try:
            # 使用LLM提取关键词
            prompt = f"请从以下文本中提取5-8个最重要的关键词，用逗号分隔：\n\n{text[:500]}..."
            keywords_text = self.llm_client.chat([{"role": "user", "content": prompt}])
            
            # 解析关键词
            keywords = [kw.strip() for kw in keywords_text.split(",")]
            return keywords[:8]  # 最多返回8个关键词
            
        except Exception:
            # 提取失败，返回空列表
            return []

    def _generate_topic_summary(self, text: str) -> str:
        """
        生成主题摘要
        
        Args:
            text: 主题文本
            
        Returns:
            主题摘要
        """
        try:
            # 使用LLM生成摘要
            prompt = f"请为以下文本生成一个简洁的主题摘要（不超过100字）：\n\n{text[:1000]}..."
            summary = self.llm_client.chat([{"role": "user", "content": prompt}])
            return summary.strip()
        except Exception as e:
            # 如果LLM调用失败，返回文本开头作为摘要
            return text[:100] + "..." if len(text) > 100 else text
    
    def _create_single_topic(self, sentences_data: List[Dict]) -> List[Dict]:
        """
        当句子数量太少时，创建单个主题
        
        Args:
            sentences_data: 句子数据列表
            
        Returns:
            单个主题的列表
        """
        if not sentences_data:
            return []
            
        topic_text = "\n".join([sent["text"] for sent in sentences_data])
        sources = list(set([sent["meta"].get("source", "") for sent in sentences_data]))
        sources = [s for s in sources if s]
        
        summary = self._generate_topic_summary(topic_text)
        
        return [{
            "id": "advanced_topic_0",
            "text": topic_text,
            "source": ",".join(sources),
            "summary": summary,
            "title": summary[:50] + "..." if len(summary) > 50 else summary,
            "sentence_count": len(sentences_data),
            "clustering_method": "single_topic"
        }]
    
    def analyze_clustering_quality(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> Dict:
        """
        分析聚类质量
        
        Args:
            embeddings: 嵌入向量矩阵
            cluster_labels: 聚类标签数组
            
        Returns:
            聚类质量分析结果
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        try:
            # 计算轮廓系数
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            
            # 计算Calinski-Harabasz指数
            ch_score = calinski_harabasz_score(embeddings, cluster_labels)
            
            # 计算聚类内部和外部相似度
            intra_cluster_sim, inter_cluster_sim = self._calculate_cluster_similarities(
                embeddings, cluster_labels
            )
            
            return {
                "silhouette_score": silhouette_avg,
                "calinski_harabasz_score": ch_score,
                "intra_cluster_similarity": intra_cluster_sim,
                "inter_cluster_similarity": inter_cluster_sim,
                "n_clusters": len(set(cluster_labels)),
                "clustering_method": self.clustering_method
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_cluster_similarities(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> Tuple[float, float]:
        """
        计算聚类内部和外部相似度
        
        Args:
            embeddings: 嵌入向量矩阵
            cluster_labels: 聚类标签数组
            
        Returns:
            (聚类内部平均相似度, 聚类间平均相似度)
        """
        similarity_matrix = cosine_similarity(embeddings)
        
        intra_similarities = []
        inter_similarities = []
        
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            
            # 聚类内部相似度
            if len(cluster_indices) > 1:
                cluster_sim_matrix = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
                # 排除对角线元素（自相似度）
                mask = ~np.eye(cluster_sim_matrix.shape[0], dtype=bool)
                intra_similarities.extend(cluster_sim_matrix[mask])
            
            # 聚类间相似度
            other_indices = np.where(cluster_labels != label)[0]
            if len(other_indices) > 0:
                inter_sim_matrix = similarity_matrix[np.ix_(cluster_indices, other_indices)]
                inter_similarities.extend(inter_sim_matrix.flatten())
        
        avg_intra_sim = np.mean(intra_similarities) if intra_similarities else 0.0
        avg_inter_sim = np.mean(inter_similarities) if inter_similarities else 0.0
        
        return avg_intra_sim, avg_inter_sim