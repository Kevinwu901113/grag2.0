#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
嵌入向量聚类器

负责对文本块的嵌入向量进行聚类分析：
1. 默认使用 KMeans 聚类
2. 支持 scikit-learn 和 faiss 两种实现
3. 聚类数量可配置或自动估算
4. 提供聚类质量评估和日志记录

作者: AI Assistant
日期: 2024
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings

# 可选的 faiss 支持
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


class Clusterer:
    """
    嵌入向量聚类器
    
    使用 KMeans 算法对嵌入向量进行聚类分析
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化聚类器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 从配置中获取聚类参数
        doc_config = config.get('document_processing', {})
        self.cluster_count = doc_config.get('cluster_count', None)  # None 表示自动估算
        self.auto_estimate_clusters = doc_config.get('auto_estimate_clusters', True)
        self.min_clusters = doc_config.get('min_clusters', 5)
        self.max_clusters = doc_config.get('max_clusters', 100)
        
        # 聚类算法配置
        self.algorithm = doc_config.get('clustering_algorithm', 'sklearn')  # 'sklearn' 或 'faiss'
        self.random_state = doc_config.get('random_state', 42)
        self.max_iter = doc_config.get('max_iter', 300)
        self.n_init = doc_config.get('n_init', 10)
        
        # 质量评估配置
        self.enable_quality_assessment = doc_config.get('enable_quality_assessment', True)
        self.log_cluster_distribution = doc_config.get('log_cluster_distribution', True)
        
        # 检查 faiss 可用性
        if self.algorithm == 'faiss' and not FAISS_AVAILABLE:
            self.logger.warning("faiss 不可用，回退到 sklearn")
            self.algorithm = 'sklearn'
        
        self.logger.info(f"聚类器初始化完成")
        self.logger.info(f"算法: {self.algorithm}, 聚类数: {self.cluster_count or '自动估算'}")
    
    def cluster_embeddings(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        """
        对嵌入向量进行聚类
        
        Args:
            embeddings: 嵌入向量矩阵，形状为 (n_samples, n_features)
            
        Returns:
            聚类标签数组，形状为 (n_samples,)
            如果失败则返回 None
        """
        if embeddings is None or len(embeddings) == 0:
            self.logger.error("输入嵌入向量为空")
            return None
        
        self.logger.info(f"开始聚类 {len(embeddings)} 个嵌入向量")
        
        try:
            # 确定聚类数量
            n_clusters = self._determine_cluster_count(embeddings)
            
            if n_clusters <= 0:
                self.logger.error("无效的聚类数量")
                return None
            
            self.logger.info(f"使用 {n_clusters} 个聚类")
            
            # 执行聚类
            if self.algorithm == 'faiss':
                labels = self._cluster_with_faiss(embeddings, n_clusters)
            else:
                labels = self._cluster_with_sklearn(embeddings, n_clusters)
            
            if labels is None:
                self.logger.error("聚类失败")
                return None
            
            # 质量评估和日志记录
            if self.enable_quality_assessment:
                self._assess_clustering_quality(embeddings, labels)
            
            if self.log_cluster_distribution:
                self._log_cluster_distribution(labels)
            
            self.logger.info(f"聚类完成，得到 {len(set(labels))} 个聚类")
            return labels
            
        except Exception as e:
            self.logger.error(f"聚类过程中出错: {e}")
            return None
    
    def _determine_cluster_count(self, embeddings: np.ndarray) -> int:
        """
        确定聚类数量
        
        Args:
            embeddings: 嵌入向量矩阵
            
        Returns:
            聚类数量
        """
        n_samples = len(embeddings)
        
        # 如果已指定聚类数量
        if self.cluster_count is not None:
            return min(self.cluster_count, n_samples)
        
        # 自动估算聚类数量
        if not self.auto_estimate_clusters:
            # 使用启发式规则
            estimated_clusters = max(self.min_clusters, min(self.max_clusters, int(np.sqrt(n_samples / 2))))
            self.logger.info(f"使用启发式规则估算聚类数: {estimated_clusters}")
            return estimated_clusters
        
        # 使用肘部法则或轮廓系数估算
        return self._estimate_optimal_clusters(embeddings)
    
    def _estimate_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """
        使用轮廓系数估算最优聚类数量
        
        Args:
            embeddings: 嵌入向量矩阵
            
        Returns:
            估算的最优聚类数量
        """
        n_samples = len(embeddings)
        max_k = min(self.max_clusters, n_samples // 2)
        min_k = max(self.min_clusters, 2)
        
        if max_k <= min_k:
            return min_k
        
        self.logger.info(f"正在估算最优聚类数量 (范围: {min_k}-{max_k})")
        
        best_score = -1
        best_k = min_k
        
        # 为了效率，只测试部分 k 值
        test_k_values = np.linspace(min_k, max_k, min(10, max_k - min_k + 1), dtype=int)
        test_k_values = np.unique(test_k_values)
        
        for k in test_k_values:
            try:
                # 使用较少的初始化次数以提高速度
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=3, max_iter=100)
                labels = kmeans.fit_predict(embeddings)
                
                # 计算轮廓系数
                score = silhouette_score(embeddings, labels)
                
                self.logger.debug(f"k={k}, 轮廓系数={score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            except Exception as e:
                self.logger.warning(f"测试 k={k} 时出错: {e}")
                continue
        
        self.logger.info(f"估算的最优聚类数: {best_k} (轮廓系数: {best_score:.3f})")
        return best_k
    
    def _cluster_with_sklearn(self, embeddings: np.ndarray, n_clusters: int) -> Optional[np.ndarray]:
        """
        使用 scikit-learn 进行聚类
        
        Args:
            embeddings: 嵌入向量矩阵
            n_clusters: 聚类数量
            
        Returns:
            聚类标签数组
        """
        try:
            self.logger.debug(f"使用 sklearn KMeans 进行聚类")
            
            # 抑制收敛警告
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    n_init=self.n_init,
                    max_iter=self.max_iter
                )
                
                labels = kmeans.fit_predict(embeddings)
            
            return labels
            
        except Exception as e:
            self.logger.error(f"sklearn 聚类失败: {e}")
            return None
    
    def _cluster_with_faiss(self, embeddings: np.ndarray, n_clusters: int) -> Optional[np.ndarray]:
        """
        使用 faiss 进行聚类
        
        Args:
            embeddings: 嵌入向量矩阵
            n_clusters: 聚类数量
            
        Returns:
            聚类标签数组
        """
        if not FAISS_AVAILABLE:
            self.logger.error("faiss 不可用")
            return None
        
        try:
            self.logger.debug(f"使用 faiss KMeans 进行聚类")
            
            # 确保数据类型为 float32
            embeddings_f32 = embeddings.astype(np.float32)
            
            # 创建 faiss kmeans 对象
            d = embeddings_f32.shape[1]  # 维度
            kmeans = faiss.Kmeans(d, n_clusters, niter=self.max_iter, verbose=False)
            
            # 训练聚类模型
            kmeans.train(embeddings_f32)
            
            # 获取聚类标签
            _, labels = kmeans.index.search(embeddings_f32, 1)
            labels = labels.flatten()
            
            return labels
            
        except Exception as e:
            self.logger.error(f"faiss 聚类失败: {e}")
            return None
    
    def _assess_clustering_quality(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        """
        评估聚类质量
        
        Args:
            embeddings: 嵌入向量矩阵
            labels: 聚类标签
        """
        try:
            # 轮廓系数 (Silhouette Score)
            silhouette_avg = silhouette_score(embeddings, labels)
            
            # Calinski-Harabasz 指数
            ch_score = calinski_harabasz_score(embeddings, labels)
            
            # 聚类内聚度 (平均类内距离)
            inertia = self._calculate_inertia(embeddings, labels)
            
            self.logger.info(f"聚类质量评估:")
            self.logger.info(f"  轮廓系数: {silhouette_avg:.3f}")
            self.logger.info(f"  Calinski-Harabasz 指数: {ch_score:.3f}")
            self.logger.info(f"  聚类内聚度: {inertia:.3f}")
            
        except Exception as e:
            self.logger.warning(f"聚类质量评估失败: {e}")
    
    def _calculate_inertia(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        计算聚类内聚度（总的类内平方和）
        
        Args:
            embeddings: 嵌入向量矩阵
            labels: 聚类标签
            
        Returns:
            内聚度值
        """
        inertia = 0.0
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_points = embeddings[labels == label]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                inertia += np.sum((cluster_points - centroid) ** 2)
        
        return inertia
    
    def _log_cluster_distribution(self, labels: np.ndarray) -> None:
        """
        记录聚类分布信息
        
        Args:
            labels: 聚类标签
        """
        try:
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            self.logger.info(f"聚类分布:")
            self.logger.info(f"  总聚类数: {len(unique_labels)}")
            self.logger.info(f"  平均每聚类样本数: {np.mean(counts):.1f}")
            self.logger.info(f"  最大聚类大小: {np.max(counts)}")
            self.logger.info(f"  最小聚类大小: {np.min(counts)}")
            self.logger.info(f"  聚类大小标准差: {np.std(counts):.1f}")
            
            # 记录每个聚类的大小
            for label, count in zip(unique_labels, counts):
                self.logger.debug(f"  聚类 {label}: {count} 个样本")
                
        except Exception as e:
            self.logger.warning(f"记录聚类分布失败: {e}")
    
    def get_cluster_centers(self, embeddings: np.ndarray, labels: np.ndarray) -> Optional[np.ndarray]:
        """
        计算聚类中心
        
        Args:
            embeddings: 嵌入向量矩阵
            labels: 聚类标签
            
        Returns:
            聚类中心矩阵，形状为 (n_clusters, n_features)
        """
        try:
            unique_labels = np.unique(labels)
            centers = []
            
            for label in unique_labels:
                cluster_points = embeddings[labels == label]
                center = np.mean(cluster_points, axis=0)
                centers.append(center)
            
            return np.array(centers)
            
        except Exception as e:
            self.logger.error(f"计算聚类中心失败: {e}")
            return None