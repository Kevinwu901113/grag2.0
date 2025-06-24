import torch
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import logging
import torch

class HuggingFaceEmbedder:
    """
    HuggingFace本地嵌入模型封装类
    支持批量处理、GPU加速和嵌入归一化
    
    注意：首次使用时模型下载较慢，建议预先下载模型到本地
    """
    
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "auto", batch_size: int = 32, local_files_only: bool = False, max_seq_length: int = 512, enable_memory_optimization: bool = True):
        """
        初始化HuggingFace嵌入器
        
        Args:
            model_name: 模型名称，默认使用BAAI/bge-m3
            device: 设备类型，'cuda', 'cpu' 或 'auto'
            batch_size: 批量处理大小
            local_files_only: 是否仅使用本地文件（离线模式）
            max_seq_length: 最大序列长度，用于内存优化
            enable_memory_optimization: 是否启用内存优化
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self.max_seq_length = max_seq_length
        self.enable_memory_optimization = enable_memory_optimization
        
        # 设备选择
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # 初始化日志记录器
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 嵌入归一化配置
        self.normalize_embeddings = True  # 默认启用归一化
        
        # 嵌入维度（将在首次编码时确定）
        self.embedding_dim = None
            
        print(f"[HuggingFaceEmbedder] 初始化嵌入模型: {model_name}")
        print(f"[HuggingFaceEmbedder] 使用设备: {self.device}")
        print(f"[HuggingFaceEmbedder] 批量大小: {batch_size}")
        print(f"[HuggingFaceEmbedder] 离线模式: {local_files_only}")
        
        if not local_files_only:
            print(f"[HuggingFaceEmbedder] 注意: 首次运行会下载模型文件，请确保网络连接正常")
        
        try:
            # 加载sentence-transformers模型
            self.model = SentenceTransformer(
                model_name, 
                device=self.device,
                local_files_only=local_files_only
            )
            
            # 内存优化配置
            if self.enable_memory_optimization and hasattr(self.model, '_modules'):
                # 设置最大序列长度
                if hasattr(self.model, 'max_seq_length'):
                    self.model.max_seq_length = self.max_seq_length
                    print(f"[HuggingFaceEmbedder] 设置最大序列长度: {self.max_seq_length}")
                
                # 启用CUDA内存优化
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()  # 清理GPU缓存
                    print(f"[HuggingFaceEmbedder] 已清理CUDA缓存")
            
            # 获取嵌入维度
            test_embedding = self.model.encode(["test"], convert_to_numpy=True)
            self.embedding_dim = test_embedding.shape[1]
            
            print(f"[HuggingFaceEmbedder] 模型加载成功")
            print(f"[HuggingFaceEmbedder] 嵌入维度: {self.embedding_dim}")
            print(f"[HuggingFaceEmbedder] 内存优化: {self.enable_memory_optimization}")
        except Exception as e:
            print(f"[HuggingFaceEmbedder] 嵌入模型加载失败: {e}")
            if not local_files_only:
                print(f"[HuggingFaceEmbedder] 提示: 如果网络不可用，请先下载模型到本地，然后设置 local_files_only=True")
            raise RuntimeError(f"嵌入模型加载失败: {e}")
        

    
    def encode(self, text_list: Union[str, List[str]]) -> List[List[float]]:
        """
        对文本列表进行嵌入编码
        
        Args:
            text_list: 单个文本字符串或文本列表
            
        Returns:
            嵌入向量列表，格式为 List[List[float]]
            如果启用归一化，返回的向量已进行L2归一化，适用于余弦相似度计算
        """
        # 统一处理输入格式
        if isinstance(text_list, str):
            text_list = [text_list]
        elif not isinstance(text_list, list):
            raise ValueError("输入必须是字符串或字符串列表")
            
        if not text_list:
            raise ValueError("输入文本列表为空")
        
        # 过滤空文本
        valid_texts = [text.strip() for text in text_list if text and text.strip()]
        if not valid_texts:
            raise ValueError("过滤后文本列表为空")
        
        try:
            # 使用模型进行批量编码
            self.logger.debug(f"开始编码 {len(valid_texts)} 条文本，批量大小: {self.batch_size}")
            
            # 内存优化：截断过长文本
            if self.enable_memory_optimization:
                valid_texts = [text[:self.max_seq_length*4] for text in valid_texts]  # 粗略估算字符数
            
            current_batch_size = self.batch_size
            max_retries = 3
            
            for retry in range(max_retries):
                try:
                    # 清理CUDA缓存
                    if self.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # SentenceTransformer自动处理批量
                    embeddings = self.model.encode(
                        valid_texts,
                        batch_size=current_batch_size,
                        show_progress_bar=len(valid_texts) > 100,  # 大量文本时显示进度条
                        convert_to_numpy=True,
                        normalize_embeddings=False  # 我们手动控制归一化
                    )
                    break  # 成功则跳出重试循环
                    
                except RuntimeError as cuda_error:
                    if "CUDA out of memory" in str(cuda_error) and retry < max_retries - 1:
                        # CUDA内存不足，减小批量大小重试
                        current_batch_size = max(1, current_batch_size // 2)
                        self.logger.warning(f"⚠️ CUDA内存不足，降低批量大小至 {current_batch_size} 并重试 (第{retry+1}次)")
                        
                        # 清理CUDA缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise cuda_error
            
            # 手动归一化（如果需要）
            if self.normalize_embeddings:
                embeddings = normalize(embeddings, norm='l2', axis=1)
            
            # 转换为列表格式
            embeddings_list = embeddings.tolist()
            
            self.logger.debug(f"✅ 编码完成，生成 {len(embeddings_list)} 个嵌入向量")
            self.logger.debug(f"向量维度: {len(embeddings_list[0]) if embeddings_list else 0}")
            if current_batch_size != self.batch_size:
                self.logger.info(f"📊 实际使用批量大小: {current_batch_size} (原设置: {self.batch_size})")
            
            return embeddings_list
            
        except Exception as e:
            self.logger.error(f"❌ 嵌入编码失败: {e}")
            # 提供内存优化建议
            if "CUDA out of memory" in str(e):
                self.logger.error("💡 建议解决方案:")
                self.logger.error("   1. 减小配置文件中的 batch_size (当前: {})")
                self.logger.error("   2. 设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
                self.logger.error("   3. 或将 device 设置为 'cpu' 使用CPU处理")
            raise RuntimeError(f"嵌入编码失败: {e}")
    
    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量维度
        
        Returns:
            嵌入向量维度
        """
        return self.embedding_dim
    
    def get_device_info(self) -> dict:
        """
        获取设备信息
        
        Returns:
            设备信息字典
        """
        return {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings
        }