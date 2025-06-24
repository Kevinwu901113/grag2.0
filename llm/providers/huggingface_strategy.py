from typing import List
from embedding.hf_embedder import HuggingFaceEmbedder
import logging

class HuggingFaceStrategy:
    """
    HuggingFace策略类
    专门用于嵌入向量生成，不支持文本生成
    """
    
    def __init__(self, config: dict):
        """
        初始化HuggingFace策略
        
        Args:
            config: 配置字典，包含嵌入相关配置
        """
        self.config = config
        embedding_config = config.get('embedding', {})
        
        # 初始化日志记录器
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 提取嵌入配置参数
        model_name = embedding_config.get('model_name', 'BAAI/bge-m3')
        device = embedding_config.get('device', 'auto')
        batch_size = embedding_config.get('batch_size', 32)
        local_files_only = embedding_config.get('local_files_only', False)
        max_seq_length = embedding_config.get('max_seq_length', 512)
        enable_memory_optimization = embedding_config.get('enable_memory_optimization', True)
        
        # 初始化HuggingFace嵌入器
        self.embedder = HuggingFaceEmbedder(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            local_files_only=local_files_only,
            max_seq_length=max_seq_length,
            enable_memory_optimization=enable_memory_optimization
        )
        
        print(f"[HuggingFaceStrategy] 初始化完成")
        print(f"[HuggingFaceStrategy] 模型: {model_name}")
        print(f"[HuggingFaceStrategy] 设备: {device}")
        print(f"[HuggingFaceStrategy] 批量大小: {batch_size}")
        print(f"[HuggingFaceStrategy] 离线模式: {local_files_only}")
        
    def generate(self, prompt: str) -> str:
        """
        HuggingFace策略不支持文本生成
        
        Args:
            prompt: 输入提示
            
        Raises:
            NotImplementedError: HuggingFace策略仅用于嵌入，不支持文本生成
        """
        raise NotImplementedError(
            "HuggingFace策略仅用于嵌入向量生成，不支持文本生成。"
            "请使用Ollama或OpenAI策略进行文本生成。"
        )
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        生成文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表，格式为 List[List[float]]
        """
        if not texts:
            raise ValueError("输入文本列表为空")
        
        try:
            # 使用HuggingFace嵌入器进行批量编码
            embeddings = self.embedder.encode(texts)
            
            self.logger.debug(f"✅ HuggingFace嵌入生成成功，处理 {len(texts)} 条文本")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"❌ HuggingFace嵌入生成失败: {e}")
            raise RuntimeError(f"HuggingFace嵌入生成失败: {e}")
    
    def get_embedding_info(self) -> dict:
        """
        获取嵌入器信息
        
        Returns:
            嵌入器信息字典
        """
        return self.embedder.get_device_info()