from typing import Dict, Any, Optional
import os
import torch
import joblib
from pathlib import Path
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
from utils.logger import setup_logger

logger = setup_logger(os.getcwd())

# 默认缓存目录配置
DEFAULT_HF_CACHE_DIR = os.getenv('HF_CACHE_DIR', './hf_models')

def ensure_cache_dir(cache_dir: str = None) -> str:
    """确保缓存目录存在并返回路径"""
    if cache_dir is None:
        cache_dir = DEFAULT_HF_CACHE_DIR
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir

class ModelCache:
    """模型缓存管理器，实现单例模式确保模型只加载一次"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
            cls._instance._models = {}
            cls._instance._tokenizers = {}
            cls._instance._sentence_transformers = {}
            cls._instance._label_encoders = {}
        return cls._instance
    
    def get_bert_model(self, model_path: str, num_labels: int, model_name: str = "bert-base-chinese", cache_dir: str = None) -> Any:
        """获取BERT模型，如果已加载则直接返回缓存的模型"""
        if model_path in self._models:
            return self._models[model_path]
        
        from classifier.train_base_classifier import BERTClassifier
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BERTClassifier(model_name=model_name, num_labels=num_labels)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            self._models[model_path] = model
            logger.info(f"已加载BERT模型: {model_path}")
            return model
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            logger.error(f"加载BERT模型失败: {e}")
            return None
    
    def get_tokenizer(self, model_name: str, cache_dir: str = None) -> Optional[BertTokenizer]:
        """获取tokenizer，如果已加载则直接返回缓存的tokenizer"""
        if model_name in self._tokenizers:
            return self._tokenizers[model_name]
        
        try:
            cache_dir = ensure_cache_dir(cache_dir)
            tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self._tokenizers[model_name] = tokenizer
            logger.info(f"已加载Tokenizer: {model_name}")
            return tokenizer
        except Exception as e:
            logger.error(f"加载Tokenizer失败: {e}")
            return None
    
    def get_sentence_transformer(self, model_name: str) -> Optional[SentenceTransformer]:
        """获取句向量模型，如果已加载则直接返回缓存的模型"""
        if model_name in self._sentence_transformers:
            return self._sentence_transformers[model_name]
        
        try:
            model = SentenceTransformer(model_name)
            self._sentence_transformers[model_name] = model
            logger.info(f"已加载SentenceTransformer: {model_name}")
            return model
        except Exception as e:
            logger.error(f"加载SentenceTransformer失败: {e}")
            return None
    
    def get_label_encoder(self, encoder_path: str) -> Any:
        """获取标签编码器，如果已加载则直接返回缓存的编码器"""
        if encoder_path in self._label_encoders:
            return self._label_encoders[encoder_path]
        
        try:
            encoder = joblib.load(encoder_path)
            self._label_encoders[encoder_path] = encoder
            logger.info(f"已加载标签编码器: {encoder_path}")
            return encoder
        except Exception as e:
            logger.error(f"加载标签编码器失败: {e}")
            return None
    
    def clear_cache(self):
        """清除所有缓存的模型"""
        self._models.clear()
        self._tokenizers.clear()
        self._sentence_transformers.clear()
        self._label_encoders.clear()
        logger.info("已清除所有模型缓存")

# 全局单例
model_cache = ModelCache()