"""冗余过滤器工厂

提供统一的冗余过滤器创建和管理接口。
"""

from typing import Dict, Any, Type, Optional
from redundancy.base_redundancy_filter import BaseRedundancyFilter, RedundancyFilterError, ConfigurationError
from redundancy.simhash_buffer import SimHashBuffer
from document.redundancy_buffer import RedundancyBuffer
from utils.config_manager import ConfigManager
import logging


class RedundancyFilterFactory:
    """冗余过滤器工厂类
    
    负责根据配置创建合适的冗余过滤器实例。
    """
    
    # 注册的过滤器类型
    _filter_types: Dict[str, Type[BaseRedundancyFilter]] = {}
    
    @classmethod
    def register_filter(cls, name: str, filter_class: Type[BaseRedundancyFilter]) -> None:
        """注册冗余过滤器类型
        
        Args:
            name: 过滤器名称
            filter_class: 过滤器类
        """
        cls._filter_types[name] = filter_class
    
    @classmethod
    def create_filter(cls, config: Dict[str, Any]) -> BaseRedundancyFilter:
        """创建冗余过滤器实例
        
        Args:
            config: 配置字典
            
        Returns:
            冗余过滤器实例
            
        Raises:
            ConfigurationError: 配置错误
        """
        method = config.get('method', 'simhash')
        
        if method not in cls._filter_types:
            raise ConfigurationError(f"不支持的冗余检测方法: {method}")
        
        filter_class = cls._filter_types[method]
        
        try:
            return filter_class(config)
        except Exception as e:
            raise ConfigurationError(f"创建 {method} 过滤器失败: {e}")
    
    @classmethod
    def create_from_config_manager(cls, config_manager: ConfigManager) -> BaseRedundancyFilter:
        """从配置管理器创建冗余过滤器
        
        Args:
            config_manager: 配置管理器实例
            
        Returns:
            冗余过滤器实例
        """
        # 验证配置
        errors = config_manager.validate_redundancy_config()
        if errors:
            raise ConfigurationError(f"配置验证失败: {'; '.join(errors)}")
        
        # 获取标准化配置
        config = config_manager.get_redundancy_config()
        
        return cls.create_filter(config)
    
    @classmethod
    def get_available_methods(cls) -> list:
        """获取可用的冗余检测方法
        
        Returns:
            可用方法列表
        """
        return list(cls._filter_types.keys())
    
    @classmethod
    def get_method_info(cls, method: str) -> Dict[str, Any]:
        """获取方法信息
        
        Args:
            method: 方法名称
            
        Returns:
            方法信息字典
        """
        if method not in cls._filter_types:
            raise ConfigurationError(f"未知方法: {method}")
        
        filter_class = cls._filter_types[method]
        
        return {
            'name': method,
            'class': filter_class.__name__,
            'module': filter_class.__module__,
            'doc': filter_class.__doc__ or "无描述"
        }


# SimHash适配器类，使其兼容BaseRedundancyFilter接口
class SimHashFilterAdapter(BaseRedundancyFilter):
    """SimHash过滤器适配器
    
    将SimHashBuffer适配为BaseRedundancyFilter接口。
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 验证SimHash特定配置
        if 'hamming_threshold' not in config:
            raise ConfigurationError("SimHash方法缺少 'hamming_threshold' 配置")
        if 'max_buffer_size' not in config:
            raise ConfigurationError("SimHash方法缺少 'max_buffer_size' 配置")
        
        # 创建SimHashBuffer实例
        self.simhash_buffer = SimHashBuffer(config)
        self.logger = logging.getLogger(__name__)
    
    def is_duplicate(self, text: str) -> bool:
        """检查文本是否为重复"""
        return self.simhash_buffer.is_duplicate(text)
    
    def is_redundant(self, sentence: str, embedding=None) -> bool:
        """检查句子是否冗余（兼容接口）"""
        return self.simhash_buffer.is_redundant(sentence)
    
    def add_text(self, text: str) -> None:
        """添加文本到缓冲区"""
        self.simhash_buffer.add_text(text)
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.simhash_buffer.clear()
    
    def get_memory_usage(self) -> float:
        """获取内存使用量 (MB)"""
        return self.simhash_buffer.get_memory_usage()
    
    @property
    def method_name(self) -> str:
        return "SimHash"


# Embedding适配器类
class EmbeddingFilterAdapter(BaseRedundancyFilter):
    """Embedding过滤器适配器
    
    将RedundancyBuffer适配为BaseRedundancyFilter接口。
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 验证Embedding特定配置
        if 'similarity_threshold' not in config:
            raise ConfigurationError("Embedding方法缺少 'similarity_threshold' 配置")
        
        # 创建RedundancyBuffer实例
        redundancy_config = {
            'threshold': config['similarity_threshold'],
            'enable_logging': config.get('enable_logging', True),
            'enable_progress': config.get('enable_progress', True),
            'log_interval': config.get('log_interval', 1000)
        }
        
        self.redundancy_buffer = RedundancyBuffer(redundancy_config)
        self.logger = logging.getLogger(__name__)
    
    def is_duplicate(self, text: str) -> bool:
        """检查文本是否为重复"""
        return self.redundancy_buffer.is_duplicate(text)
    
    def is_redundant(self, sentence: str, embedding) -> bool:
        """检查句子是否冗余（兼容接口）"""
        return self.redundancy_buffer.is_redundant(sentence, embedding)
    
    def add_text(self, text: str) -> None:
        """添加文本到缓冲区"""
        self.redundancy_buffer.add_text(text)
    
    def clear(self) -> None:
        """清空缓冲区"""
        self.redundancy_buffer.clear()
    
    def get_memory_usage(self) -> float:
        """获取内存使用量 (MB)"""
        return self.redundancy_buffer.get_memory_usage()
    
    @property
    def method_name(self) -> str:
        return "Embedding"


# 注册默认的过滤器类型
RedundancyFilterFactory.register_filter('simhash', SimHashFilterAdapter)
RedundancyFilterFactory.register_filter('embedding', EmbeddingFilterAdapter)


# 便捷函数
def create_redundancy_filter(config_path: str, method: Optional[str] = None) -> BaseRedundancyFilter:
    """便捷函数：从配置文件创建冗余过滤器
    
    Args:
        config_path: 配置文件路径
        method: 强制指定的方法，如果为None则从配置文件读取
        
    Returns:
        冗余过滤器实例
    """
    config_manager = ConfigManager(config_path)
    
    if method:
        # 临时覆盖方法
        config = config_manager.get_redundancy_config()
        config['method'] = method
        return RedundancyFilterFactory.create_filter(config)
    else:
        return RedundancyFilterFactory.create_from_config_manager(config_manager)


def get_optimal_method(data_size: int, accuracy_requirement: str = 'balanced') -> str:
    """根据数据规模和准确率要求推荐最优方法
    
    Args:
        data_size: 数据规模（文档数量）
        accuracy_requirement: 准确率要求 ('high', 'balanced', 'fast')
        
    Returns:
        推荐的方法名称
    """
    if accuracy_requirement == 'high':
        return 'embedding'
    elif accuracy_requirement == 'fast':
        return 'simhash'
    else:  # balanced
        # 根据数据规模选择
        if data_size > 50000:
            return 'simhash'  # 大规模数据优先性能
        else:
            return 'embedding'  # 小规模数据优先准确率