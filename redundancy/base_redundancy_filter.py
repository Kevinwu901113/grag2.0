"""冗余过滤器抽象基类

定义统一的冗余检测接口，确保不同实现的一致性。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class RedundancyStats:
    """冗余检测统计信息"""
    total_processed: int = 0
    duplicates_found: int = 0
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    @property
    def duplicate_rate(self) -> float:
        """重复率"""
        return self.duplicates_found / max(self.total_processed, 1)
    
    @property
    def throughput(self) -> float:
        """处理吞吐量 (文档/秒)"""
        return self.total_processed / max(self.processing_time, 0.001)


class BaseRedundancyFilter(ABC):
    """冗余过滤器抽象基类
    
    定义所有冗余检测实现必须遵循的接口。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化冗余过滤器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.stats = RedundancyStats()
        self.enable_logging = config.get('enable_logging', True)
        self.enable_progress = config.get('enable_progress', True)
        self.log_interval = config.get('log_interval', 1000)
        self._start_time = time.time()
    
    @abstractmethod
    def is_duplicate(self, text: str) -> bool:
        """检查文本是否为重复
        
        Args:
            text: 待检查的文本
            
        Returns:
            True if duplicate, False otherwise
        """
        pass
    
    @abstractmethod
    def add_text(self, text: str) -> None:
        """添加文本到缓冲区
        
        Args:
            text: 要添加的文本
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空缓冲区"""
        pass
    
    @abstractmethod
    def get_memory_usage(self) -> float:
        """获取内存使用量 (MB)
        
        Returns:
            内存使用量
        """
        pass
    
    def process_text(self, text: str) -> bool:
        """处理文本并更新统计信息
        
        Args:
            text: 待处理的文本
            
        Returns:
            True if text is unique, False if duplicate
        """
        start_time = time.time()
        
        is_dup = self.is_duplicate(text)
        
        if not is_dup:
            self.add_text(text)
        
        # 更新统计信息
        self.stats.total_processed += 1
        if is_dup:
            self.stats.duplicates_found += 1
        
        self.stats.processing_time = time.time() - self._start_time
        self.stats.memory_usage_mb = self.get_memory_usage()
        
        # 日志记录
        if (self.enable_logging and 
            self.stats.total_processed % self.log_interval == 0):
            self._log_progress()
        
        return not is_dup
    
    def process_batch(self, texts: List[str]) -> List[bool]:
        """批量处理文本
        
        Args:
            texts: 文本列表
            
        Returns:
            每个文本是否唯一的布尔列表
        """
        results = []
        for text in texts:
            results.append(self.process_text(text))
        return results
    
    def filter_duplicates(self, texts: List[str]) -> List[str]:
        """过滤重复文本
        
        Args:
            texts: 输入文本列表
            
        Returns:
            去重后的文本列表
        """
        unique_texts = []
        for text in texts:
            if self.process_text(text):
                unique_texts.append(text)
        return unique_texts
    
    def get_stats(self) -> RedundancyStats:
        """获取统计信息
        
        Returns:
            统计信息对象
        """
        return self.stats
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = RedundancyStats()
        self._start_time = time.time()
    
    def _log_progress(self) -> None:
        """记录处理进度"""
        if hasattr(self, 'logger'):
            self.logger.info(
                f"处理进度: {self.stats.total_processed} 文档, "
                f"重复率: {self.stats.duplicate_rate:.2%}, "
                f"吞吐量: {self.stats.throughput:.1f} 文档/秒, "
                f"内存使用: {self.stats.memory_usage_mb:.1f} MB"
            )
        else:
            print(
                f"[冗余检测] 处理进度: {self.stats.total_processed} 文档, "
                f"重复率: {self.stats.duplicate_rate:.2%}, "
                f"吞吐量: {self.stats.throughput:.1f} 文档/秒, "
                f"内存使用: {self.stats.memory_usage_mb:.1f} MB"
            )
    
    @property
    def method_name(self) -> str:
        """获取方法名称"""
        return self.__class__.__name__
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        if self.enable_logging:
            self._log_final_stats()
    
    def _log_final_stats(self) -> None:
        """记录最终统计信息"""
        if hasattr(self, 'logger'):
            self.logger.info(
                f"[{self.method_name}] 处理完成: "
                f"总计 {self.stats.total_processed} 文档, "
                f"发现 {self.stats.duplicates_found} 重复 ({self.stats.duplicate_rate:.2%}), "
                f"总耗时 {self.stats.processing_time:.2f}s, "
                f"平均吞吐量 {self.stats.throughput:.1f} 文档/秒"
            )
        else:
            print(
                f"[{self.method_name}] 处理完成: "
                f"总计 {self.stats.total_processed} 文档, "
                f"发现 {self.stats.duplicates_found} 重复 ({self.stats.duplicate_rate:.2%}), "
                f"总耗时 {self.stats.processing_time:.2f}s, "
                f"平均吞吐量 {self.stats.throughput:.1f} 文档/秒"
            )


class RedundancyFilterError(Exception):
    """冗余过滤器异常基类"""
    pass


class ConfigurationError(RedundancyFilterError):
    """配置错误"""
    pass


class ProcessingError(RedundancyFilterError):
    """处理错误"""
    pass