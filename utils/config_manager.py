"""统一配置管理器

提供标准化的配置加载、验证和管理功能。
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConfigValidationError(Exception):
    """配置验证错误"""
    field: str
    message: str
    
    def __str__(self):
        return f"配置字段 '{self.field}' 验证失败: {self.message}"


class ConfigManager:
    """统一配置管理器
    
    提供配置加载、验证、获取和更新功能。
    """
    
    def __init__(self, config_path: str):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._validation_rules: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """加载配置文件"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
                
        except yaml.YAMLError as e:
            raise ConfigValidationError("yaml_format", f"YAML格式错误: {e}")
        except Exception as e:
            raise ConfigValidationError("file_access", f"配置文件访问错误: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键，支持点分隔的嵌套键 (如 'redundancy.method')
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值
        
        Args:
            key: 配置键，支持点分隔的嵌套键
            value: 配置值
        """
        keys = key.split('.')
        config = self._config
        
        # 创建嵌套字典结构
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def validate_redundancy_config(self) -> List[str]:
        """验证冗余检测配置
        
        Returns:
            验证错误列表
        """
        errors = []
        
        # 检查冗余配置是否存在
        redundancy_config = self.get('redundancy')
        if not redundancy_config:
            errors.append("缺少 'redundancy' 配置节")
            return errors
        
        # 检查方法配置
        method = self.get('redundancy.method', 'simhash')
        if method not in ['simhash', 'embedding']:
            errors.append(f"不支持的冗余检测方法: {method}")
        
        # 验证SimHash配置
        if method == 'simhash':
            # 验证hamming_threshold
            threshold = self.get('redundancy.hamming_threshold')
            if threshold is None:
                errors.append("SimHash方法缺少 'hamming_threshold' 配置")
            elif not isinstance(threshold, int) or threshold < 0 or threshold > 64:
                errors.append("'hamming_threshold' 必须是0-64之间的整数")
            
            # 验证max_buffer_size
            buffer_size = self.get('redundancy.max_buffer_size')
            if buffer_size is None:
                errors.append("SimHash方法缺少 'max_buffer_size' 配置")
            elif not isinstance(buffer_size, int) or buffer_size <= 0:
                errors.append("'max_buffer_size' 必须是正整数")
        
        # 验证Embedding配置
        elif method == 'embedding':
            # 验证threshold
            threshold = self.get('redundancy.threshold')
            if threshold is None:
                errors.append("Embedding方法缺少 'threshold' 配置")
            elif not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                errors.append("'threshold' 必须是0-1之间的数值")
        
        return errors
    
    def validate_llm_config(self) -> List[str]:
        """验证LLM配置
        
        Returns:
            验证错误列表
        """
        errors = []
        
        llm_config = self.get('llm')
        if not llm_config:
            errors.append("缺少 'llm' 配置节")
            return errors
        
        # 检查provider
        provider = self.get('llm.provider')
        if not provider:
            errors.append("缺少 'llm.provider' 配置")
        elif provider not in ['openai', 'ollama', 'huggingface']:
            errors.append(f"不支持的LLM提供商: {provider}")
        
        # 检查model
        model = self.get('llm.model')
        if not model:
            errors.append("缺少 'llm.model' 配置")
        
        return errors
    
    def validate_all(self) -> List[str]:
        """验证所有配置
        
        Returns:
            所有验证错误列表
        """
        errors = []
        errors.extend(self.validate_redundancy_config())
        errors.extend(self.validate_llm_config())
        return errors
    
    def get_redundancy_config(self) -> Dict[str, Any]:
        """获取冗余检测配置
        
        Returns:
            冗余检测配置字典
        """
        return self.get('redundancy', {})
    
    def get_all_config(self) -> Dict[str, Any]:
        """获取所有配置
        
        Returns:
            完整的配置字典
        """
        return self._config.copy()
    
    def save_config(self) -> None:
        """保存配置到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        except Exception as e:
            raise ConfigValidationError("file_write", f"配置文件写入错误: {e}")
    
    @property
    def config(self) -> Dict[str, Any]:
        """获取完整配置字典"""
        return self._config.copy()
    
    def reload(self) -> None:
        """重新加载配置文件"""
        self.load_config()