#!/usr/bin/env python3
"""
配置迁移工具
帮助用户从旧的配置格式迁移到新的标准化配置格式
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigMigration:
    """配置迁移工具类"""
    
    def __init__(self):
        self.migration_log = []
    
    def migrate_config(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        将旧配置迁移到新格式
        
        Args:
            old_config: 旧的配置字典
            
        Returns:
            新格式的配置字典
        """
        new_config = {
            "llm": {},
            "redundancy": {},
            "document": {},
            "embedding": {},
            "performance": {
                "enable_monitoring": True,
                "log_memory_usage": True,
                "log_execution_time": True
            }
        }
        
        # 迁移LLM配置
        self._migrate_llm_config(old_config, new_config)
        
        # 迁移冗余检测配置
        self._migrate_redundancy_config(old_config, new_config)
        
        # 迁移文档处理配置
        self._migrate_document_config(old_config, new_config)
        
        # 迁移嵌入配置
        self._migrate_embedding_config(old_config, new_config)
        
        return new_config
    
    def _migrate_llm_config(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """迁移LLM配置"""
        llm_mapping = {
            "api_key": ["llm.api_key", "openai.api_key", "api_key"],
            "base_url": ["llm.base_url", "openai.base_url", "base_url"],
            "model": ["llm.model", "openai.model", "model"],
            "embedding_model": ["llm.embedding_model", "openai.embedding_model", "embedding_model"],
            "max_tokens": ["llm.max_tokens", "openai.max_tokens", "max_tokens"],
            "temperature": ["llm.temperature", "openai.temperature", "temperature"]
        }
        
        for new_key, old_paths in llm_mapping.items():
            value = self._find_value_by_paths(old_config, old_paths)
            if value is not None:
                new_config["llm"][new_key] = value
                self.migration_log.append(f"迁移LLM配置: {new_key} = {value}")
        
        # 设置默认值
        defaults = {
            "max_tokens": 4000,
            "temperature": 0.7
        }
        
        for key, default_value in defaults.items():
            if key not in new_config["llm"]:
                new_config["llm"][key] = default_value
                self.migration_log.append(f"设置LLM默认值: {key} = {default_value}")
    
    def _migrate_redundancy_config(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """迁移冗余检测配置"""
        # 检查旧的冗余配置结构
        redundancy_old = old_config.get("redundancy_filter", old_config.get("redundancy", {}))
        
        # 确定使用的方法
        if redundancy_old.get("enable_enhanced_filter", False):
            # 使用embedding方法
            new_config["redundancy"]["method"] = "embedding"
            
            # 迁移embedding相关配置
            threshold = redundancy_old.get("threshold", 0.95)
            new_config["redundancy"]["threshold"] = threshold
            
            self.migration_log.append(f"迁移到embedding方法，阈值: {threshold}")
        else:
            # 使用simhash方法
            new_config["redundancy"]["method"] = "simhash"
            
            # 迁移simhash相关配置
            hamming_threshold = redundancy_old.get("hamming_threshold", 3)
            max_buffer_size = redundancy_old.get("max_buffer_size", 10000)
            
            new_config["redundancy"]["hamming_threshold"] = hamming_threshold
            new_config["redundancy"]["max_buffer_size"] = max_buffer_size
            
            self.migration_log.append(f"迁移到simhash方法，汉明距离阈值: {hamming_threshold}")
        
        # 迁移通用配置
        common_mapping = {
            "enable_logging": ["redundancy_filter.enable_logging", "redundancy.enable_logging"],
            "enable_progress": ["redundancy_filter.enable_progress", "redundancy.enable_progress"],
            "log_interval": ["redundancy_filter.log_interval", "redundancy.log_interval"]
        }
        
        for new_key, old_paths in common_mapping.items():
            value = self._find_value_by_paths(old_config, old_paths)
            if value is not None:
                new_config["redundancy"][new_key] = value
            else:
                # 设置默认值
                defaults = {
                    "enable_logging": True,
                    "enable_progress": True,
                    "log_interval": 100
                }
                if new_key in defaults:
                    new_config["redundancy"][new_key] = defaults[new_key]
    
    def _migrate_document_config(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """迁移文档处理配置"""
        document_mapping = {
            "allowed_types": ["document.allowed_types"],
            "enable_redundancy_filter": ["document.enable_redundancy_filter"],
            "sentence_level": ["document.sentence_level"],
            "min_sentence_length": ["document.min_sentence_length"],
            "redundancy_threshold": ["document.redundancy_threshold"]
        }
        
        for new_key, old_paths in document_mapping.items():
            value = self._find_value_by_paths(old_config, old_paths)
            if value is not None:
                new_config["document"][new_key] = value
                self.migration_log.append(f"迁移文档配置: {new_key} = {value}")
        
        # 设置默认值
        defaults = {
            "allowed_types": [".docx", ".json", ".jsonl", ".txt"],
            "enable_redundancy_filter": True,
            "sentence_level": True,
            "min_sentence_length": 10,
            "redundancy_threshold": 0.95
        }
        
        for key, default_value in defaults.items():
            if key not in new_config["document"]:
                new_config["document"][key] = default_value
                self.migration_log.append(f"设置文档默认值: {key} = {default_value}")
    
    def _migrate_embedding_config(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """迁移嵌入配置"""
        embedding_mapping = {
            "dimension": ["embedding.dimension", "embedding_dimension"],
            "batch_size": ["embedding.batch_size", "batch_size"]
        }
        
        for new_key, old_paths in embedding_mapping.items():
            value = self._find_value_by_paths(old_config, old_paths)
            if value is not None:
                new_config["embedding"][new_key] = value
                self.migration_log.append(f"迁移嵌入配置: {new_key} = {value}")
        
        # 设置默认值
        defaults = {
            "dimension": 1536,  # text-embedding-ada-002的维度
            "batch_size": 100
        }
        
        for key, default_value in defaults.items():
            if key not in new_config["embedding"]:
                new_config["embedding"][key] = default_value
                self.migration_log.append(f"设置嵌入默认值: {key} = {default_value}")
    
    def _find_value_by_paths(self, config: Dict[str, Any], paths: list) -> Any:
        """通过多个可能的路径查找配置值"""
        for path in paths:
            value = self._get_nested_value(config, path)
            if value is not None:
                return value
        return None
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """获取嵌套配置值"""
        keys = path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def save_migrated_config(self, new_config: Dict[str, Any], output_path: str):
        """保存迁移后的配置"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=2, ensure_ascii=False)
        
        self.migration_log.append(f"配置已保存到: {output_path}")
    
    def get_migration_log(self) -> list:
        """获取迁移日志"""
        return self.migration_log.copy()

def migrate_config_file(old_config_path: str, new_config_path: str) -> bool:
    """
    迁移配置文件
    
    Args:
        old_config_path: 旧配置文件路径
        new_config_path: 新配置文件路径
        
    Returns:
        是否迁移成功
    """
    try:
        # 读取旧配置
        with open(old_config_path, 'r', encoding='utf-8') as f:
            old_config = json.load(f)
        
        # 执行迁移
        migration = ConfigMigration()
        new_config = migration.migrate_config(old_config)
        
        # 保存新配置
        migration.save_migrated_config(new_config, new_config_path)
        
        # 打印迁移日志
        print("配置迁移完成！")
        print("迁移日志:")
        for log_entry in migration.get_migration_log():
            print(f"  - {log_entry}")
        
        return True
        
    except Exception as e:
        print(f"配置迁移失败: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("用法: python config_migration.py <旧配置文件> <新配置文件>")
        sys.exit(1)
    
    old_path = sys.argv[1]
    new_path = sys.argv[2]
    
    if not os.path.exists(old_path):
        print(f"旧配置文件不存在: {old_path}")
        sys.exit(1)
    
    success = migrate_config_file(old_path, new_path)
    sys.exit(0 if success else 1)