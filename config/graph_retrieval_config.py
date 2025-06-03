#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图谱检索配置文件
注意：此文件已弃用，所有配置现在都在主配置文件 config.yaml 中统一管理
为了向后兼容，保留此类但从主配置文件读取配置
"""

import os
import sys
import yaml
from typing import Dict, Any

class GraphRetrievalConfig:
    """图谱检索配置类 - 从主配置文件读取配置"""
    
    _config_cache = None
    
    @classmethod
    def _load_main_config(cls) -> Dict[str, Any]:
        """加载主配置文件"""
        if cls._config_cache is not None:
            return cls._config_cache
            
        # 查找配置文件
        config_paths = [
            "config.yaml",
            "../config.yaml",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        cls._config_cache = yaml.safe_load(f)
                        return cls._config_cache
                except Exception as e:
                    print(f"警告：无法加载配置文件 {config_path}: {e}")
                    continue
        
        # 如果无法加载配置文件，返回默认配置
        print("警告：无法找到主配置文件，使用默认配置")
        cls._config_cache = cls._get_default_config()
        return cls._config_cache
    
    @classmethod
    def _get_default_config(cls) -> Dict[str, Any]:
        """获取默认配置（向后兼容）"""
        return {
            'graph_retrieval': {
                'entity_matching': {
                    'enable_fuzzy_match': True,
                    'enable_partial_match': True,
                    'min_similarity': 0.6,
                    'max_fuzzy_results': 20,
                    'min_token_length': 2,
                    'max_length_diff': 3,
                },
                'synonym_map': {
                    '深圳': ['深圳市', '鹏城', 'SZ'],
                    '宝安': ['宝安区', '宝安县'],
                    '村': ['村委会', '村民委员会', '行政村'],
                    '区': ['区政府', '行政区', '市辖区'],
                    '市': ['市政府', '地级市', '县级市'],
                    '县': ['县政府', '县级市'],
                    '镇': ['镇政府', '乡镇', '建制镇'],
                    '街道': ['街道办', '街道办事处'],
                    '社区': ['社区居委会', '居委会', '社区委员会'],
                    '公司': ['有限公司', '股份公司', '企业'],
                    '学校': ['小学', '中学', '学院', '大学'],
                    '医院': ['卫生院', '诊所', '医疗中心'],
                    '银行': ['信用社', '金融机构'],
                    '公园': ['广场', '绿地', '景区'],
                    '工厂': ['制造厂', '生产基地', '工业园']
                },
                'graph_retrieval': {
                    'max_hop_depth': 2,
                    'max_summary_length': 400,
                    'max_intermediate_paths': 5,
                    'entity_weight_factor': 1.0,
                    'enable_pagerank': True,
                    'pagerank_alpha': 0.85,
                },
                'performance': {
                    'max_entities_for_fuzzy': 50,
                    'fuzzy_match_timeout': 1.0,
                    'enable_caching': True,
                    'cache_size': 1000,
                },
                'debug': {
                    'enable_debug_print': True,
                    'log_matching_details': False,
                    'performance_monitoring': True,
                }
            }
        }
    
    @classmethod
    def get_entity_matching_config(cls):
        """获取实体匹配配置"""
        config = cls._load_main_config()
        return config.get('graph_retrieval', {}).get('entity_matching', {})
    
    @classmethod
    def get_synonym_map(cls):
        """获取同义词映射"""
        config = cls._load_main_config()
        return config.get('graph_retrieval', {}).get('synonym_map', {})
    
    @classmethod
    def get_graph_retrieval_config(cls):
        """获取图谱检索配置"""
        config = cls._load_main_config()
        return config.get('graph_retrieval', {}).get('graph_retrieval', {})
    
    @classmethod
    def get_performance_config(cls):
        """获取性能配置"""
        config = cls._load_main_config()
        return config.get('graph_retrieval', {}).get('performance', {})
    
    @classmethod
    def get_debug_config(cls):
        """获取调试配置"""
        config = cls._load_main_config()
        return config.get('graph_retrieval', {}).get('debug', {})
    
    @classmethod
    def update_config(cls, section, key, value):
        """更新配置项（注意：这只会更新内存中的缓存，不会写入文件）"""
        config = cls._load_main_config()
        graph_config = config.setdefault('graph_retrieval', {})
        
        if section in graph_config:
            graph_config[section][key] = value
        else:
            raise ValueError(f"Unknown config section: {section}")
    
    @classmethod
    def get_all_config(cls):
        """获取所有配置"""
        config = cls._load_main_config()
        graph_config = config.get('graph_retrieval', {})
        return {
            'entity_matching': graph_config.get('entity_matching', {}),
            'synonym_map': graph_config.get('synonym_map', {}),
            'graph_retrieval': graph_config.get('graph_retrieval', {}),
            'performance': graph_config.get('performance', {}),
            'debug': graph_config.get('debug', {})
        }
    
    # 为了向后兼容，保留类属性访问方式
    @property
    @classmethod
    def ENTITY_MATCHING(cls):
        return cls.get_entity_matching_config()
    
    @property
    @classmethod
    def SYNONYM_MAP(cls):
        return cls.get_synonym_map()
    
    @property
    @classmethod
    def GRAPH_RETRIEVAL(cls):
        return cls.get_graph_retrieval_config()
    
    @property
    @classmethod
    def PERFORMANCE(cls):
        return cls.get_performance_config()
    
    @property
    @classmethod
    def DEBUG(cls):
        return cls.get_debug_config()