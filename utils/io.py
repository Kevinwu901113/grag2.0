#!/usr/bin/env python3
"""
统一的文件I/O工具函数
整合项目中重复出现的文件加载、保存操作
"""

import os
import json
import faiss
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
from networkx.readwrite import json_graph


def load_json(file_path: str, default=None, encoding='utf-8'):
    """
    统一的JSON文件加载函数
    
    Args:
        file_path: JSON文件路径
        default: 加载失败时的默认返回值
        encoding: 文件编码
        
    Returns:
        加载的JSON数据或默认值
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        print(f"❌ 加载JSON文件失败 {file_path}: {e}")
        return default if default is not None else []


def save_json(data: Any, file_path: str, encoding='utf-8', ensure_ascii=False, indent=2):
    """
    统一的JSON文件保存函数
    
    Args:
        data: 要保存的数据
        file_path: 保存路径
        encoding: 文件编码
        ensure_ascii: 是否确保ASCII编码
        indent: 缩进空格数
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
    except (IOError, OSError) as e:
        print(f"❌ 保存JSON文件失败 {file_path}: {e}")
        raise


def load_chunks(work_dir: str) -> List[Dict[str, Any]]:
    """
    统一的文档块加载函数
    
    Args:
        work_dir: 工作目录
        
    Returns:
        文档块列表
    """
    # 优先尝试加载enhanced_chunks.json
    enhanced_chunk_path = os.path.join(work_dir, "enhanced_chunks.json")
    if file_exists(enhanced_chunk_path):
        return load_json(enhanced_chunk_path, default=[])
    
    # 如果enhanced_chunks.json不存在，尝试加载chunks.json
    chunk_path = os.path.join(work_dir, "chunks.json")
    return load_json(chunk_path, default=[])


def save_chunks(chunks: List[Dict[str, Any]], work_dir: str):
    """
    统一的文档块保存函数
    
    Args:
        chunks: 文档块列表
        work_dir: 工作目录
    """
    chunk_path = os.path.join(work_dir, "chunks.json")
    save_json(chunks, chunk_path)


def load_vector_index(work_dir: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    统一的向量索引加载函数
    
    Args:
        work_dir: 工作目录
        
    Returns:
        (faiss索引对象, ID映射字典)
    """
    try:
        index_path = os.path.join(work_dir, "vector.index")
        mapping_path = os.path.join(work_dir, "embedding_map.json")
        
        index = faiss.read_index(index_path)
        id_map = load_json(mapping_path, default={})
        
        return index, id_map
    except (FileNotFoundError, IOError) as e:
        print(f"❌ 加载向量索引失败: {e}")
        return None, {}


def save_vector_index(index: Any, id_map: Dict[str, Any], work_dir: str):
    """
    统一的向量索引保存函数
    
    Args:
        index: faiss索引对象
        id_map: ID映射字典
        work_dir: 工作目录
    """
    try:
        os.makedirs(work_dir, exist_ok=True)
        index_path = os.path.join(work_dir, "vector.index")
        mapping_path = os.path.join(work_dir, "embedding_map.json")
        
        faiss.write_index(index, index_path)
        save_json(id_map, mapping_path)
    except (IOError, OSError) as e:
        print(f"❌ 保存向量索引失败: {e}")
        raise


def load_graph(work_dir: str) -> Optional[nx.Graph]:
    """
    统一的图谱加载函数
    
    Args:
        work_dir: 工作目录
        
    Returns:
        NetworkX图对象或None
    """
    graph_path = os.path.join(work_dir, "graph.json")
    try:
        data = load_json(graph_path)
        if data:
            return json_graph.node_link_graph(data)
        return None
    except Exception as e:
        print(f"❌ 加载图谱失败: {e}")
        return None


def save_graph(graph: nx.Graph, work_dir: str):
    """
    统一的图谱保存函数
    
    Args:
        graph: NetworkX图对象
        work_dir: 工作目录
    """
    try:
        os.makedirs(work_dir, exist_ok=True)
        
        # 保存JSON格式
        graph_json = json_graph.node_link_data(graph)
        graph_path = os.path.join(work_dir, "graph.json")
        save_json(graph_json, graph_path)
        
        # 保存GraphML格式供可视化（需要清理不支持的数据类型）
        graphml_path = os.path.join(work_dir, "graph.graphml")
        _save_graphml_safe(graph, graphml_path)
    except Exception as e:
        print(f"❌ 保存图谱失败: {e}")
        raise


def _save_graphml_safe(graph: nx.Graph, graphml_path: str):
    """
    安全保存GraphML格式，清理不支持的数据类型
    
    Args:
        graph: NetworkX图对象
        graphml_path: GraphML文件路径
    """
    # 创建图的副本以避免修改原图
    graph_copy = graph.copy()
    
    # 清理节点属性中的不支持类型
    for node, data in graph_copy.nodes(data=True):
        for key, value in list(data.items()):
            if isinstance(value, (list, dict, type)):
                # 将list转换为字符串，删除dict和type类型
                if isinstance(value, list):
                    data[key] = str(value)
                else:
                    del data[key]
    
    # 清理边属性中的不支持类型
    for u, v, data in graph_copy.edges(data=True):
        for key, value in list(data.items()):
            if isinstance(value, (list, dict, type)):
                # 将list转换为字符串，删除dict和type类型
                if isinstance(value, list):
                    data[key] = str(value)
                else:
                    del data[key]
    
    # 保存清理后的图
    nx.write_graphml(graph_copy, graphml_path)


def ensure_dir_exists(dir_path: str):
    """
    确保目录存在
    
    Args:
        dir_path: 目录路径
    """
    os.makedirs(dir_path, exist_ok=True)


def file_exists(file_path: str) -> bool:
    """
    检查文件是否存在
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件是否存在
    """
    return os.path.exists(file_path) and os.path.isfile(file_path)


def load_entity_vector_index(work_dir: str) -> Tuple[Optional[Any], Dict[str, Any], Dict[str, Any]]:
    """
    加载实体向量索引
    
    Args:
        work_dir: 工作目录
        
    Returns:
        (faiss索引对象, 实体ID映射字典, 实体元数据字典)
    """
    try:
        index_path = os.path.join(work_dir, "entity_vector.index")
        mapping_path = os.path.join(work_dir, "entity_embedding_map.json")
        metadata_path = os.path.join(work_dir, "entity_metadata.json")
        
        index = faiss.read_index(index_path)
        id_map = load_json(mapping_path, default={})
        metadata = load_json(metadata_path, default={})
        
        return index, id_map, metadata
    except (FileNotFoundError, IOError) as e:
        print(f"❌ 加载实体向量索引失败: {e}")
        return None, {}, {}


def save_entity_vector_index(index: Any, id_map: Dict[str, Any], metadata: Dict[str, Any], work_dir: str):
    """
    保存实体向量索引
    
    Args:
        index: faiss索引对象
        id_map: 实体ID映射字典
        metadata: 实体元数据字典
        work_dir: 工作目录
    """
    try:
        os.makedirs(work_dir, exist_ok=True)
        index_path = os.path.join(work_dir, "entity_vector.index")
        mapping_path = os.path.join(work_dir, "entity_embedding_map.json")
        metadata_path = os.path.join(work_dir, "entity_metadata.json")
        
        faiss.write_index(index, index_path)
        save_json(id_map, mapping_path)
        save_json(metadata, metadata_path)
    except (IOError, OSError) as e:
        print(f"❌ 保存实体向量索引失败: {e}")
        raise