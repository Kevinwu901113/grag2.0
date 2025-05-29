#!/usr/bin/env python3
"""
测试多跳图推理与摘要融合功能
"""

import os
import sys
import networkx as nx
from typing import Set

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph.graph_subgraph_extractor import (
    extract_multi_hop_subgraph,
    find_intermediate_entities,
    generate_concise_graph_summary,
    get_multi_hop_retrieval_summary,
    extract_shortest_paths_summary
)
from graph.graph_utils import summarize_subgraph

def create_test_graph():
    """
    创建测试用的知识图谱，模拟复杂的人物/机构关系
    """
    graph = nx.Graph()
    
    # 添加人物实体
    people = ["张三", "李四", "王五", "赵六", "陈七"]
    for person in people:
        graph.add_node(person, type="entity")
    
    # 添加机构实体
    organizations = ["北京大学", "清华大学", "中科院", "华为公司", "阿里巴巴"]
    for org in organizations:
        graph.add_node(org, type="entity")
    
    # 添加概念实体
    concepts = ["人工智能", "机器学习", "深度学习", "自然语言处理", "计算机视觉"]
    for concept in concepts:
        graph.add_node(concept, type="entity")
    
    # 添加主题节点
    topics = ["AI研究项目", "学术合作", "技术转化", "人才培养", "产业应用"]
    for i, topic in enumerate(topics):
        graph.add_node(f"topic_{i}", type="topic", topic_id=i, label=topic)
    
    # 添加复杂的关系网络
    relations = [
        # 人物与机构关系
        ("张三", "北京大学", "任职于"),
        ("李四", "清华大学", "毕业于"),
        ("王五", "中科院", "研究员"),
        ("赵六", "华为公司", "工程师"),
        ("陈七", "阿里巴巴", "算法专家"),
        
        # 人物与概念关系
        ("张三", "人工智能", "研究"),
        ("李四", "机器学习", "专长"),
        ("王五", "深度学习", "开发"),
        ("赵六", "自然语言处理", "应用"),
        ("陈七", "计算机视觉", "优化"),
        
        # 机构与概念关系
        ("北京大学", "人工智能", "教学"),
        ("清华大学", "机器学习", "研究"),
        ("中科院", "深度学习", "创新"),
        ("华为公司", "自然语言处理", "产品化"),
        ("阿里巴巴", "计算机视觉", "商业化"),
        
        # 人物间的合作关系（通过中介实体）
        ("张三", "李四", "合作研究"),
        ("王五", "赵六", "技术交流"),
        ("李四", "陈七", "项目协作"),
        
        # 机构间关系
        ("北京大学", "清华大学", "学术合作"),
        ("中科院", "华为公司", "产学研合作"),
        ("清华大学", "阿里巴巴", "人才输送"),
        
        # 概念间关系
        ("人工智能", "机器学习", "包含"),
        ("机器学习", "深度学习", "包含"),
        ("深度学习", "自然语言处理", "应用于"),
        ("深度学习", "计算机视觉", "应用于"),
        
        # 主题连接
        ("topic_0", "张三", "参与"),
        ("topic_0", "人工智能", "涉及"),
        ("topic_1", "北京大学", "主导"),
        ("topic_1", "清华大学", "参与"),
        ("topic_2", "华为公司", "推动"),
        ("topic_2", "自然语言处理", "转化"),
        ("topic_3", "李四", "指导"),
        ("topic_3", "机器学习", "教授"),
        ("topic_4", "阿里巴巴", "应用"),
        ("topic_4", "计算机视觉", "商用"),
    ]
    
    for u, v, relation in relations:
        graph.add_edge(u, v, relation=relation)
    
    return graph

def test_multi_hop_subgraph_extraction():
    """
    测试多跳子图提取功能
    """
    print("=== 测试多跳子图提取 ===")
    
    graph = create_test_graph()
    
    # 测试单实体情况
    single_entities = {"张三"}
    single_subgraph = extract_multi_hop_subgraph(graph, single_entities, max_depth=2)
    print(f"单实体子图节点数: {len(single_subgraph.nodes)}, 边数: {len(single_subgraph.edges)}")
    
    # 测试多实体情况
    multi_entities = {"张三", "李四", "华为公司"}
    multi_subgraph = extract_multi_hop_subgraph(graph, multi_entities, max_depth=2)
    print(f"多实体子图节点数: {len(multi_subgraph.nodes)}, 边数: {len(multi_subgraph.edges)}")
    
    # 显示子图中的路径
    print("\n多实体子图中的关键路径:")
    for u, v, data in list(multi_subgraph.edges(data=True))[:5]:
        relation = data.get('relation', '关联')
        print(f"  {u} -[{relation}]-> {v}")
    
    print()

def test_intermediate_entities():
    """
    测试中介实体发现功能
    """
    print("=== 测试中介实体发现 ===")
    
    graph = create_test_graph()
    
    # 测试人物间的中介实体
    entities = {"张三", "赵六"}
    intermediate_info = find_intermediate_entities(graph, entities)
    
    print(f"发现{len(intermediate_info)}条中介路径:")
    for info in intermediate_info[:5]:
        source = info['source']
        intermediate = info['intermediate']
        target = info['target']
        relations = info['relations']
        print(f"  {source} -[{relations[0]}]-> {intermediate} -[{relations[1]}]-> {target}")
    
    print()

def test_concise_summary_generation():
    """
    测试简洁摘要生成功能
    """
    print("=== 测试简洁摘要生成 ===")
    
    graph = create_test_graph()
    
    # 测试不同实体组合的摘要
    test_cases = [
        {"张三", "北京大学"},
        {"张三", "李四", "人工智能"},
        {"华为公司", "阿里巴巴", "自然语言处理", "计算机视觉"},
        {"北京大学", "清华大学", "中科院"}
    ]
    
    for i, entities in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}: {', '.join(entities)}")
        
        subgraph = extract_multi_hop_subgraph(graph, entities, max_depth=2)
        
        # 生成简洁摘要
        concise_summary = generate_concise_graph_summary(subgraph, entities, max_summary_length=200)
        print(f"简洁摘要: {concise_summary}")
        
        # 对比传统摘要
        traditional_summary = summarize_subgraph(subgraph, max_length=200)
        print(f"传统摘要: {traditional_summary}")
        
        print(f"摘要长度: 简洁={len(concise_summary)}, 传统={len(traditional_summary)}")

def test_shortest_paths():
    """
    测试最短路径摘要功能
    """
    print("\n=== 测试最短路径摘要 ===")
    
    graph = create_test_graph()
    
    entities = {"张三", "赵六", "阿里巴巴"}
    path_summaries = extract_shortest_paths_summary(graph, entities, max_paths=5)
    
    print(f"发现{len(path_summaries)}条最短路径:")
    for path_info in path_summaries:
        source = path_info['source']
        target = path_info['target']
        length = path_info['length']
        description = path_info['description']
        print(f"  {source} -> {target} (长度{length}): {description}")

def test_complete_multi_hop_retrieval():
    """
    测试完整的多跳检索摘要功能
    """
    print("\n=== 测试完整多跳检索摘要 ===")
    
    graph = create_test_graph()
    
    # 模拟文档块
    chunks = [
        {"id": "doc1", "text": "张三在北京大学从事人工智能研究", "topic_id": 0},
        {"id": "doc2", "text": "李四是清华大学机器学习专家", "topic_id": 1},
        {"id": "doc3", "text": "华为公司在自然语言处理领域有重要应用", "topic_id": 2},
        {"id": "doc4", "text": "北京大学与清华大学开展学术合作", "topic_id": 1},
        {"id": "doc5", "text": "阿里巴巴在计算机视觉商业化方面领先", "topic_id": 4},
    ]
    
    # 测试不同查询场景
    test_queries = [
        {"张三", "李四"},  # 人物关系
        {"北京大学", "华为公司"},  # 机构关系
        {"人工智能", "自然语言处理", "计算机视觉"},  # 技术概念关系
        {"张三", "华为公司", "人工智能"}  # 混合关系
    ]
    
    for i, entities in enumerate(test_queries, 1):
        print(f"\n查询场景 {i}: {', '.join(entities)}")
        
        summary = get_multi_hop_retrieval_summary(
            graph, entities, chunks, max_summary_length=300
        )
        
        print(f"多跳检索摘要: {summary}")
        print(f"摘要长度: {len(summary)}")

def main():
    """
    主测试函数
    """
    print("开始测试多跳图推理与摘要融合功能\n")
    
    try:
        test_multi_hop_subgraph_extraction()
        test_intermediate_entities()
        test_concise_summary_generation()
        test_shortest_paths()
        test_complete_multi_hop_retrieval()
        
        print("\n✅ 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()