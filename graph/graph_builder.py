import os
import json
import networkx as nx
from typing import List, Tuple
from llm.llm import LLMClient


def run_graph_construction(config: dict, work_dir: str, logger):
    chunk_path = os.path.join(work_dir, "chunks.json")
    if not os.path.exists(chunk_path):
        logger.error("未找到 chunks.json，无法构建图结构。请先运行文档处理模块。")
        return

    with open(chunk_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    llm = LLMClient(config)
    G = nx.DiGraph()

    for chunk in chunks:
        text = chunk['text']
        try:
            logger.info(f"处理 chunk {chunk['id']}...")
            entities = llm.extract_entities(text)
            relations = llm.extract_relations(text, entities)

            for ent, ent_type in entities:
                G.add_node(ent, type=ent_type)

            for src, tgt, rel in relations:
                G.add_edge(src, tgt, relation=rel)

        except Exception as e:
            logger.warning(f"处理 chunk {chunk['id']} 时出错: {str(e)}")
            logger.debug(f"原始文本: {text[:100]}...")

    graph_json_path = os.path.join(work_dir, "graph.json")
    graph_graphml_path = os.path.join(work_dir, "graph.graphml")

    try:
        nx.write_gml(G, graph_json_path)
    except Exception as e:
        logger.error(f"保存 graph.json 失败: {str(e)}")

    try:
        nx.write_graphml(G, graph_graphml_path)
    except Exception as e:
        logger.error(f"保存 graph.graphml 失败: {str(e)}")

    logger.info(f"图结构构建完成，共包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
    logger.info(f"已保存为 {graph_json_path} 和 {graph_graphml_path}")
