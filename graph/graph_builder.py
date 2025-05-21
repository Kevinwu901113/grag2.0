import os
import re
import json
import networkx as nx
from llm.llm import LLMClient

def load_chunks(chunk_path: str) -> list[dict]:
    with open(chunk_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_entities_and_relations(text: str, llm_client: LLMClient):
    """
    调用 LLM 抽取实体与三元组，返回列表
    """
    prompt = f"""
请从以下段落中抽取所有实体及它们之间的关系，严格按照格式输出，每行一个：

格式：实体A -[关系]-> 实体B

要求：
1. 保证每行都符合上述格式；
2. 不要输出解释或说明性文字；
3. 所有实体名称与关系用原语言保留（如中文）；
4. 如果找不到任何关系，请返回空字符串。

示例输出：
马云 -[创立]-> 阿里巴巴
张勇 -[担任]-> 首席执行官

正文如下：
{text}

请开始输出（每行一个三元组）：
""".strip()

    try:
        response = llm_client.generate(prompt)
        lines = response.strip().splitlines()
        triples = []

        for line in lines:
            line = line.strip()
            if not line or "->" not in line:
                continue

            match = re.match(r"(.+?)\s*-\[(.+?)\]->\s*(.+)", line)
            if match:
                head, relation, tail = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
                if head and relation and tail:
                    triples.append((head, relation, tail))
                continue

            print(f"[⚠️ 格式异常跳过] LLM输出: {line}")

        return triples

    except Exception as e:
        print(f"抽取失败: {e}")
        return []

def build_graph(chunks: list[dict], llm_client: LLMClient):
    G = nx.DiGraph()

    for chunk in chunks:
        text = chunk["text"]
        summary = chunk.get("summary", chunk["id"])
        topic_node_id = f"topic::{summary}"

        # 创建主题节点
        G.add_node(topic_node_id, type="topic", label=summary)

        # 抽取实体三元组
        triples = extract_entities_and_relations(text, llm_client)

        for head, relation, tail in triples:
            # 添加实体节点
            G.add_node(head, type="entity")
            G.add_node(tail, type="entity")

            # 添加三元组边
            G.add_edge(head, tail, relation=relation)

            # 把实体连接回主题
            G.add_edge(topic_node_id, head, relation="包含")
            G.add_edge(topic_node_id, tail, relation="包含")

    return G

def save_graph(graph: nx.DiGraph, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 保存 graph.json（自定义格式）
    nodes = [{"id": node, **graph.nodes[node]} for node in graph.nodes]
    edges = [{"source": u, "target": v, "relation": graph[u][v]["relation"]} for u, v in graph.edges]
    with open(os.path.join(output_dir, "graph.json"), "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)

    # 保存为 graphml（可用 Gephi / Cytoscape 查看）
    nx.write_graphml(graph, os.path.join(output_dir, "graph.graphml"))

def run_graph_construction(config: dict, work_dir: str, logger):
    chunk_path = os.path.join(work_dir, "chunks.json")
    output_dir = os.path.join(work_dir)

    logger.info(f"加载文本块: {chunk_path}")
    chunks = load_chunks(chunk_path)

    llm_client = LLMClient(config["llm"])

    logger.info("开始实体图构建...")
    G = build_graph(chunks, llm_client)

    logger.info(f"图构建完成，节点数: {len(G.nodes)}, 边数: {len(G.edges)}")
    save_graph(G, output_dir)
    logger.info(f"图已保存至 {output_dir}/graph.json 和 graph.graphml")
