import json
import networkx as nx
import re

def load_graph(graph_path):
    with open(graph_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 处理自定义格式的图数据，转换为NetworkX期望的格式
    if "nodes" in data and "edges" in data and "links" not in data:
        # 将edges键重命名为links以匹配NetworkX期望的格式
        data["links"] = data.pop("edges")
    # 明确指定edges参数为"links"以避免警告
    G = nx.node_link_graph(data, edges="links")
    return G

def extract_entity_names(graph):
    names = set()
    for node, attrs in graph.nodes(data=True):
        label = attrs.get("label", "")
        match = re.search(r'\|\s*([^|]+?)\s*\|', label)
        if match:
            names.add(match.group(1).strip())
    return names

def match_entities_in_query(query, entity_names):
    return [name for name in entity_names if name in query]

def extract_subgraph(graph, entities, depth=1):
    nodes = set()
    for entity in entities:
        if entity in graph:
            nodes.add(entity)
            for neighbor in nx.single_source_shortest_path_length(graph, entity, cutoff=depth):
                nodes.add(neighbor)
    return graph.subgraph(nodes).copy()

def summarize_subgraph(subgraph):
    summary = []
    for u, v, data in subgraph.edges(data=True):
        relation = data.get("relation", "关系")
        summary.append(f"{u} 与 {v} 存在关系：{relation}。")
    return "\n".join(summary)

if __name__ == "__main__":
    # 测试示例
    g = load_graph("./result/your_run_dir/graph.json")
    names = extract_entity_names(g)
    print("图谱实体样本：", list(names)[:10])
    matched = match_entities_in_query("党委和陈义妹的历史背景", names)
    print("匹配到实体：", matched)
    subg = extract_subgraph(g, matched)
    print(summarize_subgraph(subg))
