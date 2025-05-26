import os
import re
import json
import networkx as nx
from llm.llm import LLMClient
from networkx.readwrite import json_graph

def load_chunks(chunk_path: str) -> list[dict]:
    with open(chunk_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_entities_and_relations(text: str, llm_client: LLMClient):
    """
    调用 LLM 抽取实体与三元组，返回列表
    """
    prompt = f"""
请从以下段落中抽取所有实体及它们之间的关系，严格按照以下格式输出，每行一个：

格式：实体A -[关系]-> 实体B

格式说明：
1. 实体A和实体B是文本中提到的人物、组织、地点、概念等
2. 关系是连接两个实体的动词或描述词
3. 必须使用 -[关系]-> 这种精确格式，包括方括号和箭头
4. 每个三元组占一行，不要有多余文字

要求：
1. 不要输出任何解释说明文字
2. 每行必须严格遵循上述格式
3. 如果找不到关系，直接跳过，不要输出不完整的行
4. 确保实体和关系都有实际内容，不要输出空实体或空关系

正确示例：
马云 -[创立]-> 阿里巴巴
张勇 -[担任]-> 首席执行官
中国 -[包含]-> 北京
学生 -[参加]-> 考试

错误示例（请勿这样输出）：
马云创立了阿里巴巴  （错误：没有使用规定格式）
马云 - 创立 -> 阿里巴巴  （错误：关系没有用方括号括起来）
马云-[创立]->阿里巴巴  （错误：缺少空格）
-[位于]->北京  （错误：缺少实体A）

正文如下：
{text}

请开始输出实体关系三元组（严格按照格式）：
""".strip()

    try:
        response = llm_client.generate(prompt)
        lines = response.strip().splitlines()
        triples = []

        for line in lines:
            line = line.strip()
            if not line:
                continue  # 跳过空行
                
            # 尝试修复常见的格式问题
            # 1. 修复缺少空格的情况
            line = re.sub(r'(\S)-\[', r'\1 -[', line)
            line = re.sub(r'\]-(\S)', r']-> \1', line)
            
            # 2. 修复箭头格式问题
            if "-[" in line and not "->" in line:
                line = line.replace("]", "]->")
            
            # 3. 修复缺少方括号的情况
            if "->" in line and not "-[" in line and "-" in line:
                parts = line.split("-")
                if len(parts) >= 2 and "->" in parts[1]:
                    relation = parts[1].split("->")[0].strip()
                    line = f"{parts[0].strip()} -[{relation}]-> {parts[1].split('->', 1)[1].strip()}"
            
            # 跳过仍然无效的行
            if not line or "-[" not in line or "->" not in line:
                continue
                
            # 使用更灵活的正则表达式匹配
            match = re.match(r"(.+?)\s*-\[(.+?)\]->\s*(.+)", line)
            if match:
                h, r, t = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
                # 过滤空关系或空实体
                if h and r and t:
                    triples.append((h, r, t))
                else:
                    print(f"[⚠️ 跳过空值实体或关系]: {line}")
            else:
                # 尝试更宽松的匹配
                alt_match = re.search(r"([^-\[\]]+)\s*-\s*\[?([^\[\]]+)\]?\s*-?>?\s*([^-\[\]]+)", line)
                if alt_match:
                    h, r, t = alt_match.group(1).strip(), alt_match.group(2).strip(), alt_match.group(3).strip()
                    if h and r and t:
                        print(f"[ℹ️ 修复格式]: {line} -> {h} -[{r}]-> {t}")
                        triples.append((h, r, t))
                    else:
                        print(f"[⚠️ 跳过空值实体或关系]: {line}")
                else:
                    print(f"[⚠️ 跳过格式异常]: {line}")

        return triples
    except (ValueError, json.JSONDecodeError, KeyError) as e:
        print(f"❌ 抽取失败: {e}")
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

    # ✅ 使用 node-link 格式保存 JSON
    graph_json = json_graph.node_link_data(graph)
    with open(os.path.join(output_dir, "graph.json"), "w", encoding="utf-8") as f:
        json.dump(graph_json, f, ensure_ascii=False, indent=2)

    # ✅ 可选保存 GraphML 供可视化
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
