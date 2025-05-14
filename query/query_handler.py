import os
import json
import faiss
import numpy as np
from query.query_classifier import classify_query, predict_precise_need
from llm.llm import LLMClient


def load_chunks(work_dir):
    path = os.path.join(work_dir, "chunks.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_index(work_dir):
    index_path = os.path.join(work_dir, "vector.index")
    mapping_path = os.path.join(work_dir, "embedding_map.json")
    index = faiss.read_index(index_path)
    with open(mapping_path, 'r', encoding='utf-8') as f:
        id_map = json.load(f)
    return index, id_map


def run_query_loop(config: dict, work_dir: str, logger):
    model_path = os.path.join(work_dir, "query_classifier.pkl")
    if not os.path.exists(model_path):
        logger.error("分类器模型未找到，请先训练。")
        return

    llm = LLMClient(config)
    chunks = load_chunks(work_dir)
    index, id_map = load_index(work_dir)
    cache_path = os.path.join(work_dir, "query_cache.jsonl")

    while True:
        query = input("请输入查询（输入exit退出）：")
        if query.strip().lower() == "exit":
            break

        mode = classify_query(query, model_path)
        precise = predict_precise_need(query, model_path)
        logger.info(f"当前 query 分类为: {mode}，是否精确: {precise}")

        response = ""
        if mode == "norag" and not precise:
            response = llm.generate("请根据图结构信息回答以下问题：\n" + query)
        else:
            q_embed = llm.embed([query])[0]
            D, I = index.search(np.array([q_embed]).astype('float32'), k=5)
            retrieved = [chunks[int(id_map[str(i)])]['text'] for i in I[0]]
            context = "\n".join(retrieved)
            if precise:
                prompt = f"以下是相关文本块：\n{context}\n\n请精确回答：{query}"
            else:
                prompt = f"以下是相关文本块和图信息摘要：\n{context}\n\n请结合上下文回答：{query}"
            response = llm.generate(prompt)

        print("\n[回答]:", response.strip(), "\n")

        if config.get("query", {}).get("cache_enabled", True):
            with open(cache_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"query": query, "mode": mode, "precise": precise, "response": response.strip()}, ensure_ascii=False) + "\n")
