import os
import json
import faiss
import numpy as np
import torch
import joblib
from transformers import BertTokenizer
from query.theme_matcher import ThemeMatcher
from classifier.train_base_classifier import BERTClassifier
from llm.llm import LLMClient
from graph.graph_utils import load_graph, extract_entity_names, match_entities_in_query, extract_subgraph, summarize_subgraph

def classify_query_bert(query: str, model_path: str, encoder_path: str, model_name="bert-base-chinese", cache_dir="./hf_models"):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        label_encoder = joblib.load(encoder_path)

        model = BERTClassifier(model_name=model_name, num_labels=len(label_encoder.classes_))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = label_encoder.inverse_transform([pred_idx])[0]

        is_precise = pred_label == "hybrid_precise"
        return pred_label, is_precise
    except Exception as e:
        print(f"❌ 分类器推理失败: {e}")
        return "norag", False

def load_chunks(work_dir):
    try:
        path = os.path.join(work_dir, "chunks.json")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载文档块失败: {e}")
        return []

def load_index(work_dir):
    try:
        index_path = os.path.join(work_dir, "vector.index")
        mapping_path = os.path.join(work_dir, "embedding_map.json")
        index = faiss.read_index(index_path)
        with open(mapping_path, 'r', encoding='utf-8') as f:
            id_map = json.load(f)
        return index, id_map
    except Exception as e:
        print(f"❌ 加载索引失败: {e}")
        return None, {}

def run_query_loop(config: dict, work_dir: str, logger):
    # 加载分类器模型路径，自动判断是否存在 fine_tuned 目录
    model_path_pt = os.path.join(work_dir, "query_classifier.pt")
    encoder_path = os.path.join(work_dir, "label_encoder.pkl")

    # 优先查找 fine_tuned 下的模型
    fine_tuned_dir = os.path.join(work_dir, "fine_tuned")
    fine_model_path = os.path.join(fine_tuned_dir, "query_classifier.pt")
    fine_encoder_path = os.path.join(fine_tuned_dir, "label_encoder.pkl")

    if os.path.exists(fine_model_path) and os.path.exists(fine_encoder_path):
        model_path_pt = fine_model_path
        encoder_path = fine_encoder_path
    model_name = config.get("classifier", {}).get("bert_model", "bert-base-chinese")

    if not os.path.exists(model_path_pt):
        logger.error("未找到 BERT 分类器模型，请先训练。")
        return

    llm = LLMClient(config)
    chunks = load_chunks(work_dir)
    index, id_map = load_index(work_dir)
    if index is None:
        logger.error("向量索引未正确加载，终止查询流程。")
        return

    cache_path = os.path.join(work_dir, "query_cache.jsonl")
    graph_path = os.path.join(work_dir, "graph.json")
    graph = load_graph(graph_path) if os.path.exists(graph_path) else None
    entity_names = extract_entity_names(graph) if graph else set()

    while True:
        query = input("请输入查询（输入exit退出）：")
        if query.strip().lower() == "exit":
            break

        mode, precise = classify_query_bert(query, model_path_pt, encoder_path, model_name)
        logger.info(f"当前 query 分类为: {mode}，是否精确: {precise}")

        response = ""
        if mode == "norag" and not precise:
            response = llm.generate("请根据图结构信息回答以下问题：\n" + query)
        else:
            try:
                if "matcher" not in locals():
                    matcher = ThemeMatcher(chunks)  # chunks.json 中含 summary 字段

                matches = matcher.match(query, top_k=3)
                retrieved = []
                
                # 打印召回的文档块详细信息
                print("\n[召回文档详情]")
                print("-" * 50)
                
                for i, match in enumerate(matches):
                    topic_id = match["node_id"]
                    similarity = match.get("similarity", 0.0)
                    topic_title = match.get("title", "无标题")
                    match_text = next((c['text'] for c in chunks if c['id'] == topic_id), None)
                    
                    print(f"文档块 #{i+1}:")
                    print(f"  ID: {topic_id}")
                    print(f"  相似度: {similarity:.4f}")
                    print(f"  主题: {topic_title}")
                    print(f"  内容: {match_text[:100]}..." if match_text and len(match_text) > 100 else f"  内容: {match_text}")
                    print("-" * 50)
                    
                    if match_text:
                        retrieved.append(match_text)
                        
                if not retrieved:
                    logger.warning("未检索到任何相关文本块。")
                    context = "（未检索到相关文本内容）"
                    print("未检索到任何相关文本块。")
                else:
                    logger.info(f"检索到文本块数量: {len(retrieved)}")
                    context = "\n".join(retrieved)

                # 图谱摘要
                graph_summary = ""
                if graph:
                    try:
                        matched_entities = match_entities_in_query(query, entity_names)
                        subgraph = extract_subgraph(graph, matched_entities)
                        graph_summary = summarize_subgraph(subgraph)
                        
                        # 打印图谱召回信息
                        print("\n[图谱召回详情]")
                        print("-" * 50)
                        print(f"命中实体数量: {len(matched_entities)}")
                        if matched_entities:
                            print("命中实体列表:")
                            for i, entity in enumerate(matched_entities):
                                print(f"  {i+1}. {entity}")
                        print("-" * 50)
                        
                        logger.info(f"图谱命中实体数量: {len(matched_entities)}")
                    except Exception as ge:
                        logger.warning(f"图谱摘要生成失败: {ge}")

                if precise:
                    prompt = f"以下是相关文本块：\n{context}\n\n请精确回答：{query}"
                else:
                    prompt = f"以下是相关文本块和图信息摘要：\n{context}\n\n图谱摘要：\n{graph_summary}\n\n请结合上下文回答：{query}"

                response = llm.generate(prompt)
            except Exception as e:
                logger.error(f"查询处理失败: {e}")
                response = "对不起，查询处理过程中发生错误。"

        print("\n[回答]:", response.strip(), "\n")

        if config.get("query", {}).get("cache_enabled", True):
            try:
                with open(cache_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"query": query, "mode": mode, "precise": precise, "response": response.strip()}, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"⚠️ 缓存写入失败: {e}")
