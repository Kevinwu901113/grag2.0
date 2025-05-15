import os
import json
import faiss
import numpy as np
import torch
import joblib
from transformers import BertTokenizer
from classifier.train_base_classifier import BERTClassifier
from llm.llm import LLMClient

def classify_query_bert(query: str, model_path: str, encoder_path: str, model_name="bert-base-chinese", cache_dir="./hf_models"):
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

    is_precise = float(torch.max(probs)) < 0.85 or "精确" in pred_label
    return pred_label, is_precise

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
    model_path_pt = os.path.join(work_dir, "query_classifier.pt")
    encoder_path = os.path.join(work_dir, "label_encoder.pkl")
    model_name = config.get("classifier", {}).get("bert_model", "bert-base-chinese")

    if not os.path.exists(model_path_pt):
        logger.error("未找到 BERT 分类器模型，请先训练。")
        return

    llm = LLMClient(config)
    chunks = load_chunks(work_dir)
    index, id_map = load_index(work_dir)
    cache_path = os.path.join(work_dir, "query_cache.jsonl")

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
            q_embed = llm.embed([query])[0]
            D, I = index.search(np.array([q_embed]).astype('float32'), k=5)
            retrieved = []
            for i in I[0]:
                chunk_id = id_map[str(i)]
                match = next((c['text'] for c in chunks if c['id'] == chunk_id), None)
                if match:
                    retrieved.append(match)
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
