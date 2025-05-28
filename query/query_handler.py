import os
import json
import faiss
import numpy as np
import torch
import joblib
from transformers import BertTokenizer
from query.optimized_theme_matcher import ThemeMatcher
from query.reranker import SimpleReranker, LLMReranker
from classifier.train_base_classifier import BERTClassifier
from llm.llm import LLMClient
from graph.graph_utils import load_graph, extract_entity_names, match_entities_in_query, extract_subgraph, summarize_subgraph

def classify_query_bert(query: str, model, tokenizer, label_encoder):
    """使用预加载模型的BERT分类器"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = label_encoder.inverse_transform([pred_idx])[0]

        is_precise = pred_label == "hybrid_precise"
        return pred_label, is_precise
    except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"❌ 分类器推理失败: {e}")
        return "hybrid_imprecise", False  # 暂时关闭norag选项

def load_chunks(work_dir):
    try:
        path = os.path.join(work_dir, "chunks.json")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
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
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        print(f"❌ 加载索引失败: {e}")
        return None, {}

def direct_vector_search(query: str, index, id_map: dict, chunks: list, config: dict, top_k: int = 5) -> list:
    """
    直接使用Faiss向量索引进行精确查询，跳过主题摘要
    
    Args:
        query: 查询文本
        index: Faiss索引
        id_map: ID映射字典
        chunks: 文档块列表
        config: 配置字典
        top_k: 返回的最大结果数量
        
    Returns:
        检索到的候选结果列表（包含text和similarity字段）
    """
    from utils.model_cache import model_cache
    
    try:
        # 获取嵌入模型
        embedding_config = config.get("embedding", {})
        model_mapping = {
            "bge-m3": "BAAI/bge-m3",
            "text2vec": "shibing624/text2vec-base-chinese", 
            "text-embedding-ada-002": "all-MiniLM-L6-v2",
            "all-MiniLM-L6-v2": "all-MiniLM-L6-v2"
        }
        
        model_name = embedding_config.get("model_name", "bge-m3")
        actual_model_name = model_mapping.get(model_name, model_name)
        
        model = model_cache.get_sentence_transformer(actual_model_name)
        if model is None:
            print("❌ 无法加载嵌入模型进行直接向量搜索")
            return []
        
        # 编码查询
        query_emb = model.encode([query])
        
        # 搜索最相似的向量
        scores, indices = index.search(query_emb, top_k)
        
        candidates = []
        print("\n[直接向量检索详情]")
        print("-" * 50)
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # Faiss返回-1表示无效结果
                continue
                
            # 从ID映射中获取原始chunk ID
            chunk_id = id_map.get(str(idx))
            if chunk_id is None:
                continue
                
            # 查找对应的文档块
            chunk_text = next((c['text'] for c in chunks if c['id'] == chunk_id), None)
            if chunk_text:
                candidate = {
                    'text': chunk_text,
                    'similarity': float(score),
                    'id': chunk_id,
                    'source': 'direct_vector'
                }
                candidates.append(candidate)
                
                print(f"文档块 #{i+1}:")
                print(f"  ID: {chunk_id}")
                print(f"  相似度: {score:.4f}")
                print(f"  内容: {chunk_text[:100]}..." if len(chunk_text) > 100 else f"  内容: {chunk_text}")
                print("-" * 50)
        
        return candidates
        
    except Exception as e:
        print(f"❌ 直接向量搜索失败: {e}")
        return []

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

    # 初始化重排序器
    rerank_config = config.get("rerank", {})
    use_rerank = rerank_config.get("enabled", True)
    rerank_method = rerank_config.get("method", "simple")  # simple 或 llm
    
    if use_rerank:
        if rerank_method == "llm":
            reranker = LLMReranker(llm, rerank_config)
        else:
            reranker = SimpleReranker(rerank_config)
        logger.info(f"启用重排序机制: {rerank_method}")

    cache_path = os.path.join(work_dir, "query_cache.jsonl")
    graph_path = os.path.join(work_dir, "graph.json")
    graph = load_graph(graph_path) if os.path.exists(graph_path) else None
    entity_names = extract_entity_names(graph) if graph else set()
    
    # 预加载BERT分类器模型以改善用户体验
    logger.info("正在预加载BERT分类器模型...")
    from utils.model_cache import model_cache
    
    # 预加载标签编码器
    label_encoder = model_cache.get_label_encoder(encoder_path)
    if label_encoder is None:
        logger.error(f"❌ 无法加载标签编码器: {encoder_path}")
        return
    
    # 预加载tokenizer
    cache_dir = config.get("classifier", {}).get("cache_dir", None)
    tokenizer = model_cache.get_tokenizer(model_name, cache_dir)
    if tokenizer is None:
        logger.error(f"❌ 无法加载tokenizer: {model_name}")
        return
    
    # 预加载BERT模型
    bert_model = model_cache.get_bert_model(model_path_pt, len(label_encoder.classes_), model_name, cache_dir)
    if bert_model is None:
        logger.error(f"❌ 无法加载BERT模型: {model_path_pt}")
        return
    
    logger.info("BERT分类器模型预加载完成")
    
    # 预加载主题匹配器以改善用户体验
    logger.info("正在预加载主题匹配器...")
    matcher = ThemeMatcher(chunks, config)
    logger.info("主题匹配器预加载完成")

    while True:
        query = input("请输入查询（输入exit退出）：")
        if query.strip().lower() == "exit":
            break

        # 初始分类
        original_mode, original_precise = classify_query_bert(query, bert_model, tokenizer, label_encoder)
        logger.info(f"初始分类结果: {original_mode}，是否精确: {original_precise}")
        
        # 应用二次判断增强
        from query.query_enhancer import enhance_query_classification
        mode, precise = enhance_query_classification(query, original_mode, original_precise)
        
        # 如果分类结果被修改，记录日志
        if mode != original_mode or precise != original_precise:
            logger.info(f"二次判断后分类为: {mode}，是否精确: {precise}")
        else:
            logger.info(f"当前 query 分类为: {mode}，是否精确: {precise}")

        response = ""
        # if mode == "norag" and not precise:
        #     response = llm.generate("请根据图结构信息回答以下问题：\n" + query)
        # else:
        if True:  # 暂时关闭norag选项
            try:
                candidates = []
                
                # 对精确查询，优先使用直接向量检索
                if precise and index is not None:
                    candidates = direct_vector_search(query, index, id_map, chunks, config, top_k=10)
                else:
                    # 使用预加载的主题匹配器
                    matches = matcher.match(query, top_k=10, min_score=0.3)
                    
                    # 转换为候选格式
                    for match in matches:
                        topic_id = match["node_id"]
                        similarity = match.get("similarity", 0.0)
                        topic_title = match.get("title", "无标题")
                        match_text = next((c['text'] for c in chunks if c['id'] == topic_id), None)
                        
                        if match_text:
                            candidate = {
                                'text': match_text,
                                'similarity': similarity,
                                'id': topic_id,
                                'title': topic_title,
                                'source': 'theme_matcher'
                            }
                            candidates.append(candidate)
                
                # 应用重排序
                if use_rerank and candidates:
                    logger.info(f"重排序前候选数量: {len(candidates)}")
                    candidates = reranker.rerank(query, candidates, top_k=5)
                    logger.info(f"重排序后候选数量: {len(candidates)}")
                    
                    print("\n[重排序后结果]")
                    print("-" * 50)
                    for i, candidate in enumerate(candidates):
                        print(f"文档块 #{i+1}:")
                        print(f"  ID: {candidate.get('id', 'N/A')}")
                        print(f"  原始相似度: {candidate.get('similarity', 0.0):.4f}")
                        if 'rerank_score' in candidate:
                            print(f"  重排序分数: {candidate['rerank_score']:.4f}")
                        print(f"  内容: {candidate['text'][:100]}..." if len(candidate['text']) > 100 else f"  内容: {candidate['text']}")
                        print("-" * 50)
                else:
                    # 不使用重排序，直接截取前5个
                    candidates = candidates[:5]
                    
                    print("\n[召回文档详情]")
                    print("-" * 50)
                    for i, candidate in enumerate(candidates):
                        print(f"文档块 #{i+1}:")
                        print(f"  ID: {candidate.get('id', 'N/A')}")
                        print(f"  相似度: {candidate.get('similarity', 0.0):.4f}")
                        print(f"  内容: {candidate['text'][:100]}..." if len(candidate['text']) > 100 else f"  内容: {candidate['text']}")
                        print("-" * 50)
                
                # 提取文本用于生成回答
                retrieved = [c['text'] for c in candidates]
                        
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
                    except (KeyError, ValueError, TypeError) as ge:
                        logger.warning(f"图谱摘要生成失败: {ge}")

                if precise:
                    prompt = f"以下是相关文本块：\n{context}\n\n请精确回答：{query}"
                else:
                    prompt = f"以下是相关文本块和图信息摘要：\n{context}\n\n图谱摘要：\n{graph_summary}\n\n请结合上下文回答：{query}"

                response = llm.generate(prompt)
            except (ValueError, ConnectionError, TimeoutError) as e:
                logger.error(f"查询处理失败: {e}")
                response = "对不起，查询处理过程中发生错误。"

        print("\n[回答]:", response.strip(), "\n")

        if config.get("query", {}).get("cache_enabled", True):
            try:
                with open(cache_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"query": query, "mode": mode, "precise": precise, "response": response.strip()}, ensure_ascii=False) + "\n")
            except (IOError, PermissionError) as e:
                logger.warning(f"⚠️ 缓存写入失败: {e}")
