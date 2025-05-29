import os
import json
import faiss
import numpy as np
from query.optimized_theme_matcher import ThemeMatcher
from query.reranker import SimpleReranker, LLMReranker
from query.enhanced_retriever import EnhancedRetriever
from llm.llm import LLMClient
from llm.answer_selector import AnswerSelector
from graph.graph_utils import extract_entity_names, match_entities_in_query, extract_subgraph, summarize_subgraph
from query.query_classifier import classify_query_lightweight
from utils.io import load_chunks, load_vector_index, load_graph

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
    try:
        # 统一使用LLMClient进行嵌入向量生成
        llm_client = LLMClient(config)
        
        # 编码查询
        query_emb = llm_client.embed(query)
        query_emb = np.array(query_emb).astype('float32')
        
        # 归一化查询向量以确保计算余弦相似度
        import faiss
        faiss.normalize_L2(query_emb)
        
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

def handle_query(query: str, config: dict, work_dir: str, mode: str = "auto", precise: bool = False, use_reranker: str = "simple", use_enhanced_retrieval: bool = True) -> dict:
    """
    处理单个查询
    
    Args:
        query: 查询文本
        config: 配置字典
        work_dir: 工作目录
        mode: 查询模式
        precise: 是否精确查询
        use_reranker: 重排序器类型
        use_enhanced_retrieval: 是否使用增强检索
        
    Returns:
        包含答案、来源和处理时间的字典
    """
    import time
    start_time = time.time()
    
    llm = LLMClient(config)
    chunks = load_chunks(work_dir)
    index, id_map = load_vector_index(work_dir)
    
    # 加载图谱
    graph_path = os.path.join(work_dir, "graph.json")
    graph = load_graph(work_dir) if os.path.exists(graph_path) else None
    entity_names = extract_entity_names(graph) if graph else set()
    
    # 如果是auto模式，进行自动分类
    if mode == "auto":
        original_mode, original_precise = classify_query_lightweight(query, config)
        from query.query_enhancer import enhance_query_classification
        mode, precise = enhance_query_classification(query, original_mode, original_precise)
    
    # 处理norag模式
    if mode == "norag":
        response = llm.generate(f"请回答以下问题：\n{query}")
        return {
            'answer': response.strip(),
            'sources': [],
            'processing_time': time.time() - start_time,
            'enhanced_retrieval': False
        }
    
    # 检索候选文档
    candidates = []
    
    if use_enhanced_retrieval:
        # 使用增强检索器
        retriever = EnhancedRetriever(config, work_dir)
        candidates = retriever.retrieve(query, top_k=10)
    else:
        # 传统检索方法
        if precise and index is not None:
            candidates = direct_vector_search(query, index, id_map, chunks, config, top_k=10)
        else:
            matcher = ThemeMatcher(chunks, config)
            matches = matcher.match(query, top_k=10, min_score=0.3)
            
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
    if use_reranker != "none" and candidates:
        rerank_config = config.get("rerank", {})
        if use_reranker == "llm":
            reranker = LLMReranker(llm, rerank_config)
        else:
            reranker = SimpleReranker(rerank_config)
        
        candidates = reranker.rerank(query, candidates, top_k=5)
    else:
        candidates = candidates[:5]
    
    # 提取文本用于生成回答
    retrieved = [c['text'] for c in candidates]
    context = "\n".join(retrieved) if retrieved else "（未检索到相关文本内容）"
    
    # 图谱摘要
    graph_summary = ""
    if graph:
        try:
            matched_entities = match_entities_in_query(query, entity_names)
            subgraph = extract_subgraph(graph, matched_entities)
            graph_summary = summarize_subgraph(subgraph)
        except Exception:
            pass
    
    # 生成回答 - 使用答案选择器
    answer_selector_config = config.get('answer_selector', {})
    answer_selector = AnswerSelector(llm, answer_selector_config)
    
    # 构建完整上下文
    full_context = context
    if graph_summary:
        full_context += f"\n\n图谱摘要：\n{graph_summary}"
    
    # 提取实体用于复杂度判断
    entities = []
    if graph:
        try:
            entities = match_entities_in_query(query, entity_names)
        except Exception:
            pass
    
    # 使用答案选择器生成最佳答案
    answer, selection_metadata = answer_selector.select_best_answer(
        query=query, 
        context=full_context, 
        entities=entities
    )
    
    return {
        'answer': answer.strip(),
        'sources': candidates,
        'processing_time': time.time() - start_time,
        'enhanced_retrieval': use_enhanced_retrieval,
        'answer_selection': selection_metadata
    }

def run_query_loop(config: dict, work_dir: str, logger=None):
    """
    运行查询循环
    
    Args:
        config: 配置字典
        work_dir: 工作目录
        logger: 日志对象
    """
    if logger:
        logger.info("开始查询循环")
    
    print("\n🔍 RAG查询系统已启动")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'mode <模式>' 切换查询模式 (norag, hybrid_precise, hybrid_imprecise, auto)")
    print("输入 'reranker <类型>' 切换重排序器 (simple, llm, none)")
    print("输入 'enhanced <on/off>' 切换增强检索")
    print("-" * 50)
    
    current_mode = "auto"
    current_reranker = "simple"
    use_enhanced = True
    
    while True:
        try:
            user_input = input("\n请输入查询: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("再见！")
                break
            
            if user_input.startswith('mode '):
                new_mode = user_input[5:].strip()
                if new_mode in ['norag', 'hybrid_precise', 'hybrid_imprecise', 'auto']:
                    current_mode = new_mode
                    print(f"✅ 查询模式已切换为: {current_mode}")
                else:
                    print("❌ 无效模式，可选: norag, hybrid_precise, hybrid_imprecise, auto")
                continue
            
            if user_input.startswith('reranker '):
                new_reranker = user_input[9:].strip()
                if new_reranker in ['simple', 'llm', 'none']:
                    current_reranker = new_reranker
                    print(f"✅ 重排序器已切换为: {current_reranker}")
                else:
                    print("❌ 无效重排序器，可选: simple, llm, none")
                continue
                
            if user_input.startswith('enhanced '):
                enhanced_setting = user_input[9:].strip().lower()
                if enhanced_setting in ['on', 'true', '1']:
                    use_enhanced = True
                    print("✅ 增强检索已启用")
                elif enhanced_setting in ['off', 'false', '0']:
                    use_enhanced = False
                    print("✅ 增强检索已禁用")
                else:
                    print("❌ 无效设置，请使用 on/off")
                continue
            
            if not user_input:
                continue
            
            print(f"\n🔍 查询模式: {current_mode}, 重排序器: {current_reranker}, 增强检索: {'开启' if use_enhanced else '关闭'}")
            
            # 处理查询
            result = handle_query(
                query=user_input,
                config=config,
                work_dir=work_dir,
                mode=current_mode,
                use_reranker=current_reranker,
                use_enhanced_retrieval=use_enhanced
            )
            
            # 显示结果
            print(f"\n📝 回答:")
            print(result['answer'])
            
            # 显示答案选择信息
            if 'answer_selection' in result:
                selection_info = result['answer_selection']
                method = selection_info.get('method', 'unknown')
                candidates_count = selection_info.get('candidates_count', 0)
                
                if method == 'multi' and candidates_count > 1:
                    best_score = selection_info.get('best_score', 0)
                    print(f"\n🎯 答案选择: 多候选模式 ({candidates_count}个候选答案, 最佳评分: {best_score:.1f})")
                    if 'best_reasoning' in selection_info:
                        print(f"   选择理由: {selection_info['best_reasoning']}")
                elif method == 'multi':
                    print(f"\n🎯 答案选择: 多候选模式 ({candidates_count}个候选答案)")
                else:
                    print(f"\n🎯 答案选择: 单一答案模式")
            
            if result['sources']:
                print(f"\n📚 参考来源 ({len(result['sources'])}个):")
                for i, source in enumerate(result['sources'], 1):
                    similarity = source.get('similarity', 0)
                    retrieval_info = ""
                    if 'retrieval_types' in source:
                        retrieval_info = f" [{','.join(source['retrieval_types'])}]"
                    elif 'retrieval_type' in source:
                        retrieval_info = f" [{source['retrieval_type']}]"
                    print(f"{i}. [相似度: {similarity:.3f}]{retrieval_info} {source['text']}")
            
            processing_time = result['processing_time']
            enhanced_status = "(增强检索)" if result.get('enhanced_retrieval') else "(传统检索)"
            print(f"\n⏱️ 处理时间: {processing_time:.2f}秒 {enhanced_status}")
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n❌ 处理查询时出错: {e}")
            import traceback
            traceback.print_exc()
