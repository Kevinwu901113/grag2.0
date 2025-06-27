#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略调优器模块

根据反馈信息动态调整系统配置，优化检索和生成策略。
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging


class StrategyTuner:
    """
    策略调优器
    
    根据上一轮反馈内容判断是否修改配置内容，如关闭BM25检索、
    启用图谱增强、多样性调度等。所有修改都在传入config上进行原地更新。
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, work_dir: Optional[str] = None):
        """
        初始化策略调优器
        
        Args:
            logger: 日志记录器，如果为None则使用默认logger
            work_dir: 工作目录路径，用于统一管理所有产物
        """
        self.logger = logger or logging.getLogger(__name__)
        self.work_dir = work_dir
        
        # 调优阈值配置
        self.thresholds = {
            'low_retrieval_quality': 0.6,  # 检索质量低阈值
            'high_processing_time': 5.0,   # 处理时间高阈值（秒）
            'low_diversity': 0.3,          # 多样性低阈值
            'low_answer_quality': 0.5,     # 答案质量低阈值
            'few_sources': 3,              # 来源数量少阈值
            'low_graph_coverage': 0.2      # 图谱覆盖率低阈值
        }
        
        # 策略调整历史
        self.adjustment_history = []
        
        # 保存初始配置状态（用于reset）
        self.initial_config = None
    
    def update_strategy(self, config: Dict[str, Any], feedback_dict: Dict[str, Any]) -> Dict[str, str]:
        """
        根据反馈内容更新配置策略
        
        Args:
            config: 系统配置字典（原地修改）
            feedback_dict: 反馈信息字典
            
        Returns:
            调整说明字典，包含所有进行的调整及其原因
        """
        adjustments = {}
        
        try:
            # 分析反馈信息
            analysis = self._analyze_feedback(feedback_dict)
            
            # 根据分析结果调整配置
            if analysis['poor_retrieval_quality']:
                adjustments.update(self._adjust_retrieval_strategy(config, analysis))
            
            if analysis['slow_processing']:
                adjustments.update(self._optimize_performance(config, analysis))
            
            if analysis['low_diversity']:
                adjustments.update(self._enhance_diversity(config, analysis))
            
            if analysis['insufficient_sources']:
                adjustments.update(self._expand_source_coverage(config, analysis))
            
            if analysis['poor_graph_utilization']:
                adjustments.update(self._optimize_graph_usage(config, analysis))
            
            if analysis['high_redundancy'] or analysis['context_scheduler_needed']:
                adjustments.update(self._enable_context_scheduler(config, analysis))
            
            if analysis['bm25_ineffective']:
                adjustments.update(self._disable_bm25_retrieval(config, analysis))
            
            if analysis['vector_insufficient']:
                adjustments.update(self._enhance_vector_retrieval(config, analysis))
            
            # Prompt模板和嵌入策略调整
            adjustments.update(self._adjust_prompt_and_embedding_strategy(config, analysis, feedback_dict))
            
            # 记录调整历史
            if adjustments:
                self.adjustment_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'feedback_analysis': analysis,
                    'adjustments': adjustments
                })
                
                self.logger.info(f"策略调整完成，共进行 {len(adjustments)} 项调整")
            
            return adjustments
            
        except Exception as e:
            self.logger.error(f"策略调优过程中出错: {e}")
            return {}
    
    def _analyze_feedback(self, feedback_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析反馈信息，识别需要优化的问题
        
        Args:
            feedback_dict: 反馈信息字典
            
        Returns:
            分析结果字典
        """
        analysis = {
            'poor_retrieval_quality': False,
            'slow_processing': False,
            'low_diversity': False,
            'insufficient_sources': False,
            'poor_graph_utilization': False,
            'high_redundancy': False,
            'bm25_ineffective': False,
            'vector_insufficient': False,
            'context_scheduler_needed': False
        }
        
        # 检索质量分析
        retrieval_info = feedback_dict.get('retrieval', {})
        sources = feedback_dict.get('sources', [])
        
        # 平均相似度低
        if sources:
            avg_similarity = sum(s.get('similarity', 0) for s in sources) / len(sources)
            if avg_similarity < self.thresholds['low_retrieval_quality']:
                analysis['poor_retrieval_quality'] = True
        
        # 处理时间过长
        processing_time = feedback_dict.get('processing_time', 0)
        if processing_time > self.thresholds['high_processing_time']:
            analysis['slow_processing'] = True
        
        # 来源数量不足
        final_candidates = retrieval_info.get('final_candidates', len(sources))
        if final_candidates < self.thresholds['few_sources']:
            analysis['insufficient_sources'] = True
        
        # 多样性分析
        retrieval_types = set()
        for source in sources:
            retrieval_types.update(source.get('retrieval_types', []))
        
        if len(retrieval_types) < 2:  # 检索方法单一
            analysis['low_diversity'] = True
        
        # 图谱利用率分析
        graph_candidates = retrieval_info.get('graph_candidates', 0)
        graph_sources = sum(1 for s in sources if 'graph' in s.get('retrieval_types', []) or 'graph_entity' in s.get('retrieval_types', []))
        
        # 如果图谱检索开启但没有找到候选或图谱来源占比很低
        if retrieval_info.get('graph_enabled', False):
            if graph_candidates == 0 or (sources and graph_sources / len(sources) < self.thresholds['low_graph_coverage']):
                analysis['poor_graph_utilization'] = True
        
        # 冗余度分析
        candidates_after_dedup = retrieval_info.get('candidates_after_dedup', final_candidates)
        total_candidates = (retrieval_info.get('vector_candidates', 0) + 
                          retrieval_info.get('bm25_candidates', 0) + 
                          retrieval_info.get('graph_candidates', 0))
        
        if total_candidates > 0 and candidates_after_dedup / total_candidates < 0.6:
            analysis['high_redundancy'] = True
            analysis['context_scheduler_needed'] = True
        
        # BM25效果分析
        bm25_candidates = retrieval_info.get('bm25_candidates', 0)
        bm25_sources = sum(1 for s in sources if 'bm25' in s.get('retrieval_types', []))
        
        if retrieval_info.get('bm25_enabled', False) and bm25_candidates > 0:
            # 如果BM25找到了候选但最终没有被选中，说明BM25效果不好
            if bm25_sources == 0:
                analysis['bm25_ineffective'] = True
        
        # 向量检索不足分析
        vector_candidates = retrieval_info.get('vector_candidates', 0)
        if vector_candidates < final_candidates:
            analysis['vector_insufficient'] = True
        
        # 评估信息分析
        evaluation_info = feedback_dict.get('evaluation', {})
        best_score = evaluation_info.get('best_score', 0)
        if best_score and best_score < self.thresholds['low_answer_quality']:
            analysis['poor_retrieval_quality'] = True
        
        return analysis
    
    def _adjust_retrieval_strategy(self, config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        调整检索策略，包括检索融合权重调整
        
        Args:
            config: 配置字典
            analysis: 分析结果
            
        Returns:
            调整说明字典
        """
        adjustments = {}
        
        # 降低相似度阈值，增加召回率
        if 'query' in config:
            old_threshold = config['query'].get('similarity_threshold', 0.5)
            new_threshold = max(0.3, old_threshold - 0.1)
            config['query']['similarity_threshold'] = new_threshold
            adjustments['similarity_threshold'] = f"降低相似度阈值从 {old_threshold} 到 {new_threshold}，提高召回率"
        
        # 增加检索结果数量
        if 'enhanced_retrieval' in config:
            old_vector_k = config['enhanced_retrieval'].get('vector_top_k', 15)
            new_vector_k = min(25, old_vector_k + 5)
            config['enhanced_retrieval']['vector_top_k'] = new_vector_k
            adjustments['vector_top_k'] = f"增加向量检索数量从 {old_vector_k} 到 {new_vector_k}"
            
            old_final_k = config['enhanced_retrieval'].get('final_top_k', 8)
            new_final_k = min(15, old_final_k + 3)
            config['enhanced_retrieval']['final_top_k'] = new_final_k
            adjustments['final_top_k'] = f"增加最终结果数量从 {old_final_k} 到 {new_final_k}"
        
        # 启用查询扩展
        if 'enhanced_retrieval' in config:
            config['enhanced_retrieval']['enable_query_expansion'] = True
            adjustments['query_expansion'] = "启用查询扩展以提高检索覆盖率"
        
        # 调整检索融合权重
        if 'retrieval_fusion' not in config:
            config['retrieval_fusion'] = {}
        
        # 根据分析结果调整各检索渠道权重
        fusion_weights = config['retrieval_fusion'].get('weights', {
            'vector': 0.6,
            'bm25': 0.3,
            'graph': 0.1
        })
        
        # 如果图谱检索效果好，增加其权重
        if not analysis.get('poor_graph_utilization', False):
            fusion_weights['graph'] = min(0.25, fusion_weights.get('graph', 0.1) + 0.05)
            adjustments['graph_weight'] = f"增加图谱检索权重到 {fusion_weights['graph']}"
        
        # 如果BM25效果不好，降低其权重
        if analysis.get('bm25_ineffective', False):
            fusion_weights['bm25'] = max(0.1, fusion_weights.get('bm25', 0.3) - 0.1)
            fusion_weights['vector'] = min(0.8, fusion_weights.get('vector', 0.6) + 0.1)
            adjustments['bm25_weight'] = f"降低BM25权重到 {fusion_weights['bm25']}，增加向量权重到 {fusion_weights['vector']}"
        
        config['retrieval_fusion']['weights'] = fusion_weights
        
        # 设置多源命中加分和图谱实体加分
        config['retrieval_fusion']['multi_source_bonus'] = 0.05
        config['retrieval_fusion']['graph_entity_bonus'] = 0.15 if not analysis.get('poor_graph_utilization', False) else 0.25
        
        if config['retrieval_fusion']['graph_entity_bonus'] == 0.25:
            adjustments['graph_bonus'] = "增加图谱实体加分到0.25以提升图谱片段排序"
        
        return adjustments
    
    def _optimize_performance(self, config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        优化性能配置
        
        Args:
            config: 配置字典
            analysis: 分析结果
            
        Returns:
            调整说明字典
        """
        adjustments = {}
        
        # 禁用BM25检索以提高速度
        if 'enhanced_retrieval' in config and config['enhanced_retrieval'].get('enable_bm25', True):
            config['enhanced_retrieval']['enable_bm25'] = False
            adjustments['disable_bm25'] = "禁用BM25检索以提高处理速度"
        
        # 减少重排序候选数量
        if 'query' in config:
            old_rerank_k = config['query'].get('rerank_top_k', 20)
            new_rerank_k = max(10, old_rerank_k - 5)
            config['query']['rerank_top_k'] = new_rerank_k
            adjustments['rerank_top_k'] = f"减少重排序候选数量从 {old_rerank_k} 到 {new_rerank_k}"
        
        # 禁用并行处理避免资源竞争
        if 'graph' in config:
            config['graph']['enable_parallel_processing'] = False
            adjustments['disable_parallel'] = "禁用图谱并行处理以避免资源竞争"
        
        return adjustments
    
    def _enhance_diversity(self, config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        增强结果多样性
        
        Args:
            config: 配置字典
            analysis: 分析结果
            
        Returns:
            调整说明字典
        """
        adjustments = {}
        
        # 启用上下文调度器
        if 'context_scheduler' not in config:
            config['context_scheduler'] = {}
        
        config['context_scheduler']['enabled'] = True
        config['context_scheduler']['weights'] = {
            'relevance': 0.4,
            'structure': 0.3,
            'diversity': 0.3  # 增加多样性权重
        }
        adjustments['context_scheduler'] = "启用上下文调度器并增加多样性权重"
        
        # 调整重排序权重，增加多样性
        if 'rerank_weights' not in config:
            config['rerank_weights'] = {}
        
        config['rerank_weights']['diversity'] = 0.15  # 增加多样性权重
        config['rerank_weights']['vector'] = 0.5      # 相应减少向量权重
        adjustments['rerank_diversity'] = "增加重排序中的多样性权重"
        
        return adjustments
    
    def _expand_source_coverage(self, config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        扩展来源覆盖范围
        
        Args:
            config: 配置字典
            analysis: 分析结果
            
        Returns:
            调整说明字典
        """
        adjustments = {}
        
        # 启用所有检索方法
        if 'enhanced_retrieval' in config:
            config['enhanced_retrieval']['enable_bm25'] = True
            config['enhanced_retrieval']['enable_graph_retrieval'] = True
            config['enhanced_retrieval']['enable_query_expansion'] = True
            adjustments['enable_all_retrieval'] = "启用所有检索方法以扩展来源覆盖"
        
        # 增加BM25检索数量
        if 'enhanced_retrieval' in config:
            old_bm25_k = config['enhanced_retrieval'].get('bm25_top_k', 3)
            new_bm25_k = min(8, old_bm25_k + 2)
            config['enhanced_retrieval']['bm25_top_k'] = new_bm25_k
            adjustments['bm25_top_k'] = f"增加BM25检索数量从 {old_bm25_k} 到 {new_bm25_k}"
        
        return adjustments
    
    def _optimize_graph_usage(self, config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        优化图谱使用，包括动态开关和参数调整
        
        Args:
            config: 配置字典
            analysis: 分析结果
            
        Returns:
            调整说明字典
        """
        adjustments = {}
        
        # 如果图谱利用率低且连续多轮无效，考虑暂时关闭
        if analysis.get('poor_graph_utilization', False):
            # 检查是否应该关闭图谱检索
            if 'graph' in config:
                config['graph']['enable_graph'] = False
                adjustments['disable_graph'] = "暂时关闭图谱检索，因为连续多轮未找到相关实体"
            
            if 'enhanced_retrieval' in config:
                config['enhanced_retrieval']['enable_graph_retrieval'] = False
                config['enhanced_retrieval']['use_enhanced_graph'] = False
                adjustments['disable_enhanced_graph'] = "关闭增强图谱检索以减少开销"
        else:
            # 图谱效果好，启用并优化图谱检索
            if 'graph' in config:
                config['graph']['enable_graph'] = True
                adjustments['enable_graph'] = "启用知识图谱检索"
            
            if 'enhanced_retrieval' in config:
                config['enhanced_retrieval']['enable_graph_retrieval'] = True
                config['enhanced_retrieval']['use_enhanced_graph'] = True
                adjustments['enhanced_graph'] = "启用增强图谱检索进行多跳分析"
            
            # 降低实体相似度阈值，增加图谱匹配
            if 'graph' in config:
                old_threshold = config['graph'].get('entity_similarity_threshold', 0.8)
                new_threshold = max(0.6, old_threshold - 0.1)
                config['graph']['entity_similarity_threshold'] = new_threshold
                adjustments['entity_threshold'] = f"降低实体相似度阈值从 {old_threshold} 到 {new_threshold}"
            
            # 启用图谱并行处理以提高效率
            if 'graph' in config:
                config['graph']['enable_parallel_processing'] = True
                adjustments['graph_parallel'] = "启用图谱并行处理以提高效率"
        
        return adjustments
    
    def _enable_context_scheduler(self, config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        启用上下文调度器以减少冗余
        
        Args:
            config: 配置字典
            analysis: 分析结果
            
        Returns:
            调整说明字典
        """
        adjustments = {}
        
        # 启用上下文调度器
        if 'context_scheduler' not in config:
            config['context_scheduler'] = {}
        
        config['context_scheduler']['enabled'] = True
        config['context_scheduler']['diversity_threshold'] = 0.85  # 提高多样性阈值
        
        # 设置权重，增加多样性和结构权重
        config['context_scheduler']['weights'] = {
            'relevance': 0.4,
            'structure': 0.3,
            'diversity': 0.3
        }
        
        # 如果有图谱实体，增加图谱实体加分
        config['context_scheduler']['graph_entity_bonus'] = 0.2
        
        adjustments['context_scheduler'] = "启用上下文调度器以减少冗余，提高多样性阈值到0.85"
        
        return adjustments
    
    def _disable_bm25_retrieval(self, config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        禁用BM25检索（当其效果不佳时）
        
        Args:
            config: 配置字典
            analysis: 分析结果
            
        Returns:
            调整说明字典
        """
        adjustments = {}
        
        # 禁用BM25检索
        if 'enhanced_retrieval' in config:
            config['enhanced_retrieval']['enable_bm25'] = False
            adjustments['disable_bm25'] = "禁用BM25检索，因为其候选未被最终选中"
        
        # 相应增加向量检索数量以补偿
        if 'enhanced_retrieval' in config:
            old_vector_k = config['enhanced_retrieval'].get('vector_top_k', 15)
            new_vector_k = min(25, old_vector_k + 3)
            config['enhanced_retrieval']['vector_top_k'] = new_vector_k
            adjustments['compensate_vector'] = f"增加向量检索数量从 {old_vector_k} 到 {new_vector_k} 以补偿BM25禁用"
        
        return adjustments
    
    def _enhance_vector_retrieval(self, config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        增强向量检索
        
        Args:
            config: 配置字典
            analysis: 分析结果
            
        Returns:
            调整说明字典
        """
        adjustments = {}
        
        # 增加向量检索数量
        if 'enhanced_retrieval' in config:
            old_vector_k = config['enhanced_retrieval'].get('vector_top_k', 15)
            new_vector_k = min(30, old_vector_k + 5)
            config['enhanced_retrieval']['vector_top_k'] = new_vector_k
            adjustments['vector_top_k'] = f"增加向量检索数量从 {old_vector_k} 到 {new_vector_k}"
        
        # 启用查询扩展以提高召回
        if 'enhanced_retrieval' in config:
            config['enhanced_retrieval']['enable_query_expansion'] = True
            adjustments['query_expansion'] = "启用查询扩展以提高向量检索召回率"
        
        # 降低相似度阈值
        if 'query' in config:
            old_threshold = config['query'].get('similarity_threshold', 0.5)
            new_threshold = max(0.3, old_threshold - 0.1)
            config['query']['similarity_threshold'] = new_threshold
            adjustments['similarity_threshold'] = f"降低相似度阈值从 {old_threshold} 到 {new_threshold}"
        
        return adjustments
    
    def _adjust_prompt_and_embedding_strategy(self, config: Dict[str, Any], analysis: Dict[str, Any], feedback_dict: Dict[str, Any]) -> Dict[str, str]:
        """
        调整Prompt模板和嵌入策略
        
        Args:
            config: 配置字典
            analysis: 分析结果
            feedback_dict: 完整反馈信息
            
        Returns:
            调整说明字典
        """
        adjustments = {}
        
        # 分析答案质量和评分情况
        evaluation_info = feedback_dict.get('evaluation', {})
        best_score = evaluation_info.get('best_score', 0)
        all_scores = evaluation_info.get('all_scores', [])
        candidates_count = evaluation_info.get('candidates_count', 1)
        
        # 答案生成策略调整
        if 'llm' not in config:
            config['llm'] = {}
        
        # 如果答案质量低，调整生成策略
        if best_score and best_score < self.thresholds['low_answer_quality']:
            # 增加候选答案数量
            if 'answer_selection' not in config:
                config['answer_selection'] = {}
            
            old_candidates = config['answer_selection'].get('num_candidates', 3)
            new_candidates = min(5, old_candidates + 1)
            config['answer_selection']['num_candidates'] = new_candidates
            adjustments['answer_candidates'] = f"增加候选答案数量从 {old_candidates} 到 {new_candidates}"
            
            # 调整温度参数，增加创新性
            old_temp = config['llm'].get('temperature', 0.1)
            new_temp = min(0.3, old_temp + 0.05)
            config['llm']['temperature'] = new_temp
            adjustments['temperature'] = f"增加生成温度从 {old_temp} 到 {new_temp}，提高答案创新性"
            
            # 启用多样化生成策略
            config['answer_selection']['enable_diverse_generation'] = True
            config['answer_selection']['generation_strategies'] = [
                {'temperature': 0.1, 'weight': 0.4},  # 保守
                {'temperature': 0.2, 'weight': 0.4},  # 中庸
                {'temperature': 0.3, 'weight': 0.2}   # 创新
            ]
            adjustments['diverse_generation'] = "启用多样化答案生成策略"
        
        # 如果多个候选答案评分接近，说明需要更多上下文
        if len(all_scores) > 1:
            score_variance = max(all_scores) - min(all_scores) if all_scores else 0
            if score_variance < 0.2:  # 评分差距很小
                # 增加上下文片段数量
                if 'enhanced_retrieval' in config:
                    old_final_k = config['enhanced_retrieval'].get('final_top_k', 8)
                    new_final_k = min(12, old_final_k + 2)
                    config['enhanced_retrieval']['final_top_k'] = new_final_k
                    adjustments['more_context'] = f"增加上下文片段数量从 {old_final_k} 到 {new_final_k}，帮助LLM区分"
                
                # 放宽多样性阈值，允许适当重复以保留关键信息
                if 'context_scheduler' in config:
                    old_threshold = config['context_scheduler'].get('diversity_threshold', 0.8)
                    new_threshold = max(0.7, old_threshold - 0.1)
                    config['context_scheduler']['diversity_threshold'] = new_threshold
                    adjustments['relax_diversity'] = f"放宽多样性阈值从 {old_threshold} 到 {new_threshold}，保留关键信息"
        
        # 嵌入策略调整
        if analysis.get('high_redundancy', False):
            # 收紧检索范围
            if 'enhanced_retrieval' in config:
                old_vector_k = config['enhanced_retrieval'].get('vector_top_k', 15)
                new_vector_k = max(10, old_vector_k - 3)
                config['enhanced_retrieval']['vector_top_k'] = new_vector_k
                adjustments['tighten_retrieval'] = f"收紧向量检索范围从 {old_vector_k} 到 {new_vector_k}，减少冗余"
            
            # 提高相似度阈值
            if 'query' in config:
                old_threshold = config['query'].get('similarity_threshold', 0.5)
                new_threshold = min(0.7, old_threshold + 0.1)
                config['query']['similarity_threshold'] = new_threshold
                adjustments['raise_similarity'] = f"提高相似度阈值从 {old_threshold} 到 {new_threshold}，减少低质量匹配"
        
        # 查询扩展策略调整
        if analysis.get('insufficient_sources', False):
            if 'enhanced_retrieval' not in config:
                config['enhanced_retrieval'] = {}
            
            config['enhanced_retrieval']['enable_query_expansion'] = True
            config['enhanced_retrieval']['expansion_methods'] = ['synonym', 'paraphrase']
            adjustments['query_expansion'] = "启用查询扩展（同义词+改写）以提高召回率"
        
        return adjustments
    
    def get_adjustment_history(self) -> List[Dict[str, Any]]:
        """
        获取调整历史记录
        
        Returns:
            调整历史列表
        """
        return self.adjustment_history.copy()
    
    def reset_history(self) -> None:
        """
        重置调整历史记录
        """
        self.adjustment_history.clear()
        self.logger.info("策略调整历史已重置")
    
    def export_adjustments(self, filepath: Optional[str] = None) -> None:
        """
        导出调整历史到文件
        
        Args:
            filepath: 导出文件路径，如果为None则使用工作目录下的默认路径
        """
        if filepath is None:
            if self.work_dir:
                filepath = os.path.join(self.work_dir, "strategy_adjustments.json")
            else:
                filepath = "strategy_adjustments.json"
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.adjustment_history, f, ensure_ascii=False, indent=2)
            self.logger.info(f"调整历史已导出到 {filepath}")
        except Exception as e:
            self.logger.error(f"导出调整历史失败: {e}")
    
    def reset(self, config: Dict[str, Any]) -> None:
        """
        重置策略调优状态，将配置恢复到初始值
        
        Args:
            config: 当前配置字典（将被重置为初始值）
        """
        try:
            if self.initial_config is None:
                # 如果没有保存初始配置，则使用默认配置
                self.initial_config = self._get_default_config()
                self.logger.warning("未找到初始配置，使用默认配置进行重置")
            
            # 清空调整历史
            self.adjustment_history.clear()
            
            # 恢复配置到初始状态
            config.clear()
            config.update(self.initial_config)
            
            self.logger.info("策略调优状态已重置，配置已恢复到初始值")
            
        except Exception as e:
            self.logger.error(f"重置策略调优状态失败: {e}")
    
    def save_initial_config(self, config: Dict[str, Any]) -> None:
        """
        保存初始配置状态
        
        Args:
            config: 初始配置字典
        """
        import copy
        self.initial_config = copy.deepcopy(config)
        self.logger.info("初始配置已保存")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            默认配置字典
        """
        return {
            "query": {
                "similarity_threshold": 0.5,
                "rerank_top_k": 20
            },
            "enhanced_retrieval": {
                "vector_top_k": 15,
                "final_top_k": 8,
                "enable_query_expansion": False,
                "enable_bm25": True,
                "enable_graph_retrieval": False,
                "use_enhanced_graph": False,
                "bm25_top_k": 3
            },
            "context_scheduler": {
                "enabled": False,
                "weights": {
                    "relevance": 0.5,
                    "structure": 0.3,
                    "diversity": 0.2
                }
            },
            "rerank_weights": {
                "diversity": 0.1,
                "vector": 0.6
            },
            "graph": {
                "enable_graph": False,
                "enable_parallel_processing": True,
                "entity_similarity_threshold": 0.8
            }
        }