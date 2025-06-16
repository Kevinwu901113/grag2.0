#!/usr/bin/env python3

import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class AnswerCandidate:
    """答案候选项"""
    content: str
    score: float = 0.0
    reasoning: str = ""
    generation_params: Dict = None

class AnswerSelector:
    """多候选答案生成器和选择器"""
    
    def __init__(self, llm_client, config: Dict = None):
        self.llm_client = llm_client
        self.config = config or {}
        
        # 默认配置
        self.default_num_candidates = self.config.get('num_candidates', 3)
        self.max_answer_length = self.config.get('max_answer_length', 500)
        self.enable_for_complex_queries = self.config.get('enable_for_complex_queries', True)
        self.complexity_threshold = self.config.get('complexity_threshold', {
            'min_length': 50,
            'min_entities': 3
        })
        
        # 生成参数变化
        self.generation_variants = [
            {'temperature': 0.3, 'description': '保守生成'},
            {'temperature': 0.7, 'description': '平衡生成'},
            {'temperature': 0.9, 'description': '创新生成'}
        ]
    
    def should_use_multi_candidate(self, query: str, entities: List = None) -> bool:
        """判断是否应该使用多候选答案机制"""
        if not self.enable_for_complex_queries:
            return False
            
        # 基于查询长度判断
        if len(query) < self.complexity_threshold['min_length']:
            return False
            
        # 基于实体数量判断
        if entities and len(entities) < self.complexity_threshold['min_entities']:
            return False
            
        # 检查是否包含复杂问题关键词
        complex_keywords = ['比较', '分析', '评估', '如何', '为什么', '原因', '影响', '关系']
        if any(keyword in query for keyword in complex_keywords):
            return True
            
        return len(query) >= self.complexity_threshold['min_length']
    
    def generate_answer_candidates(self, query: str, context: str, 
                                 num_candidates: Optional[int] = None) -> List[AnswerCandidate]:
        """生成多个候选答案"""
        num_candidates = num_candidates or self.default_num_candidates
        candidates = []
        
        for i in range(num_candidates):
            # 使用不同的生成参数
            variant = self.generation_variants[i % len(self.generation_variants)]
            
            # 构建提示词
            prompt = self._build_answer_prompt(query, context, variant)
            
            try:
                # 生成答案
                answer = self.llm_client.generate(prompt)
                
                # 限制答案长度
                if len(answer) > self.max_answer_length:
                    answer = answer[:self.max_answer_length] + "..."
                
                candidate = AnswerCandidate(
                    content=answer,
                    generation_params=variant
                )
                candidates.append(candidate)
                
            except Exception as e:
                print(f"[AnswerSelector] 生成第{i+1}个候选答案失败: {e}")
                continue
        
        return candidates
    
    def _build_answer_prompt(self, query: str, context: str, variant: Dict) -> str:
        """构建答案生成提示词"""
        base_prompt = f"""
基于以下上下文信息，回答用户问题。请确保答案准确、相关且有帮助。

上下文信息：
{context}

用户问题：
{query}

请提供一个清晰、准确的答案：
"""
        
        # 根据生成变体调整提示词
        if variant.get('temperature', 0.5) > 0.8:
            base_prompt += "\n注意：请提供创新性和多角度的回答。"
        elif variant.get('temperature', 0.5) < 0.4:
            base_prompt += "\n注意：请提供准确、保守的回答。"
        
        return base_prompt
    
    def evaluate_candidates(self, candidates: List[AnswerCandidate], 
                          query: str, context: str) -> List[AnswerCandidate]:
        """评估候选答案质量"""
        if len(candidates) <= 1:
            return candidates
        
        # 构建评分提示词
        evaluation_prompt = self._build_evaluation_prompt(candidates, query, context)
        
        try:
            # 获取评分结果
            evaluation_result = self.llm_client.generate(evaluation_prompt)
            
            # 解析评分结果
            scored_candidates = self._parse_evaluation_result(evaluation_result, candidates)
            
            # 按分数排序
            scored_candidates.sort(key=lambda x: x.score, reverse=True)
            
            return scored_candidates
            
        except Exception as e:
            print(f"[AnswerSelector] 评估候选答案失败: {e}")
            # 如果评估失败，返回原始候选答案
            return candidates
    
    def _build_evaluation_prompt(self, candidates: List[AnswerCandidate], 
                               query: str, context: str) -> str:
        """构建评估提示词"""
        candidates_text = ""
        for i, candidate in enumerate(candidates, 1):
            candidates_text += f"\n答案{i}：\n{candidate.content}\n"
        
        prompt = f"""
请评估以下候选答案的质量，并为每个答案打分（1-10分）。评估标准包括：
1. 准确性：答案是否基于给定上下文准确回答问题
2. 相关性：答案是否直接回答了用户问题
3. 完整性：答案是否全面覆盖了问题的关键方面
4. 清晰性：答案是否表达清晰、易于理解
5. 有用性：答案是否对用户有实际帮助

上下文信息：
{context}

用户问题：
{query}

候选答案：{candidates_text}

请按以下格式输出评分结果：
答案1: 分数=X, 理由=具体评价理由
答案2: 分数=Y, 理由=具体评价理由
答案3: 分数=Z, 理由=具体评价理由

最后请指出最佳答案：最佳答案=答案X
"""
        return prompt
    
    def _parse_evaluation_result(self, evaluation_text: str, 
                               candidates: List[AnswerCandidate]) -> List[AnswerCandidate]:
        """解析评估结果"""
        lines = evaluation_text.strip().split('\n')
        
        for i, candidate in enumerate(candidates):
            # 查找对应答案的评分
            pattern = rf'答案{i+1}[：:]\s*分数[=＝]([0-9.]+).*?理由[=＝](.*)'
            
            for line in lines:
                match = re.search(pattern, line)
                if match:
                    try:
                        score = float(match.group(1))
                        reasoning = match.group(2).strip()
                        candidate.score = score
                        candidate.reasoning = reasoning
                        break
                    except ValueError:
                        continue
            
            # 如果没有找到评分，给默认分数
            if candidate.score == 0.0:
                candidate.score = 5.0
                candidate.reasoning = "未找到具体评分"
        
        return candidates
    
    def select_best_answer(self, query: str, context: str, 
                         entities: List = None) -> Tuple[str, Dict]:
        """选择最佳答案的主要接口"""
        # 判断是否使用多候选机制
        if not self.should_use_multi_candidate(query, entities):
            # 直接生成单个答案
            prompt = self._build_answer_prompt(query, context, {'temperature': 0.5})
            answer = self.llm_client.generate(prompt)
            return answer, {'method': 'single', 'candidates_count': 1}
        
        # 生成多个候选答案
        candidates = self.generate_answer_candidates(query, context)
        
        if not candidates:
            # 如果没有生成任何候选答案，返回错误信息
            return "抱歉，无法生成答案。", {'method': 'multi', 'candidates_count': 0, 'error': 'no_candidates'}
        
        if len(candidates) == 1:
            # 只有一个候选答案
            return candidates[0].content, {
                'method': 'multi', 
                'candidates_count': 1,
                'generation_params': candidates[0].generation_params
            }
        
        # 评估候选答案
        evaluated_candidates = self.evaluate_candidates(candidates, query, context)
        
        # 返回最佳答案
        best_candidate = evaluated_candidates[0]
        
        metadata = {
            'method': 'multi',
            'candidates_count': len(candidates),
            'best_score': best_candidate.score,
            'best_reasoning': best_candidate.reasoning,
            'generation_params': best_candidate.generation_params,
            'all_scores': [c.score for c in evaluated_candidates]
        }
        
        return best_candidate.content, metadata
    
    def get_evaluation_prompt_template(self) -> str:
        """获取评估提示词模板，供外部使用"""
        return """
请评估以下候选答案的质量，并为每个答案打分（1-10分）。评估标准包括：
1. 准确性：答案是否基于给定上下文准确回答问题
2. 相关性：答案是否直接回答了用户问题
3. 完整性：答案是否全面覆盖了问题的关键方面
4. 清晰性：答案是否表达清晰、易于理解
5. 有用性：答案是否对用户有实际帮助

上下文信息：
{context}

用户问题：
{query}

候选答案：
{candidates}

请按以下格式输出评分结果：
答案1: 分数=X, 理由=具体评价理由
答案2: 分数=Y, 理由=具体评价理由
答案3: 分数=Z, 理由=具体评价理由

最后请指出最佳答案：最佳答案=答案X
"""