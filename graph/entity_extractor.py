#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实体抽取器模块
使用预训练的中文NER模型进行实体识别，并结合规则方法提高抽取质量
"""

import re
import jieba
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict
import logging

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers库未安装，将使用基于规则的实体抽取")

from llm.llm import LLMClient

class EntityExtractor:
    """
    实体抽取器，支持多种抽取策略
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.ner_model = None
        self.ner_pipeline = None
        
        # 新增配置参数
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.min_entity_length = self.config.get("min_entity_length", 2)
        self.enable_context_validation = self.config.get("enable_context_validation", True)
        self.generic_word_filter = self.config.get("generic_word_filter", True)
        
        # 初始化NER模型
        if HAS_TRANSFORMERS and self.config.get("use_ner_model", True):
            self._init_ner_model()
        
        # 实体类型映射
        self.entity_type_mapping = {
            'PERSON': '人物',
            'PER': '人物', 
            'ORG': '组织',
            'ORGANIZATION': '组织',
            'LOC': '地点',
            'LOCATION': '地点',
            'GPE': '地点',
            'TIME': '时间',
            'DATE': '时间',
            'MISC': '其他',
            'MONEY': '金额',
            'PERCENT': '百分比'
        }
        
        # 预定义的实体模式（正则表达式）- 增强版
        self.entity_patterns = {
            '人物': [
                r'[\u4e00-\u9fff]{2,4}(?:先生|女士|同志)',
                r'[\u4e00-\u9fff]{2,4}(?=任|担任|出任|就任)',  # 人名+任职
                r'[\u4e00-\u9fff]{2,4}(?=在|于|向|对|表示|指出|强调)',  # 发言人模式
                r'[\u4e00-\u9fff]{1,3}(?=书记|主任|经理|部长|局长|处长|科长)',  # 姓+职位
                r'[\u4e00-\u9fff]{2,4}(?=主持|召开|负责|分管)',  # 行为主体
            ],
            '职位': [
                r'(?:党委|纪委|工委)?(?:书记|副书记)',
                r'(?:总|副|常务副)?(?:经理|主任|部长|局长|处长|科长)',
                r'(?:市长|县长|镇长|村长|主席|副主席)',
                r'(?:董事长|总裁|CEO|CTO|CFO|总监|主管)',
                r'(?:财务|技术|人事|行政|市场)(?:部)?(?:经理|主任)',
            ],
            '组织': [
                r'[\u4e00-\u9fff]{2,8}(?:公司|集团|企业)',
                r'[\u4e00-\u9fff]{2,6}(?:委员会|政府|部门)',
                r'[\u4e00-\u9fff]{2,6}(?:党委|纪委|人大|政协)',
                r'[\u4e00-\u9fff]{2,8}(?:大学|学院|学校|医院|银行)',
                r'(?:技术|财务|人事|行政|市场)部(?:门)?',
                r'[\u4e00-\u9fff]{1,4}(?:镇|县|市)(?:党委|政府)',
            ],
            '地点': [
                r'[\u4e00-\u9fff]{2,6}(?:省|市|县|区|镇|乡|村)',
                r'[\u4e00-\u9fff]{2,8}(?:街道|路|街|巷)',
                r'[\u4e00-\u9fff]{2,8}(?:开发区|工业园|科技园|高新区)',
            ],
            '时间': [
                r'\d{4}年(?:\d{1,2}月)?(?:\d{1,2}日)?',
                r'\d{1,2}月\d{1,2}日',
                r'(?:上午|下午|晚上|凌晨)\d{1,2}(?:点|时)',
                r'(?:今年|去年|明年|本月|上月|下月)',
            ]
        }
    
    def _init_ner_model(self):
        """
        初始化NER模型
        """
        try:
            # 使用中文NER模型，这里使用一个轻量级的模型
            model_name = self.config.get("ner_model_name", "ckiplab/bert-base-chinese-ner")
            
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                tokenizer=model_name,
                aggregation_strategy="simple",
                device=-1  # 使用CPU
            )
            
            logging.info(f"NER模型加载成功: {model_name}")
            
        except Exception as e:
            logging.warning(f"NER模型加载失败: {e}，将使用基于规则的方法")
            self.ner_pipeline = None
    
    def extract_entities_with_ner(self, text: str) -> List[Dict[str, Any]]:
        """
        使用NER模型抽取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表，每个实体包含text, label, score等信息
        """
        if not self.ner_pipeline:
            return []
        
        try:
            # 使用NER模型进行实体识别
            entities = self.ner_pipeline(text)
            
            # 过滤和标准化实体
            filtered_entities = []
            for entity in entities:
                entity_text = entity['word'].strip()
                entity_label = entity['entity_group']
                confidence = entity.get('score', 0.0)
                
                # 过滤条件 - 使用配置化参数
                if (len(entity_text) >= self.min_entity_length and 
                    confidence >= self.confidence_threshold and 
                    (not self.generic_word_filter or not self._is_generic_word(entity_text))):
                    
                    filtered_entities.append({
                        'text': entity_text,
                        'label': self.entity_type_mapping.get(entity_label, entity_label),
                        'confidence': confidence,
                        'method': 'ner_model'
                    })
            
            return filtered_entities
            
        except Exception as e:
            logging.error(f"NER模型抽取失败: {e}")
            return []
    
    def extract_entities_with_rules(self, text: str) -> List[Dict[str, Any]]:
        """
        使用规则方法抽取实体 - 优化版
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity_text = match.group().strip()
                    
                    # 增强过滤条件
                    if (len(entity_text) >= self.min_entity_length and 
                        len(entity_text) <= 10 and  # 限制最大长度
                        (not self.generic_word_filter or not self._is_generic_word(entity_text)) and
                        self._is_valid_entity_structure(entity_text, entity_type)):
                        
                        entities.append({
                            'text': entity_text,
                            'label': entity_type,
                            'confidence': 0.8,  # 规则方法的固定置信度
                            'method': 'rule_based'
                        })
        
        return entities
    
    def _is_valid_entity_structure(self, entity: str, entity_type: str) -> bool:
        """
        验证实体结构是否合理
        
        Args:
            entity: 实体文本
            entity_type: 实体类型
            
        Returns:
            是否为有效实体结构
        """
        # 基本长度检查
        if len(entity) < 2 or len(entity) > 10:
            return False
        
        # 人物名称验证
        if entity_type == '人物':
            # 人名通常2-4个字符，不包含动词或长短语
            if (len(entity) > 4 or 
                any(verb in entity for verb in ['担任', '主持', '召开', '负责', '管理', '讨论']) or
                any(suffix in entity for suffix in ['会议', '工作', '规划'])):
                return False
        
        # 职位验证
        elif entity_type == '职位':
            # 职位不应包含动作或人名
            if any(action in entity for action in ['担任', '主持', '召开', '负责', '讨论']):
                return False
        
        # 组织验证
        elif entity_type == '组织':
            # 组织名称不应包含动作词或过长短语
            if (any(action in entity for action in ['担任', '主持', '召开', '负责', '管理', '讨论']) or
                any(suffix in entity for suffix in ['会议', '工作', '规划'])):
                return False
        
        return True
    
    def _is_generic_word(self, word: str) -> bool:
        """
        判断是否为无意义的通用词
        
        Args:
            word: 待判断的词
            
        Returns:
            是否为通用词
        """
        generic_words = {
            '市', '县', '镇', '村', '区', '省', '街道', '路', '街',
            '公司', '企业', '组织', '机构', '部门', '委员会',
            '主任', '经理', '书记', '主席', '市长', '县长',
            '今天', '昨天', '明天', '上午', '下午', '晚上',
            '这里', '那里', '地方', '时候', '时间', '地点',
            '工作', '会议', '活动', '事情', '问题', '情况',
            '发展', '建设', '管理', '服务', '领导', '同志'
        }
        
        # 增强过滤逻辑
        if len(word) == 1:
            return True
        
        if word in generic_words:
            return True
            
        # 过滤纯数字或纯标点
        if word.isdigit() or not any(c.isalnum() for c in word):
            return True
            
        return False
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        综合抽取实体（NER模型 + 规则方法）
        
        Args:
            text: 输入文本
            
        Returns:
            去重后的实体列表
        """
        all_entities = []
        
        # 使用NER模型抽取
        if self.ner_pipeline:
            ner_entities = self.extract_entities_with_ner(text)
            all_entities.extend(ner_entities)
        
        # 使用规则方法抽取
        rule_entities = self.extract_entities_with_rules(text)
        all_entities.extend(rule_entities)
        
        # 去重和合并
        # 去重和合并
        deduplicated_entities = self._deduplicate_entities(all_entities)
        
        # 如果启用上下文验证，进一步过滤
        if self.enable_context_validation:
            deduplicated_entities = self._validate_entities_with_context(text, deduplicated_entities)
            
        return deduplicated_entities
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        实体去重和合并 - 优化版
        
        Args:
            entities: 原始实体列表
            
        Returns:
            去重后的实体列表
        """
        entity_dict = {}
        
        for entity in entities:
            text = entity['text']
            
            if text not in entity_dict:
                entity_dict[text] = entity
            else:
                # 保留置信度更高的实体
                if entity['confidence'] > entity_dict[text]['confidence']:
                    entity_dict[text] = entity
                # 如果置信度相同，优先选择NER模型的结果
                elif (entity['confidence'] == entity_dict[text]['confidence'] and 
                      entity['method'] == 'ner_model'):
                    entity_dict[text] = entity
        
        # 进一步处理包含关系的实体
        final_entities = list(entity_dict.values())
        filtered_entities = []
        
        for i, entity in enumerate(final_entities):
            is_contained = False
            for j, other_entity in enumerate(final_entities):
                if i != j and entity['text'] in other_entity['text'] and len(entity['text']) < len(other_entity['text']):
                    # 如果当前实体被包含在另一个实体中，且置信度不明显更高，则过滤掉
                    if entity['confidence'] <= other_entity['confidence'] + 0.1:
                        is_contained = True
                        break
            
            if not is_contained:
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def _validate_entities_with_context(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用上下文验证实体的有效性
        
        Args:
            text: 原始文本
            entities: 实体列表
            
        Returns:
            验证后的实体列表
        """
        validated_entities = []
        
        for entity in entities:
            entity_text = entity['text']
            
            # 检查实体在文本中的出现次数和位置
            occurrences = [m.start() for m in re.finditer(re.escape(entity_text), text)]
            
            if len(occurrences) == 0:
                continue
                
            # 简单的上下文验证：检查实体前后是否有合理的词汇
            is_valid = False
            for pos in occurrences:
                # 获取前后文
                start = max(0, pos - 10)
                end = min(len(text), pos + len(entity_text) + 10)
                context = text[start:end]
                
                # 简单验证：实体不应该是其他词的一部分
                if self._is_valid_entity_in_context(entity_text, context, pos - start):
                    is_valid = True
                    break
                    
            if is_valid:
                validated_entities.append(entity)
                
        return validated_entities
    
    def _is_valid_entity_in_context(self, entity: str, context: str, entity_pos: int) -> bool:
        """
        检查实体在上下文中是否有效
        
        Args:
            entity: 实体文本
            context: 上下文
            entity_pos: 实体在上下文中的位置
            
        Returns:
            是否有效
        """
        # 检查实体前后是否有分隔符（空格、标点等）
        before_char = context[entity_pos - 1] if entity_pos > 0 else ' '
        after_char = context[entity_pos + len(entity)] if entity_pos + len(entity) < len(context) else ' '
        
        # 如果前后都是字母或数字，可能是其他词的一部分
        if (before_char.isalnum() and entity[0].isalnum()) or (after_char.isalnum() and entity[-1].isalnum()):
            return False
            
        return True

def _find_best_entity_match(target: str, entities: List[str]) -> str:
    """
    找到最匹配的实体 - 优化版
    
    Args:
        target: 目标实体文本
        entities: 候选实体列表
        
    Returns:
        最匹配的实体，如果没有找到返回None
    """
    if not target or not entities:
        return None
        
    target = target.strip()
    
    # 1. 精确匹配
    if target in entities:
        return target
    
    # 2. 包含匹配（优先选择被包含的较短实体）
    contained_matches = []
    containing_matches = []
    
    for entity in entities:
        if target in entity:
            containing_matches.append(entity)
        elif entity in target:
            contained_matches.append(entity)
    
    # 优先返回被包含的实体（通常更精确）
    if contained_matches:
        return max(contained_matches, key=len)  # 选择最长的被包含实体
    
    if containing_matches:
        return min(containing_matches, key=len)  # 选择最短的包含实体
    
    # 3. 字符重叠匹配（提高阈值）
    best_match = None
    max_overlap_ratio = 0
    
    for entity in entities:
        # 计算重叠字符数和比例
        overlap = len(set(target) & set(entity))
        min_len = min(len(target), len(entity))
        
        if min_len > 0:
            overlap_ratio = overlap / min_len
            # 要求至少50%的字符重叠，且至少2个字符
            if overlap >= 2 and overlap_ratio >= 0.5 and overlap_ratio > max_overlap_ratio:
                max_overlap_ratio = overlap_ratio
                best_match = entity
    
    return best_match


def extract_relations_with_llm(text: str, entities: List[str], llm_client: LLMClient) -> List[Tuple[str, str, str]]:
    """
    使用LLM抽取实体间的关系 - 优化版
    
    Args:
        text: 原始文本
        entities: 已识别的实体列表
        llm_client: LLM客户端
        
    Returns:
        关系三元组列表 (头实体, 关系, 尾实体)
    """
    if len(entities) < 2:
        return []
    
    # 按类型分组实体，提供更好的上下文
    entity_groups = defaultdict(list)
    for entity in entities:
        # 简单的实体类型推断
        if any(keyword in entity for keyword in ['书记', '主任', '经理', '市长', '县长', '主席']):
            entity_groups['职位'].append(entity)
        elif any(keyword in entity for keyword in ['公司', '委员会', '政府', '部门', '党委']):
            entity_groups['组织'].append(entity)
        elif len(entity) <= 4 and all('\u4e00' <= c <= '\u9fff' for c in entity):
            entity_groups['人物'].append(entity)
        else:
            entity_groups['其他'].append(entity)
    
    entities_str = "、".join(entities)
    
    prompt = f"""
请从以下文本中抽取实体间的关系。

文本：{text}

已识别实体：{entities_str}

抽取要求：
1. 严格使用上述实体列表中的精确名称，禁止修改、组合或创造新实体
2. 输出格式：实体A -[关系]-> 实体B
3. 关系类型包括但不限于：担任、管理、隶属于、位于、属于、负责、领导、参与
4. 只输出明确存在的关系，不要推测或创造
5. 每行一个关系，不要添加解释
6. 实体名称必须与提供列表完全一致

示例输出：
张三 -[担任]-> 总经理
财务部门 -[隶属于]-> 公司

关系抽取结果：
""".strip()
    
    try:
        response = llm_client.generate(prompt)
        lines = response.strip().splitlines()
        triples = []
        
        for line in lines:
            line = line.strip()
            if not line or "-[" not in line or "->" not in line:
                continue
            
            # 解析关系三元组
            match = re.match(r"(.+?)\s*-\[(.+?)\]->\s*(.+)", line)
            if match:
                head, relation, tail = match.groups()
                head, relation, tail = head.strip(), relation.strip(), tail.strip()
                
                # 首先尝试精确匹配
                if head in entities and tail in entities and head != tail:
                    triples.append((head, relation, tail))
                    continue
                
                # 尝试模糊匹配
                matched_head = _find_best_entity_match(head, entities)
                matched_tail = _find_best_entity_match(tail, entities)
                
                if (matched_head and matched_tail and 
                    matched_head != matched_tail and
                    matched_head in entities and matched_tail in entities):
                    triples.append((matched_head, relation, matched_tail))
                    logging.info(f"实体模糊匹配成功: '{head}' -> '{matched_head}', '{tail}' -> '{matched_tail}'")
                else:
                    logging.warning(f"无法匹配实体: '{head}', '{tail}' 在实体列表 {entities} 中")
        
        # 去重
        unique_triples = list(set(triples))
        logging.info(f"成功抽取 {len(unique_triples)} 个关系三元组")
        
        return unique_triples
        
    except Exception as e:
        logging.error(f"LLM关系抽取失败: {e}")
        return []