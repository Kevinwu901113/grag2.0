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
        
        # 预定义的实体模式（正则表达式）
        self.entity_patterns = {
            '人物': [
                r'[\u4e00-\u9fff]{2,4}(?:先生|女士|同志|主席|书记|市长|县长|镇长|村长|主任|经理|总裁|董事长)',
                r'(?:主席|书记|市长|县长|镇长|村长|主任|经理|总裁|董事长)[\u4e00-\u9fff]{2,4}',
            ],
            '组织': [
                r'[\u4e00-\u9fff]+(?:公司|集团|企业|机构|组织|委员会|政府|部门|局|处|科|股|室)',
                r'(?:中共|中国共产党)[\u4e00-\u9fff]*(?:委员会|党委|支部)',
            ],
            '地点': [
                r'[\u4e00-\u9fff]+(?:省|市|县|区|镇|乡|村|街道|路|街|巷|号)',
                r'[\u4e00-\u9fff]+(?:大学|学院|医院|银行|商场|公园|广场)',
            ],
            '时间': [
                r'\d{4}年(?:\d{1,2}月)?(?:\d{1,2}日)?',
                r'\d{1,2}月\d{1,2}日',
                r'(?:上午|下午|晚上|凌晨)\d{1,2}(?:点|时)',
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
                
                # 过滤条件
                if (len(entity_text) >= 2 and 
                    confidence >= 0.5 and 
                    not self._is_generic_word(entity_text)):
                    
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
        使用规则方法抽取实体
        
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
                    if (len(entity_text) >= 2 and 
                        not self._is_generic_word(entity_text)):
                        
                        entities.append({
                            'text': entity_text,
                            'label': entity_type,
                            'confidence': 0.8,  # 规则方法的固定置信度
                            'method': 'rule_based'
                        })
        
        return entities
    
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
            '这里', '那里', '地方', '时候', '时间', '地点'
        }
        
        return word in generic_words or len(word) == 1
    
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
        return self._deduplicate_entities(all_entities)
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        实体去重和合并
        
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
                # 如果已存在，选择置信度更高的
                if entity['confidence'] > entity_dict[text]['confidence']:
                    entity_dict[text] = entity
                # 如果置信度相同，优先选择NER模型的结果
                elif (entity['confidence'] == entity_dict[text]['confidence'] and 
                      entity['method'] == 'ner_model'):
                    entity_dict[text] = entity
        
        return list(entity_dict.values())

def extract_relations_with_llm(text: str, entities: List[str], llm_client: LLMClient) -> List[Tuple[str, str, str]]:
    """
    使用LLM抽取实体间的关系
    
    Args:
        text: 原始文本
        entities: 已识别的实体列表
        llm_client: LLM客户端
        
    Returns:
        关系三元组列表 (头实体, 关系, 尾实体)
    """
    if len(entities) < 2:
        return []
    
    entities_str = "、".join(entities)
    
    prompt = f"""
请从以下文本中抽取实体间的关系，只考虑这些已识别的实体：{entities_str}

文本：{text}

要求：
1. 只输出涉及上述实体的关系
2. 每行一个关系，格式：实体A -[关系]-> 实体B
3. 关系要具体明确，如"担任"、"位于"、"隶属于"等
4. 不要输出解释文字
5. 如果没有明确关系，不要强行创造

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
                
                # 确保实体在已识别列表中
                if head in entities and tail in entities and head != tail:
                    triples.append((head, relation, tail))
        
        return triples
        
    except Exception as e:
        logging.error(f"LLM关系抽取失败: {e}")
        return []