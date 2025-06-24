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
        
        # 获取entity_extraction配置
        entity_config = self.config.get('graph', {}).get('entity_extraction', {})
        
        # 新增配置参数
        self.confidence_threshold = entity_config.get("confidence_threshold", 0.5)
        self.min_entity_length = entity_config.get("min_entity_length", 2)
        self.enable_context_validation = entity_config.get("enable_context_validation", True)
        self.generic_word_filter = entity_config.get("generic_word_filter", True)
        self.use_ner_model = entity_config.get("use_ner_model", True)
        
        # 初始化NER模型
        if HAS_TRANSFORMERS and self.use_ner_model:
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
        
        # 预定义的实体模式（正则表达式）- 支持中英文
        self.entity_patterns = {
            '人物': [
                # 中文人名
                r'[\u4e00-\u9fff]{2,4}(?:先生|女士|同志)',
                r'[张李王刘陈杨赵黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文][\u4e00-\u9fff]{1,2}(?=[，。；！？\s在是的工作学习任职担任]|$)',
                r'[\u4e00-\u9fff]{2,4}(?=表示|指出|强调|说|称|认为|提到)',
                r'[\u4e00-\u9fff]{2,4}(?=在[\u4e00-\u9fff]+(?:大学|学院|公司|企业|机构))',
                r'(?:张三|李四|王五|赵六|孙七|周八|吴九|郑十|马云|马化腾)',
                r'[A-Za-z]+·[A-Za-z]+',  # 外国人名，如蒂姆·库克
                r'比尔·盖茨|史蒂夫·乔布斯|埃隆·马斯克|蒂姆·库克',
                # 英文人名
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # 标准英文人名格式
                r'\b(?:Steve Jobs|Bill Gates|Tim Cook|Elon Musk|Larry Page|Sergey Brin|Mark Zuckerberg|Jeff Bezos)\b',  # 知名人物
                r'\b[A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+\b',  # 带中间名缩写的人名
                r'\b(?:Mr|Ms|Mrs|Dr|Prof)\. [A-Z][a-z]+ [A-Z][a-z]+\b',  # 带称谓的人名
            ],
            '职位': [
                r'(?:党委|纪委|工委)?(?:书记|副书记)',
                r'(?:总|副|常务副)?(?:经理|主任|部长|局长|处长|科长)',
                r'(?:市长|县长|镇长|村长|主席|副主席)',
                r'(?:董事长|总裁|CEO|CTO|CFO|总监|主管)',
                r'(?:财务|技术|人事|行政|市场)(?:部)?(?:经理|主任)',
                r'工程师|程序员|设计师|分析师|顾问|专家|教授|医生|律师|会计师',
            ],
            '组织': [
                # 中文组织
                r'(?<![\u4e00-\u9fff])[\u4e00-\u9fff]{2,6}(?:公司|集团|企业)(?![\u4e00-\u9fff])',
                r'(?<![\u4e00-\u9fff])[\u4e00-\u9fff]{2,6}(?:委员会|政府|部门)(?![\u4e00-\u9fff])',
                r'(?<![\u4e00-\u9fff])[\u4e00-\u9fff]{2,6}(?:党委|纪委|人大|政协)(?![\u4e00-\u9fff])',
                r'(?:阿里巴巴|腾讯|百度|京东|美团|字节跳动|小米|华为|苹果|微软|谷歌|亚马逊|特斯拉)(?:公司|集团)?',
                r'(?:北京|清华|复旦|上海交通|中山|华中科技|西安交通|哈尔滨工业|南京|浙江|中南|东南|华南理工|大连理工|北京理工|西北工业|电子科技|重庆|兰州|东北|湖南|郑州|苏州|华东师范|中国人民|北京师范|南开|天津|山东|厦门|同济|华东理工|中国农业|北京航空航天|中国海洋|西北农林科技|中央民族|华中师范|陕西师范|东北师范|西南|中南财经政法|对外经济贸易|北京外国语|上海外国语|中国政法|北京邮电|北京科技|华北电力|中国石油|北京化工|北京林业|中国地质|中国矿业|河海|江南|合肥工业|西南交通|西南财经|中央财经|首都师范|北京工业|上海|天津医科|中国医科|首都医科|南方医科|中国药科|沈阳药科|北京中医药|上海中医药|广州中医药|成都中医药|南京中医药|天津中医药|黑龙江中医药|辽宁中医药|长春中医药|山东中医药|河南中医药|湖北中医药|湖南中医药|广西中医药|云南中医药|贵州中医药|陕西中医药|甘肃中医药|新疆医科|石河子|青海|宁夏|内蒙古|西藏|延边|广西|海南|贵州|云南|西北|青海师范|宁夏|内蒙古师范|西藏|新疆师范|石河子)大学',
                r'(?:技术|财务|人事|行政|市场)部(?:门)?(?![\u4e00-\u9fff])',
                # 英文组织
                r'\b[A-Z][a-zA-Z]+ (?:Inc|Corp|Corporation|Company|Ltd|LLC|Group|Institute|Foundation|Association|Organization)\b',
                r'\b(?:Apple Inc|Microsoft Corporation|Google|Amazon|Facebook|Tesla|IBM|Oracle|Intel|Cisco|Adobe|Salesforce|Netflix|Uber|Twitter|LinkedIn|PayPal|eBay|Zoom|Slack|Spotify|Dropbox|Airbnb|Pinterest|Snapchat|TikTok|WhatsApp|Instagram)\b',
                r'\b[A-Z][a-zA-Z]+ University\b',
                r'\b(?:Stanford University|Harvard University|MIT|Yale University|Princeton University|Columbia University|University of California|University of Pennsylvania|Duke University|Northwestern University|Johns Hopkins University|University of Chicago|Cornell University|Brown University|Dartmouth College|Vanderbilt University|Rice University|Washington University|Emory University|Georgetown University|Carnegie Mellon University|University of Notre Dame|University of Virginia|Wake Forest University|Tufts University|Boston College|New York University|University of Rochester|Brandeis University|Case Western Reserve University|Tulane University|Boston University|Northeastern University|Rensselaer Polytechnic Institute|University of Miami|Pepperdine University|University of Southern California|California Institute of Technology|Georgia Institute of Technology)\b',
                r'\b(?:Department of|Ministry of|Bureau of|Office of|Agency of|Commission of|Committee of|Council of|Board of) [A-Z][a-zA-Z ]+\b',
            ],
            '地点': [
                # 中文地点
                r'(?<![\u4e00-\u9fff位于在])(?:[\u4e00-\u9fff]{2,4}(?:省|市|县|区|镇|乡|村))(?![\u4e00-\u9fff])',
                r'(?<![\u4e00-\u9fff])[\u4e00-\u9fff]{2,8}(?:街道|路|街|巷)(?![\u4e00-\u9fff])',
                r'(?<![\u4e00-\u9fff])[\u4e00-\u9fff]{2,8}(?:开发区|工业园|科技园|高新区)(?![\u4e00-\u9fff])',
                r'(?<![\u4e00-\u9fff位于在])(?:北京市|上海市|天津市|重庆市|深圳市)(?![\u4e00-\u9fff])',
                r'(?<=位于)([\u4e00-\u9fff]{2,4}(?:省|市|县|区))',
                r'(?<![\u4e00-\u9fff])(?:海淀区|朝阳区|西城区|东城区|浦东新区)(?![\u4e00-\u9fff])',
                r'(?<![\u4e00-\u9fff])中关村(?![\u4e00-\u9fff])',
                # 英文地点
                r'\b[A-Z][a-zA-Z]+ (?:City|County|State|Province|District|Region|Area|Zone|Park|Center|Square|Street|Avenue|Road|Boulevard|Drive|Lane|Way|Place|Court|Circle|Plaza)\b',
                r'\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose|Austin|Jacksonville|Fort Worth|Columbus|Charlotte|San Francisco|Indianapolis|Seattle|Denver|Washington|Boston|El Paso|Nashville|Detroit|Oklahoma City|Portland|Las Vegas|Memphis|Louisville|Baltimore|Milwaukee|Albuquerque|Tucson|Fresno|Sacramento|Kansas City|Long Beach|Mesa|Atlanta|Colorado Springs|Virginia Beach|Raleigh|Omaha|Miami|Oakland|Minneapolis|Tulsa|Wichita|New Orleans|Arlington|Cleveland|Tampa|Bakersfield|Aurora|Honolulu|Anaheim|Santa Ana|Corpus Christi|Riverside|Lexington|Stockton|Toledo|St. Paul|Newark|Greensboro|Plano|Henderson|Lincoln|Buffalo|Jersey City|Chula Vista|Fort Wayne|Orlando|St. Petersburg|Chandler|Laredo|Norfolk|Durham|Madison|Lubbock|Irvine|Winston-Salem|Glendale|Garland|Hialeah|Reno|Chesapeake|Gilbert|Baton Rouge|Irving|Scottsdale|North Las Vegas|Fremont|Boise|Richmond|San Bernardino|Birmingham|Spokane|Rochester|Des Moines|Modesto|Fayetteville|Tacoma|Oxnard|Fontana|Columbus|Montgomery|Moreno Valley|Shreveport|Aurora|Yonkers|Akron|Huntington Beach|Little Rock|Augusta|Amarillo|Glendale|Mobile|Grand Rapids|Salt Lake City|Tallahassee|Huntsville|Grand Prairie|Knoxville|Worcester|Newport News|Brownsville|Overland Park|Santa Clarita|Providence|Garden Grove|Chattanooga|Oceanside|Jackson|Fort Lauderdale|Santa Rosa|Rancho Cucamonga|Port St. Lucie|Tempe|Ontario|Vancouver|Cape Coral|Sioux Falls|Springfield|Peoria|Pembroke Pines|Elk Grove|Salem|Lancaster|Corona|Eugene|Palmdale|Salinas|Springfield|Pasadena|Fort Collins|Hayward|Pomona|Cary|Rockford|Alexandria|Escondido|McKinney|Kansas City|Joliet|Sunnyvale|Torrance|Bridgeport|Lakewood|Hollywood|Paterson|Naperville|Syracuse|Mesquite|Dayton|Savannah|Clarksville|Orange|Pasadena|Fullerton|Killeen|Frisco|Hampton|McAllen|Warren|Bellevue|West Valley City|Columbia|Olathe|Sterling Heights|New Haven|Miramar|Waco|Thousand Oaks|Cedar Rapids|Charleston|Sioux City|Round Rock|Fargo|Columbia|Coral Springs|Stamford|Concord|Daly City|Richardson|Gainesville|Carrollton|Surprise|Roseville|Thornton|Allentown|Inglewood|Pearland|Vallejo|Ann Arbor|Berkeley|Richardson|Odessa|Arvada|Cambridge|Sugar Land|Lansing|Evansville|College Station|Fairfield|Clearwater|West Jordan|Westminster|Ventura|Carlsbad|St. George|North Charleston|Murfreesboro|Wilmington|Pueblo|Portsmouth|Denton|Midland)\b',
                r'\b(?:California|Texas|Florida|New York|Pennsylvania|Illinois|Ohio|Georgia|North Carolina|Michigan|New Jersey|Virginia|Washington|Arizona|Massachusetts|Tennessee|Indiana|Missouri|Maryland|Wisconsin|Colorado|Minnesota|South Carolina|Alabama|Louisiana|Kentucky|Oregon|Oklahoma|Connecticut|Utah|Iowa|Nevada|Arkansas|Mississippi|Kansas|New Mexico|Nebraska|West Virginia|Idaho|Hawaii|New Hampshire|Maine|Montana|Rhode Island|Delaware|South Dakota|North Dakota|Alaska|Vermont|Wyoming)\b',
                r'\b(?:United States|USA|Canada|Mexico|United Kingdom|UK|France|Germany|Italy|Spain|Japan|China|India|Australia|Brazil|Russia|South Korea|Netherlands|Belgium|Switzerland|Sweden|Norway|Denmark|Finland|Austria|Portugal|Greece|Poland|Czech Republic|Hungary|Romania|Bulgaria|Croatia|Slovenia|Slovakia|Estonia|Latvia|Lithuania|Ireland|Luxembourg|Malta|Cyprus)\b',
            ],
            '时间': [
                r'\d{4}年(?:\d{1,2}月)?(?:\d{1,2}日)?',
                r'\d{1,2}月\d{1,2}日',
                r'(?:上午|下午|晚上|凌晨)\d{1,2}(?:点|时)',
                r'(?:今年|去年|明年|本月|上月|下月)',
                r'2023年12月15日',  # 具体日期
            ]
        }
    
    def _init_ner_model(self):
        """
        初始化NER模型 - 支持中英文
        """
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            import os
            
            # 获取entity_extraction配置
            entity_config = self.config.get('graph', {}).get('entity_extraction', {})
            
            # 从配置文件读取模型名称，支持中英文模型
            model_name = entity_config.get("ner_model_name", "dbmdz/bert-large-cased-finetuned-conll03-english")
            local_files_only = entity_config.get("local_files_only", False)  # 默认允许在线下载
            cache_dir = entity_config.get("cache_dir", "/home/wjk/workplace/rag/cache/models")
            
            # 设置环境变量
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            os.environ['HF_HUB_CACHE'] = cache_dir
            if local_files_only:
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                os.environ['HF_DATASETS_OFFLINE'] = '1'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            logging.info(f"模型配置 - 离线模式: {local_files_only}, 缓存目录: {cache_dir}")
            
            # 构建模型和tokenizer的加载参数
            load_kwargs = {
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
                "trust_remote_code": False,  # 安全设置
            }
            
            # 分别加载模型和tokenizer
            try:
                logging.info(f"开始加载NER模型: {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
                model = AutoModelForTokenClassification.from_pretrained(model_name, **load_kwargs)
                
                # 创建pipeline，使用合适的聚合策略
                self.ner_pipeline = pipeline(
                    "ner",
                    model=model,
                    tokenizer=tokenizer,
                    aggregation_strategy="simple",  # 使用simple策略，更稳定
                    device=-1  # 使用CPU
                )
                
                logging.info(f"✅ NER模型加载成功: {model_name}")
                
            except Exception as load_error:
                logging.warning(f"详细加载失败: {load_error}")
                # 尝试使用简化的pipeline方式
                try:
                    pipeline_kwargs = {
                        "model": model_name,
                        "aggregation_strategy": "simple",
                        "device": -1,
                    }
                    if local_files_only:
                        pipeline_kwargs["model_kwargs"] = {"local_files_only": True, "cache_dir": cache_dir}
                        pipeline_kwargs["tokenizer_kwargs"] = {"local_files_only": True, "cache_dir": cache_dir}
                    
                    self.ner_pipeline = pipeline("ner", **pipeline_kwargs)
                    logging.info(f"✅ 使用简化方式加载NER模型: {model_name}")
                except Exception as e2:
                    logging.error(f"简化方式也失败: {e2}")
                    if local_files_only:
                        logging.error("请确保模型已下载到本地缓存目录，或设置 local_files_only: false 允许在线下载")
                    self.ner_pipeline = None
            
        except Exception as e:
            logging.warning(f"NER模型加载失败: {e}，将使用基于规则的方法")
            self.ner_pipeline = None
    
    def _detect_language(self, text: str) -> str:
        """
        检测文本语言
        
        Args:
            text: 输入文本
            
        Returns:
            语言代码 ('zh' 为中文, 'en' 为英文, 'mixed' 为混合)
        """
        # 统计中文字符数量
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 统计英文字符数量
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = chinese_chars + english_chars
        if total_chars == 0:
            return 'en'  # 默认英文
        
        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if chinese_ratio > 0.7:
            return 'zh'
        elif english_ratio > 0.7:
            return 'en'
        else:
            return 'mixed'
    
    def extract_entities_with_ner(self, text: str) -> List[Dict[str, Any]]:
        """
        使用NER模型抽取实体 - 优化版本
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表，每个实体包含text, label, score等信息
        """
        if not self.ner_pipeline:
            return []
        
        try:
            # 文本长度限制，避免过长文本导致GPU内存溢出
            max_text_length = 512  # 限制最大文本长度
            if len(text) > max_text_length:
                # 分段处理长文本
                segments = [text[i:i+max_text_length] for i in range(0, len(text), max_text_length-50)]  # 50字符重叠
                all_entities = []
                for segment in segments:
                    segment_entities = self.ner_pipeline(segment)
                    all_entities.extend(segment_entities)
                entities = all_entities
            else:
                # 使用NER模型进行实体识别
                entities = self.ner_pipeline(text)
            
            # 过滤和标准化实体
            filtered_entities = []
            seen_entities = set()  # 去重
            
            for entity in entities:
                entity_text = entity['word'].strip()
                entity_label = entity['entity_group']
                confidence = entity.get('score', 0.0)
                
                # 去重检查
                if entity_text in seen_entities:
                    continue
                seen_entities.add(entity_text)
                
                # 过滤条件 - 使用配置化参数
                if (len(entity_text) >= self.min_entity_length and 
                    confidence >= self.confidence_threshold and 
                    (not self.generic_word_filter or not self._is_generic_word(entity_text))):
                    
                    filtered_entities.append({
                        'text': entity_text,
                        'type': self.entity_type_mapping.get(entity_label, entity_label),
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
                    
                    # 调试信息
                    logging.debug(f"规则匹配: {entity_type} - {pattern} -> {entity_text}")
                    
                    # 检查各个过滤条件
                    length_ok = len(entity_text) >= self.min_entity_length
                    max_length_ok = len(entity_text) <= 10
                    not_generic = not self.generic_word_filter or not self._is_generic_word(entity_text)
                    valid_structure = self._is_valid_entity_structure(entity_text, entity_type)
                    
                    logging.debug(f"过滤检查 - 长度:{length_ok}, 最大长度:{max_length_ok}, 非通用词:{not_generic}, 有效结构:{valid_structure}")
                    
                    # 增强过滤条件
                    if length_ok and max_length_ok and not_generic and valid_structure:
                        entities.append({
                            'text': entity_text,
                            'type': entity_type,
                            'confidence': 0.8,  # 规则方法的固定置信度
                            'method': 'rule_based'
                        })
                        logging.debug(f"规则实体已添加: {entity_text} ({entity_type})")
                    else:
                        logging.debug(f"规则实体被过滤: {entity_text} ({entity_type})")
        
        logging.debug(f"规则方法总共抽取到 {len(entities)} 个实体")
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
        if len(entity) < 1 or len(entity) > 15:
            return False
        
        # 过滤明显不合理的实体
        invalid_patterns = [
            r'^[的了在是有和与或但而因所如那这个一二三四五六七八九十]$',  # 单独的虚词
            r'^(方案|计划|项目|任务|工作|活动|会议|讨论|研究|分析|方法|方式|措施|办法|政策|制度|规定|标准|要求|条件|情况|状态|过程|结果|问题|原因|目的|意义|作用|影响|效果|高等|教育|学校|机构|部门|单位|组织|系统|平台|网站|应用|软件|技术|科技|发展|建设|管理|服务|支持|帮助|提供|实现|完成|进行|开展|推进|加强|改善|提高|优化|创新|改革|高等学府|学术机构|研究院所)$',  # 过滤常见的非人名词汇
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, entity):
                return False
        
        # 人物名称验证
        if entity_type == '人物':
            # 人名通常2-4个字符，不包含动词或长短语
            if (len(entity) > 6 or 
                any(verb in entity for verb in ['担任', '主持', '召开', '负责', '管理', '讨论', '工作', '学习']) or
                any(suffix in entity for suffix in ['会议', '工作', '规划', '大学', '学院'])):
                return False
                
        # 组织验证 - 更严格
        elif entity_type == '组织':
            # 过滤包含人名+动作的错误匹配
            if (any(action in entity for action in ['在', '工作', '学习', '担任', '主持', '召开', '负责', '管理', '讨论']) and
                len(entity) > 8):  # 长度超过8且包含动作词的可能是错误匹配
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
        # 只过滤真正无意义的通用词，减少过度过滤
        generic_words = {
            '这里', '那里', '地方', '时候', 
            '的', '了', '在', '是', '有', '和', '与', '或',
            '但', '而', '因为', '所以', '如果', '那么'
        }
        
        # 过滤单字符虚词
        if len(word) == 1:
            meaningless_chars = {'的', '了', '在', '是', '有', '和', '与', '或', '但', '而', '因', '所', '如', '那', '这', '个'}
            if word in meaningless_chars:
                return True
            # 保留其他单字符，包括人名中的单字
            return False
        
        # 检查是否在通用词列表中
        if word in generic_words:
            return True
            
        # 过滤纯数字或纯标点
        if word.isdigit() or (not any(c.isalnum() for c in word) and len(word) < 3):
            return True
            
        return False
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        综合抽取实体（NER模型 + 规则方法）- 支持中英文
        
        Args:
            text: 输入文本
            
        Returns:
            去重后的实体列表
        """
        all_entities = []
        
        # 检测文本语言
        language = self._detect_language(text)
        logging.debug(f"检测到文本语言: {language}")
        
        # 使用NER模型抽取（优先使用，特别是对英文文本）
        if self.ner_pipeline:
            ner_entities = self.extract_entities_with_ner(text)
            all_entities.extend(ner_entities)
            logging.debug(f"NER模型抽取到 {len(ner_entities)} 个实体")
        
        # 使用规则方法抽取（作为补充，特别是对中文文本）
        rule_entities = self.extract_entities_with_rules(text)
        all_entities.extend(rule_entities)
        logging.debug(f"规则方法抽取到 {len(rule_entities)} 个实体")
        
        # 去重和合并
        deduplicated_entities = self._deduplicate_entities(all_entities)
        logging.debug(f"去重后剩余 {len(deduplicated_entities)} 个实体")
        
        # 如果启用上下文验证，进一步过滤
        if self.enable_context_validation:
            deduplicated_entities = self._validate_entities_with_context(text, deduplicated_entities)
            logging.debug(f"上下文验证后剩余 {len(deduplicated_entities)} 个实体")
            
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
        # 对于中文文本，上下文验证应该更宽松
        # 中文实体通常不需要严格的边界检查
        if any('\u4e00' <= c <= '\u9fff' for c in entity):
            # 中文实体，基本都认为有效
            # 只排除明显错误的情况：实体是英文单词的一部分
            before_char = context[entity_pos - 1] if entity_pos > 0 else ' '
            after_char = context[entity_pos + len(entity)] if entity_pos + len(entity) < len(context) else ' '
            
            # 如果实体前后都是英文字母，且实体本身包含英文，可能是错误匹配
            if (before_char.isalpha() and after_char.isalpha() and 
                any(c.isalpha() and not ('\u4e00' <= c <= '\u9fff') for c in entity)):
                return False
            return True
        else:
            # 英文实体，需要更严格的验证
            before_char = context[entity_pos - 1] if entity_pos > 0 else ' '
            after_char = context[entity_pos + len(entity)] if entity_pos + len(entity) < len(context) else ' '
            
            if (before_char.isalnum() and entity[0].isalnum()) or (after_char.isalnum() and entity[-1].isalnum()):
                return False
            return True

def _extract_relations_with_rules(text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
    """
    基于规则的关系抽取方法
    
    Args:
        text: 原始文本
        entities: 已识别的实体列表
        
    Returns:
        关系三元组列表 (头实体, 关系, 尾实体)
    """
    triples = []
    
    # 定义关系模式
    relation_patterns = [
        # 人物-职位-组织关系
        (r'([^，。；！？\s]+)(?:是|担任|任职)([^，。；！？\s]*(?:教授|经理|主任|书记|市长|县长|主席|董事长|总裁|CEO|CTO|CFO))', '担任'),
        (r'([^，。；！？\s]+)在([^，。；！？\s]*(?:大学|学院|公司|企业|机构|委员会|政府|部门))(?:工作|学习|任职)', '隶属于'),
        (r'([^，。；！？\s]+)创立了([^，。；！？\s]*(?:公司|集团|企业))', '创立'),
        (r'([^，。；！？\s]*(?:公司|集团|企业))的(?:CEO|总裁|董事长|创始人)是([^，。；！？\s]+)', '领导'),
        # 组织关系
        (r'([^，。；！？\s]*(?:部门|部))(?:隶属于|属于)([^，。；！？\s]*(?:公司|集团|企业|机构))', '隶属于'),
        # 地理关系
        (r'([^，。；！？\s]+)位于([^，。；！？\s]*(?:省|市|县|区|镇|乡|村))', '位于'),
    ]
    
    for pattern, relation_type in relation_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            entity1, entity2 = match.groups()
            entity1, entity2 = entity1.strip(), entity2.strip()
            
            # 检查实体是否在识别列表中
            matched_entity1 = _find_best_entity_match(entity1, entities)
            matched_entity2 = _find_best_entity_match(entity2, entities)
            
            if (matched_entity1 and matched_entity2 and 
                matched_entity1 != matched_entity2 and
                matched_entity1 in entities and matched_entity2 in entities):
                triples.append((matched_entity1, relation_type, matched_entity2))
                logging.info(f"规则抽取关系: {matched_entity1} -[{relation_type}]-> {matched_entity2}")
    
    # 去重
    unique_triples = list(set(triples))
    logging.info(f"基于规则抽取到 {len(unique_triples)} 个关系三元组")
    
    return unique_triples


def _find_best_entity_match(target: str, entities: List[str]) -> str:
    """
    找到最匹配的实体 - 增强版
    
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
    
    # 2. 忽略大小写的精确匹配
    target_lower = target.lower()
    for entity in entities:
        if target_lower == entity.lower():
            return entity
    
    # 3. 包含匹配（优先选择被包含的较短实体）
    contained_matches = []
    containing_matches = []
    
    for entity in entities:
        if target in entity:
            containing_matches.append(entity)
        elif entity in target:
            contained_matches.append(entity)
        # 忽略大小写的包含匹配
        elif target_lower in entity.lower():
            containing_matches.append(entity)
        elif entity.lower() in target_lower:
            contained_matches.append(entity)
    
    # 优先返回被包含的实体（通常更精确）
    if contained_matches:
        return max(contained_matches, key=len)  # 选择最长的被包含实体
    
    if containing_matches:
        return min(containing_matches, key=len)  # 选择最短的包含实体
    
    # 4. 词汇级别的部分匹配（针对复合词）
    target_words = set(re.findall(r'\w+', target.lower()))
    if target_words:
        best_word_match = None
        max_word_overlap = 0
        
        for entity in entities:
            entity_words = set(re.findall(r'\w+', entity.lower()))
            if entity_words:
                overlap = len(target_words & entity_words)
                overlap_ratio = overlap / min(len(target_words), len(entity_words))
                
                # 如果有词汇重叠且重叠比例较高
                if overlap > 0 and overlap_ratio >= 0.5 and overlap > max_word_overlap:
                    max_word_overlap = overlap
                    best_word_match = entity
        
        if best_word_match:
            return best_word_match
    
    # 5. 字符重叠匹配（降低阈值以提高匹配率）
    best_match = None
    max_overlap_ratio = 0
    
    for entity in entities:
        # 计算重叠字符数和比例
        overlap = len(set(target.lower()) & set(entity.lower()))
        min_len = min(len(target), len(entity))
        
        if min_len > 0:
            overlap_ratio = overlap / min_len
            # 降低阈值：要求至少30%的字符重叠，且至少2个字符
            if overlap >= 2 and overlap_ratio >= 0.3 and overlap_ratio > max_overlap_ratio:
                max_overlap_ratio = overlap_ratio
                best_match = entity
    
    # 6. 编辑距离匹配（作为最后手段）
    if not best_match and len(target) >= 3:
        min_distance = float('inf')
        for entity in entities:
            if len(entity) >= 3:
                distance = _levenshtein_distance(target.lower(), entity.lower())
                max_len = max(len(target), len(entity))
                similarity = 1 - distance / max_len
                
                # 如果相似度超过60%
                if similarity >= 0.6 and distance < min_distance:
                    min_distance = distance
                    best_match = entity
    
    return best_match


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    计算两个字符串的编辑距离
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def _relaxed_entity_match(target: str, entities: List[str]) -> str:
    """
    更宽松的实体匹配策略，用于处理复杂的实体名称
    
    Args:
        target: 目标实体文本
        entities: 候选实体列表
        
    Returns:
        最匹配的实体，如果没有找到返回None
    """
    if not target or not entities:
        return None
        
    target = target.strip()
    
    # 1. 移除常见的修饰词后再匹配
    target_cleaned = target
    for modifier in ['所在的', '出生的', '位于', '属于', '的', '之', '等']:
        target_cleaned = target_cleaned.replace(modifier, '')
    
    if target_cleaned != target:
        match = _find_best_entity_match(target_cleaned, entities)
        if match:
            return match
    
    # 2. 提取核心词汇进行匹配
    try:
        # 对中文进行分词
        target_words = list(jieba.cut(target))
        # 过滤掉停用词和短词
        meaningful_words = [w for w in target_words if len(w) >= 2 and w not in ['的', '在', '是', '有', '和', '与', '或', '但', '而', '所', '之', '等']]
        
        for word in meaningful_words:
            match = _find_best_entity_match(word, entities)
            if match:
                return match
                
    except Exception as e:
        # 如果jieba分词失败，使用简单的词汇提取
        logging.debug(f"jieba分词失败: {e}，使用简单词汇提取")
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', target)
        for word in words:
            if len(word) >= 2:
                match = _find_best_entity_match(word, entities)
                if match:
                    return match
    
    # 3. 基于地名、人名等特殊模式的匹配
    # 地名模式：XX州、XX市、XX县等
    location_patterns = [r'(.+?)州', r'(.+?)市', r'(.+?)县', r'(.+?)省', r'(.+?)区']
    for pattern in location_patterns:
        match_obj = re.search(pattern, target)
        if match_obj:
            location_name = match_obj.group(1)
            for entity in entities:
                if location_name in entity or entity in location_name:
                    return entity
    
    # 4. 机构名称模式：XX学校、XX公司、XX大学等
    org_patterns = [r'(.+?)学校', r'(.+?)大学', r'(.+?)公司', r'(.+?)医院', r'(.+?)银行']
    for pattern in org_patterns:
        match_obj = re.search(pattern, target)
        if match_obj:
            org_name = match_obj.group(1)
            for entity in entities:
                if org_name in entity or entity in org_name:
                    return entity
    
    return None


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
    
    # 实体数量限制，避免prompt过长导致LLM处理缓慢
    max_entities = 15  # 限制最大实体数量
    if len(entities) > max_entities:
        # 优先选择较短的实体（通常更重要）
        entities = sorted(entities, key=len)[:max_entities]
        logging.info(f"实体数量过多({len(entities)})，已限制为前{max_entities}个")
    
    # 文本长度限制，避免prompt过长
    max_text_length = 800  # 限制文本长度
    if len(text) > max_text_length:
        # 截取包含更多实体的文本段
        entity_positions = []
        for entity in entities:
            pos = text.find(entity)
            if pos != -1:
                entity_positions.append(pos)
        
        if entity_positions:
            start_pos = max(0, min(entity_positions) - 100)
            end_pos = min(len(text), max(entity_positions) + 200)
            text = text[start_pos:end_pos]
            logging.info(f"文本过长，已截取关键段落: {start_pos}-{end_pos}")
        else:
            text = text[:max_text_length]
    
    entities_str = "、".join(entities)
    
    # 简化prompt，减少LLM处理时间
    prompt = f"""
从文本中找出实体间的关系：

文本：{text}

实体：{entities_str}

要求：
1. 只使用上述实体，不要修改名称
2. 格式：实体A -[关系]-> 实体B
3. 关系类型：担任、管理、隶属于、位于、属于、负责、领导、参与
4. 每行一个关系

关系：
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
                elif matched_head and not matched_tail:
                    logging.debug(f"部分匹配成功: '{head}' -> '{matched_head}', 但无法匹配 '{tail}'")
                elif matched_tail and not matched_head:
                    logging.debug(f"部分匹配成功: '{tail}' -> '{matched_tail}', 但无法匹配 '{head}'")
                else:
                    # 尝试更宽松的匹配策略
                    relaxed_head = _relaxed_entity_match(head, entities)
                    relaxed_tail = _relaxed_entity_match(tail, entities)
                    
                    if (relaxed_head and relaxed_tail and 
                        relaxed_head != relaxed_tail):
                        triples.append((relaxed_head, relation, relaxed_tail))
                        logging.info(f"宽松匹配成功: '{head}' -> '{relaxed_head}', '{tail}' -> '{relaxed_tail}'")
                    else:
                        logging.debug(f"无法匹配实体: '{head}', '{tail}' 在实体列表 {entities[:5]}{'...' if len(entities) > 5 else ''} 中")
        
        # 去重
        unique_triples = list(set(triples))
        logging.info(f"成功抽取 {len(unique_triples)} 个关系三元组")
        
        return unique_triples
        
    except Exception as e:
        logging.error(f"LLM关系抽取失败: {e}")
        logging.info("尝试使用基于规则的关系抽取方法")
        return _extract_relations_with_rules(text, entities)