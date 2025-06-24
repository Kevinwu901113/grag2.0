import hashlib
import re
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict, deque
import time
from loguru import logger
import jieba
import psutil
import os


class SimHashBuffer:
    """
    基于SimHash的快速冗余检测缓冲区
    使用64位SimHash签名和Hamming距离进行近似冗余检测
    """
    
    def __init__(self, config=None):
        """
        初始化SimHash缓冲区
        
        Args:
            config: 配置字典，包含以下参数：
                - hamming_threshold: Hamming距离阈值，默认3
                - max_buffer_size: 最大缓冲区大小，默认100000
                - enable_logging: 是否启用日志，默认True
                - log_interval: 日志记录间隔，默认1000
                - enable_progress: 是否启用进度条，默认True
        """
        # 处理配置参数
        if config is None:
            config = {}
        
        if isinstance(config, dict):
            self.hamming_threshold = config.get('hamming_threshold', 3)
            self.max_buffer_size = config.get('max_buffer_size', 100000)
            self.enable_logging = config.get('enable_logging', True)
            self.enable_progress = config.get('enable_progress', True)
            self.log_interval = config.get('log_interval', 1000)
        else:
            # 兼容旧的直接传递threshold的方式
            self.hamming_threshold = 3
            self.max_buffer_size = 100000
            self.enable_logging = True
            self.enable_progress = True
            self.log_interval = 1000
        
        # 存储SimHash签名的集合
        self.signatures: Set[int] = set()
        # 存储签名到句子的映射（用于调试和记录）
        self.signature_to_sentence: Dict[int, str] = {}
        # 使用deque来维护插入顺序，便于清理旧记录
        self.signature_queue: deque = deque()
        
        # 冗余记录
        self.redundant_records = []
        
        # 统计信息
        self.processed_count = 0
        self.redundant_count = 0
        self.start_time = None
        
        # 进度条
        self.progress_bar = None
        
        if self.enable_logging:
            logger.info(f"初始化SimHash冗余过滤器，Hamming距离阈值: {self.hamming_threshold}, "
                       f"最大缓冲区大小: {self.max_buffer_size}")
    
    def _normalize_text(self, text: str) -> str:
        """
        文本规范化：去除多余空格、转小写、去除标点符号
        
        Args:
            text: 原始文本
            
        Returns:
            规范化后的文本
        """
        # 转小写
        text = text.lower()
        # 去除多余空格和换行符
        text = re.sub(r'\s+', ' ', text).strip()
        # 去除标点符号（保留中文字符、英文字母、数字）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """
        文本分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        # 使用jieba进行中文分词
        tokens = list(jieba.cut(text))
        # 过滤空白token
        tokens = [token.strip() for token in tokens if token.strip()]
        return tokens
    
    def _hash_token(self, token: str) -> int:
        """
        对单个token计算hash值
        
        Args:
            token: 输入token
            
        Returns:
            64位hash值
        """
        # 使用MD5生成hash，然后取前8字节作为64位整数
        hash_obj = hashlib.md5(token.encode('utf-8'))
        hash_bytes = hash_obj.digest()[:8]
        return int.from_bytes(hash_bytes, byteorder='big', signed=False)
    
    def _compute_simhash(self, text: str) -> int:
        """
        计算文本的SimHash签名
        
        Args:
            text: 输入文本
            
        Returns:
            64位SimHash签名
        """
        # 文本规范化
        normalized_text = self._normalize_text(text)
        
        if not normalized_text:
            return 0
        
        # 分词
        tokens = self._tokenize(normalized_text)
        
        if not tokens:
            return 0
        
        # 初始化64位向量
        vector = [0] * 64
        
        # 对每个token计算hash并累加到向量中
        for token in tokens:
            token_hash = self._hash_token(token)
            
            # 检查每一位，如果是1则+1，如果是0则-1
            for i in range(64):
                if (token_hash >> i) & 1:
                    vector[i] += 1
                else:
                    vector[i] -= 1
        
        # 生成最终的SimHash签名
        simhash = 0
        for i in range(64):
            if vector[i] > 0:
                simhash |= (1 << i)
        
        return simhash
    
    def _hamming_distance(self, hash1: int, hash2: int) -> int:
        """
        计算两个hash值的Hamming距离
        
        Args:
            hash1: 第一个hash值
            hash2: 第二个hash值
            
        Returns:
            Hamming距离
        """
        # XOR操作得到不同的位
        xor_result = hash1 ^ hash2
        # 计算1的个数
        return bin(xor_result).count('1')
    
    def _find_similar_signature(self, signature: int) -> Optional[int]:
        """
        在已存储的签名中查找相似的签名
        
        Args:
            signature: 待查找的签名
            
        Returns:
            相似的签名，如果没有找到则返回None
        """
        for stored_signature in self.signatures:
            if self._hamming_distance(signature, stored_signature) <= self.hamming_threshold:
                return stored_signature
        return None
    
    def _cleanup_old_records(self):
        """
        清理旧记录，保持缓冲区大小在限制范围内
        """
        while len(self.signatures) >= self.max_buffer_size:
            # 移除最旧的记录
            if self.signature_queue:
                old_signature = self.signature_queue.popleft()
                self.signatures.discard(old_signature)
                self.signature_to_sentence.pop(old_signature, None)
    
    def is_redundant(self, sentence: str) -> bool:
        """
        检查句子是否冗余
        
        Args:
            sentence: 待检查的句子
            
        Returns:
            True表示冗余，False表示不冗余
        """
        # 初始化计时
        if self.start_time is None:
            self.start_time = time.time()
        
        self.processed_count += 1
        
        # 计算SimHash签名
        signature = self._compute_simhash(sentence)
        
        # 查找相似签名
        similar_signature = self._find_similar_signature(signature)
        
        if similar_signature is not None:
            # 找到相似签名，判定为冗余
            self.redundant_count += 1
            
            # 记录冗余信息
            matched_sentence = self.signature_to_sentence.get(similar_signature, "未知")
            hamming_dist = self._hamming_distance(signature, similar_signature)
            
            self.redundant_records.append({
                "duplicate": sentence,
                "matched_to": matched_sentence,
                "hamming_distance": hamming_dist,
                "signature": signature,
                "matched_signature": similar_signature
            })
            
            if self.enable_logging:
                logger.debug(f"检测到冗余句子 (Hamming距离: {hamming_dist}): {sentence[:50]}...")
            
            # 更新进度条
            if self.progress_bar:
                self.progress_bar.update(1)
                self.progress_bar.set_postfix({
                    '冗余': f'{self.redundant_count}/{self.processed_count}',
                    '冗余率': f'{self.redundant_count/self.processed_count:.1%}'
                })
            
            return True
        else:
            # 没有找到相似签名，添加到缓冲区
            self._cleanup_old_records()
            
            self.signatures.add(signature)
            self.signature_to_sentence[signature] = sentence
            self.signature_queue.append(signature)
            
            if self.enable_logging and self.processed_count % self.log_interval == 0:
                elapsed_time = time.time() - self.start_time
                speed = self.processed_count / elapsed_time if elapsed_time > 0 else 0
                remaining = getattr(self, 'total_sentences', 0) - self.processed_count
                logger.info(f"已处理 {self.processed_count} 个句子，检测到 {self.redundant_count} 个冗余句子，"
                           f"剩余 {remaining} 个句子待处理 (冗余率: {self.redundant_count/self.processed_count:.1%})，"
                           f"处理速度: {speed:.1f} 句/秒，缓冲区大小: {len(self.signatures)}")
            
            # 更新进度条
            if self.progress_bar:
                self.progress_bar.update(1)
                if self.processed_count % 10 == 0:  # 每10个句子更新一次后缀
                    self.progress_bar.set_postfix({
                        '冗余': f'{self.redundant_count}/{self.processed_count}',
                        '冗余率': f'{self.redundant_count/self.processed_count:.1%}'
                    })
            
            return False
    
    def is_redundant_batch(self, sentences: List[str]) -> List[bool]:
        """
        批量冗余检测
        
        Args:
            sentences: 句子列表
            
        Returns:
            布尔列表，表示每个句子是否冗余
        """
        if not sentences:
            return []
        
        results = []
        for sentence in sentences:
            results.append(self.is_redundant(sentence))
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            包含统计信息的字典
        """
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        processing_speed = self.processed_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'processed_count': self.processed_count,
            'redundant_count': self.redundant_count,
            'redundancy_rate': self.redundant_count / self.processed_count if self.processed_count > 0 else 0.0,
            'buffer_size': len(self.signatures),
            'max_buffer_size': self.max_buffer_size,
            'hamming_threshold': self.hamming_threshold,
            'processing_speed': processing_speed,
            'elapsed_time': elapsed_time
        }
    
    def get_redundant_records(self) -> List[Dict[str, Any]]:
        """
        获取冗余记录
        
        Returns:
            冗余记录列表
        """
        return self.redundant_records.copy()
    
    def clear(self):
        """
        清空缓冲区
        """
        self.signatures.clear()
        self.signature_to_sentence.clear()
        self.signature_queue.clear()
        self.redundant_records.clear()
        self.processed_count = 0
        self.redundant_count = 0
        self.start_time = None
        
        if self.enable_logging:
            logger.info("SimHash缓冲区已清空")
    
    def set_progress_bar(self, progress_bar):
        """
        设置进度条
        
        Args:
            progress_bar: tqdm进度条对象
        """
        self.progress_bar = progress_bar
    
    def set_total_sentences(self, total: int):
        """
        设置总句子数（用于进度显示）
        
        Args:
            total: 总句子数
        """
        self.total_sentences = total
    
    # 兼容BaseRedundancyFilter接口的方法
    def is_duplicate(self, text: str) -> bool:
        """
        检查文本是否为重复（兼容BaseRedundancyFilter接口）
        
        Args:
            text: 待检查的文本
            
        Returns:
            True if duplicate, False otherwise
        """
        return self.is_redundant(text)
    
    def add_text(self, text: str) -> None:
        """
        添加文本到缓冲区（兼容BaseRedundancyFilter接口）
        
        Args:
            text: 要添加的文本
        """
        # 如果文本不是重复的，则会自动添加到缓冲区
        # 这里我们直接调用is_redundant，它会处理添加逻辑
        self.is_redundant(text)
    
    def get_memory_usage(self) -> float:
        """
        获取内存使用量 (MB)
        
        Returns:
            内存使用量
        """
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # 转换为MB
        except Exception:
            # 如果无法获取内存信息，返回估算值
            # 每个签名大约8字节，加上字典开销
            estimated_mb = len(self.signatures) * 16 / 1024 / 1024
            return estimated_mb