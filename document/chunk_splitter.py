#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档切分器

负责将文档按段落或固定字数切分为小块（chunk），确保：
1. 不允许块间重叠
2. 中文文本不会中途截断句子
3. 每个块保留原文文本、所在文件、块序号等信息
4. 支持多种文档格式（DOCX、JSON、JSONL、TXT）

作者: AI Assistant
日期: 2024
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Optional
from docx import Document


class ChunkSplitter:
    """
    文档切分器
    
    将文档切分为固定大小的文本块，确保语义完整性
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化文档切分器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 从配置中获取切分参数
        doc_config = config.get('document_processing', {})
        self.chunk_length = doc_config.get('chunk_length', 200)  # 字符数
        self.min_chunk_length = doc_config.get('min_chunk_length', 50)  # 最小块长度
        
        # 句子分割符（中英文）
        self.sentence_endings = r'[。！？；.!?;]'
        self.paragraph_separators = r'\n\s*\n|\r\n\s*\r\n'
        
        self.logger.info(f"文档切分器初始化完成，块长度: {self.chunk_length}")
    
    def split_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        切分单个文档
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            切分后的文本块列表，每个块包含：
            - text: 文本内容
            - source_file: 源文件路径
            - chunk_id: 块序号
            - char_count: 字符数
            - start_pos: 在原文中的起始位置（可选）
        """
        self.logger.debug(f"开始切分文档: {file_path}")
        
        try:
            # 根据文件扩展名选择读取方法
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.docx':
                content = self._read_docx(file_path)
            elif file_ext == '.json':
                content = self._read_json(file_path)
            elif file_ext == '.jsonl':
                content = self._read_jsonl(file_path)
            elif file_ext in ['.txt', '.md']:
                content = self._read_text(file_path)
            else:
                self.logger.warning(f"不支持的文件格式: {file_ext}")
                return []
            
            if not content:
                self.logger.warning(f"文档内容为空: {file_path}")
                return []
            
            # 切分文本
            chunks = self._split_text(content, file_path)
            
            self.logger.debug(f"文档 {file_path} 切分完成，得到 {len(chunks)} 个块")
            return chunks
            
        except Exception as e:
            self.logger.error(f"切分文档 {file_path} 失败: {e}")
            return []
    
    def _read_docx(self, file_path: str) -> str:
        """
        读取DOCX文件内容
        
        Args:
            file_path: DOCX文件路径
            
        Returns:
            文档文本内容
        """
        doc = Document(file_path)
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        return '\n\n'.join(paragraphs)
    
    def _read_json(self, file_path: str) -> str:
        """
        读取JSON文件内容
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            文档文本内容
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            contents = []
            for item in data:
                if isinstance(item, dict):
                    content = (
                        item.get("content") or 
                        item.get("text") or 
                        item.get("body") or 
                        item.get("message")
                    )
                    if content and isinstance(content, str):
                        contents.append(content.strip())
            return '\n\n'.join(contents)
        elif isinstance(data, dict):
            content = (
                data.get("content") or 
                data.get("text") or 
                data.get("body") or 
                data.get("message")
            )
            return content.strip() if content else ""
        else:
            return str(data)
    
    def _read_jsonl(self, file_path: str) -> str:
        """
        读取JSONL文件内容
        
        Args:
            file_path: JSONL文件路径
            
        Returns:
            文档文本内容
        """
        contents = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict):
                            content = (
                                data.get("content") or 
                                data.get("text") or 
                                data.get("body") or 
                                data.get("message") or
                                data.get("question") or
                                data.get("answer")
                            )
                            if content and isinstance(content, str):
                                contents.append(content.strip())
                            
                            # 处理paragraphs字段（如musique数据集）
                            if "paragraphs" in data and isinstance(data["paragraphs"], list):
                                for para in data["paragraphs"]:
                                    if isinstance(para, dict) and "text" in para:
                                        contents.append(para["text"].strip())
                                    elif isinstance(para, str):
                                        contents.append(para.strip())
                    except json.JSONDecodeError:
                        continue
        
        return '\n\n'.join(contents)
    
    def _read_text(self, file_path: str) -> str:
        """
        读取纯文本文件内容
        
        Args:
            file_path: 文本文件路径
            
        Returns:
            文档文本内容
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def _split_text(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """
        将文本切分为固定大小的块
        
        Args:
            text: 要切分的文本
            source_file: 源文件路径
            
        Returns:
            切分后的文本块列表
        """
        chunks = []
        chunk_id = 0
        
        # 首先按段落分割
        paragraphs = re.split(self.paragraph_separators, text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        current_chunk = ""
        current_start_pos = 0
        
        for paragraph in paragraphs:
            # 如果当前段落加上现有块超过长度限制
            if len(current_chunk) + len(paragraph) + 2 > self.chunk_length and current_chunk:
                # 保存当前块
                if len(current_chunk) >= self.min_chunk_length:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "source_file": source_file,
                        "chunk_id": chunk_id,
                        "char_count": len(current_chunk.strip()),
                        "start_pos": current_start_pos
                    })
                    chunk_id += 1
                
                # 开始新块
                current_chunk = paragraph
                current_start_pos = text.find(paragraph, current_start_pos)
            else:
                # 添加到当前块
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    current_start_pos = text.find(paragraph, current_start_pos)
            
            # 如果单个段落就超过长度限制，需要进一步切分
            if len(current_chunk) > self.chunk_length:
                # 按句子切分
                sentence_chunks = self._split_by_sentences(current_chunk, source_file, chunk_id)
                chunks.extend(sentence_chunks)
                chunk_id += len(sentence_chunks)
                current_chunk = ""
        
        # 处理最后一个块
        if current_chunk and len(current_chunk) >= self.min_chunk_length:
            chunks.append({
                "text": current_chunk.strip(),
                "source_file": source_file,
                "chunk_id": chunk_id,
                "char_count": len(current_chunk.strip()),
                "start_pos": current_start_pos
            })
        
        return chunks
    
    def _split_by_sentences(self, text: str, source_file: str, start_chunk_id: int) -> List[Dict[str, Any]]:
        """
        按句子切分过长的文本
        
        Args:
            text: 要切分的文本
            source_file: 源文件路径
            start_chunk_id: 起始块ID
            
        Returns:
            切分后的文本块列表
        """
        chunks = []
        chunk_id = start_chunk_id
        
        # 按句子分割
        sentences = re.split(self.sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = ""
        
        for sentence in sentences:
            # 如果加上这个句子会超过长度限制
            if len(current_chunk) + len(sentence) + 1 > self.chunk_length and current_chunk:
                # 保存当前块
                if len(current_chunk) >= self.min_chunk_length:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "source_file": source_file,
                        "chunk_id": chunk_id,
                        "char_count": len(current_chunk.strip()),
                        "start_pos": -1  # 句子级别切分时不记录精确位置
                    })
                    chunk_id += 1
                
                # 开始新块
                current_chunk = sentence
            else:
                # 添加到当前块
                if current_chunk:
                    current_chunk += "。" + sentence  # 恢复句号
                else:
                    current_chunk = sentence
            
            # 如果单个句子就超过长度限制，强制切分
            if len(current_chunk) > self.chunk_length:
                # 按字符强制切分，但尽量在标点处切分
                force_chunks = self._force_split_text(current_chunk, source_file, chunk_id)
                chunks.extend(force_chunks)
                chunk_id += len(force_chunks)
                current_chunk = ""
        
        # 处理最后一个块
        if current_chunk and len(current_chunk) >= self.min_chunk_length:
            chunks.append({
                "text": current_chunk.strip(),
                "source_file": source_file,
                "chunk_id": chunk_id,
                "char_count": len(current_chunk.strip()),
                "start_pos": -1
            })
        
        return chunks
    
    def _force_split_text(self, text: str, source_file: str, start_chunk_id: int) -> List[Dict[str, Any]]:
        """
        强制按字符切分过长的文本
        
        Args:
            text: 要切分的文本
            source_file: 源文件路径
            start_chunk_id: 起始块ID
            
        Returns:
            切分后的文本块列表
        """
        chunks = []
        chunk_id = start_chunk_id
        
        # 定义较好的切分点（标点符号和空格）
        good_split_chars = '，、；：,;: \t\n'
        
        start = 0
        while start < len(text):
            end = start + self.chunk_length
            
            if end >= len(text):
                # 最后一块
                chunk_text = text[start:].strip()
                if len(chunk_text) >= self.min_chunk_length:
                    chunks.append({
                        "text": chunk_text,
                        "source_file": source_file,
                        "chunk_id": chunk_id,
                        "char_count": len(chunk_text),
                        "start_pos": -1
                    })
                break
            
            # 寻找较好的切分点
            best_split = end
            for i in range(end - 1, max(start + self.min_chunk_length, end - 50), -1):
                if text[i] in good_split_chars:
                    best_split = i + 1
                    break
            
            chunk_text = text[start:best_split].strip()
            if len(chunk_text) >= self.min_chunk_length:
                chunks.append({
                    "text": chunk_text,
                    "source_file": source_file,
                    "chunk_id": chunk_id,
                    "char_count": len(chunk_text),
                    "start_pos": -1
                })
                chunk_id += 1
            
            start = best_split
        
        return chunks