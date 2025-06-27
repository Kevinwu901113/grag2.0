import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from ..output_manager import resolve_work_dir

class FeedbackLogger:
    """
    反馈日志记录器，用于记录每轮问答的输入输出、检索统计、LLM评分等信息
    """
    
    def __init__(self, log_file_path: Optional[str] = None, work_dir: Optional[str] = None):
        """
        初始化反馈日志记录器
        
        Args:
            log_file_path: 日志文件路径，如果为None则使用默认路径
            work_dir: 工作目录路径，用于统一管理所有产物
        """
        if log_file_path is None:
            if work_dir:
                # 使用指定的工作目录
                self.log_file_path = os.path.join(work_dir, "feedback.jsonl")
            else:
                # 默认在当前工作目录下创建feedback.jsonl文件
                self.log_file_path = os.path.join(os.getcwd(), "feedback.jsonl")
        else:
            self.log_file_path = log_file_path
            
        # 确保日志文件目录存在
        log_dir = os.path.dirname(self.log_file_path)
        if log_dir:  # 只有当目录路径不为空时才创建
            os.makedirs(log_dir, exist_ok=True)
    
    def log_feedback(self, query: str, result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        记录一轮问答的反馈信息
        
        Args:
            query: 原始查询文本
            result: 查询结果字典，包含答案、来源、处理时间等信息
            config: 配置字典，包含系统配置信息
            
        Returns:
            反馈记录字典，用于策略调优
        """
        try:
            # 构建反馈记录
            feedback_record = self._build_feedback_record(query, result, config)
            
            # 写入JSONL文件
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback_record, ensure_ascii=False) + '\n')
            
            # 返回反馈记录用于策略调优
            return feedback_record
                
        except Exception as e:
            print(f"记录反馈日志时出错: {e}")
            return {}
    
    def _build_feedback_record(self, query: str, result: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建反馈记录字典
        
        Args:
            query: 原始查询文本
            result: 查询结果字典
            config: 配置字典
            
        Returns:
            反馈记录字典
        """
        # 基础记录结构
        record = {
            "timestamp": datetime.now().timestamp(),
            "query_original": query,
            "query_final": result.get("query_rewrite", {}).get("final_query", query),
            "mode": result.get("mode", config.get("mode", "auto")),
            "enhanced_retrieval": result.get("enhanced_retrieval", False),
            "answer": result.get("answer", "")
        }
        
        # 检索统计信息 - 按照需求规范格式
        retrieval_info = {
            "vector_candidates": result.get("vector_candidates", 0),
            "bm25_candidates": result.get("bm25_candidates", 0), 
            "graph_candidates": result.get("graph_candidates", 0),
            "enhanced_graph_candidates": result.get("enhanced_graph_candidates", 0),
            "candidates_after_dedup": result.get("candidates_after_dedup", len(result.get("sources", []))),
            "final_candidates": len(result.get("sources", [])),
            "bm25_enabled": config.get("retrieval", {}).get("enable_bm25", False),
            "graph_enabled": config.get("graph", {}).get("enabled", False),
            "diversity_enabled": config.get("retrieval", {}).get("enable_diversity", False)
        }
        record["retrieval"] = retrieval_info
        
        # 来源信息 - 按照需求规范格式
        sources_info = []
        for idx, source in enumerate(result.get("sources", [])):
            source_record = {
                "rank": idx + 1,
                "id": source.get("id", f"doc_{idx}"),
                "similarity": source.get("similarity", 0.0),
                "source_type": source.get("retrieval_type", source.get("source", "unknown")),
                "retrieval_types": source.get("retrieval_types", [source.get("retrieval_type", "unknown")])
            }
            sources_info.append(source_record)
        record["sources"] = sources_info
        
        # LLM评分和选择信息 - 按照需求规范格式
        selection_info = result.get("answer_selection", {})
        evaluation_info = {
            "method": selection_info.get("method", "single"),
            "candidates_count": selection_info.get("candidates_count", 1),
            "all_scores": selection_info.get("all_scores", []),
            "best_score": selection_info.get("best_score", None),
            "best_reasoning": selection_info.get("best_reasoning", "")
        }
        record["evaluation"] = evaluation_info
        
        # 可选的额外信息
        if "query_type" in result:
            record["query_type"] = result["query_type"]
        
        if "processing_time" in result:
            record["processing_time"] = result["processing_time"]
        
        return record
    
    def _extract_retrieval_methods(self, result: Dict[str, Any]) -> list:
        """
        从结果中提取使用的检索方法
        
        Args:
            result: 查询结果字典
            
        Returns:
            检索方法列表
        """
        methods = set()
        
        # 从来源中提取检索方法
        for source in result.get("sources", []):
            if "retrieval_type" in source:
                methods.add(source["retrieval_type"])
            if "retrieval_types" in source:
                methods.update(source["retrieval_types"])
            if "source" in source:
                methods.add(source["source"])
        
        # 添加其他检索方法标识
        if result.get("enhanced_retrieval"):
            methods.add("enhanced")
        if result.get("graph_info"):
            methods.add("graph")
        
        return list(methods)
    
    def _get_text_preview(self, source: Dict[str, Any], max_length: int = 200) -> str:
        """
        获取来源文本的预览
        
        Args:
            source: 来源字典
            max_length: 最大预览长度
            
        Returns:
            文本预览
        """
        text = ""
        
        # 处理不同的文本格式
        if "text" in source:
            text = source["text"]
        elif "sentences" in source:
            if isinstance(source["sentences"], list):
                text = "\n".join(source["sentences"])
            else:
                text = str(source["sentences"])
        
        # 截断文本
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    def get_log_file_path(self) -> str:
        """
        获取日志文件路径
        
        Returns:
            日志文件路径
        """
        return self.log_file_path
    
    def clear_log(self) -> None:
        """
        清空日志文件
        """
        try:
            if os.path.exists(self.log_file_path):
                os.remove(self.log_file_path)
        except Exception as e:
            print(f"清空日志文件时出错: {e}")
    
    def read_feedback_logs(self, limit: Optional[int] = None) -> list:
        """
        读取反馈日志记录
        
        Args:
            limit: 限制读取的记录数量，None表示读取全部
            
        Returns:
            反馈记录列表
        """
        records = []
        
        try:
            if not os.path.exists(self.log_file_path):
                return records
            
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if limit and line_num >= limit:
                        break
                    
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError as e:
                            print(f"解析第{line_num + 1}行JSON时出错: {e}")
                            
        except Exception as e:
            print(f"读取反馈日志时出错: {e}")
        
        return records