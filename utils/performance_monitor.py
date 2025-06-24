"""性能监控和分析工具

提供性能分析装饰器、指标收集器和性能报告功能。
"""

import time
import functools
import psutil
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import json
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    function_name: str
    execution_time: float
    memory_before: float
    memory_after: float
    memory_peak: float
    cpu_percent: float
    timestamp: float
    args_info: str = ""
    kwargs_info: str = ""
    
    @property
    def memory_delta(self) -> float:
        """内存变化量 (MB)"""
        return self.memory_after - self.memory_before
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'function_name': self.function_name,
            'execution_time': self.execution_time,
            'memory_before': self.memory_before,
            'memory_after': self.memory_after,
            'memory_peak': self.memory_peak,
            'memory_delta': self.memory_delta,
            'cpu_percent': self.cpu_percent,
            'timestamp': self.timestamp,
            'args_info': self.args_info,
            'kwargs_info': self.kwargs_info
        }


class PerformanceCollector:
    """性能指标收集器"""
    
    def __init__(self, max_records: int = 1000):
        """初始化性能收集器
        
        Args:
            max_records: 最大记录数量
        """
        self.max_records = max_records
        self.metrics: deque = deque(maxlen=max_records)
        self.function_stats: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        self._process = psutil.Process(os.getpid())
    
    def add_metric(self, metric: PerformanceMetrics) -> None:
        """添加性能指标
        
        Args:
            metric: 性能指标对象
        """
        with self.lock:
            self.metrics.append(metric)
            self.function_stats[metric.function_name].append(metric.execution_time)
    
    def get_function_stats(self, function_name: str) -> Dict[str, float]:
        """获取函数统计信息
        
        Args:
            function_name: 函数名称
            
        Returns:
            统计信息字典
        """
        with self.lock:
            times = self.function_stats.get(function_name, [])
            if not times:
                return {}
            
            return {
                'count': len(times),
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'median_time': sorted(times)[len(times) // 2] if times else 0
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有函数统计信息
        
        Returns:
            所有函数的统计信息
        """
        with self.lock:
            return {func: self.get_function_stats(func) 
                   for func in self.function_stats.keys()}
    
    def get_recent_metrics(self, count: int = 10) -> List[PerformanceMetrics]:
        """获取最近的性能指标
        
        Args:
            count: 获取数量
            
        Returns:
            最近的性能指标列表
        """
        with self.lock:
            return list(self.metrics)[-count:]
    
    def clear(self) -> None:
        """清空所有指标"""
        with self.lock:
            self.metrics.clear()
            self.function_stats.clear()
    
    def export_to_json(self, filepath: str) -> None:
        """导出指标到JSON文件
        
        Args:
            filepath: 文件路径
        """
        with self.lock:
            data = {
                'metrics': [metric.to_dict() for metric in self.metrics],
                'function_stats': self.get_all_stats(),
                'export_time': time.time()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用量 (MB)"""
        try:
            memory_info = self._process.memory_info()
            return memory_info.rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def get_cpu_percent(self) -> float:
        """获取CPU使用率"""
        try:
            return self._process.cpu_percent()
        except Exception:
            return 0.0


# 全局性能收集器实例
_global_collector = PerformanceCollector()


def performance_monitor(include_args: bool = False, 
                       include_memory: bool = True,
                       include_cpu: bool = True):
    """性能监控装饰器
    
    Args:
        include_args: 是否包含参数信息
        include_memory: 是否监控内存使用
        include_cpu: 是否监控CPU使用
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 记录开始时间和资源使用
            start_time = time.time()
            memory_before = _global_collector.get_memory_usage() if include_memory else 0.0
            cpu_before = _global_collector.get_cpu_percent() if include_cpu else 0.0
            
            # 记录参数信息
            args_info = ""
            kwargs_info = ""
            if include_args:
                try:
                    args_info = str(args)[:100] + "..." if len(str(args)) > 100 else str(args)
                    kwargs_info = str(kwargs)[:100] + "..." if len(str(kwargs)) > 100 else str(kwargs)
                except Exception:
                    args_info = "<无法序列化>"
                    kwargs_info = "<无法序列化>"
            
            # 执行函数
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # 记录结束时间和资源使用
                end_time = time.time()
                execution_time = end_time - start_time
                memory_after = _global_collector.get_memory_usage() if include_memory else 0.0
                cpu_after = _global_collector.get_cpu_percent() if include_cpu else 0.0
                
                # 创建性能指标
                metric = PerformanceMetrics(
                    function_name=f"{func.__module__}.{func.__name__}",
                    execution_time=execution_time,
                    memory_before=memory_before,
                    memory_after=memory_after,
                    memory_peak=max(memory_before, memory_after),
                    cpu_percent=(cpu_before + cpu_after) / 2,
                    timestamp=start_time,
                    args_info=args_info,
                    kwargs_info=kwargs_info
                )
                
                # 添加到收集器
                _global_collector.add_metric(metric)
        
        return wrapper
    return decorator


def get_performance_collector() -> PerformanceCollector:
    """获取全局性能收集器
    
    Returns:
        性能收集器实例
    """
    return _global_collector


def generate_performance_report(output_file: Optional[str] = None) -> str:
    """生成性能报告
    
    Args:
        output_file: 输出文件路径，如果为None则返回字符串
        
    Returns:
        性能报告内容
    """
    collector = get_performance_collector()
    all_stats = collector.get_all_stats()
    recent_metrics = collector.get_recent_metrics(20)
    
    report_lines = [
        "# 性能分析报告",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 函数性能统计",
        ""
    ]
    
    # 函数统计表格
    if all_stats:
        report_lines.extend([
            "| 函数名 | 调用次数 | 总耗时(s) | 平均耗时(s) | 最小耗时(s) | 最大耗时(s) |",
            "|--------|----------|-----------|-------------|-------------|-------------|"
        ])
        
        for func_name, stats in sorted(all_stats.items(), 
                                      key=lambda x: x[1].get('total_time', 0), 
                                      reverse=True):
            report_lines.append(
                f"| {func_name} | {stats['count']} | "
                f"{stats['total_time']:.3f} | {stats['avg_time']:.3f} | "
                f"{stats['min_time']:.3f} | {stats['max_time']:.3f} |"
            )
    else:
        report_lines.append("暂无性能数据")
    
    report_lines.extend([
        "",
        "## 最近执行记录",
        ""
    ])
    
    # 最近执行记录
    if recent_metrics:
        report_lines.extend([
            "| 函数名 | 执行时间(s) | 内存变化(MB) | CPU使用率(%) | 时间戳 |",
            "|--------|-------------|--------------|--------------|--------|"
        ])
        
        for metric in recent_metrics:
            timestamp_str = time.strftime('%H:%M:%S', time.localtime(metric.timestamp))
            report_lines.append(
                f"| {metric.function_name} | {metric.execution_time:.3f} | "
                f"{metric.memory_delta:.2f} | {metric.cpu_percent:.1f} | {timestamp_str} |"
            )
    else:
        report_lines.append("暂无执行记录")
    
    # 性能建议
    report_lines.extend([
        "",
        "## 性能优化建议",
        ""
    ])
    
    if all_stats:
        # 找出最耗时的函数
        slowest_func = max(all_stats.items(), key=lambda x: x[1].get('total_time', 0))
        most_called_func = max(all_stats.items(), key=lambda x: x[1].get('count', 0))
        
        report_lines.extend([
            f"1. **最耗时函数**: {slowest_func[0]} (总耗时: {slowest_func[1]['total_time']:.3f}s)",
            f"2. **调用最频繁函数**: {most_called_func[0]} (调用次数: {most_called_func[1]['count']})",
            "3. **优化建议**:",
            "   - 对于耗时较长的函数，考虑算法优化或并行处理",
            "   - 对于调用频繁的函数，考虑缓存机制",
            "   - 监控内存使用，避免内存泄漏",
            "   - 使用批处理减少函数调用开销"
        ])
    
    report_content = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    return report_content


# 便捷函数
def monitor_function(func: Callable) -> Callable:
    """简单的函数监控装饰器
    
    Args:
        func: 要监控的函数
        
    Returns:
        装饰后的函数
    """
    return performance_monitor()(func)


def clear_performance_data() -> None:
    """清空性能数据"""
    _global_collector.clear()


def export_performance_data(filepath: str) -> None:
    """导出性能数据
    
    Args:
        filepath: 导出文件路径
    """
    _global_collector.export_to_json(filepath)