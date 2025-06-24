#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控和内存管理工具
用于监控图构建过程中的性能瓶颈和GPU内存使用情况
"""

import time
import psutil
import logging
import gc
from typing import Dict, Any, Optional
from functools import wraps
from contextlib import contextmanager

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

class PerformanceMonitor:
    """
    性能监控器
    """
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
    def start_timer(self, name: str):
        """开始计时"""
        self.start_times[name] = time.time()
        
    def end_timer(self, name: str) -> float:
        """结束计时并返回耗时"""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.metrics[name] = elapsed
            del self.start_times[name]
            return elapsed
        return 0.0
        
    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        memory_info = {
            'cpu_memory_mb': psutil.virtual_memory().used / 1024 / 1024,
            'cpu_memory_percent': psutil.virtual_memory().percent
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            memory_info.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'gpu_memory_cached_mb': torch.cuda.memory_cached() / 1024 / 1024
            })
            
        if GPUTIL_AVAILABLE:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 假设使用第一个GPU
                memory_info.update({
                    'gpu_utilization_percent': gpu.load * 100,
                    'gpu_memory_used_mb': gpu.memoryUsed,
                    'gpu_memory_total_mb': gpu.memoryTotal,
                    'gpu_memory_free_mb': gpu.memoryFree
                })
                
        return memory_info
        
    def log_performance(self, operation: str):
        """记录性能信息"""
        memory_info = self.get_memory_usage()
        
        logging.info(f"[性能监控] {operation}:")
        logging.info(f"  CPU内存: {memory_info.get('cpu_memory_mb', 0):.1f}MB ({memory_info.get('cpu_memory_percent', 0):.1f}%)")
        
        if 'gpu_memory_allocated_mb' in memory_info:
            logging.info(f"  GPU内存(已分配): {memory_info['gpu_memory_allocated_mb']:.1f}MB")
            logging.info(f"  GPU内存(已保留): {memory_info['gpu_memory_reserved_mb']:.1f}MB")
            
        if 'gpu_utilization_percent' in memory_info:
            logging.info(f"  GPU利用率: {memory_info['gpu_utilization_percent']:.1f}%")
            logging.info(f"  GPU内存使用: {memory_info['gpu_memory_used_mb']:.1f}MB / {memory_info['gpu_memory_total_mb']:.1f}MB")
            
    def clear_gpu_cache(self):
        """清理GPU缓存"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("[内存管理] GPU缓存已清理")
            
    def force_garbage_collection(self):
        """强制垃圾回收"""
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("[内存管理] 垃圾回收完成")

# 全局性能监控器实例
performance_monitor = PerformanceMonitor()

def monitor_performance(operation_name: str):
    """性能监控装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            performance_monitor.start_timer(operation_name)
            performance_monitor.log_performance(f"{operation_name} - 开始")
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = performance_monitor.end_timer(operation_name)
                performance_monitor.log_performance(f"{operation_name} - 完成 (耗时: {elapsed:.2f}s)")
                
                # 如果操作耗时较长，进行内存清理
                if elapsed > 10:  # 超过10秒
                    performance_monitor.force_garbage_collection()
                    
        return wrapper
    return decorator

@contextmanager
def memory_management_context(operation_name: str):
    """内存管理上下文管理器"""
    performance_monitor.log_performance(f"{operation_name} - 开始")
    initial_memory = performance_monitor.get_memory_usage()
    
    try:
        yield
    finally:
        final_memory = performance_monitor.get_memory_usage()
        
        # 计算内存增长
        cpu_growth = final_memory.get('cpu_memory_mb', 0) - initial_memory.get('cpu_memory_mb', 0)
        gpu_growth = final_memory.get('gpu_memory_allocated_mb', 0) - initial_memory.get('gpu_memory_allocated_mb', 0)
        
        logging.info(f"[内存监控] {operation_name} - 内存变化:")
        logging.info(f"  CPU内存增长: {cpu_growth:.1f}MB")
        if gpu_growth != 0:
            logging.info(f"  GPU内存增长: {gpu_growth:.1f}MB")
            
        # 如果内存增长过多，进行清理
        if cpu_growth > 500 or gpu_growth > 200:  # CPU增长超过500MB或GPU增长超过200MB
            logging.warning(f"[内存警告] {operation_name} 内存增长过多，执行清理")
            performance_monitor.force_garbage_collection()

def check_gpu_memory_usage() -> bool:
    """检查GPU内存使用情况，如果使用率过高返回True"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return False
        
    memory_info = performance_monitor.get_memory_usage()
    gpu_memory_percent = (memory_info.get('gpu_memory_used_mb', 0) / 
                         memory_info.get('gpu_memory_total_mb', 1)) * 100
    
    if gpu_memory_percent > 85:  # GPU内存使用超过85%
        logging.warning(f"[GPU内存警告] GPU内存使用率过高: {gpu_memory_percent:.1f}%")
        return True
        
    return False

def optimize_for_memory():
    """内存优化建议"""
    memory_info = performance_monitor.get_memory_usage()
    
    suggestions = []
    
    # CPU内存检查
    if memory_info.get('cpu_memory_percent', 0) > 80:
        suggestions.append("CPU内存使用率过高，建议减少批处理大小或启用分段处理")
        
    # GPU内存检查
    if 'gpu_memory_used_mb' in memory_info:
        gpu_usage_percent = (memory_info['gpu_memory_used_mb'] / memory_info['gpu_memory_total_mb']) * 100
        if gpu_usage_percent > 80:
            suggestions.append("GPU内存使用率过高，建议减少模型批处理大小或使用CPU模式")
            
    # GPU利用率检查
    if memory_info.get('gpu_utilization_percent', 0) > 95:
        suggestions.append("GPU利用率过高但处理缓慢，可能存在内存瓶颈或模型加载问题")
        
    if suggestions:
        logging.warning("[性能优化建议]:")
        for suggestion in suggestions:
            logging.warning(f"  - {suggestion}")
            
    return suggestions