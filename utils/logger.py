import logging
import os
from loguru import logger
import sys
from typing import Optional

def setup_logger(work_dir: str, log_level: str = "INFO") -> logger:
    """
    设置统一的日志系统，使用loguru替代标准logging
    
    Args:
        work_dir: 工作目录，日志文件将保存在此目录下
        log_level: 日志级别，默认为INFO
        
    Returns:
        logger: loguru日志对象
    """
    # 创建日志目录
    log_path = os.path.join(work_dir, "log.txt")
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台处理器
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 添加文件处理器
    logger.add(
        log_path,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8"
    )
    
    return logger

# 兼容旧版API的函数
def init_logger(work_dir: str) -> logger:
    """
    兼容旧版API的日志初始化函数
    
    Args:
        work_dir: 工作目录
        
    Returns:
        logger: loguru日志对象
    """
    return setup_logger(work_dir)

# 创建一个进度条包装器
def get_progress_bar(iterable=None, total=None, desc="Processing", **kwargs):
    """
    获取一个进度条对象，统一使用tqdm
    
    Args:
        iterable: 可迭代对象
        total: 总数
        desc: 描述
        **kwargs: 其他参数
        
    Returns:
        tqdm对象
    """
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, **kwargs)
    except ImportError:
        # 如果没有安装tqdm，返回一个简单的进度显示
        class SimpleProg:
            def __init__(self, iterable=None, total=None, desc=""):
                self.iterable = iterable
                self.total = total or (len(iterable) if iterable is not None else 100)
                self.desc = desc
                self.n = 0
                self.last_print = 0
                print(f"{desc}: 0%")
                
            def update(self, n=1):
                self.n += n
                pct = int(self.n / self.total * 100)
                if pct > self.last_print + 10:  # 每10%打印一次
                    print(f"{self.desc}: {pct}%")
                    self.last_print = pct
                    
            def __iter__(self):
                if self.iterable is None:
                    return range(self.total)
                self.iter = iter(self.iterable)
                return self
                
            def __next__(self):
                try:
                    obj = next(self.iter)
                    self.update()
                    return obj
                except StopIteration:
                    print(f"{self.desc}: 100%")
                    raise
                    
            def close(self):
                pass
                
        return SimpleProg(iterable, total, desc)