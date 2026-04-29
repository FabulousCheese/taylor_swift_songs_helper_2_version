"""
日志工具模块
"""
import logging
import sys
from .config import LOG_LEVEL, LOG_FORMAT

def get_logger(name: str) -> logging.Logger:
    """获取logger实例"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL))
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, LOG_LEVEL))
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
    
    return logger
