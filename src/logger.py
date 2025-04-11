#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
PyDAS Logger Module
Provides logging functionality for the PyDAS system.
"""
import logging

# 日志级别映射
LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

# 创建logger
logger = logging.getLogger('pydas')

def setup_logger(level='info'):
    """
    Configure the logger for the PyDAS system.
    
    Parameters:
    -----------
    level : str, optional
        Logging level ('debug', 'info', 'warning', 'error', 'critical'), default is 'info'
    
    Returns:
    --------
    logging.Logger
        Configured logger instance
    
    Notes:
    ------
    - Sets the logging level for the PyDAS logger
    - Available levels: 'debug', 'info', 'warning', 'error', 'critical'
    """
    level = level.lower()
    if level not in LOG_LEVELS:
        level = 'info'
        
    # Set the logger level
    log_level = LOG_LEVELS[level]
    logger.setLevel(log_level)
    
    # Add handler if needed
    if not logger.handlers:
        # Avoid adding handlers multiple times
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.info(f"Logger level set to: {level.upper()}")
    
    return logger 