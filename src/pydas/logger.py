#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
PyDAS Logger Module
Provides logging functionality for the PyDAS system.
"""

import logging

LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

logger = logging.getLogger('pydas')


def get_logger(name):
    """Get a logger with the specified name.

    Parameters
    ----------
    name : str
        Name of the logger.

    Returns
    -------
    logging.Logger
        Logger instance with the specified name.

    Notes
    -----
    - Returns a child logger of the PyDAS logger.
    - Inherits level and handlers from the parent logger.
    """
    return logging.getLogger(name)


def setup_logger(level='info'):
    """Configure the logger for the PyDAS system.

    Parameters
    ----------
    level : str, optional
        Logging level ('debug', 'info', 'warning', 'error', 'critical'),
        default is 'info'.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    level = level.lower()
    if level not in LOG_LEVELS:
        level = 'info'

    # Set the logger level
    log_level = LOG_LEVELS[level]
    logger.setLevel(log_level)

    # Add handler if needed
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info(f"Logger level set to: {level.upper()}")
    return logger