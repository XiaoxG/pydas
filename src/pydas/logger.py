"""PyDAS Logger Module

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


def get_logger(name=None):
    """Return a logger under the ``pydas`` hierarchy.

    Parameters
    ----------
    name : str, optional
        Logger name. Bare names become children of ``pydas``. Names that
        already start with ``pydas`` are used as-is. ``None`` returns the
        package logger.

    Returns
    -------
    logging.Logger
    """
    if not name:
        return logging.getLogger('pydas')
    if name == 'pydas' or name.startswith('pydas.'):
        return logging.getLogger(name)
    return logging.getLogger('pydas').getChild(name)


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