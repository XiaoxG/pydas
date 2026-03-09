"""
PyDAS - Python Data Analysis System
====================================
A comprehensive data analysis system for processing and analyzing
large time series data in ocean engineering and signal processing.

Usage::

    from pydas import PyDAS
    data = PyDAS(filename='data.out', lam=36)
    data.print_statistics()
"""

from .core.pydas_obj import PyDAS
from .utils import diff1d, data_change_fs

__all__ = ["PyDAS", "diff1d", "data_change_fs"]
