#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
PyDAS Python Data Analysis System

A Python package for processing and analyzing data from model tests and other sources.

Main modules:
- pydas: Core functionality for data loading, processing and analysis
- plot: Visualization functions
- analysis: Statistical and spectral analysis functions
- reporting: Excel report generation and data export functions

Version: 1.0.3
Author: Xiaoxiang Guo
"""

__version__ = '1.0.3'

try:
    from .pydas import PyDAS
except ImportError:
    import warnings
    warnings.warn("Unable to import PyDAS class. Some dependencies might be missing.")

try:
    from . import plot
    from . import analysis
    from . import reporting
except ImportError:
    import warnings
    warnings.warn("Unable to import one or more modules. Some functionality may be limited.")

from .utils import diff1d, data_change_fs  # 从utils模块导入通用函数，保持向后兼容性
from .output import write_data, export_to_dat, export_to_mat  # 从output模块导入数据输出函数
from .process import (  # 从process模块导入数据处理函数
    apply_lowpass_filter,
    apply_highpass_filter,
    remove_mean,
    add_value,
    multiply_value,
    move_data,
    data_wash,
    add_diff1,
    add_diff2
)

__author__ = "Xiaoxiang Guo" 