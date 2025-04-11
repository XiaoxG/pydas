#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
PyDAS - Python Data Analysis System
==================================

A comprehensive data analysis system for processing and analyzing large time series data.

Version: 1.0.3
Author: Xiaoxiang Guo
"""

from .pydas import PyDAS
from .utils import diff1d, data_change_fs  # 从utils模块导入通用函数，保持向后兼容性
from .output import write_data, export_to_dat, export_to_mat  # 从output模块导入数据输出函数

__version__ = "1.0.3"
__author__ = "Xiaoxiang Guo" 