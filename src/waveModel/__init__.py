"""
WaveModel - Wave Modeling and Spectral Analysis Library
======================================================

A library for wave modeling, spectral analysis, and oceanographic calculations.

This module is part of the PyDAS package.
"""

# Core functionality - 这些不太可能引起循环导入
from .core import discretize, nextpow2, ecross
from .misc import (findcross, findpeaks, findrfc, moment, polar2cart, cart2polar,
                   gravity, moving_average)


# 定义__all__列表，但不导入可能引起循环导入的类
__all__ = [
    # Core functionality
    'discretize', 'nextpow2', 'ecross',
    
    # Data classes
    'SpecData1D', 'SpecData2D', 'TimeSeries', 'CovData1D',
    
    # Wave models
    'Jonswap', 'Torsethaugen',
    
    # Dispersion relations
    'k2w', 'w2k',
    
    # Utility functions
    'findcross', 'findpeaks', 'findrfc', 'moment', 
    'polar2cart', 'cart2polar', 'gravity', 'moving_average'
]

# 延迟导入可能引起循环导入的类
# 注意: 这些导入会在__init__.py完全加载后执行
from .timeseries import TimeSeries
from .covdata import CovData1D
from .wavemodels import Jonswap, Torsethaugen, k2w, w2k
from .specdata import SpecData1D, SpecData2D