"""
WaveModel - Wave Modeling and Spectral Analysis Library
======================================================

A library for wave modeling, spectral analysis, and oceanographic calculations.

This module is part of the PyDAS package.
"""

# Core functionality
from .core import discretize, nextpow2, ecross

# Spectral data and models
from .data import SpecData1D, SpecData2D, TimeSeries, CovData1D
from .wavemodels import Jonswap, Torsethaugen, k2w, w2k

# Common utility functions
from .misc import (findcross, findpeaks, findrfc, moment, polar2cart, cart2polar,
                   gravity, moving_average)

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
