"""
Data Processing Module for WaveModel
------------------------------------

This module consolidates all data processing classes from the WaveModel package.
"""

# Import Base container class
from waveModel.datacontainer import DataContainer, AxisLabels

# Import specific data container classes
from waveModel.timeseries import TimeSeries
from waveModel.specdata import SpecData1D, SpecData2D
from waveModel.covdata import CovData1D

__all__ = [
    'DataContainer', 'AxisLabels',
    'TimeSeries', 
    'SpecData1D', 'SpecData2D',
    'CovData1D'
]


def array2timeseries(x):
    """
    Convert 2D arrays to TimeSeries object
        assuming 1st column is time and the remaining columns contain data.
    
    Parameters
    ----------
    x : array-like
        2D array with first column as time and remaining columns as data
        
    Returns
    -------
    ts : TimeSeries
        TimeSeries object
    """
    import numpy as np
    return TimeSeries(x[:, 1::], x[:, 0].ravel()) 