"""PyDAS Core - Main Object"""
import logging
import pandas as pd
import numpy as np
import os
import struct
import math

from .io_mixin import IOMixin
from .channel_mixin import ChannelMixin
from .processing_mixin import ProcessingMixin
from .plot_mixin import PlotMixin
from .analysis_mixin import AnalysisMixin
from .report_mixin import ReportMixin

from ..logger import setup_logger

logger = logging.getLogger(__name__)

class PyDAS(IOMixin, ChannelMixin, ProcessingMixin, PlotMixin, AnalysisMixin, ReportMixin):
    """
    Python Data Analysis System for processing and analyzing time series data.
    """
    def __init__(self, filename, lam, sseg='all', log_level='info'):
        """
        Initialize PyDAS object and read data file.
        
        Parameters:
        -----------
        filename : str
            Path to the data file
        lam : float, optional
            Scale factor for data conversion, default is 1
        sseg : int or str, optional
            Segment index to read, 'all' for all segments, default is 'all'
        log_level : str, optional
            Logging level ('debug', 'info', 'warning', 'error', 'critical'), default is 'info'
            
        Notes:
        ------
        - Automatically detects file format and reads accordingly
        - Processes channel information and data segments
        - Calculates basic statistics for each channel
        """
        # Configure logger
        # self.set_logger(log_level)  # 使用新的setup_logger函数
        setup_logger(log_level)
        
        # Initialize basic properties
        self.__lam__ = lam
        self.__fs__ = 1  # Default sampling frequency, will be updated during reading
        self.__chN__ = 0  # Number of channels
        self.__segN__ = 0  # Number of segments
        self.__scale__ = 'model'  # Default scale is model scale
        
        # Validate segment selection
        if not (isinstance(sseg, int) or sseg == 'all'):
            logger.error("Input 'sseg' is illegal (should be int or 'all').")
            raise ValueError("sseg must be an integer or 'all'")
        
        # If filename is provided, read the data file
        if filename is not None:
            # Validate file existence
            if os.path.exists(filename):
                self.__filename__ = filename
            else:
                raise FileNotFoundError(f"File {filename} does not exist.")
            
            # Read the data file
            self.__read__(sseg)
        else:
            # Initialize empty data structures for manual data setting
            self.__filename__ = None
            self.chInfo = pd.DataFrame(columns=['Name', 'Unit', 'Coef'])
            self.data = {}
            logger.info("Created empty PyDAS object. Use load() method to read data or set data manually.")

