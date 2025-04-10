#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
PyDAS - Python Data Analysis System
A comprehensive data analysis system for processing and analyzing time series data.

Function Categories:
------------------
1. Channel Operations
   a) Channel Management:
      - add_channel: Add a new channel
      - delete_channel: Delete specified channel
      - select_channels: Select and keep specified channels
      - rename_channel: Rename channel
      - change_channel_order: Change channel order
   
   b) Channel Data Processing:
      - remove_mean: Remove mean from channel data
      - add_value: Add constant value to channel data
      - multiply_value: Multiply channel data by constant
      - cut_series: Cut time series to specified range
      - move_data: Move channel data
      - data_wash: Clean data, detect and interpolate outliers

2. Channel Calculations
   a) Differential Operations:
      - add_diff1: Calculate and add first derivative
      - add_diff2: Calculate and add second derivative
   
   b) Filtering:
      - apply_lowpass_filter: Apply lowpass filter
      - apply_highpass_filter: Apply highpass filter
   
   c) Data Alignment:
      - move_ccor: Move channel data using cross-correlation
      - find_move_ccor: Find points to move between channels

3. Data Output
   a) File Output:
      - to_dat: Export data to DAT file
      - to_mat: Export data to MAT file
      - write: Write data file
   
   b) Information Output:
      - print_info: Print basic information
      - print_channel_info: Print channel information
      - print_statistics: Print statistical information

4. Data Visualization
   - plot_channel: Plot channel data with interactive features and performance optimization
   - plot_histogram: Generate histograms with statistics and Gaussian fitting capabilities
   - plot_xy: Create XY scatter plots with density visualization, downsampling, and linear regression
   - spectral_analysis: Perform spectral analysis on channels with customizable parameters

5. Data Import
   - read_waveCal: Read wave calibration data
   - read_motion: Read motion data and add as channels

6. Data Conversion
   - fix_unit: Fix channel unit
   - to_fullscale: Convert model scale data to prototype scale

7. Data Update
   - updateST: Update statistical information
   - updateChN: Update channel count

8. Global Utility Functions
   - diff1d: Calculate derivative of one-dimensional array
   - data_change_fs: Change data sampling frequency

Performance Optimizations:
------------------------
1. Numba Acceleration
   - JIT compilation for compute-intensive functions
   - Applied to derivative calculation and data resampling

2. Vectorized Operations
   - Pandas vectorized operations for statistics
   - Numpy vectorized operations for dataset processing

3. Cache Optimization
   - Cache for unit conversion calculations
   - Pre-calculation of unique unit conversions

4. Other Optimizations
   - Reduced data copying and conversion
   - Efficient algorithms and data structures
   - Automatic downsampling for large datasets
   - WebGL rendering for interactive visualization

Dependencies:
------------
- numpy: Numerical computing
- pandas: Data manipulation
- numba: JIT compilation
- plotly: Interactive visualization
- scipy: Scientific computing
- matplotlib: Static visualization and fallback rendering

Author: Xiaoxian Guo
Date: 2024-03-20
Version: 1.0.1
"""
import re
import sys
import os
import struct
import math
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import correlate
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
import logging
import warnings
from waveModel.timeseries import TimeSeries
import datetime

# Import numba for acceleration
try:
    from numba import jit, float64, int32, void, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available. Some functions will run slower.")

# Initialize module-level logger
logger = logging.getLogger(__name__)

# 日志级别映射
LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

def diff1d(series, dx=1.0):
    """
    Calculate the derivative of a one-dimensional array with optimized performance.
    
    This function uses Numba JIT compilation for large arrays to improve performance.
    For different array sizes, it automatically selects the most efficient implementation:
    - Small arrays (< 1000): Standard numpy implementation
    - Medium arrays (1000-10000): Basic Numba implementation
    - Large arrays (10000-100000): Parallel Numba implementation
    - Huge arrays (> 100000): Optimized parallel Numba implementation
    
    Parameters:
    -----------
    series : numpy.ndarray
        Input array to calculate derivative
    dx : float, optional
        Time step, default is 1.0
        
    Returns:
    --------
    numpy.ndarray
        Derivative array with the same length as input
        
    Notes:
    ------
    - Uses Numba JIT compilation for performance optimization
    - Automatically handles different array sizes
    - Maintains numerical accuracy for various input sizes
    """
    # Convert input to numpy array
    series_array = np.asarray(series, dtype=np.float64)
    
    # Check input array size and select the most appropriate implementation
    if len(series_array) <= 1:
        return np.zeros_like(series_array)
        
    # If numba is available, use the accelerated version
    if NUMBA_AVAILABLE:
        # Select the appropriate numba optimized version
        if len(series_array) > 10000000:  # Extremely large dataset
            return _diff1d_numba_huge(series_array, dx)
        elif len(series_array) > 1000000:  # Large dataset
            return _diff1d_numba_large(series_array, dx)
        else:  # Small to medium dataset
            return _diff1d_numba(series_array, dx)
    
    # Original implementation (fallback if numba is not available)
    n = len(series_array)
    dy = np.zeros_like(series_array)
    
    if n <= 6:
        # For very small arrays, use simple central difference
        if n == 1:
            return np.zeros_like(series_array)
        elif n == 2:
            dy[0] = (series_array[1] - series_array[0]) / dx
            dy[1] = dy[0]
            return dy

# 使用numba加速的版本
if NUMBA_AVAILABLE:
    @jit(float64[:](float64[:], float64), nopython=True, parallel=True, fastmath=True, cache=True)
    def _diff1d_numba(y, dx):
        """
        Numba-accelerated implementation of the derivative calculation.
        
        This function uses Numba JIT compilation to accelerate the derivative calculation
        for medium-sized arrays.
        """
        n = len(y)
        dy = np.zeros(n, dtype=np.float64)
        
        # Check if array is large enough
        if n <= 5:
            return dy
        
        # Forward difference (first point, 2nd order)
        dy[0] = (-y[2] + 4 * y[1] - 3 * y[0]) / (2 * dx)
        
        # Forward difference (second point, 3rd order)
        dy[1] = (-y[3] + 6 * y[2] - 3 * y[1] - 2 * y[0]) / (6 * dx)
        
        # Central difference (third point, 4th order)
        dy[2] = (8 * (y[3] - y[1]) - (y[4] - y[0])) / (12 * dx)
        
        # Use prange for parallel processing of interior points
        for i in prange(3, n - 3):
            dy[i] = (45 * (y[i+1] - y[i-1]) - 9 * (y[i+2] - y[i-2]) + (y[i+3] - y[i-3])) / (60 * dx)
        
        # Central difference (third-to-last point, 4th order)
        dy[n-3] = (8 * (y[n-2] - y[n-4]) - (y[n-1] - y[n-5])) / (12 * dx)
        
        # Backward difference (second-to-last point, 3rd order)
        dy[n-2] = (2 * y[n-1] + 3 * y[n-2] - 6 * y[n-3] + y[n-4]) / (6 * dx)
        
        # Backward difference (last point, 2nd order)
        dy[n-1] = (3 * y[n-1] - 4 * y[n-2] + y[n-3]) / (2 * dx)
        
        return dy
    
    # Optimized version for large data
    @jit(float64[:](float64[:], float64), nopython=True, fastmath=True, cache=True)
    def _diff1d_numba_large(y, dx):
        """
        Optimized implementation for large arrays.
        
        This function uses a simplified approach for large arrays to balance
        accuracy and performance.
        """
        n = len(y)
        dy = np.zeros(n, dtype=np.float64)
        
        # Handle boundary points
        dy[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * dx)
        dy[1] = (-2 * y[0] - 3 * y[1] + 6 * y[2] - y[3]) / (6 * dx)
        dy[2] = (y[0] - 8 * y[1] + 8 * y[3] - y[4]) / (12 * dx)
        
        # Interior points using vectorized operations (more efficient)
        # This loop will be automatically optimized in numba
        for i in range(3, n - 3):
            dy[i] = (y[i+1] - y[i-1]) / (2 * dx)
        
        dy[n-3] = (y[n-5] - 8 * y[n-3] + 8 * y[n-1]) / (12 * dx)
        dy[n-2] = (y[n-4] - 6 * y[n-3] + 3 * y[n-2] + 2 * y[n-1]) / (6 * dx)
        dy[n-1] = (y[n-3] - 4 * y[n-2] + 3 * y[n-1]) / (2 * dx)
        
        return dy
    
    # Optimized version for huge data using chunked processing
    @jit(float64[:](float64[:], float64), nopython=True, parallel=True, fastmath=True, cache=True)
    def _diff1d_numba_huge(y, dx):
        """
        Optimized implementation for extremely large arrays.
        
        This function uses chunked processing and parallel computation to handle
        very large arrays efficiently.
        """
        n = len(y)
        dy = np.zeros(n, dtype=np.float64)
        
        if n <= 1000000:  # For smaller arrays, use regular method
            return _diff1d_numba_large(y, dx)
        
        # Handle boundary points (first 3 points)
        dy[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * dx)
        dy[1] = (-2 * y[0] - 3 * y[1] + 6 * y[2] - y[3]) / (6 * dx)
        dy[2] = (y[0] - 8 * y[1] + 8 * y[3] - y[4]) / (12 * dx)
        
        # Handle boundary points (last 3 points)
        dy[n-3] = (y[n-5] - 8 * y[n-3] + 8 * y[n-1]) / (12 * dx)
        dy[n-2] = (y[n-4] - 6 * y[n-3] + 3 * y[n-2] + 2 * y[n-1]) / (6 * dx)
        dy[n-1] = (y[n-3] - 4 * y[n-2] + 3 * y[n-1]) / (2 * dx)
        
        # Process interior points in chunks
        chunk_size = 1000000  # Size per chunk
        n_chunks = (n - 6 + chunk_size - 1) // chunk_size  # Round up to get number of chunks
        
        # Precompute coefficients
        coef = 1.0 / (2.0 * dx)
        
        # Process each chunk in parallel
        for chunk in prange(n_chunks):
            start = 3 + chunk * chunk_size
            end = min(n - 3, start + chunk_size)
            
            for i in range(start, end):
                dy[i] = (y[i+1] - y[i-1]) * coef
        
        return dy

def data_change_fs(series, fs, fs_new):
    """
    Change the sampling frequency of data using linear interpolation with optimized performance.
    
    This function provides multiple implementations for different data sizes:
    - Small arrays: Standard scipy interpolation
    - Medium arrays: Basic Numba implementation
    - Large arrays: Parallel Numba implementation
    - Huge arrays: Optimized parallel Numba implementation
    
    Parameters:
    -----------
    series : numpy.ndarray
        Input time series data
    fs : float
        Original sampling frequency in Hz
    fs_new : float
        New sampling frequency in Hz
        
    Returns:
    --------
    numpy.ndarray
        Resampled data at the new sampling frequency
        
    Notes:
    ------
    - Uses linear interpolation for resampling
    - Automatically selects optimal implementation based on data size
    - Maintains signal integrity during resampling
    - Handles edge cases and potential extrapolation issues
    """
    # 检查输入数组大小，选择最合适的实现
    series_array = np.asarray(series, dtype='float64')
    
    # 如果numba可用，使用加速版本
    if NUMBA_AVAILABLE:
        # 选择适合的numba优化版本
        if len(series_array) > 10000000:  # 超大数据集
            return _data_change_fs_numba_huge(series_array, fs, fs_new)
        elif len(series_array) > 1000000:  # 大数据集
            return _data_change_fs_numba_fast(series_array, fs, fs_new)
        else:  # 中小型数据集
            return _data_change_fs_numba(series_array, fs, fs_new)
    else:
        # 原始实现
        # Calculate the total time duration of the original signal
        total_time = 1 / fs * len(series)
        
        # Create time vectors for original and new sampling rates
        # Note: Subtracting 5/fs_new to avoid potential extrapolation issues
        x_new = np.arange(0, total_time - 5 / fs_new, 1 / fs_new)
        x_series = np.arange(0, total_time, 1 / fs)
        
        # Ensure x_series matches the length of the input series
        x_series = x_series[:len(series)]
        
        # Create interpolation function and apply it
        series_interp_func = interpolate.interp1d(
            x_series, series, kind='linear', axis=0, fill_value=(0, 0))
        series_interp = series_interp_func(x_new)
        
        return series_interp

# 使用numba加速的版本
if NUMBA_AVAILABLE:
    @jit(float64[:](float64[:], float64, float64), nopython=True, fastmath=True, cache=True)
    def _data_change_fs_numba(series, fs, fs_new):
        """Numba加速版本的data_change_fs函数"""
        # 计算原始信号的总时长
        total_time = 1 / fs * len(series)
        
        # 创建原始和新的时间向量
        x_new = np.arange(0, total_time - 5 / fs_new, 1 / fs_new)
        x_series = np.arange(0, total_time, 1 / fs)
        
        # 确保x_series与输入序列长度匹配
        if len(x_series) > len(series):
            x_series = x_series[:len(series)]
        
        # 使用numpy的interp函数进行插值
        return np.interp(x_new, x_series, series)
    
    @jit(float64[:](float64[:], float64, float64), nopython=True, fastmath=True, parallel=True, cache=True)
    def _data_change_fs_numba_fast(series, fs, fs_new):
        """针对大型数据集优化的data_change_fs函数"""
        # 由于大数据集上直接使用interp可能占用大量内存，这里使用分块处理方法
        # 计算原始信号的总时长
        total_time = 1 / fs * len(series)
        
        # 创建新的时间向量
        x_new = np.arange(0, total_time - 5 / fs_new, 1 / fs_new)
        x_series = np.arange(0, total_time, 1 / fs)
        
        # 确保x_series与输入序列长度匹配
        if len(x_series) > len(series):
            x_series = x_series[:len(series)]
        
        # 创建结果数组
        result = np.zeros(len(x_new))
        
        # 计算转换比例
        ratio = fs / fs_new
        
        # 对于每个目标时间点，找到最近的两个源时间点并进行线性插值
        for i in range(len(x_new)):
            # 找到x_new[i]对应的在原数组中的位置（非整数）
            pos = x_new[i] * fs
            
            # 找到左右两个整数索引
            pos_left = int(pos)
            pos_right = pos_left + 1
            
            # 确保索引在有效范围内
            if pos_right >= len(series):
                pos_right = len(series) - 1
            
            # 计算插值权重
            weight_right = pos - pos_left
            weight_left = 1.0 - weight_right
            
            # 线性插值
            if pos_left < len(series):
                result[i] = weight_left * series[pos_left] + weight_right * series[pos_right]
            
        return result
        
    @jit(float64[:](float64[:], float64, float64), nopython=True, parallel=True, fastmath=True, cache=True)
    def _data_change_fs_numba_huge(series, fs, fs_new):
        """针对超大型数据集优化的data_change_fs函数，使用分块并行处理"""
        # 计算原始信号的总时长
        total_time = 1 / fs * len(series)
        
        # 创建新的时间向量
        x_new = np.arange(0, total_time - 5 / fs_new, 1 / fs_new)
        
        # 创建结果数组
        result = np.zeros(len(x_new))
        
        # 分块处理
        chunk_size = 1000000  # 每块大小
        n_chunks = (len(x_new) + chunk_size - 1) // chunk_size  # 向上取整得到块数
        
        # 并行处理每个块
        for chunk in prange(n_chunks):
            start = chunk * chunk_size
            end = min(start + chunk_size, len(x_new))
            
            # 处理当前块
            for i in range(start, end):
                # 找到x_new[i]对应的在原数组中的位置（非整数）
                pos = x_new[i] * fs
                
                # 找到左右两个整数索引
                pos_left = int(pos)
                pos_right = pos_left + 1
                
                # 确保索引在有效范围内
                if pos_right >= len(series):
                    pos_right = len(series) - 1
                
                # 计算插值权重
                weight_right = pos - pos_left
                weight_left = 1.0 - weight_right
                
                # 线性插值
                if pos_left < len(series):
                    result[i] = weight_left * series[pos_left] + weight_right * series[pos_right]
        
        return result


class PyDAS:
    """
    Python Data Analysis System for processing and analyzing time series data.
    
    This class provides comprehensive tools for reading, processing, and analyzing
    experimental data with support for filtering, scaling, and visualization.
    
    Attributes:
    -----------
    data : list of pandas.DataFrame
        List of data segments, each containing channel data
    chInfo : pandas.DataFrame
        Channel information including names, units, and coefficients
    segInfo : pandas.DataFrame
        Segment information including sample counts and timestamps
    __fs__ : float
        Sampling frequency in Hz
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
        self.set_logger(log_level)
        
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
                logger.error(f"File {filename} does not exist. Breaking!")
                sys.exit()
            
            # Read the data file
            self.__read__(sseg)
        else:
            # Initialize empty data structures for manual data setting
            self.__filename__ = None
            self.chInfo = pd.DataFrame(columns=['Name', 'Unit'])
            self.data = {}
            logger.info("Created empty PyDAS object. Use load() method to read data or set data manually.")

    def set_logger(self, level='info'):
        """
        Configure the logger for the PyDAS class.
        
        Parameters:
        -----------
        level : str, optional
            Logging level ('debug', 'info', 'warning', 'error', 'critical'), default is 'info'
        
        Returns:
        --------
        None
        
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

    def __read__(self, sseg):
        """
        Read data from the *.out file.
        
        This method reads the file header, channel information, and data segments.
        It populates the object's properties with the read data.
        
        Parameters:
        -----------
        sseg : int or 'all'
            Selected segment number to load, or 'all' to load all segments
        """
        with open(self.__filename__, 'rb') as fIn:
            # Read file header (256 bytes)
            fmtstr = '=hhlhh2s2s240s'
            buf = fIn.read(256)
            if not buf:
                logger.warning(f"Reading data file {self.__filename__} failed, exiting...")
                return
                
            # Unpack header data
            tmp = struct.unpack(fmtstr, buf)
            index, self.__chN__, self.__fs__, self.__segN__ = tmp[0], tmp[1], tmp[3], tmp[4]
            
            # Extract date information
            datemm, datedd = tmp[5].decode('utf-8'), tmp[6].decode('utf-8')
            self.__date__ = f'{datemm}-{datedd}'
            
            # Extract global description
            self.__desc__ = tmp[7].decode('utf-8').rstrip()

            # Read channel names (16 bytes per channel)
            chName = [namei.decode('utf-8').rstrip() for namei in
                      struct.unpack(self.__chN__ * '16s', fIn.read(self.__chN__ * 16))]
            
            # Read channel units (4 bytes per channel)
            chUnit = [uniti.decode('utf-8').rstrip() for uniti in
                      struct.unpack(self.__chN__ * '4s', fIn.read(self.__chN__ * 4))]
            
            # Read channel coefficients (4 bytes per channel)
            chCoef = struct.unpack('=' + self.__chN__ * 'f',
                                   fIn.read(self.__chN__ * 4))

            # Read channel IDs if available
            if (index < -1):
                chIdx = struct.unpack(
                    '=' + self.__chN__ * 'h', fIn.read(self.__chN__ * 2))
            else:
                chIdx = list(range(1, self.__chN__ + 1))

            # Create channel information dictionary and DataFrame
            chInfoDict = {'Index': chIdx, 'Name': chName, 'Unit': chUnit,
                          'Coef': chCoef}
            column = ['Name', 'Unit', 'Coef']
            self.chInfo = pd.DataFrame(chInfoDict, columns=column)
            self.chInfo.index = range(1, self.__chN__ + 1)

            # Initialize arrays for segment data
            sampNum = [0] * self.__segN__  # Number of samples in each segment
            segInfo = [[] for _ in range(self.__segN__)]  # Segment information
            segStatis = [[] for _ in range(self.__segN__)]  # Statistical values
            dataRaw = [[] for _ in range(self.__segN__)]  # Raw data
            note = [[] for _ in range(self.__segN__)]  # Notes for each segment
            
            # 记录文件位置，用于后续内存映射
            segment_positions = []
            segment_sizes = []

            # 首先读取所有段的信息，但不立即读取数据
            for iseg in range(self.__segN__):
                # Align to 128-byte boundary
                p_cur = fIn.tell()
                aligned_pos = 128 * math.ceil(p_cur / 128)
                fIn.seek(aligned_pos)
                segment_positions.append(aligned_pos)

                # Read segment information (256 bytes)
                fmtstr = '=hhlBBBBBBBB240s'
                buf = fIn.read(256)
                segInfo[iseg] = struct.unpack(fmtstr, buf)

                # Extract segment details
                segChN = segInfo[iseg][1]  # Number of channels in this segment
                sampNum[iseg] = segInfo[iseg][2] - 5  # Number of samples (minus 5)
                note[iseg] = segInfo[iseg][11].decode('utf-8').rstrip()  # Segment note

                # Read statistical values for each channel
                fmtstr = '=' + segChN * 'h' + segChN * 'f' + segChN * 2 * 'h'
                buf = fIn.read(segChN * (2 * 3 + 4))
                segStatis[iseg] = struct.unpack(fmtstr, buf)
                
                # 记录数据段的位置和大小，但不立即读取
                data_pos = fIn.tell()
                data_size = sampNum[iseg] * segChN * 2
                segment_sizes.append(data_size)
                
                # 跳过数据段
                fIn.seek(data_pos + data_size)

        # 使用内存映射读取大数据段
        with open(self.__filename__, 'rb') as fIn:
            for iseg in range(self.__segN__):
                segChN = segInfo[iseg][1]
                
                # 使用内存映射读取数据
                fIn.seek(segment_positions[iseg] + 256 + segChN * (2 * 3 + 4))
                
                # 对于大数据段，使用内存映射
                if sampNum[iseg] * segChN > 1000000:  # 阈值可以根据实际情况调整
                    # 使用numpy的memmap直接从文件读取数据
                    mm = np.memmap(self.__filename__, dtype=np.int16, mode='r',
                                  offset=fIn.tell(),
                                  shape=(sampNum[iseg], segChN))
                    # 复制到内存中以避免后续操作影响原文件
                    dataRaw[iseg] = np.array(mm, dtype=np.int16)
                    # 关闭内存映射
                    del mm
                else:
                    # 对于小数据段，直接读取
                    dataRaw[iseg] = np.frombuffer(
                        fIn.read(sampNum[iseg] * segChN * 2),
                        dtype=np.int16
                    ).reshape((sampNum[iseg], segChN))

        # Process segment information
        segType = []
        startTime = []
        stopTime = []
        index = []
        duration = []
        
        for n in range(self.__segN__):
            # Segment type: 0-sampling, 1-pre-calibration, 2-post-calibration
            segType.append(segInfo[n][0])
            
            # Format start and stop times
            startTime.append('{0:02d}:{1:02d}:{2:02d}.{3:1d}'.format(
                segInfo[n][6], segInfo[n][5], segInfo[n][4], segInfo[n][3]))
            stopTime.append('{0:02d}:{1:02d}:{2:02d}.{3:1d}'.format(
                segInfo[n][10], segInfo[n][9], segInfo[n][8], segInfo[n][7]))
            
            # Segment index and duration
            index.append(f'Seg{n:2d}')
            duration.append(f'{(sampNum[n] - 1) / self.__fs__:8.1f}s')
        
        # Create segment information DataFrame
        segInfoDict = {
            'Type': segType,
            'Start': startTime,
            'Stop': stopTime,
            'Duration': duration,
            'N sample': sampNum,
            'Note': note
        }
        column = ['Type', 'Start', 'Stop', 'Duration', 'N sample', 'Note']
        segInfo = pd.DataFrame(segInfoDict, index=index, columns=column)

        # 使用并行处理转换统计数据和原始数据
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing
        
        # 确定使用的CPU核心数
        num_cores = min(multiprocessing.cpu_count(), self.__segN__)
        
        # 转换统计数据的函数
        def process_statistics(iseg):
            segStatis_temp = np.reshape(
                np.array(segStatis[iseg], dtype='float64'),
                (4, self.__chN__)).transpose()
            
            for m in range(self.__chN__):
                segStatis_temp[m] *= chCoef[m]
            
            # Create DataFrame with statistics
            column = ['Mean', 'STD', 'Max', 'Min']
            stats_df = pd.DataFrame(
                segStatis_temp, index=chName, columns=column)
            stats_df['Unit'] = chUnit
            return stats_df
        
        # 转换原始数据的函数
        def process_raw_data(iseg):
            # 使用numpy的向量化操作加速
            data_temp = dataRaw[iseg].astype('float64')
            
            # 使用广播机制一次性应用所有系数
            coef_array = np.array(chCoef, dtype='float64')
            data_temp = data_temp * coef_array
            
            # 创建DataFrame
            return pd.DataFrame(data_temp, columns=chName, dtype='float64')
        
        # 并行处理统计数据
        self.segStatis = [None] * self.__segN__
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            for iseg, result in enumerate(executor.map(process_statistics, range(self.__segN__))):
                self.segStatis[iseg] = result
        
        # 并行处理原始数据
        self.data = [None] * self.__segN__
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            for iseg, result in enumerate(executor.map(process_raw_data, range(self.__segN__))):
                self.data[iseg] = result

        # Handle segment selection
        if sseg == 'all':
            self.segInfo = segInfo
        else:
            # If a specific segment is selected, keep only that segment
            self.__segN__ = 1
            self.segInfo = segInfo[sseg:sseg + 1]
            self.segInfo = segInfo[sseg:sseg + 1].rename(index={'Seg{0:2d}'.format(sseg): 'Seg 0'})
            self.segStatis = [self.segStatis[sseg]]
            self.data = [self.data[sseg]]

    def write(self, filename, sseg='all', ch='all'):
        """
        Write data to a new *.out file.
        
        Parameters:
        -----------
        filename : str
            Path to the output *.out file
        sseg : int, list, or 'all', optional
            Segment(s) to write to the file, default is 'all'
        ch : list or 'all', optional
            Channels to write to the file, default is 'all'
        
        Notes:
        ------
        This method will automatically append '.out' extension if not provided.
        """
        # Ensure filename has .out extension
        if not filename.endswith('.out'):
            filename += '.out'

        # Determine which segments to write
        if sseg == 'all':
            sseg = list(range(self.__segN__))
        elif isinstance(sseg, int):
            sseg = [sseg]
        else:
            logger.warning("Unsupported segment number, using 'all'.")
            sseg = list(range(self.__segN__))

        logger.info(f'Saving segment(s) No. {sseg} to file {filename}')

        with open(filename, 'wb') as fOut:
            # Write file header (256 bytes)
            datemmdd = self.__date__.split('-')
            
            # Pack header information
            buf = struct.pack('=hhlhh',
                              -2,                # File format version
                              self.__chN__,      # Number of channels
                              0x0d,              # Reserved
                              self.__fs__,       # Sampling frequency
                              len(sseg))         # Number of segments
            
            # Pack date and description
            buf += struct.pack('2s2s240s',
                               datemmdd[0].encode('utf-8'),
                               datemmdd[1].encode('utf-8'),
                               self.__desc__.encode('utf-8')).replace(b'\x00', b' ')
            
            # Write header
            if fOut.write(buf) != 256:
                logger.error("Error when saving out file!")
                raise IOError("Failed to write file header")

            # Write channel names (16 bytes per channel)
            fOut.write(struct.pack(self.__chN__ * '16s',
                                   *[self.chInfo['Name'].iloc[i].encode('utf-8')
                                     for i in range(self.__chN__)]).replace(b'\x00', b' '))

            # Write channel units (4 bytes per channel)
            fOut.write(struct.pack(self.__chN__ * '4s',
                                   *[self.chInfo['Unit'].iloc[i].encode('utf-8')
                                     for i in range(self.__chN__)]).replace(b'\x00', b' '))

            # Calculate new coefficients for optimal data range
            # Find maximum absolute value for each channel across selected segments
            chMagMax = np.amax(np.array(
                [np.amax(abs(self.data[i].values), axis=0) for i in sseg]),
                axis=0)
            
            # Calculate coefficients to scale data to 16-bit range (-32767 to 32767)
            chCoef_ = (chMagMax / 32767).astype(np.float32)
            
            # Write channel coefficients (4 bytes per channel)
            fOut.write(struct.pack('=' + self.__chN__ * 'f', *chCoef_))

            # Write channel indices (2 bytes per channel)
            fOut.write(struct.pack('=' + self.__chN__ * 'h', *self.chInfo.index))

            # Write each segment
            for iseg in sseg:
                # Align to 128-byte boundary
                p_cur = fOut.tell()
                fOut.seek(128 * math.ceil(p_cur / 128))

                # Write segment information (256 bytes)
                # Segment type
                fOut.write(struct.pack('=h', self.segInfo['Type'][iseg]))
                # Number of channels
                fOut.write(struct.pack('=h', self.__chN__))
                # Number of samples (+5 for compatibility)
                fOut.write(struct.pack(
                    '=l', self.segInfo['N sample'][iseg] + 5))
                
                # Write start and stop times (8 bytes)
                # Convert time strings to bytes: HH:MM:SS.s -> [s, SS, MM, HH]
                start_time_parts = re.split(':|\.', self.segInfo.Start[iseg])[::-1]
                stop_time_parts = re.split(':|\.', self.segInfo.Stop[iseg])[::-1]
                time_parts = start_time_parts + stop_time_parts
                time_parts_int = list(map(int, time_parts))
                fOut.write(struct.pack(8 * 'B', *time_parts_int))
                
                # Write segment note (240 bytes)
                fOut.write(struct.pack('240s', self.segInfo.Note[iseg].encode(
                    'utf-8')).replace(b'\x00', b' '))

                # Calculate statistical information for each channel
                # Mean values (as short integers)
                mean_ = np.mean(self.data[iseg].values, axis=0) / chCoef_
                # Standard deviations (as floats)
                std_ = np.std(self.data[iseg].values, axis=0) / chCoef_
                # Maximum and minimum values (as short integers)
                max_ = np.amax(self.data[iseg].values, axis=0) / chCoef_
                min_ = np.amin(self.data[iseg].values, axis=0) / chCoef_
                
                # Write statistical information
                fOut.write(struct.pack('=' + self.__chN__ * 'h',
                                       *np.round(mean_).astype(np.int16)))
                fOut.write(struct.pack('=' + self.__chN__ * 'f', *std_))
                fOut.write(struct.pack('=' + self.__chN__ * 'h',
                                       *np.round(max_).astype(np.int16)))
                fOut.write(struct.pack('=' + self.__chN__ * 'h',
                                       *np.round(min_).astype(np.int16)))

                # Convert data to 16-bit integers and write
                raw_ = np.round(self.data[iseg].values / np.repeat(chCoef_.reshape(
                    1, -1), self.segInfo['N sample'][iseg], axis=0)).astype(np.int16)
                fOut.write(raw_.tobytes())

    def add_channel(self, name, unit, series, fs, coef=1, point_of_move=0, sseg=0):
        """
        Add a new channel to the data.
        
        Parameters:
        -----------
        name : str
            Name of the new channel
        unit : str
            Unit of measurement
        series : numpy.ndarray
            Channel data
        fs : float
            Sampling frequency in Hz
        coef : float, optional
            Coefficient for data scaling, default is 1
        point_of_move : int, optional
            Number of points to shift the data, default is 0
        sseg : int, optional
            Segment index to add channel to, default is 0
            
        Notes:
        ------
        - Validates input data and parameters
        - Handles data alignment and scaling
        - Updates channel information and statistics
        """
        if name not in self.chInfo['Name'].values:
            # Resample if necessary
            if fs != self.__fs__:
                series = data_change_fs(series, fs, self.__fs__)
                
            # Adjust data length if necessary
            n_sample = self.segInfo.iloc[sseg]['N sample']
            if len(series) > n_sample:
                series = series[:n_sample]
            elif len(series) < n_sample:
                series = np.pad(series, (0, n_sample - len(series)), 'constant', constant_values=0)
            
            # Add to data
            self.data[sseg][name] = series
            
            # Update channel info
            new_idx = self.__chN__ + 1
            self.chInfo.loc[new_idx] = [name, unit, coef]
            
            # Update statistics
            self.segStatis[sseg].loc[name] = [
                np.mean(series), np.std(series), np.amax(series), np.amin(series), unit]
                
            # Update channel count
            self.__chN__ += 1
            
            # Move data if requested
            if point_of_move != 0:
                self.move_data(name, point_of_move, sseg=sseg)
                
            logger.info(f"Channel '{name}' has been added")
        else:
            logger.warning(f"Channel '{name}' already exists.")

    def delete_channel(self, name):
        """
        Delete a specified channel from the data.
        
        Parameters:
        -----------
        name : str
            Name of the channel to delete
            
        Notes:
        ------
        - Removes channel from all data segments
        - Updates channel information
        - Recalculates statistics
        """
        if name in self.chInfo['Name'].values:
            # Find the index of the channel
            idx = self.chInfo.index[self.chInfo['Name'] == name].tolist()[0]
            
            # Remove from channel info
            self.chInfo = self.chInfo.drop(idx)
            
            # Remove from data in all segments
            for sseg in range(self.__segN__):
                self.data[sseg] = self.data[sseg].drop(name, axis=1)
                self.segStatis[sseg] = self.segStatis[sseg].drop(name)
            
            # Update channel count
            self.__chN__ -= 1
            
            # Reset index
            self.chInfo.index = range(1, self.__chN__ + 1)
            
            logger.info(f"Channel '{name}' has been removed")
        else:
            logger.warning(f"Channel '{name}' does not exist.")

    def select_channels(self, chnames):
        """
        Select and keep only specified channels, removing others.
        
        Parameters:
        -----------
        chnames : str or list of str
            Channel name(s) to keep. Can be a single string for one channel
            or a list of strings for multiple channels.
            
        Returns:
        --------
        bool
            True if successful, False otherwise
            
        Notes:
        ------
        - Validates channel names
        - Removes all channels not in the specified list
        - Updates channel information and statistics
        """
        # Check if input is a string (single channel) and convert to list
        if isinstance(chnames, str):
            chnames = [chnames]
            
        # Validate channel names
        valid_chnames = []
        for name in chnames:
            if name in self.chInfo['Name'].values:
                valid_chnames.append(name)
            else:
                logger.warning(f"Channel '{name}' does not exist and will be ignored.")
        
        if not valid_chnames:
            logger.warning("No valid channels specified.")
            return False
            
        # Get indices of channels to keep
        keep_indices = []
        for name in valid_chnames:
            idx = self.chInfo.index[self.chInfo['Name'] == name].tolist()[0]
            keep_indices.append(idx)
            
        # Keep only selected channels in channel info
        self.chInfo = self.chInfo.loc[keep_indices]
        
        # Keep only selected channels in data and statistics
        for sseg in range(self.__segN__):
            self.data[sseg] = self.data[sseg][valid_chnames]
            self.segStatis[sseg] = self.segStatis[sseg].loc[valid_chnames]
            
        # Update channel count and reset indices
        self.__chN__ = len(valid_chnames)
        self.chInfo.index = range(1, self.__chN__ + 1)
        
        logger.info(f"Selected {len(valid_chnames)} channels: {', '.join(valid_chnames)}")
        return True

    def print_info(self, printTxt=False, printExcel=False):
        """
        Print and optionally export general information about the data.
        
        Parameters:
        -----------
        printTxt : bool, optional
            If True, export information to a text file, default is False
        printExcel : bool, optional
            If True, export information to an Excel file, default is False
            
        Returns:
        --------
        DataFrame
            DataFrame containing general information about the data
        """
        # Create information DataFrame
        info = pd.DataFrame(columns=['Value'])
        info.loc['Filename'] = self.__filename__
        info.loc['Date'] = self.__date__
        info.loc['Scale'] = self.__scale__
        info.loc['Lambda'] = self.__lam__
        info.loc['Sampling frequency'] = '{0:5.2f} Hz'.format(self.__fs__)
        info.loc['Number of channels'] = self.__chN__
        info.loc['Number of segments'] = self.__segN__
        
        # Print to console
        logger.info('\nGeneral Information:')
        logger.info(info.to_string())
        
        # Export to text file if requested
        if printTxt:
            txt_filename = os.path.splitext(self.__filename__)[0] + '_info.txt'
            with open(txt_filename, 'w') as f:
                f.write('General Information:\n')
                f.write(info.to_string())
                f.write('\n\nSegment Information:\n')
                f.write(self.segInfo.to_string())
                f.write('\n\nChannel Information:\n')
                f.write(self.chInfo.to_string())
            logger.info(f"Information exported to: {txt_filename}")
            
        # Export to Excel file if requested
        if printExcel:
            excel_filename = os.path.splitext(self.__filename__)[0] + '_info.xlsx'
            with pd.ExcelWriter(excel_filename) as writer:
                info.to_excel(writer, sheet_name='General Info')
                self.segInfo.to_excel(writer, sheet_name='Segment Info')
                self.chInfo.to_excel(writer, sheet_name='Channel Info')
            logger.info(f"Information exported to: {excel_filename}")
            
        return info

    def print_channel_info(self, printTxt=False, printExcel=False):
        """
        Print and optionally export channel information.
        
        Parameters:
        -----------
        printTxt : bool, optional
            If True, export information to a text file, default is False
        printExcel : bool, optional
            If True, export information to an Excel file, default is False
            
        Returns:
        --------
        DataFrame
            DataFrame containing channel information
        """
        # Print to console
        logger.info('Channel Information:')
        logger.info('\n' + self.chInfo.to_string())
        
        # Export to text file if requested
        if printTxt:
            txt_filename = os.path.splitext(self.__filename__)[0] + '_channel_info.txt'
            with open(txt_filename, 'w') as f:
                f.write('Channel Information:\n')
                f.write(self.chInfo.to_string())
            logger.info(f"Channel information exported to: {txt_filename}")
            
        # Export to Excel file if requested
        if printExcel:
            excel_filename = os.path.splitext(self.__filename__)[0] + '_channel_info.xlsx'
            self.chInfo.to_excel(excel_filename)
            logger.info(f"Channel information exported to: {excel_filename}")
            
        return self.chInfo

    def to_dat(self, Time=True, sseg='all'):
        """
        Export data to DAT file format.
        
        Parameters:
        -----------
        Time : bool, optional
            If True, include time column in the output, default is True
        sseg : int or 'all', optional
            Segment(s) to export, default is 'all'
            
        Notes:
        ------
        The output file will be named based on the original filename with
        segment number and scale (model or full) appended.
        """
        def writefile(self, idx):
            """
            Helper function to write a single segment to a DAT file.
            
            Parameters:
            -----------
            idx : int
                Index of the segment to write
            """
            # Prepare file path and name
            path = os.getcwd()
            
            # Create filename based on scale (model or prototype)
            if self.__scale__ == 'model':
                filename = f"{path}/{os.path.splitext(self.__filename__)[0]}_seg{idx:02d}-model.dat"
            else:
                filename = f"{path}/{os.path.splitext(self.__filename__)[0]}_seg{idx:02d}-full.dat"
            
            # Prepare header information
            header = [
                f"OUTFILE NAME: {self.__filename__}",
                f"CHANNEL NO.: {self.__chN__}",
                f"SAMPLING FREQUENCY: {self.__fs__:.1f}"
            ]
            
            # Add channel names and units to header
            if Time:
                header.append("Time " + " ".join(self.chInfo['Name']))
                header.append("S " + " ".join(self.chInfo['Unit']))
            else:
                header.append(" ".join(self.chInfo['Name']))
                header.append(" ".join(self.chInfo['Unit']))
            
            # Combine header lines
            header_str = "\n".join(header)
            
            # Get number of samples for this segment
            n_sample = self.segInfo.iloc[idx]['N sample']
            
            # Write data with or without time column
            if Time:
                # Create time vector
                time_vector = np.arange(0, n_sample / self.__fs__, 1 / self.__fs__)
                
                # Create output array with time as first column
                datawrite = np.zeros((n_sample, self.__chN__ + 1))
                datawrite[:, 0] = time_vector
                datawrite[:, 1:] = self.data[idx].values
                
                # Save to file
                np.savetxt(
                    filename,
                    datawrite,
                    fmt='% .5E',
                    delimiter=' ',
                    header=header_str
                )
            else:
                # Convert DataFrame to string with proper formatting
                with open(filename, 'w') as f:
                    f.write(header_str + "\n")
                    f.write(
                        self.data[idx].to_string(
                            header=False,
                            index=False,
                            justify='left',
                            float_format='% .5E'
                        )
                    )
            
            logger.info(f"Data exported to: {filename}")

        # Determine which segments to export
        if sseg == 'all':
            # Export all segments
            for idx in range(self.__segN__):
                writefile(self, idx)
        elif isinstance(sseg, int):
            # Export single segment if valid
            if sseg < self.__segN__:
                writefile(self, sseg)
            else:
                logger.warning(f"Segment {sseg} exceeds the maximum segment number ({self.__segN__ - 1}).")
        else:
            logger.warning("Invalid segment selection. Use an integer or 'all'.")

    def print_statistics(self, printTxt=False, printExcel=False):
        """
        Print and optionally export statistical information for all channels.
        
        Parameters:
        -----------
        printTxt : bool, optional
            If True, export statistics to a text file, default is False
        printExcel : bool, optional
            If True, export statistics to an Excel file, default is False
            
        Returns:
        --------
        None
            Statistics are printed to the console and optionally exported to files
        """
        # Update statistics for all segments
        self.updateST(sseg=0)
        
        # Print separator line and segment count
        logger.info(f'Segment total: {self.__segN__:02d}')
        
        # Print statistics for each segment
        for idx, segment_stats in enumerate(self.segStatis):
            logger.info(f'Seg{idx:02d}')
            logger.info('\n' + segment_stats.to_string(float_format='% .3E', justify='center'))
        
        
        # Export to files if requested
        if printTxt or printExcel:
            # Prepare file path
            path = os.getcwd()
            base_filename = os.path.splitext(self.__filename__)[0]
            
            # Export to text file
            if printTxt:
                txt_filename = f"{path}/{base_filename}_statistic.txt"
                
                # Write to file
                with open(txt_filename, 'w') as infoFile:
                    infoFile.write(f'Segment total: {self.__segN__:02d}\n')
                    
                    # Write statistics for each segment
                    for idx, segment_stats in enumerate(self.segStatis):
                        infoFile.write('\n')
                        infoFile.write(f'Seg{idx:02d}\n')
                        infoFile.write(segment_stats.to_string(
                            float_format='% .3E', justify='center'))
                
                logger.info(f"Statistics exported to: {txt_filename}")
            
            # Export to Excel file
            if printExcel:
                excel_filename = f"{path}/{base_filename}_statistic.xlsx"
                
                # Write each segment to a separate sheet
                with pd.ExcelWriter(excel_filename) as writer:
                    for idx, segment_stats in enumerate(self.segStatis):
                        segment_stats.to_excel(writer, sheet_name=f'SEG{idx:02d}')
                
                logger.info(f"Statistics exported to: {excel_filename}")

    def to_mat(self, sseg=0):
        """
        Export data to MATLAB MAT file format.
        
        Parameters:
        -----------
        sseg : int, optional
            Segment index to export, default is 0
            
        Returns:
        --------
        bool
            True if export was successful, False otherwise
            
        Notes:
        ------
        The output file will be named based on the original filename.
        """
        # Validate segment index
        if not isinstance(sseg, int):
            logger.warning("Selected segment id must be an integer.")
            return False
            
        if sseg >= self.__segN__:
            logger.warning(f"Segment {sseg} exceeds the maximum segment number ({self.__segN__ - 1}).")
            return False
            
        # Create dictionary with data to export
        data_dic = {
            'Data': self.data[sseg].values,
            'chName': self.chInfo['Name'].values,
            'chUnit': self.chInfo['Unit'].values,
            'Date': self.__date__,
            'fs': self.__fs__,
            'chN': self.__chN__,
            'Readme': 'Generated by PyDAS from python, SKLOE/SJTU'
        }
        
        # Prepare file path and name
        path = os.getcwd()
        mat_filename = f"{path}/{os.path.splitext(self.__filename__)[0]}.mat"
        
        # Save to MAT file
        try:
            sio.savemat(mat_filename, data_dic)
            logger.info(f"Data exported to: {mat_filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to MAT file: {str(e)}")
            return False

    def fix_unit(self, chName, newunit, pInfo=False):
        """
        Fix channel unit.
        
        Parameters:
        -----------
        chName : str
            Channel name
        newunit : str
            New unit to set
        pInfo : bool, optional
            Whether to print information, default is False
            
        Notes:
        ------
        - Updates channel unit in information
        - Validates unit conversion
        - Maintains data integrity
        """
        # 检查通道名是否存在
        if chName not in self.chInfo['Name'].values:
            logger.warning(f"Channel '{chName}' does not exist.")
            return False
            
        # 找到对应的索引
        idx = self.chInfo.index[self.chInfo['Name'] == chName].tolist()[0]
        
        # 更新单位
        self.chInfo.loc[idx, 'Unit'] = newunit
        logger.info(f"Channel '{chName}' unit updated to: {newunit}")
        
        if pInfo:
            logger.info('-' * 50)
            logger.info('\n' + self.chInfo.to_string(justify='center'))
            logger.info('-' * 50)


    def to_fullscale(self, rho=1.025, g=9.807, pInfo=False):
        """
        Convert model scale data to prototype scale.
        
        Parameters:
        -----------
        lam : float
            Scale factor
        rho : float, optional
            Water density in kg/m³, default is 1.025
        g : float, optional
            Gravitational acceleration in m/s², default is 9.807
        pInfo : bool, optional
            Whether to print information, default is False
            
        Notes:
        ------
        - Applies Froude scaling laws
        - Handles unit conversions
        - Updates channel information
        - Maintains data consistency
        """
        if self.__scale__ == 'prototype':
            logger.warning('The data is already upscaled.')
            return
        else:
            logger.info('Please make sure the channel units are all checked!')
            if pInfo:
                logger.info(self.chInfo.to_string(
                    justify='center', columns=['Name', 'Unit']))
            self.rho = rho
            self.__scale__ = 'prototype'
            
            # Predefined unit conversion dictionary
            transDict = {
                'kg': ['kN', np.array([g * 0.001, 1.0, 3.0])],
                'cm': ['m', np.array([0.01, 0.0, 1.0])],
                'mm': ['m', np.array([0.001, 0.0, 1.0])],
                'm': ['m', np.array([1, 0.0, 1.0])],
                's': ['s', np.array([1, 0.0, 0.5])],
                'deg': ['deg', np.array([1, 0.0, 0.0])],
                'rad': ['rad', np.array([1, 0.0, 0.0])],
                'n': ['kn', np.array([0.001, 1.0, 3.0])],  # Added lowercase unit to avoid conversion issues
                'kn': ['kn', np.array([1, 0.0, 0.0])],
                '%': ['%', np.array([1, 0.0, 0.0])],
                '-': ['-', np.array([1, 0.0, 0.0])]
            }
            
            # Create cache dictionary to avoid recalculating the same unit conversions
            trans_cache = {}
            
            def findtrans(transDict, unit):
                # Check if result is already in cache
                unit = unit.lower().strip()
                if unit in trans_cache:
                    return trans_cache[unit]
                
                if unit in transDict:
                    trans = transDict[unit]
                    trans_cache[unit] = trans
                    return trans
                elif '/' in unit:
                    unitUpper, unitLower = unit.split('/')
                    transUpper = findtrans(transDict, unitUpper)
                    transLower = findtrans(transDict, unitLower)
                    trans = [transUpper[0] + '/' +
                             transLower[0], np.array([0.0, 0.0, 0.0])]
                    trans[1][0] = transUpper[1][0] / transLower[1][0]
                    trans[1][1] = transUpper[1][1] - transLower[1][1]
                    trans[1][2] = transUpper[1][2] - transLower[1][2]
                    trans_cache[unit] = trans
                    return trans
                elif '.' in unit:
                    unitWithDot = unit.split('.')
                    transU = []
                    transN1 = np.array([])
                    transN2 = np.array([])
                    transN3 = np.array([])
                    for uWithDot in unitWithDot:
                        transWithDot = findtrans(transDict, uWithDot)
                        transU.append(transWithDot[0])
                        transN1 = np.append(transN1, transWithDot[1][0])
                        transN2 = np.append(transN2, transWithDot[1][1])
                        transN3 = np.append(transN3, transWithDot[1][2])
                    trans = ['.'.join(transU), np.array([1.0, 0.0, 0.0])]
                    for x in np.nditer(transN1):
                        trans[1][0] *= x
                    trans[1][1] = transN2.sum()
                    trans[1][2] = transN3.sum()
                    trans_cache[unit] = trans
                    return trans
                elif unit[-1].isdigit():
                    n = int(unit[-1])
                    unit_base = unit[0:-1]
                    if unit_base in transDict:
                        trans_temp = transDict[unit_base]
                        trans = [trans_temp[0] +
                                 str(n), np.array([1.0, 0.0, 0.0])]
                        trans[1][0] = trans_temp[1][0]**n
                        trans[1][1] = trans_temp[1][1] * n
                        trans[1][2] = trans_temp[1][2] * n
                        trans_cache[unit] = trans
                        return trans
                    else:
                        logger.warning(
                            f"Input unit '{unit}' cannot be identified, using default values.")
                        return [unit, np.array([1.0, 0.0, 0.0])]
                else:
                    logger.warning(
                        f"Input unit '{unit}' cannot be identified, using default values.")
                    return [unit, np.array([1.0, 0.0, 0.0])]
            
            # Preprocess: batch get all unit conversions
            unique_units = self.chInfo['Unit'].unique()
            unit_to_trans = {}
            
            # Parallel precompute all unique unit conversions
            for unit in unique_units:
                if not pd.isna(unit):  # Handle potential NaN values
                    unit_to_trans[unit] = findtrans(transDict, unit)
            
            # Prepare batch update data
            transUnit = []
            transCoeffUnit = np.zeros(self.__chN__)
            transCoeffRho = np.zeros(self.__chN__)
            transCoeffLam = np.zeros(self.__chN__)
            
            # Use vectorized operations to update channel information
            for idx, unit in enumerate(self.chInfo['Unit']):
                if pd.isna(unit):  # Handle potential NaN values
                    transUnit.append('')
                    transCoeffUnit[idx] = 1.0
                    transCoeffRho[idx] = 0.0
                    transCoeffLam[idx] = 0.0
                    continue
                    
                trans_temp = unit_to_trans[unit]
                transUnit.append(trans_temp[0])
                transCoeffUnit[idx] = trans_temp[1][0]
                transCoeffRho[idx] = trans_temp[1][1]
                transCoeffLam[idx] = trans_temp[1][2]
            
            # Update channel information
            self.chInfo['Unit'] = transUnit
            self.chInfo['CoeffUnit'] = transCoeffUnit
            self.chInfo['CoeffRho'] = transCoeffRho
            self.chInfo['CoeffLam'] = transCoeffLam
            
            # Update sampling rate
            self.__fs__ = self.__fs__ / np.sqrt(self.__lam__)
            logger.info(f'lambda = {self.__lam__:2d}')
            
            if pInfo:
                logger.info(self.chInfo.to_string(justify='center'))

            # Use vectorized operations to update data (process all channels at once)
            for idx1 in range(self.__segN__):
                # Create conversion coefficient array
                coeffs = np.ones(self.__chN__)
                
                # Calculate conversion coefficient for each channel
                for idx2 in range(self.__chN__):
                    C1 = self.chInfo['CoeffUnit'].iloc[idx2]
                    C2 = rho ** self.chInfo['CoeffRho'].iloc[idx2]
                    C3 = self.__lam__ ** self.chInfo['CoeffLam'].iloc[idx2]
                    coeffs[idx2] = C1 * C2 * C3
                
                # Apply conversion coefficients (vectorized operation)
                for idx2, name in enumerate(self.chInfo['Name']):
                    self.data[idx1][name] *= coeffs[idx2]
            
            # Update statistical information
            self.updateST(sseg=0)

    def read_waveCal(self, wavefname, sseg=0, YBname='YBS', YBcalname='YBS', alignFlag=True):
        """
        Read wave calibration data.
        
        Parameters:
        -----------
        wavefname : str
            Path to wave calibration file
        sseg : int, optional
            Segment index to process, default is 0
        YBname : str, optional
            Name of wave gauge channel, default is 'YBS'
        YBcalname : str, optional
            Name of calibration wave gauge channel, default is 'YBS'
        alignFlag : bool, optional
            Whether to align data, default is True
            
        Notes:
        ------
        - Reads wave calibration data from file
        - Supports data alignment
        - Handles multiple wave gauges
        - Updates channel information
        """
        wavecase_cal = PyDAS(wavefname, lam = self.__lam__)
        fs_cal = wavecase_cal.__fs__
        nch = wavecase_cal.data[0].shape[1]
        for i in range(nch):
            iname = wavecase_cal.chInfo['Name'].loc[i+1]
            iunit = wavecase_cal.chInfo['Unit'].loc[i+1]
            icoef = wavecase_cal.chInfo['Coef'].loc[i+1]
            iseries = wavecase_cal.data[0][iname].values
            self.add_channel('Cal.'+iname, iunit, iseries, fs_cal, coef=icoef, point_of_move=0, sseg=sseg)
        if alignFlag:
            for ich in wavecase_cal.chInfo['Name'].values:
                self.move_ccor('Cal.'+ich, 'Cal.'+YBcalname, YBname, sseg=sseg)

    def move_ccor(self,
                  to_move_chName,
                  base_chName,
                  reference_ch,
                  sseg=0):
        """
        Move channel data using cross-correlation.
        
        Parameters:
        -----------
        to_move_chName : str
            Name of channel to move
        base_chName : str
            Name of base channel
        reference_ch : str
            Name of reference channel
        sseg : int, optional
            Segment index to process, default is 0
            
        Notes:
        ------
        - Uses cross-correlation for alignment
        - Handles data shifting
        - Maintains data quality
        - Updates channel information
        """
        try:
            # Check if channels exist
            if to_move_chName not in self.data[sseg].columns:
                logger.error(f"Channel to move '{to_move_chName}' not found in segment {sseg}")
                raise KeyError(f"Channel to move '{to_move_chName}' not found")
                
            if reference_ch not in self.data[sseg].columns:
                logger.error(f"Reference channel '{reference_ch}' not found in segment {sseg}")
                raise KeyError(f"Reference channel '{reference_ch}' not found")
                
            if base_chName not in self.data[sseg].columns:
                logger.error(f"Base channel '{base_chName}' not found in segment {sseg}")
                raise KeyError(f"Base channel '{base_chName}' not found")
            
            # Get channel data
            reference = self.data[sseg][reference_ch]
            base = self.data[sseg][base_chName]
            
            # Remove mean from signals for better correlation
            try:
                base_rmmean = base.values - np.mean(base.values)
                reference_remean = reference.values - np.mean(reference.values)
            except Exception as e:
                logger.error(f"Error removing mean from signals: {str(e)}")
                raise ValueError(f"Failed to prepare signals for correlation: {str(e)}")
            
            # Get number of samples
            try:
                n_sample = self.segInfo['N sample'].iloc[sseg]
            except Exception as e:
                logger.error(f"Error getting sample count: {str(e)}")
                raise ValueError(f"Failed to get sample count: {str(e)}")
            
            # Calculate cross-correlation and find lag
            try:
                correlation = correlate(base_rmmean, reference_remean, method='fft')
                max_corr_idx = np.argmax(correlation)
                lag = max_corr_idx - n_sample + 1
                
                # Log correlation strength
                max_corr = correlation[max_corr_idx]
                norm_factor = np.sqrt(np.sum(base_rmmean**2) * np.sum(reference_remean**2))
                if norm_factor > 0:
                    normalized_corr = max_corr / norm_factor
                    logger.debug(f"Maximum correlation between '{base_chName}' and '{reference_ch}': {normalized_corr:.4f} at lag {lag}")
                else:
                    logger.warning("Could not normalize correlation (division by zero)")
            except Exception as e:
                logger.error(f"Error calculating correlation: {str(e)}")
                raise ValueError(f"Correlation calculation failed: {str(e)}")
            
            # Move the target channel by the calculated lag
            try:
                self.move_data(to_move_chName, -lag, sseg=sseg)
                logger.info(f"Moved channel '{to_move_chName}' by {-lag} points based on correlation")
            except Exception as e:
                logger.error(f"Error moving channel '{to_move_chName}': {str(e)}")
                raise ValueError(f"Failed to move channel: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in move_ccor: {str(e)}")
            raise

    def find_move_ccor(self,
                  base_chName,
                  reference_ch,
                  sseg=0):
        """
        Find the number of points to move between channels using cross-correlation.
        
        Parameters:
        -----------
        base_chName : str
            Name of base channel
        reference_ch : str
            Name of reference channel
        sseg : int, optional
            Segment index to process, default is 0
            
        Returns:
        --------
        int
            Number of points to move
            
        Notes:
        ------
        - Uses cross-correlation for alignment
        - Handles large datasets efficiently
        - Maintains numerical accuracy
        """
        try:
            # Check if channels exist
            if reference_ch not in self.data[sseg].columns:
                logger.error(f"Reference channel '{reference_ch}' not found in segment {sseg}")
                raise KeyError(f"Reference channel '{reference_ch}' not found")
                
            if base_chName not in self.data[sseg].columns:
                logger.error(f"Base channel '{base_chName}' not found in segment {sseg}")
                raise KeyError(f"Base channel '{base_chName}' not found")
            
            # Get channel data
            reference = self.data[sseg][reference_ch]
            base = self.data[sseg][base_chName]
            
            # Remove mean from signals for better correlation
            try:
                base_rmmean = base.values - np.mean(base.values)
                reference_remean = reference.values - np.mean(reference.values)
            except Exception as e:
                logger.error(f"Error removing mean from signals: {str(e)}")
                raise ValueError(f"Failed to prepare signals for correlation: {str(e)}")
            
            # Get number of samples
            try:
                n_sample = self.segInfo['N sample'].iloc[sseg]
            except Exception as e:
                logger.error(f"Error getting sample count: {str(e)}")
                raise ValueError(f"Failed to get sample count: {str(e)}")
            
            # Calculate cross-correlation and find lag
            try:
                correlation = correlate(base_rmmean, reference_remean, method='fft')
                max_corr_idx = np.argmax(correlation)
                lag = max_corr_idx - n_sample + 1
                lag = - lag
                
                # Log correlation strength
                max_corr = correlation[max_corr_idx]
                norm_factor = np.sqrt(np.sum(base_rmmean**2) * np.sum(reference_remean**2))
                if norm_factor > 0:
                    normalized_corr = max_corr / norm_factor
                    logger.info(f"Maximum correlation between '{base_chName}' and '{reference_ch}': {normalized_corr:.4f} at lag {lag}")
                else:
                    logger.warning("Could not normalize correlation (division by zero)")
                    
                return lag
            except Exception as e:
                logger.error(f"Error calculating correlation: {str(e)}")
                raise ValueError(f"Correlation calculation failed: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in find_move_ccor: {str(e)}")
            raise

    def change_channel_order(self,
                     newOrder,
                     sseg=0):
        """
        Change channel order in the data.
        
        Parameters:
        -----------
        newOrder : list of str
            New order of channel names
        sseg : int, optional
            Segment index to process, default is 0
            
        Notes:
        ------
        - Reorders channels in data structure
        - Updates channel information
        - Maintains data integrity
        - Validates channel names
        """
        if len(newOrder) == self.__chN__:
            indexNew = []
            for inewOrder in newOrder:
                indexNew.append(
                    list(
                        self.data[sseg].columns).index(inewOrder) +
                    1)
            self.chInfo = self.chInfo.reindex(indexNew)
            self.segStatis[sseg] = self.segStatis[sseg].reindex(newOrder)
            self.chInfo.index = np.arange(1, len(self.chInfo) + 1)
            self.data[sseg] = self.data[sseg][newOrder]
            self.updateChN()
            logger.info('Changed the Channel order.')
        else:
            raise ValueError("Number of channels does not match!")
        #self.chInfo.index = range(1,self.__chN__+1)

    def apply_lowpass_filter(self,
                      chName,
                      cutoffull=2,
                      replace=True,
                      returnValue=False,
                      sseg=0,
                      order=6,
                      plot=False):
        """
        Apply a lowpass filter to a channel.
        
        Parameters:
        -----------
        chName : str or list
            Name of the channel to filter, or list of channel names
        cutoffull : float, optional
            Cutoff frequency in Hz, default is 2
        replace : bool, optional
            Whether to replace original data, default is True
        returnValue : bool, optional
            Whether to return filtered data, default is False
        sseg : int, optional
            Segment index, default is 0
        order : int, optional
            Filter order, default is 6
        plot : bool, optional
            Whether to plot before/after comparison, default is False
            
        Returns:
        --------
        numpy.ndarray, optional
            Filtered data if returnValue is True
            
        Notes:
        ------
        - Uses Butterworth filter design
        - Maintains phase response
        - Handles edge effects
        """
        from scipy.signal import butter, filtfilt
        import copy
        import pandas as pd
        
        def _butter_lowpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            if normal_cutoff >= 1.0:
                logger.warning(f"Cutoff frequency ({cutoff} Hz) is too high for sampling frequency ({fs} Hz). "
                              f"Setting cutoff to 0.99*nyquist.")
                normal_cutoff = 0.99
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        def _butter_lowpass_filter(data, cutoff, fs, order=5):
            try:
                # 计算滤波器所需的最小数据长度（一般为2*order + 1）
                min_data_length = 2 * order + 1
                
                # 检查数据长度是否足够
                if len(data) < min_data_length:
                    logger.warning(f"Data length ({len(data)}) is less than minimum required ({min_data_length}). Filter may not be effective.")
                
                # 计算Nyquist频率
                nyq = 0.5 * fs
                
                # 归一化截止频率
                normal_cutoff = cutoff / nyq
                
                # 设计滤波器
                b, a = _butter_lowpass(cutoff, fs, order)
                
                # 应用滤波器
                y = filtfilt(b, a, data)
                return y
            except Exception as e:
                logger.error(f"Filter error: {str(e)}. Returning original data.")
                return data

        # 检查通道是否存在
        if isinstance(chName, str) and chName not in self.chInfo['Name'].values:
            logger.error(f"Channel '{chName}' not found.")
            return None
        
        # 模型尺度下调整截止频率
        if self.__scale__ == 'model':
            cutoff = cutoffull / 2 / np.pi * np.sqrt(self.__lam__)
        else:
            cutoff = cutoffull / 2 / np.pi

        # 处理通道列表
        if isinstance(chName, list):
            results = []
            for ch in chName:
                if ch in self.chInfo['Name'].values:
                    result = self.apply_lowpass_filter(ch, cutoffull, replace, returnValue, sseg, order, plot)
                    if returnValue:
                        results.append(result)
                else:
                    logger.warning(f"Channel '{ch}' not found, skipping.")
            if returnValue:
                return results
            return None
                    
        # 获取数据
        try:
            data = self.data[sseg][chName].values
        except Exception as e:
            logger.error(f"Error accessing data for channel {chName}: {str(e)}")
            return None
            
        # 检查数据是否存在且长度足够
        if data is None or len(data) <= 0:
            logger.error(f"No data found for channel {chName} in segment {sseg}")
            return None
        
        # 保存原始数据，用于后续对比或绘图
        original_data = copy.deepcopy(data)
        
        # 应用滤波器
        try:
            filtered_data = _butter_lowpass_filter(data, cutoff, self.__fs__, order)
            
            # 如果需要绘图对比
            if plot:
                try:
                    # 为了绘图对比，我们需要创建一个临时通道
                    temp_channel_name = f"{chName}_filtered"
                    
                    # 创建一个临时PyDAS对象的副本，用于比较
                    temp_pydas = copy.deepcopy(self)
                    
                    # 添加滤波后的临时通道
                    unit = temp_pydas.chInfo.loc[temp_pydas.chInfo['Name'] == chName, 'Unit'].values[0]
                    temp_pydas.add_channel(
                        name=temp_channel_name,
                        unit=unit,
                        series=filtered_data,
                        fs=self.__fs__,
                        sseg=sseg
                    )
                    
                    # Use Plotly for interactive comparison
                    from pydas_plot import plot_channel
                    # Plot original and filtered data together
                    channels = [chName, temp_channel_name]
                    logger.info(f"Displaying interactive comparison plot for {chName} before/after filtering (cutoff={cutoffull} Hz, order={order})")
                    plot_channel(
                        pydas_obj=temp_pydas,
                        ch_name=channels,
                        sseg=sseg,
                        title=f"Lowpass Filter Comparison - {chName} (cutoff={cutoffull} Hz, order={order})",
                        alpha=[0.5, 0.8],  # 原始数据透明度0.5，滤波后数据保持默认0.8
                    )
                except Exception as e:
                    logger.error(f"Error creating comparison plot: {str(e)}")
            
            # 如果需要替换数据
            if replace:
                self.data[sseg][chName] = filtered_data
                logger.info(f'Lowpass for {chName} filter = {cutoffull:3.2f} Hz, Lambda = {self.__lam__:02d}')
                self.updateST(chName=chName)
        except Exception as e:
            logger.error(f"Failed to apply filter to {chName}: {str(e)}")
            return None
                
        # 返回结果（如果需要）
        if returnValue:
            return filtered_data
        
        return None

    def apply_highpass_filter(self,
                       chName,
                       cutoffull=2,
                       replace=True,
                       returnValue=False,
                       sseg=0,
                       order=6,
                       plot=False):
        """
        Apply a highpass filter to a channel.
        
        Parameters:
        -----------
        chName : str or list
            Name of the channel to filter, or list of channel names
        cutoffull : float, optional
            Cutoff frequency in Hz, default is 2
        replace : bool, optional
            Whether to replace original data, default is True
        returnValue : bool, optional
            Whether to return filtered data, default is False
        sseg : int, optional
            Segment index, default is 0
        order : int, optional
            Filter order, default is 6
        plot : bool, optional
            Whether to plot before/after comparison, default is False
            
        Returns:
        --------
        numpy.ndarray, optional
            Filtered data if returnValue is True
            
        Notes:
        ------
        - Uses Butterworth filter design
        - Maintains phase response
        - Handles edge effects
        """
        from scipy.signal import butter, filtfilt
        import copy
        
        def _butter_highpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            if normal_cutoff >= 1.0:
                logger.warning(f"Cutoff frequency ({cutoff} Hz) is too high for sampling frequency ({fs} Hz). "
                              f"Setting cutoff to 0.99*nyquist.")
                normal_cutoff = 0.99
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return b, a

        def _butter_highpass_filter(data, cutoff, fs, order=5):
            try:
                # 计算滤波器所需的最小数据长度（一般为2*order + 1）
                min_data_length = 2 * order + 1
                
                # 检查数据长度是否足够
                if len(data) < min_data_length:
                    logger.warning(f"Data length ({len(data)}) is less than minimum required ({min_data_length}). Filter may not be effective.")
                
                # 计算Nyquist频率
                nyq = 0.5 * fs
                
                # 归一化截止频率
                normal_cutoff = cutoff / nyq
                
                # 设计滤波器
                b, a = _butter_highpass(cutoff, fs, order)
                
                # 应用滤波器
                y = filtfilt(b, a, data)
                return y
            except Exception as e:
                logger.error(f"Filter error: {str(e)}. Returning original data.")
                return data

        # 检查通道是否存在
        if isinstance(chName, str) and chName not in self.chInfo['Name'].values:
            logger.error(f"Channel '{chName}' not found.")
            return None
        
        # 模型尺度下调整截止频率
        if self.__scale__ == 'model':
            cutoff = cutoffull / 2 / np.pi * np.sqrt(self.__lam__)
        else:
            cutoff = cutoffull / 2 / np.pi
            
        # 处理通道列表
        if isinstance(chName, list):
            results = []
            for ch in chName:
                if ch in self.chInfo['Name'].values:
                    result = self.apply_highpass_filter(ch, cutoffull, replace, returnValue, sseg, order, plot)
                    if returnValue:
                        results.append(result)
                else:
                    logger.warning(f"Channel '{ch}' not found, skipping.")
            if returnValue:
                return results
            return None
        
        # 获取数据
        try:
            data = self.data[sseg][chName].values
        except Exception as e:
            logger.error(f"Error accessing data for channel {chName}: {str(e)}")
            return None
            
        # 检查数据是否存在且长度足够
        if data is None or len(data) <= 0:
            logger.error(f"No data found for channel {chName} in segment {sseg}")
            return None
        
        # 保存原始数据，用于后续对比或绘图
        original_data = copy.deepcopy(data)
        
        # 应用滤波器
        try:
            filtered_data = _butter_highpass_filter(data, cutoff, self.__fs__, order)
            
            # 如果需要绘图对比
            if plot:
                try:
                    # 为了绘图对比，我们需要创建一个临时通道
                    temp_channel_name = f"{chName}_filtered"
                    
                    # 创建一个临时PyDAS对象的副本，用于比较
                    temp_pydas = copy.deepcopy(self)
                    
                    # 添加滤波后的临时通道
                    unit = temp_pydas.chInfo.loc[temp_pydas.chInfo['Name'] == chName, 'Unit'].values[0]
                    temp_pydas.add_channel(
                        name=temp_channel_name,
                        unit=unit,
                        series=filtered_data,
                        fs=self.__fs__,
                        sseg=sseg
                    )
                    
                    # Use Plotly for interactive comparison
                    from pydas_plot import plot_channel
                    # Plot original and filtered data together
                    channels = [chName, temp_channel_name]
                    logger.info(f"Displaying interactive comparison plot for {chName} before/after filtering (cutoff={cutoffull} Hz, order={order})")
                    plot_channel(
                        pydas_obj=temp_pydas,
                        ch_name=channels,
                        sseg=sseg,
                        title=f"Highpass Filter Comparison - {chName} (cutoff={cutoffull} Hz, order={order})",
                        alpha=[0.5, 0.8],  # 原始数据透明度0.5，滤波后数据保持默认0.8
                    )
                except Exception as e:
                    logger.error(f"Error creating comparison plot: {str(e)}")
            
            # 如果需要替换数据
            if replace:
                self.data[sseg][chName] = filtered_data
                logger.info(f'Highpass for {chName} filter = {cutoffull:3.2f} Hz, Lambda = {self.__lam__:02d}')
                self.updateST(chName=chName)
        except Exception as e:
            logger.error(f"Failed to apply filter to {chName}: {str(e)}")
            return None
                
        # 返回结果（如果需要）
        if returnValue:
            return filtered_data
            
        return None

    def remove_mean(self,
               chName,
               sseg=0):
        """
        Remove the mean value from one or more channels.
        
        Parameters:
        -----------
        chName : str or list
            Name of the channel(s) to process
        sseg : int, optional
            Segment index, default is 0
            
        Raises:
        -------
        ValueError
            If chName is neither a string nor a list
        """
        if isinstance(chName, list):
            for ichName in chName:
                data = self.data[sseg][ichName].values
                self.data[sseg][ichName] = data - data.mean()
                self.updateST(chName=ichName)
            logger.info('remove mean for Channels: ' + ', '.join(chName))
        elif isinstance(chName, str):
            data = self.data[sseg][chName].values
            self.data[sseg][chName] = data - data.mean()
            self.updateST(chName=chName)
            logger.info('remove mean for ' + chName)
        else:
            logger.warning('Unknown type for ChName!')

    def add_value(self,
               chName,
               value2add,
               sseg=0):
        """
        Add a constant value to one or more channels.
        
        Parameters:
        -----------
        chName : str or list
            Name of the channel(s) to process
        value2add : float
            Value to add to the channel(s)
        sseg : int, optional
            Segment index, default is 0
            
        Raises:
        -------
        ValueError
            If chName is neither a string nor a list
        """
        if isinstance(chName, list):
            for ichName in chName:
                data = self.data[sseg][ichName].values
                self.data[sseg][ichName] = data + value2add
                self.updateST(chName=ichName)
        elif isinstance(chName, str):
            data = self.data[sseg][chName].values
            self.data[sseg][chName] = data + value2add
            self.updateST(chName=chName)
        else:
            logger.warning('Unknown type for ChName!')  

    def multiply_value(self,
               chName,
               value2mul,
               sseg=0):
        """
        Multiply one or more channels by a constant value.
        
        Parameters:
        -----------
        chName : str or list
            Name of the channel(s) to process
        value2mul : float
            Value to multiply the channel(s) by
        sseg : int, optional
            Segment index, default is 0
            
        Raises:
        -------
        ValueError
            If chName is neither a string nor a list
        """
        if isinstance(chName, list):
            for ichName in chName:
                data = self.data[sseg][ichName].values
                self.data[sseg][ichName] = data * value2mul
                self.updateST(chName=ichName)
        elif isinstance(chName, str):
            data = self.data[sseg][chName].values
            self.data[sseg][chName] = data * value2mul
            self.updateST(chName=chName)
        else:
            logger.warning('Unknown type for ChName!')

    def cut_series(self,
                  start,
                  stop,
                  sseg=0):
        """
        Cut a time series to a specified range.
        
        Parameters:
        -----------
        start : str or float
            Start time of the cut (time string or seconds)
        stop : str or float
            End time of the cut (time string or seconds)
        sseg : int, optional
            Segment index, default is 0
            
        Raises:
        -------
        ValueError
            If start or stop times are invalid
        """
        def moveTimestr(Timestr, seconds_float):
            seconds = int(seconds_float)
            milliseconds = int((seconds_float-seconds)*1000)
            startTime = datetime.datetime.strptime(Timestr,"%H:%M:%S.%f")
            startTime_new = (startTime + datetime.timedelta(seconds=seconds, milliseconds=milliseconds)).strftime("%H:%M:%S.%f")
            return startTime_new[:-5]

        startIndx = int(start * self.__fs__)
        stopIndx = int(stop * self.__fs__)
        lngth = self.data[sseg].index[-1]
        self.data[sseg] = self.data[sseg].drop(range(startIndx + 1))
        self.data[sseg] = self.data[sseg].drop(range(stopIndx, lngth + 1))
        self.data[sseg] = self.data[sseg].reset_index(drop=True)

        sampNum = self.data[sseg].shape[0]

        self.segInfo.loc['Seg{0:2d}'.format(
            sseg),'Start'] = moveTimestr(self.segInfo['Start'].values[0], start)
        self.segInfo.loc['Seg{0:2d}'.format(
            sseg),'Stop'] = moveTimestr(self.segInfo['Start'].values[0], stop)
        self.segInfo.loc['Seg{0:2d}'.format(
            sseg),'Duration'] = '{0:8.1f}s'.format((sampNum - 1) / self.__fs__)
        self.segInfo.loc['Seg{0:2d}'.format(
            sseg),'N sample'] = sampNum
        self.updateST(sseg=sseg)
        logger.info('Cut time series from {0:5.2f}s to {1:5.2f}s'.format(
                start, stop))

    def move_data(self, chName, point_of_move, sseg=0):
        """
        Move data in a channel by a specified number of points.
        
        Parameters:
        -----------
        chName : str
            Name of the channel to move
        point_of_move : int
            Number of points to move the data (positive for forward, negative for backward)
        sseg : int, optional
            Segment index, default is 0
            
        Raises:
        -------
        KeyError
            If the channel does not exist
        """
        if chName in self.chInfo['Name'].values:
            data = self.data[sseg][chName].values
            n_sample = self.segInfo.iloc[sseg]['N sample']
            
            if point_of_move > 0:
                # Move forward (right shift)
                data_new = np.zeros(n_sample)
                data_new[point_of_move:] = data[:n_sample - point_of_move]
            else:
                # Move backward (left shift)
                data_new = np.zeros(n_sample)
                data_new[:n_sample + point_of_move] = data[-point_of_move:]
                
            self.data[sseg][chName] = data_new
            self.updateST(chName=chName)
            logger.info(f'Moved {chName} by {point_of_move} points')
        else:
            logger.error(f'ERROR! {chName:8s} not found.')

    def read_motion(self, motionfname, alignAccName=None, alignMethod='acc', zerofilename='', lowpassfilter=-1, rotation=True, NameList=['Platform']):
        """
        Read motion data and add as channels.
        
        Parameters:
        -----------
        motionfname : str
            Path to motion data file
        alignAccName : str, optional
            Acceleration channel name for alignment, default is None
        alignMethod : str, optional
            Alignment method ('time' or 'cross_correlation'), default is 'time'
        zerofilename : str, optional
            Path to zero reference file, default is ''
        lowpassfilter : float, optional
            Lowpass filter cutoff frequency, default is -1 (no filter)
        rotation : bool, optional
            Whether to apply rotation, default is True
        NameList : list of str, optional
            List of object names to process, default is ['Platform']
            
        Notes:
        ------
        - Reads motion data from file
        - Supports data alignment and filtering
        - Handles coordinate transformations
        - Updates channel information
        """
        try:
            # Try to open and read the motion file
            with open(motionfname, 'r') as f:
                try:
                    f.seek(0)
                    lines = f.readlines()
                    # Parse header information
                    try:
                        n_body = int(lines[2].replace('\n', '').split('\t')[1])
                        # n_frames = int(lines[0].replace('\n', '').split('\t')[1])
                        motion_fs = float(lines[3].replace('\n', '').split('\t')[1])
                        Time_start = pd.Timestamp(lines[7].replace('\n', '').split('\t')[1])
                        rotationname = lines[10].replace('\n', '').split('\t')[3:6]
                    except (IndexError, ValueError) as e:
                        logger.error(f"Invalid motion file format: {str(e)}")
                        raise ValueError(f"Motion file format is invalid: {str(e)}")
                except Exception as e:
                    logger.error(f"Error reading motion file: {str(e)}")
                    raise
        except FileNotFoundError:
            logger.error(f"Motion file not found: {motionfname}")
            raise

        motionName = ['Surge','Sway','Heave'] + rotationname
        motionDataRawList = []
        
        # Process zero reference file if provided
        if zerofilename:
            try:
                for ibody in range(n_body):
                    try:
                        # Read zero reference data
                        ZeromotionDataRaw = np.genfromtxt(
                            zerofilename, 
                            skip_header=12, 
                            delimiter='\t', 
                            usecols=(0+ibody*17, 1+ibody*17, 2+ibody*17, 3+ibody*17, 4+ibody*17, 5+ibody*17)
                        )
                        Zeromean = ZeromotionDataRaw.mean(axis=0)
                        Zeromean[3] = 0  # Don't apply zero correction to yaw
                        logger.info(f"Zero reference data: {Zeromean}")
                        # Read motion data and apply zero correction
                        motionDataRawList.append(
                            np.genfromtxt(
                                motionfname, 
                                skip_header=12, 
                                delimiter='\t', 
                                usecols=(0+ibody*17, 1+ibody*17, 2+ibody*17, 3+ibody*17, 4+ibody*17, 5+ibody*17)
                            ) - Zeromean
                        )
                    except Exception as e:
                        logger.error(f"Error processing data for body {ibody}: {str(e)}")
                        raise ValueError(f"Failed to process motion data for body {ibody}")
            except FileNotFoundError:
                logger.error(f"Zero reference file not found: {zerofilename}")
                raise
        else:
            # Read motion data without zero correction
            try:
                for ibody in range(n_body):
                    motionDataRawList.append(
                        np.genfromtxt(
                            motionfname, 
                            skip_header=12, 
                            delimiter='\t', 
                            usecols=(0+ibody*17, 1+ibody*17, 2+ibody*17, 3+ibody*17, 4+ibody*17, 5+ibody*17)
                        )
                    )
            except Exception as e:
                logger.error(f"Error reading motion data: {str(e)}")
                raise ValueError(f"Failed to read motion data: {str(e)}")

        # Process each body's motion data
        for ibody, motionDataRaw in enumerate(motionDataRawList):
            try:
                # Convert units from mm to cm for position data
                motionDataRaw[:, 0:3] /= 10
                
                # Apply rotation if requested
                if rotation:
                    try:
                        Yaw = np.mean(motionDataRaw[:, motionName.index('Yaw')])
                        # Yaw = 180
                        r = R.from_euler('z', Yaw, degrees=True)
                        motionDataRaw[:, 0:3] = r.apply(motionDataRaw[:, 0:3])
                        logger.info(f'motion rotated: {Yaw: 0.2f} DEG')
                    except Exception as e:
                        logger.warning(f"Failed to apply rotation: {str(e)}")
                        # Continue without rotation

                # Add channels for each motion component
                unit = ['cm']*3 + ['deg']*3
                for i, iName in enumerate(motionName):
                    try:
                        self.add_channel(
                            NameList[ibody] + '.' + iName, 
                            unit[i], 
                            motionDataRaw[:, i], 
                            fs=motion_fs
                        )
                    except Exception as e:
                        logger.error(f"Failed to add channel {NameList[ibody]}.{iName}: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing motion data for body {ibody}: {str(e)}")
                # Continue with next body
        
        # Align motion data with existing data
        try:
            if alignMethod == 'acc':
                try:
                    # Calculate acceleration from heave motion
                    heave = self.apply_lowpass_filter(NameList[0]+'.Heave', replace=False, returnValue=True)
                    vz = diff1d(heave/100, 1 / self.__fs__)
                    az = diff1d(vz, 1 / self.__fs__) * -1
                    n_sample = self.segInfo.iloc[0]['N sample']
                    
                    # Find correlation with acceleration channel
                    base = self.apply_lowpass_filter(alignAccName, replace=False, returnValue=True)
                    lag = np.argmax(correlate(base, az, method='fft')) - n_sample + 1
                    
                    # Add calculated acceleration channel
                    self.add_channel('azfromHeave', unit='m/s2', series=az, point_of_move=lag, fs=self.__fs__)
                    
                    # Move all motion channels by the calculated lag
                    for ibody in NameList:
                        for iname in motionName:
                            try:
                                self.move_data(ibody+'.'+iname, point_of_move=lag)
                            except Exception as e:
                                logger.warning(f"Failed to move channel {ibody}.{iname}: {str(e)}")
                except Exception as e:
                    logger.error(f"Failed to align using acceleration method: {str(e)}")
                    logger.warning("Continuing without alignment")
            elif alignMethod == 'time':
                try:
                    # Calculate time difference and convert to sample points
                    timedelta = Time_start - pd.Timestamp('2023-'+self.__date__+' '+self.segInfo['Start']['Seg 0'])
                    lag = round(timedelta.total_seconds() * self.__fs__)
                    
                    # Move all motion channels by the calculated lag
                    for ibody in NameList:
                        for iname in motionName:
                            try:
                                self.move_data(ibody+'.'+iname, point_of_move=lag)
                            except Exception as e:
                                logger.warning(f"Failed to move channel {ibody}.{iname}: {str(e)}")
                except Exception as e:
                    logger.error(f"Failed to align using time method: {str(e)}")
                    logger.warning("Continuing without alignment")
            elif alignMethod != 'none':
                logger.warning(f"Unknown alignment method: {alignMethod}. No alignment applied.")
        except Exception as e:
            logger.error(f"Error during alignment: {str(e)}")
            logger.warning("Continuing without alignment")

        # Apply lowpass filter if requested
        if lowpassfilter > 0:
            try:
                for ibody in NameList:
                    for iname in motionName:
                        try:
                            self.apply_lowpass_filter(ibody+'.'+iname, cutoffull=lowpassfilter)
                        except Exception as e:
                            logger.warning(f"Failed to apply filter to {ibody}.{iname}: {str(e)}")
            except Exception as e:
                logger.error(f"Error applying lowpass filter: {str(e)}")
                logger.warning("Continuing without filtering")

    def updateST(self, chName='all', sseg=0):
        """
        Update statistical information for channels.
        
        Parameters:
        -----------
        chName : str, optional
            Channel name to update, 'all' for all channels, default is 'all'
        sseg : int, optional
            Segment index to process, default is 0
            
        Notes:
        ------
        - Calculates basic statistics (mean, std, min, max)
        - Updates channel information
        - Handles multiple channels efficiently
        - Uses vectorized operations for performance
        """
        if chName == 'all':
            # 使用pandas的优化方法一次性计算所有统计量
            try:
                # 获取数据帧
                data_frame = self.data[sseg]
                
                # 检查数据大小，对于大型数据使用分块处理
                if data_frame.shape[0] * data_frame.shape[1] > 10000000:  # 阈值可调整
                    # 使用dask进行大数据并行计算
                    try:
                        import dask.dataframe as dd
                        
                        # 将pandas DataFrame转换为dask DataFrame
                        dask_df = dd.from_pandas(data_frame, npartitions=min(32, data_frame.shape[1]))
                        
                        # 并行计算统计量
                        mean_result = dask_df.mean().compute()
                        std_result = dask_df.std().compute()
                        max_result = dask_df.max().compute()
                        min_result = dask_df.min().compute()
                        
                        # 创建结果DataFrame
                        stats = pd.DataFrame({
                            'Mean': mean_result,
                            'Std': std_result,
                            'Max': max_result,
                            'Min': min_result
                        })
                        
                        # 添加单位列
                        stats['Unit'] = self.chInfo.set_index('Name')['Unit']
                        
                        # 更新统计信息
                        self.segStatis[sseg] = stats
                        
                    except ImportError:
                        # 如果dask不可用，使用分块处理
                        logger.info("Dask not available, using chunked processing for large dataset")
                        
                        # 分块大小
                        chunk_size = 1000000 // data_frame.shape[1]
                        chunk_size = max(chunk_size, 1000)  # 确保至少有1000行
                        
                        # 初始化结果
                        means = pd.Series(index=data_frame.columns)
                        stds = pd.Series(index=data_frame.columns)
                        maxs = pd.Series(index=data_frame.columns)
                        mins = pd.Series(index=data_frame.columns)
                        
                        # 分块处理
                        n_chunks = (data_frame.shape[0] + chunk_size - 1) // chunk_size
                        
                        # 使用Welford算法进行在线计算均值和标准差
                        count = 0
                        M2 = pd.Series(0, index=data_frame.columns)
                        mean = pd.Series(0, index=data_frame.columns)
                        
                        # 初始化最大最小值
                        maxs = data_frame.iloc[0]
                        mins = data_frame.iloc[0]
                        
                        for i in range(n_chunks):
                            start_idx = i * chunk_size
                            end_idx = min((i + 1) * chunk_size, data_frame.shape[0])
                            chunk = data_frame.iloc[start_idx:end_idx]
                            
                            # 更新最大最小值
                            maxs = pd.concat([maxs, chunk.max()]).max(level=0)
                            mins = pd.concat([mins, chunk.min()]).min(level=0)
                            
                            # 更新均值和方差（Welford算法）
                            for _, row in chunk.iterrows():
                                count += 1
                                delta = row - mean
                                mean += delta / count
                                delta2 = row - mean
                                M2 += delta * delta2
                        
                        # 计算标准差
                        stds = np.sqrt(M2 / count)
                        means = mean
                        
                        # 创建结果DataFrame
                        stats = pd.DataFrame({
                            'Mean': means,
                            'Std': stds,
                            'Max': maxs,
                            'Min': mins,
                            'Unit': self.chInfo.set_index('Name')['Unit']
                        })
                        
                        # 更新统计信息
                        self.segStatis[sseg] = stats
                else:
                    # 对于小型数据，使用pandas的优化方法
                    # 并行计算统计量
                    stats = data_frame.agg(['mean', 'std', 'max', 'min'])
                    
                    # 转置结果，使其与所需格式匹配
                    stats = stats.T
                    stats.columns = ['Mean', 'Std', 'Max', 'Min']
                    
                    # 添加单位列
                    stats['Unit'] = self.chInfo.set_index('Name')['Unit']
                    
                    # 更新统计信息
                    self.segStatis[sseg] = stats
                
            except Exception as e:
                logger.error(f"统计计算错误: {str(e)}")
                # 回退到原始方法
                data_frame = self.data[sseg]
                means = data_frame.mean()
                stds = data_frame.std()
                maxs = data_frame.max()
                mins = data_frame.min()
                
                # 创建统计数据DataFrame
                stats_data = {
                    'Mean': means,
                    'Std': stds,
                    'Max': maxs,
                    'Min': mins,
                    'Unit': self.chInfo.set_index('Name')['Unit']
                }
                
                # 更新统计信息
                self.segStatis[sseg] = pd.DataFrame(stats_data)
        else:
            # 只更新指定通道的统计信息
            if chName in self.chInfo['Name'].values:
                # 使用pandas的Series方法快速计算统计量
                series = self.data[sseg][chName]
                
                # 对于大型序列，使用分块处理
                if len(series) > 10000000:  # 阈值可调整
                    # 分块大小
                    chunk_size = 1000000
                    
                    # 初始化结果
                    count = 0
                    mean = 0
                    M2 = 0
                    max_val = series.iloc[0]
                    min_val = series.iloc[0]
                    
                    # 分块处理
                    n_chunks = (len(series) + chunk_size - 1) // chunk_size
                    
                    for i in range(n_chunks):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, len(series))
                        chunk = series.iloc[start_idx:end_idx]
                        
                        # 更新最大最小值
                        max_val = max(max_val, chunk.max())
                        min_val = min(min_val, chunk.min())
                        
                        # 更新均值和方差（Welford算法）
                        for val in chunk:
                            count += 1
                            delta = val - mean
                            mean += delta / count
                            delta2 = val - mean
                            M2 += delta * delta2
                    
                    # 计算标准差
                    std = np.sqrt(M2 / count)
                    
                    stats = {
                        'mean': mean,
                        'std': std,
                        'max': max_val,
                        'min': min_val
                    }
                else:
                    # 对于小型序列，直接使用pandas方法
                    stats = series.agg(['mean', 'std', 'max', 'min'])
                
                # 获取单位
                unit_idx = self.chInfo['Name'].values == chName
                unit = self.chInfo.loc[unit_idx, 'Unit'].values[0]
                
                # 更新统计信息
                self.segStatis[sseg].loc[chName] = [
                    stats['mean'], stats['std'], stats['max'], stats['min'], unit]
            else:
                logger.error(f'ERROR! {chName:8s} not found.')

    def updateChN(self, sseg=0):
        """
        Update channel count information.
        
        Parameters:
        -----------
        sseg : int, optional
            Segment index to process, default is 0
            
        Notes:
        ------
        - Updates channel count in segment information
        - Validates channel consistency
        - Maintains data integrity
        """
        if self.data[sseg].shape[1] == self.chInfo.shape[0] == self.segStatis[0].shape[0]:
            self.__chN__ = self.chInfo.shape[0]
        else:
            raise ValueError("Number of channels does not match!")

    def rename_channel(self,
                 chOld,
                 chNew,
                 sseg=0):
        """
        Rename a channel in the dataset.
        
        Parameters:
        -----------
        chOld : str
            Original channel name
        chNew : str
            New channel name
        sseg : int, optional
            Segment index to process, default is 0
            
        Notes:
        ------
        - Updates channel name in data, chInfo and statistics
        - Maintains all data and properties
        """
        try:
            # Check if the old channel exists
            if chOld not in self.data[sseg].columns:
                logger.error(f"Channel '{chOld}' not found in segment {sseg}")
                raise KeyError(f"Channel '{chOld}' not found")
                
            # Check if the new channel name already exists
            if chNew in self.data[sseg].columns:
                logger.error(f"Channel '{chNew}' already exists in segment {sseg}")
                raise ValueError(f"Channel '{chNew}' already exists")
                
            # 1. 重命名数据DataFrame中的列
            self.data[sseg].rename(columns={chOld: chNew}, inplace=True)
            
            # 2. 更新chInfo中的通道名
            # 找到具有旧通道名的行索引
            ch_mask = self.chInfo['Name'] == chOld
            # 直接使用布尔掩码更新名称
            self.chInfo.loc[ch_mask, 'Name'] = chNew
            
            # 3. 重命名统计表中的索引
            if chOld in self.segStatis[sseg].index:
                self.segStatis[sseg].rename(index={chOld: chNew}, inplace=True)
            
            logger.info(f"Renamed channel '{chOld}' to '{chNew}' in segment {sseg}")
            return True
            
        except Exception as e:
            logger.error(f"Channel renaming failed: {str(e)}")
            return False
    
    
    def data_wash(self,
                 ChName,
                 method='linear',
                 order=5, 
                 threshold=3,
                 sseg=0):
        """
        Clean data by detecting and interpolating outliers.
        
        Parameters:
        -----------
        ChName : str
            Name of the channel to clean
        method : str, optional
            Interpolation method ('linear' or 'polynomial'), default is 'linear'
        order : int, optional
            Order of polynomial interpolation, default is 5
        threshold : float, optional
            Standard deviation threshold for outlier detection, default is 3
        sseg : int, optional
            Segment index to process, default is 0
            
        Notes:
        ------
        - Uses statistical methods to detect outliers
        - Supports different interpolation methods
        - Optimized for large datasets
        - Maintains data continuity
        """
        try:
            # Check if the channel exists
            if ChName not in self.data[sseg].columns:
                logger.error(f"Channel '{ChName}' not found in segment {sseg}")
                raise KeyError(f"Channel '{ChName}' not found")
                
            # 获取数据Series
            data_series = self.data[sseg][ChName]
            data_length = len(data_series)
            
            # 为了更高效地处理大数据集，根据数据大小选择不同的处理方法
            if data_length > 1000000:  # 超大数据集
                return self._data_wash_large(ChName, method, order, threshold, sseg)
            
            logger.info(f"Cleaning channel '{ChName}' with {method} interpolation (threshold={threshold}σ)")
            
            # 使用pandas的优化方法计算均值和标准差
            arr_mean = data_series.mean()
            arr_std = data_series.std()
            
            # 检测异常值（使用向量化操作）
            outlier_mask = np.abs(data_series - arr_mean) > threshold * arr_std
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                logger.info(f"Found {outlier_count} outliers in channel '{ChName}'")
                
                # 创建带有NaN值的Series用于插值
                cleaned_series = data_series.copy()
                cleaned_series[outlier_mask] = np.nan
                
                # 使用pandas的优化插值方法
                try:
                    if method in ['spline', 'polynomial']:
                        # 这些方法需要order参数
                        filled_series = cleaned_series.interpolate(method=method, order=order, limit_direction='both')
                        logger.info(f"Applied {method} interpolation with order {order}")
                    else:
                        # 其他方法不需要order参数
                        filled_series = cleaned_series.interpolate(method=method, limit_direction='both')
                        logger.info(f"Applied {method} interpolation")
                    
                    # 检查是否还有NaN值
                    remaining_nans = filled_series.isna().sum()
                    if remaining_nans > 0:
                        logger.warning(f"{remaining_nans} NaN values could not be interpolated")
                        
                        # 尝试用前向和后向填充处理剩余的NaN值
                        filled_series = filled_series.fillna(method='ffill').fillna(method='bfill')
                        
                        # 再次检查
                        remaining_nans = filled_series.isna().sum()
                        if remaining_nans > 0:
                            logger.error(f"{remaining_nans} NaN values still remain after additional filling")
                        else:
                            logger.info("Remaining NaN values filled with forward/backward fill")
                    
                    # 更新数据
                    self.data[sseg][ChName] = filled_series.values
                    
                except Exception as e:
                    logger.error(f"Interpolation failed: {str(e)}")
                    raise ValueError(f"Interpolation method '{method}' failed: {str(e)}")
            else:
                logger.info(f"No outliers found in channel '{ChName}'")
            
            # 手动更新统计信息，避免列不匹配问题
            series = self.data[sseg][ChName].values
            
            # 获取单位
            unit_idx = np.where(self.chInfo['Name'].values == ChName)[0][0]
            unit = self.chInfo['Unit'].values[unit_idx]
            
            # 手动计算统计量并更新
            self.segStatis[sseg].loc[ChName] = [
                np.mean(series), np.std(series), np.amax(series), np.amin(series), unit]
                
            return True
            
        except Exception as e:
            logger.error(f"Data washing failed: {str(e)}")
            return False
    
    def _data_wash_large(self, ChName, method='linear', order=5, threshold=3, sseg=0):
        """
        优化的处理大型数据集的数据清洗方法。
        通过分块处理来减少内存占用。
        
        Parameters:
        -----------
        同data_wash方法
        """
        try:
            # 获取数据
            data = self.data[sseg][ChName].values
            data_length = len(data)
            
            logger.info(f"Using optimized method for large dataset ({data_length} points)")
            
            # 计算全局均值和标准差
            global_mean = np.mean(data)
            global_std = np.std(data)
            
            # 分块大小
            chunk_size = min(100000, data_length // 10)  # 确保至少分10块
            
            # 创建输出数组
            output_data = np.copy(data)
            total_outliers = 0
            
            # 分块处理
            for start in range(0, data_length, chunk_size):
                end = min(start + chunk_size, data_length)
                chunk = data[start:end]
                
                # 在当前块中检测异常值
                outlier_mask = np.abs(chunk - global_mean) > threshold * global_std
                outlier_indices = np.where(outlier_mask)[0] + start
                chunk_outlier_count = len(outlier_indices)
                total_outliers += chunk_outlier_count
                
                if chunk_outlier_count > 0:
                    # 将异常值设为NaN
                    output_data[outlier_indices] = np.nan
            
            if total_outliers > 0:
                logger.info(f"Found {total_outliers} outliers in channel '{ChName}'")
                
                # 使用pandas的Series进行高效插值
                series = pd.Series(output_data)
                
                try:
                    if method in ['spline', 'polynomial']:
                        filled_series = series.interpolate(method=method, order=order, limit_direction='both')
                    else:
                        filled_series = series.interpolate(method=method, limit_direction='both')
                    
                    # 处理边缘的NaN值
                    filled_series = filled_series.fillna(method='ffill').fillna(method='bfill')
                    
                    # 检查是否还有NaN值
                    remaining_nans = filled_series.isna().sum()
                    if remaining_nans > 0:
                        logger.warning(f"{remaining_nans} NaN values could not be filled")
                    
                    # 更新数据
                    self.data[sseg][ChName] = filled_series.values
                    
                except Exception as e:
                    logger.error(f"Large dataset interpolation failed: {str(e)}")
                    raise ValueError(f"Interpolation method '{method}' failed for large dataset: {str(e)}")
            else:
                logger.info(f"No outliers found in channel '{ChName}'")
            
            # 手动更新统计信息，避免列不匹配问题
            cleaned_data = self.data[sseg][ChName].values
            
            # 获取单位
            unit_idx = np.where(self.chInfo['Name'].values == ChName)[0][0]
            unit = self.chInfo['Unit'].values[unit_idx]
            
            # 手动计算统计量并更新
            self.segStatis[sseg].loc[ChName] = [
                np.mean(cleaned_data), np.std(cleaned_data), np.amax(cleaned_data), np.amin(cleaned_data), unit]
                
            return True
            
        except Exception as e:
            logger.error(f"Large dataset washing failed: {str(e)}")
            return False

    def add_diff1(self, name, sseg=0, filter=False, filter_cutoff=2):
        """
        Calculate and add first derivative of a channel.
        
        Parameters:
        -----------
        name : str
            Name of the channel to differentiate
        sseg : int, optional
            Segment index to process, default is 0
        filter : bool, optional
            Whether to apply lowpass filter, default is False
        filter_cutoff : float, optional
            Cutoff frequency for filtering in Hz, default is 2
            
        Notes:
        ------
        - Uses optimized numerical differentiation
        - Optional lowpass filtering to reduce noise
        - Maintains data alignment
        """
        try:
            if name in self.chInfo['Name'].values:
                # Get channel data and unit
                data = self.data[sseg][name].values
                unit_idx = np.where(self.chInfo['Name'].values == name)[0][0]
                unit = self.chInfo['Unit'].values[unit_idx]
                
                # Calculate derivative
                dt = 1.0 / self.__fs__
                diff_data = diff1d(data, dt)
                logger.info(f"Calculated first derivative of {name}")
                
                # Apply filter if requested
                if filter:
                    diff_data = self.apply_lowpass_filter(diff_data, cutoffull=filter_cutoff, 
                                                         replace=False, returnValue=True)
                
                # Determine new unit
                if 'm' in unit and not '/' in unit:
                    new_unit = unit + '/s'
                elif 'deg' in unit and not '/' in unit:
                    new_unit = unit + '/s'
                else:
                    new_unit = unit + '/s'
                
                # Add new channel
                new_name = name + '_d1'
                self.add_channel(new_name, new_unit, diff_data, self.__fs__)
                return True
            else:
                logger.error(f"Channel '{name}' does not exist.")
                return False
        except Exception as e:
            logger.error(f"Failed to add derivative channel: '{name}'")
            logger.error(f"Error in add_diff1: {str(e)}")
            return False

    def add_diff2(self, name, sseg=0, filter=False, filter_cutoff=2):
        """
        Calculate and add second derivative of a channel.
        
        Parameters:
        -----------
        name : str
            Name of the channel to differentiate
        sseg : int, optional
            Segment index to process, default is 0
        filter : bool, optional
            Whether to apply lowpass filter, default is False
        filter_cutoff : float, optional
            Cutoff frequency for filtering in Hz, default is 2
            
        Notes:
        ------
        - Uses optimized numerical differentiation
        - Optional lowpass filtering to reduce noise
        - Maintains data alignment
        """
        try:
            if name in self.chInfo['Name'].values:
                # Get channel data and unit
                data = self.data[sseg][name].values
                unit_idx = np.where(self.chInfo['Name'].values == name)[0][0]
                unit = self.chInfo['Unit'].values[unit_idx]
                
                # Calculate first derivative
                dt = 1.0 / self.__fs__
                diff1_data = diff1d(data, dt)
                logger.info(f"Calculated first derivative of {name}")
                
                # Apply filter if requested
                if filter:
                    diff1_data = self.apply_lowpass_filter(diff1_data, cutoffull=filter_cutoff, 
                                                          replace=False, returnValue=True)
                
                # Calculate second derivative
                diff2_data = diff1d(diff1_data, dt)
                logger.info(f"Calculated second derivative of {name}")
                
                # Apply filter if requested
                if filter:
                    diff2_data = self.apply_lowpass_filter(diff2_data, cutoffull=filter_cutoff, 
                                                          replace=False, returnValue=True)
                
                # Determine new unit
                if 'm' in unit and not '/' in unit:
                    new_unit = unit + '/s2'
                elif 'deg' in unit and not '/' in unit:
                    new_unit = unit + '/s2'
                else:
                    new_unit = unit + '/s2'
                
                # Add the new channel
                new_name = name + '_d2'
                self.add_channel(new_name, new_unit, diff2_data, self.__fs__, sseg=sseg)
                logger.info(f"Added second derivative channel {new_name}")
                
                self.add_channel(new_name, new_unit, diff2_data, self.__fs__)
                return True
            else:
                logger.error(f"Channel '{name}' does not exist.")
                return False
        except Exception as e:
            logger.error(f"Failed to add derivative channel: '{name}'")
            logger.error(f"Error in add_diff2: {str(e)}")
            return False

    def plot_channel(self, ch_idx, sseg=0, figsize=(10, 6), title=None, 
                    xlabel='Time (s)', ylabel=None, grid=True, 
                    color='blue', linewidth=1.0, alpha=0.8,
                    xlim=None, ylim=None, interactive=True,
                 downsampling=True, max_points=40000, 
                 save_path=None, show=True, use_plotly=True,
                 height=None, width=None, save_html=None,
                 table_width=0.25, column_widths=None,
                 stats=True, dpi=300):
        """
        Plot a channel from the PyDAS object with interactive features.
        
        Parameters:
        -----------
        ch_idx : int, str, or list
            Channel index, name, or list of channel names to plot
        sseg : int, optional
            Segment index to plot, default is 0
        figsize : tuple, optional
            Figure size in inches (width, height), default is (10, 6)
        title : str, optional
            Plot title, default is None (auto-generated)
        xlabel : str, optional
            X-axis label, default is 'Time (s)'
        ylabel : str, optional
            Y-axis label, default is None (auto-generated)
        grid : bool, optional
            Whether to show grid, default is True
        color : str, optional
            Line color for single channel plot, default is 'blue'
        linewidth : float, optional
            Line width, default is 1.0
        alpha : float, optional
            Line transparency, default is 0.8
        xlim : tuple, optional
            X-axis limits (min, max), default is None (auto)
        ylim : tuple, optional
            Y-axis limits (min, max), default is None (auto)
        interactive : bool, optional
            Whether to enable interactive features, default is True
        downsampling : bool, optional
            Whether to enable downsampling for large datasets, default is True
        max_points : int, optional
            Maximum number of points to plot before downsampling, default is 20000
        save_path : str, optional
            Path to save the figure, default is None (don't save)
        show : bool, optional
            Whether to display the plot, default is True
        use_plotly : bool, optional
            Whether to use Plotly for interactive web-based plotting, default is True
        height : int, optional
            Height in pixels for Plotly plot, default is None (auto)
        width : int, optional
            Width in pixels for Plotly plot, default is None (auto)
        save_html : str, optional
            Path to save the interactive HTML plot, default is None (don't save)
        table_width : float, optional
            Relative width of the statistics table (0-1), default is 0.25
        column_widths : list, optional
            Relative widths for each column in the statistics table, default is None
        stats : bool, optional
            Whether to show statistics, default is True
        dpi : int, optional
            DPI for saved image, default is 300
            
        Returns:
        --------
        Figure object (matplotlib.Figure or plotly.graph_objects.Figure)
        """
        try:
            from pydas_plot import plot_channel as plot_channel_func
            
            # 将ch_idx转换为ch_name
            if isinstance(ch_idx, int):
                if ch_idx < len(self.chInfo):
                    ch_name = self.chInfo.iloc[ch_idx]['Name']
                else:
                    logger.error(f"Channel index {ch_idx} out of bounds.")
                    return None
            elif isinstance(ch_idx, str):
                if ch_idx in self.chInfo['Name'].values:
                    ch_name = ch_idx
                else:
                    logger.error(f"Channel '{ch_idx}' not found.")
                    return None
            elif isinstance(ch_idx, list):
                # 如果是通道名称列表，直接使用
                if all(isinstance(item, str) for item in ch_idx):
                    ch_name = ch_idx
                # 如果是索引列表，转换为名称列表
                elif all(isinstance(item, int) for item in ch_idx):
                    ch_name = [self.chInfo.iloc[i]['Name'] for i in ch_idx if i < len(self.chInfo)]
                    if not ch_name:
                        logger.error("No valid channels to plot.")
                        return None
                else:
                    logger.error("Channel list must contain all strings or all integers.")
                    return None
            else:
                logger.error("Channel identifier must be an integer, string, or list.")
                return None
            
            # 调用外部模块的plot_channel函数
            return plot_channel_func(
                pydas_obj=self,
                ch_name=ch_name,
                sseg=sseg,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=xlim,
                ylim=ylim,
                grid=grid,
                show=show,
                save_path=save_path,
                use_plotly=use_plotly,
                downsampling=downsampling,
                max_points=max_points,
                save_html=save_html,
                dpi=dpi,
                width=width,
                height=height,
                color=color,
                alpha=alpha,
                linewidth=linewidth,
                figsize=figsize,
                stats=stats,
                table_width=table_width,
                column_widths=column_widths
            )
            
        except ImportError:
            logger.error("pydas_plot module not found. Please ensure it's installed and in the Python path.")
            return None
        except Exception as e:
            logger.error(f"Error in plot_channel: {str(e)}")
            raise

    def plot_histogram(self, ch_idx, sseg=0, title=None, xlabel=None, ylabel='Count',
                    bins=50, xlim=None, ylim=None, grid=True, show=True, 
                    save_path=None, use_plotly=True, save_html=None, 
                    dpi=300, width=None, height=None, color=None, 
                    alpha=0.6, figsize=(12, 6), fit_gaussian=True, fit_color='red'):
        """
        Plot a histogram of a channel from the PyDAS object.
        
        Parameters:
        -----------
        ch_idx : int, str, or list
            Channel index, name, or list of channel indices/names to plot histogram for
        sseg : int, optional
            Segment index to plot, default is 0
        title : str, optional
            Plot title, default is None (auto-generated)
        xlabel : str, optional
            X-axis label, default is None (auto-generated from channel name and unit)
        ylabel : str, optional
            Y-axis label, default is 'Count'
        bins : int, optional
            Number of histogram bins, default is 50
        xlim : tuple, optional
            X-axis limits as (min, max), default is None (auto)
        ylim : tuple, optional
            Y-axis limits as (min, max), default is None (auto)
        grid : bool, optional
            Whether to show grid, default is True
        show : bool, optional
            Whether to display the plot, default is True
        save_path : str, optional
            Path to save the plot, default is None (don't save)
        use_plotly : bool, optional
            Use Plotly for interactive web-based plotting, default is True
        save_html : str, optional
            Path to save the interactive HTML plot, default is None (don't save)
        dpi : int, optional
            DPI for saved image, default is 300
        width : int, optional
            Width in pixels for Plotly plot, default is None (auto)
        height : int, optional
            Height in pixels for Plotly plot, default is None (auto)
        color : str or list, optional
            Histogram color or list of colors, default is None (auto-generated)
        alpha : float, optional
            Histogram transparency, default is 0.6
        figsize : tuple, optional
            Figure size for matplotlib in inches, default is (12, 6)
        fit_gaussian : bool, optional
            Whether to fit a Gaussian distribution to the data, default is True
        fit_color : str or list, optional
            Color of the Gaussian fit curve or list of colors, default is 'red'
            
        Returns:
        --------
        Figure object (matplotlib.figure.Figure or plotly.graph_objects.Figure)
        """
        try:
            from pydas_plot import plot_histogram as plot_histogram_func
            
            # 将ch_idx转换为ch_name
            if isinstance(ch_idx, int):
                    if ch_idx < len(self.chInfo):
                        ch_name = self.chInfo.iloc[ch_idx]['Name']
                    else:
                        logger.error(f"Channel index {ch_idx} out of bounds.")
                        return None
            elif isinstance(ch_idx, str):
                if ch_idx in self.chInfo['Name'].values:
                    ch_name = ch_idx
                else:
                    logger.error(f"Channel '{ch_idx}' not found.")
                    return None
            elif isinstance(ch_idx, list):
                # 处理通道列表 - 支持索引列表或名称列表
                ch_name = []
                
                # 如果是通道名称列表，验证每个名称
                if all(isinstance(item, str) for item in ch_idx):
                    for name in ch_idx:
                        if name in self.chInfo['Name'].values:
                            ch_name.append(name)
                        else:
                            logger.warning(f"Channel '{name}' not found, skipping.")
                    
                    if not ch_name:
                        logger.error("No valid channels to plot.")
                        return None
                    
                # 如果是索引列表，转换为名称列表
                elif all(isinstance(item, int) for item in ch_idx):
                    for idx in ch_idx:
                        if idx < len(self.chInfo):
                            ch_name.append(self.chInfo.iloc[idx]['Name'])
                        else:
                            logger.warning(f"Channel index {idx} out of bounds, skipping.")
                    
                    if not ch_name:
                        logger.error("No valid channels to plot.")
                        return None
                    else:
                        logger.error("Channel list must contain all strings or all integers.")
                    return None
            else:
                logger.error("Channel identifier must be an integer, string, or list.")
                return None
            
            # 调用外部模块的plot_histogram函数
            return plot_histogram_func(
                pydas_obj=self,
                ch_name=ch_name,
                sseg=sseg,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                bins=bins,
                xlim=xlim,
                ylim=ylim,
                grid=grid,
                show=show,
                save_path=save_path,
                use_plotly=use_plotly,
                save_html=save_html,
                dpi=dpi,
                width=width,
                height=height,
                color=color,
                alpha=alpha,
                figsize=figsize,
                fit_gaussian=fit_gaussian,
                fit_color=fit_color
            )
            
        except ImportError:
            logger.error("pydas_plot module not found. Please ensure it's installed and in the Python path.")
            return None
        except Exception as e:
            logger.error(f"Error in plot_histogram: {str(e)}")
            raise
            
    def plot_xy(self, x_ch_idx, y_ch_idx, sseg=0, title=None, 
              xlabel=None, ylabel=None, xlim=None, ylim=None, grid=True, 
              show=True, save_path=None, use_plotly=True, save_html=None,
              dpi=300, width=None, height=None, color='blue', alpha=0.8, 
              marker_size=5, figsize=(8, 8), line=False, fit_line=False,
              fit_color='red', fit_line_width=2, fit_alpha=0.8,
              show_stats=False, downsampling=True, max_points=10000,
              density_plot=False, density_colorscale='Viridis', 
              density_opacity=0.7, use_webgl=True, adaptive_sampling=False,
              datashade=False, contour_levels=20, sampling_algorithm='lttb',
              memory_efficient=True, bin_size=None):
        """
        Create an XY scatter plot with one channel on the X-axis and another on the Y-axis.
        
        Args:
            x_ch_idx (int, str): Index or name of the channel to plot on the X-axis.
            y_ch_idx (int, str): Index or name of the channel to plot on the Y-axis.
            sseg (int, optional): Segment index. Defaults to 0.
            title (str, optional): Plot title. Defaults to None.
            xlabel (str, optional): X-axis label. Defaults to None (uses channel name).
            ylabel (str, optional): Y-axis label. Defaults to None (uses channel name).
            xlim (tuple, optional): X-axis limits (min, max). Defaults to None.
            ylim (tuple, optional): Y-axis limits (min, max). Defaults to None.
            grid (bool, optional): Show grid. Defaults to True.
            show (bool, optional): Display the plot. Defaults to True.
            save_path (str, optional): Path to save the plot. Defaults to None.
            use_plotly (bool, optional): Use Plotly for interactive plot. Defaults to True.
            save_html (str, optional): Path to save interactive Plotly plot as HTML. Defaults to None.
            dpi (int, optional): DPI for saved plot. Defaults to 300.
            width (int, optional): Width of the plot in pixels. Defaults to None.
            height (int, optional): Height of the plot in pixels. Defaults to None.
            color (str, optional): Color for the scatter points. Defaults to 'blue'.
            alpha (float, optional): Opacity for the scatter points. Defaults to 0.8.
            marker_size (float, optional): Size of the scatter points. Defaults to 5.
            figsize (tuple, optional): Figure size for matplotlib. Defaults to (8, 8).
            line (bool, optional): Connect points with lines. Defaults to False.
            fit_line (bool, optional): Show linear regression fit line. Defaults to False.
            fit_color (str, optional): Color for fit line. Defaults to 'red'.
            fit_line_width (float, optional): Width of fit line. Defaults to 2.
            fit_alpha (float, optional): Opacity of fit line. Defaults to 0.8.
            show_stats (bool, optional): Show statistical information on the plot. Defaults to False.
            downsampling (bool, optional): Apply downsampling for large datasets. Defaults to True.
            max_points (int, optional): Maximum number of points to show before downsampling. Defaults to 10000.
            density_plot (bool, optional): Show density contour plot for large datasets. Defaults to False.
            density_colorscale (str, optional): Colorscale for density plot. Defaults to 'Viridis'.
            density_opacity (float, optional): Opacity for density contours. Defaults to 0.7.
            use_webgl (bool, optional): Use WebGL rendering for better performance with large datasets. Defaults to True.
            adaptive_sampling (bool, optional): Use adaptive sampling to preserve signal features. Defaults to False.
            datashade (bool, optional): Use datashading for very large datasets (>100k points). Defaults to False.
            contour_levels (int, optional): Number of contour levels for density plot. Defaults to 20.
            sampling_algorithm (str, optional): Algorithm for downsampling: 'lttb' (Largest Triangle Three Buckets), 
                                               'uniform', or 'peak'. Defaults to 'lttb'.
            memory_efficient (bool, optional): Use memory-efficient methods for very large datasets. Defaults to True.
            bin_size (tuple, optional): Bin size for 2D histogram (x_bins, y_bins). Defaults to None (auto).
            
        Returns:
            tuple: (pandas.DataFrame with x and y data, figure object)
        """
        try:
            # 将通道索引或名称转换为通道名称
            x_ch_name = self._validate_channel(x_ch_idx)
            y_ch_name = self._validate_channel(y_ch_idx)
            
            if x_ch_name is None or y_ch_name is None:
                return None
            
            # 尝试导入pydas_plot模块
            try:
                from pydas_plot import plot_xy as plot_xy_func
            except ImportError:
                logger.error("pydas_plot module not found. Please ensure it's installed and in the Python path.")
                return None
            
            # 调用外部模块的plot_xy函数
            return plot_xy_func(
                pydas_obj=self,
                x_ch_name=x_ch_name,
                y_ch_name=y_ch_name,
                sseg=sseg,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                xlim=xlim,
                ylim=ylim,
                grid=grid,
                show=show,
                save_path=save_path,
                use_plotly=use_plotly,
                save_html=save_html,
                dpi=dpi,
                width=width,
                height=height,
                color=color,
                alpha=alpha,
                marker_size=marker_size,
                figsize=figsize,
                line=line,
                fit_line=fit_line,
                fit_color=fit_color,
                fit_line_width=fit_line_width,
                fit_alpha=fit_alpha,
                show_stats=show_stats,
                downsampling=downsampling,
                max_points=max_points,
                density_plot=density_plot,
                density_colorscale=density_colorscale,
                density_opacity=density_opacity,
                use_webgl=use_webgl,
                adaptive_sampling=adaptive_sampling,
                datashade=datashade,
                contour_levels=contour_levels,
                sampling_algorithm=sampling_algorithm,
                memory_efficient=memory_efficient,
                bin_size=bin_size
            )
            
        except ImportError:
            logger.error("pydas_plot module not found. Please ensure it's installed and in the Python path.")
            return None
        except Exception as e:
            logger.error(f"Error in plot_xy: {str(e)}")
            raise
    
    def _validate_channel(self, ch_idx):
        """
        Validate channel index and return the normalized index.
        
        Parameters:
        -----------
        ch_idx : int or str
            Channel index or name
            
        Returns:
        --------
        int
            Normalized channel index
        """
        if isinstance(ch_idx, str):
            if ch_idx in self.chInfo['Name'].values:
                return self.chInfo[self.chInfo['Name'] == ch_idx].index[0]
            else:
                logger.error(f"Channel name '{ch_idx}' not found.")
                return -1
        elif isinstance(ch_idx, (int, np.integer)):
            if 0 <= ch_idx < self.__chN__:
                return ch_idx
            else:
                logger.error(f"Channel index {ch_idx} out of range [0, {self.__chN__-1}].")
                return -1
        else:
            logger.error(f"Invalid channel identifier type: {type(ch_idx)}")
            return -1
            
    def _get_default_transDict(self, g=9.807):
        """
        获取默认的单位转换字典
        
        Parameters:
        -----------
        g : float, optional
            重力加速度，默认值为9.807 m/s²
            
        Returns:
        --------
        dict
            单位转换字典，包含常用单位的转换规则
        """
        return {
            'kg': ['kN', np.array([g * 0.001, 1.0, 3.0])],
            'cm': ['m', np.array([0.01, 0.0, 1.0])],
            'mm': ['m', np.array([0.001, 0.0, 1.0])],
            'm': ['m', np.array([1, 0.0, 1.0])],
            's': ['s', np.array([1, 0.0, 0.5])],
            'deg': ['deg', np.array([1, 0.0, 0.0])],
            'rad': ['rad', np.array([1, 0.0, 0.0])],
            'n': ['kn', np.array([0.001, 1.0, 3.0])],
            'kn': ['kn', np.array([1, 0.0, 0.0])],
            '%': ['%', np.array([1, 0.0, 0.0])],
            '-': ['-', np.array([1, 0.0, 0.0])]
        }

    def _findtrans(self, unit, transDict=None, trans_cache=None):
        """
        查找单位的转换参数
        
        Parameters:
        -----------
        unit : str
            需要转换的单位
        transDict : dict, optional
            单位转换字典，如果为None，使用默认字典
        trans_cache : dict, optional
            用于存储已计算过的转换结果的缓存
            
        Returns:
        --------
        list
            [new_unit, coefficients]，其中coefficients是[coeff_unit, coeff_rho, coeff_lambda]
        """
        if transDict is None:
            transDict = self._get_default_transDict()
            
        # 初始化缓存
        if trans_cache is None:
            trans_cache = {}
            
        # 检查结果是否已在缓存中
        unit = unit.lower().strip()
        if unit in trans_cache:
            return trans_cache[unit]
        
        if unit in transDict:
            trans = transDict[unit]
            trans_cache[unit] = trans
            return trans
        elif '/' in unit:
            unitUpper, unitLower = unit.split('/')
            transUpper = self._findtrans(unitUpper, transDict, trans_cache)
            transLower = self._findtrans(unitLower, transDict, trans_cache)
            trans = [transUpper[0] + '/' +
                     transLower[0], np.array([0.0, 0.0, 0.0])]
            trans[1][0] = transUpper[1][0] / transLower[1][0]
            trans[1][1] = transUpper[1][1] - transLower[1][1]
            trans[1][2] = transUpper[1][2] - transLower[1][2]
            trans_cache[unit] = trans
            return trans
        elif '.' in unit:
            unitWithDot = unit.split('.')
            transU = []
            transN1 = np.array([])
            transN2 = np.array([])
            transN3 = np.array([])
            for uWithDot in unitWithDot:
                transWithDot = self._findtrans(uWithDot, transDict, trans_cache)
                transU.append(transWithDot[0])
                transN1 = np.append(transN1, transWithDot[1][0])
                transN2 = np.append(transN2, transWithDot[1][1])
                transN3 = np.append(transN3, transWithDot[1][2])
            trans = ['.'.join(transU), np.array([1.0, 0.0, 0.0])]
            for x in np.nditer(transN1):
                trans[1][0] *= x
            trans[1][1] = transN2.sum()
            trans[1][2] = transN3.sum()
            trans_cache[unit] = trans
            return trans
        elif unit[-1].isdigit():
            n = int(unit[-1])
            unit_base = unit[0:-1]
            if unit_base in transDict:
                trans_temp = transDict[unit_base]
                trans = [trans_temp[0] +
                         str(n), np.array([1.0, 0.0, 0.0])]
                trans[1][0] = trans_temp[1][0]**n
                trans[1][1] = trans_temp[1][1] * n
                trans[1][2] = trans_temp[1][2] * n
                trans_cache[unit] = trans
                return trans
            else:
                logger.warning(
                    f"Input unit '{unit}' cannot be identified, using default values.")
                return [unit, np.array([1.0, 0.0, 0.0])]
        else:
            logger.warning(
                f"Input unit '{unit}' cannot be identified, using default values.")
            return [unit, np.array([1.0, 0.0, 0.0])]
            
    def _plot_spectrum(self, spec, channel_name, title=None, xlim=None, ylim=None, 
                      figsize=(10, 6), show=True, save_path=None, dpi=300,
                      use_plotly=True, save_html=None, width=None, height=None,
                      fullscale=False):
        """Helper method to plot spectrum."""
        try:
            # Check if spec exists
            if spec is None:
                print(f"Unable to compute spectrum for channel {channel_name}")
                return None
            
            # Get frequency (Hz) and spectral density
            if hasattr(spec, 'args'):
                if isinstance(spec.args, tuple) and len(spec.args) > 0:
                    # Handle case where spec.args is a tuple
                    f = spec.args[0]  # 保持角频率单位 (rad/s)
                else:
                    # Handle case where args is directly available but not a tuple
                    f = spec.args  # 保持角频率单位 (rad/s)
            else:
                # Handle other cases where frequency might be stored
                raise ValueError("Could not find frequency data in spectrum object")
                
            if hasattr(spec, 'data'):
                S = spec.data  # Spectral density from data attribute
            elif hasattr(spec, 'S'):
                S = spec.S  # Spectral density from S attribute
            else:
                raise ValueError("Could not find spectral density data in spectrum object")
            
            # Check if f and S are numpy arrays
            if not isinstance(f, np.ndarray) or not isinstance(S, np.ndarray):
                warnings.warn("Frequency or spectrum data is not a numpy array.")
                return None
            
            # Ensure f and S have matching dimensions
            if f.ndim > 1:
                f = f.flatten()
                warnings.warn("Flattened frequency array of dimension > 1")
            if S.ndim > 1:
                S = S.flatten()
                warnings.warn("Flattened spectral density array of dimension > 1")
            
            if len(f) != len(S):
                warnings.warn(f"Frequency and spectral density arrays have different lengths: {len(f)} vs {len(S)}")
                min_len = min(len(f), len(S))
                f = f[:min_len]
                S = S[:min_len]
            
            # Try to compute spectral characteristics
            try:
                # Calculate spectral moments
                moment_0 = spec.moment(0)
                if isinstance(moment_0, tuple) and len(moment_0) > 0:
                    if isinstance(moment_0[0], list) and len(moment_0[0]) > 0:
                        m0 = float(moment_0[0][0])  # Extract from list in tuple
                    else:
                        m0 = float(moment_0[0])  # Extract from tuple
                else:
                    m0 = float(moment_0)  # Direct value
                
                # Try to get higher moments with the same approach
                try:
                    moment_1 = spec.moment(1)
                    if isinstance(moment_1, tuple) and len(moment_1) > 0:
                        if isinstance(moment_1[0], list) and len(moment_1[0]) > 0:
                            m1 = float(moment_1[0][0])
                        else:
                            m1 = float(moment_1[0])
                    else:
                        m1 = float(moment_1)
                except Exception:
                    m1 = None
                
                try:
                    moment_2 = spec.moment(2)
                    if isinstance(moment_2, tuple) and len(moment_2) > 0:
                        if isinstance(moment_2[0], list) and len(moment_2[0]) > 0:
                            m2 = float(moment_2[0][0])
                        else:
                            m2 = float(moment_2[0])
                    else:
                        m2 = float(moment_2)
                except Exception:
                    m2 = None
                
                # Calculate standard spectral parameters
                Hm0 = 4.0 * np.sqrt(m0) if m0 is not None else None
                
                # 计算峰值周期Tp (s)，将角频率转换为周期
                if len(S) > 0:
                    max_idx = np.argmax(S)
                    if max_idx < len(f) and f[max_idx] > 0:
                        Tp = 2 * np.pi / f[max_idx]  # 从角频率(rad/s)计算周期(s)
                    else:
                        Tp = None
                else:
                    Tp = None
                
                # 计算平均周期  
                Tm01 = 2 * np.pi * m0 / m1 if m0 is not None and m1 is not None and m1 != 0 else None
                Tm02 = 2 * np.pi * np.sqrt(m0 / m2) if m0 is not None and m2 is not None and m2 != 0 else None
                
                # Format the spectral characteristics text
                stats_text = []
                if Hm0 is not None:
                    if fullscale:
                        # 由于数据已经在spectral_analysis函数中被缩放,
                        # 这里的谱矩和Hm0已经反映了全尺度的值，所以不需要额外缩放
                        stats_text.append(f"Hm0 = {Hm0:.2f} m (full scale)")
                    else:
                        stats_text.append(f"Hm0 = {Hm0:.2f} m")
                if Tp is not None:
                    if fullscale:
                        # 峰值周期已经反映了全尺度的值，因为频率已经在計算中被调整
                        stats_text.append(f"Tp = {Tp:.2f} s (full scale)")
                    else:
                        stats_text.append(f"Tp = {Tp:.2f} s")
                if Tm01 is not None:
                    if fullscale:
                        # Tm01已经反映了全尺度的值，因为频率已经在計算中被调整
                        stats_text.append(f"Tm01 = {Tm01:.2f} s (full scale)")
                    else:
                        stats_text.append(f"Tm01 = {Tm01:.2f} s")
                if Tm02 is not None:
                    if fullscale:
                        # Tm02已经反映了全尺度的值，因为频率已经在計算中被调整
                        stats_text.append(f"Tm02 = {Tm02:.2f} s (full scale)")
                    else:
                        stats_text.append(f"Tm02 = {Tm02:.2f} s")
                stats_text = "\n".join(stats_text)
            except Exception as e:
                warnings.warn(f"Unable to calculate all spectral characteristics: {str(e)}")
                stats_text = "Spectral characteristics unavailable"
            
            # Set title if not provided
            if title is None:
                if fullscale:
                    title = f"Full Scale Spectrum of {channel_name}"
                else:
                    title = f"Spectrum of {channel_name}"
            
            if use_plotly:
                import plotly.graph_objects as go
                from plotly.offline import plot
                
                # Set the width and height if provided
                if width is None:
                    width = figsize[0] * 100
                if height is None:
                    height = figsize[1] * 100
                
                # Create figure
                fig = go.Figure()
                
                # Add spectrum trace
                fig.add_trace(
                    go.Scatter(
                        x=f,
                        y=S,
                        mode='lines',
                        line=dict(color='blue', width=2),
                        name='Spectrum'
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=title,
                    xaxis_title='Angular Frequency (rad/s)',
                    yaxis_title='Spectral Density',
                    width=width,
                    height=height,
                    margin=dict(l=50, r=50, b=50, t=70, pad=4),
                    annotations=[
                        dict(
                            x=0.99,
                            y=0.99,
                            xref="paper",
                            yref="paper",
                            text=stats_text,
                            showarrow=False,
                            align="right",
                            xanchor="right",
                            yanchor="top",
                            bgcolor="rgba(255, 255, 255, 0.7)",
                            bordercolor="black",
                            borderwidth=1,
                            font=dict(size=12),
                        )
                    ]
                )
                
                # Set axis limits if provided
                if xlim is not None:
                    fig.update_xaxes(range=xlim)
                else:
                    # 设置x轴从0开始
                    fig.update_xaxes(range=[0, max(f) * 1.05])
                
                if ylim is not None:
                    fig.update_yaxes(range=ylim)
                else:
                    # 设置y轴从0开始
                    fig.update_yaxes(range=[0, max(S) * 1.05])
                
                # Show or save the figure
                if show:
                    fig.show()
                
                if save_html is not None:
                    plot(fig, filename=save_html, auto_open=False)
                
                if save_path is not None:
                    fig.write_image(save_path, scale=2)
                    
                return fig
            else:
                # Original matplotlib implementation
                import matplotlib.pyplot as plt
                
                # Create figure
                fig, ax = plt.subplots(figsize=figsize)
                
                # Plot spectrum
                ax.plot(f, S, 'b-', linewidth=2)
                
                # Set labels and title
                ax.set_xlabel('Angular Frequency (rad/s)')
                ax.set_ylabel('Spectral Density')
                ax.set_title(title)
                    
                # Add text box with spectral characteristics
                if stats_text:
                    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
                    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                            verticalalignment='top', horizontalalignment='right', 
                            bbox=props, fontsize=10)
                
                # Set axis limits if provided
                if xlim is not None:
                    ax.set_xlim(xlim)
                else:
                    # 设置x轴从0开始
                    ax.set_xlim(0, max(f) * 1.05)
                    
                if ylim is not None:
                    ax.set_ylim(ylim)
                else:
                    # 设置y轴从0开始
                    ax.set_ylim(0, max(S) * 1.05)
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Show or save the figure
                if save_path is not None:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                
                if show:
                    plt.show()
                else:
                    plt.close()
                
                return fig
        except Exception as e:
            logger.error(f"Error plotting spectrum: {str(e)}")
            return None

    def channel2fullscale(self, channel_name, lam, rho=1.025, g=9.807):
        """
        Convert a single channel from model scale to prototype scale and return timeseries.
        
        Parameters:
        -----------
        channel_name : str
            Name of the channel to convert
        lam : float
            Scale factor
        rho : float, optional
            Water density in kg/m³, default is 1.025
        g : float, optional
            Gravitational acceleration in m/s², default is 9.807
            
        Returns:
        --------
        ts : waveModel.TimeSeries
            TimeSeries object containing the converted data with time in seconds
        
        Notes:
        ------
        - Applies Froude scaling laws without modifying the original data
        - Uses the same conversion logic as to_fullscale method
        - Returns a TimeSeries object from waveModel module
        """

        # Validate channel exists
        if channel_name not in self.chInfo['Name'].values:
            logger.error(f"Channel {channel_name} not found")
            return None
            
        # Get channel index
        ch_idx = self.chInfo[self.chInfo['Name'] == channel_name].index[0]
        
        # Get channel unit
        unit = self.chInfo.loc[ch_idx, 'Unit']
        
        # Define unit conversion dictionary (same as in to_fullscale)
        transDict = self._get_default_transDict(g)
        
        # Get conversion factors
        trans_temp = self._findtrans(unit, transDict)
        C1 = trans_temp[1][0]  # CoeffUnit
        C2 = rho ** trans_temp[1][1]  # CoeffRho
        C3 = lam ** trans_temp[1][2]  # CoeffLam
        coeff = C1 * C2 * C3
        
        # Calculate time array based on the scaling
        fs_scaled = self.__fs__ / np.sqrt(lam)
        
        # 只处理第一段数据（如果用户需要多段，可以拓展此功能）
        idx1 = 0
        if self.__segN__ > 1:
            logger.info(f"Multiple segments found. Only converting first segment.")
            
        # Extract original data
        data = self.data[idx1][channel_name].copy()
        
        # Apply conversion coefficient
        data_scaled = data * coeff
        
        # Create time array
        T = np.arange(0, len(data)) / fs_scaled
        
        # 创建TimeSeries对象
        # TimeSeries构造函数需要data和args参数，其中args是时间向量
        ts = TimeSeries(data_scaled,T)
        
        return ts

    def spectral_analysis(self, channel_name, method='cov', L=1024, plot=False, title=None, show=True, save_path=None, use_plotly=True, save_html=None,fullscale=False, lam=None, rho=1.025, g=9.807, freq_range=(0, 2)):
        """
        Perform spectral analysis on a single channel and return a spectral data object
        
        Parameters:
        -----------
        channel_name : str
            Name of the channel to analyze
        method : str, optional
            Spectral analysis method ('cov' or 'psd'), default is 'cov'
        L : int, optional
            Window size for spectral analysis, default is 1024
        plot : bool, optional
            Whether to generate a plot, default is False
        title : str, optional
            Title for the plot, default is None
        show : bool, optional
            Whether to display the plot, default is True
        save_path : str, optional
            Path to save the plot, default is None
        use_plotly : bool, optional
            Use Plotly for interactive plotting, default is True
        save_html : str, optional
            Path to save interactive HTML plot, default is None
        fullscale : bool, optional
            Whether to convert data to full scale before analysis, default is False
        lam : float, optional
            Scale factor, used only when fullscale=True, default uses object's __lam__ attribute
        rho : float, optional
            Water density (kg/m³), default is 1.025
        g : float, optional
            Gravitational acceleration (m/s²), default is 9.807
        freq_range : tuple, optional
            Frequency range in full scale (rad/s), default is (0, 2)
            
        Returns:
        --------
        spec : waveModel.SpecData1D
            Spectral data object
            
        Notes:
        ------
        - Spectral analysis is performed using the waveModel toolkit
        - The spectrum shows spectral density vs. angular frequency (rad/s)
        - The returned object can be used for further analysis or custom plotting
        - When fullscale=True, data is converted to full scale using channel2fullscale method before analysis
        - freq_range specifies the valid frequency range in full scale, which is automatically converted for model scale
        """
        # Check if channel exists
        if channel_name not in self.chInfo['Name'].values:
            logger.error(f"Channel '{channel_name}' does not exist")
            return None
            
        # Ensure valid scale factor
        if lam is None:
            if hasattr(self, '__lam__'):
                lam = self.__lam__
            else:
                if fullscale:
                    logger.error("No scale factor lam provided and object has no default __lam__ attribute")
                    return None
                else:
                    # If no full scale conversion needed, set a default value for frequency range calculation
                    lam = 1
                    
        # Calculate corresponding frequency range
        # In Froude scaling, frequency scale is sqrt(λ)
        if fullscale:
            # Full scale uses the directly specified range
            w_range = freq_range
        else:
            # Model scale, convert frequency range
            # f_model = f_full * sqrt(λ)
            w_range = (freq_range[0] * np.sqrt(lam), freq_range[1] * np.sqrt(lam))
            logger.info(f"Model scale frequency range conversion: {freq_range} rad/s -> {w_range} rad/s")
            
        # If full scale conversion is requested
        if fullscale:                    
            try:
                # Get full scale TimeSeries using channel2fullscale
                ts = self.channel2fullscale(channel_name, lam, rho, g)
                if ts is None:
                    logger.error(f"Full scale conversion failed for channel: {channel_name}")
                    return None
                    
                # Calculate spectrum
                spec = ts.tospecdata(L=L, method=method)
            except Exception as e:
                logger.error(f"Full scale spectral analysis failed: {str(e)}")
                return None
        else:
            # Default using the first data segment
            sseg = 0
            
            # Get channel data
            data = self.data[sseg][channel_name].values.copy()
            
            # Create time vector (assuming equal sampling intervals)
            fs = self.__fs__
            t = np.arange(0, len(data)) / fs
            
            # Create TimeSeries object
            try:
                # 修改TimeSeries初始化方式，遵循其定义
                ts = TimeSeries(data, t)
                
                # Calculate spectrum
                spec = ts.tospecdata(L=L, method=method)
            except Exception as e:
                logger.error(f"Spectral analysis failed: {str(e)}")
                return None
        
        # Apply frequency range limitation
        try:
            # Get frequencies and corresponding spectral density
            freqs = spec.args
            density = spec.data
            
            # Find indices within specified range
            idx = np.logical_and(freqs >= w_range[0], freqs <= w_range[1])
            
            # If no data points found, warn but continue
            if not np.any(idx):
                logger.warning(f"No data points within specified frequency range {w_range} rad/s")
            else:
                # Update spectral object data
                spec.args = freqs[idx]
                spec.data = density[idx]
                logger.info(f"Spectral data limited to range {w_range[0]:.3f}-{w_range[1]:.3f} rad/s")
                
                # If spec object has other attributes that need to be synchronized, update them too
                # For example, if spec.S exists, it needs to be updated
                if hasattr(spec, 'S') and spec.S is not None:
                    spec.S = spec.S[idx]
        except Exception as e:
            logger.warning(f"Error applying frequency range limitation: {str(e)}")
        
        # If plotting is requested
        if plot:
            # Set title
            if title is None:
                title_prefix = "Full Scale " if fullscale else ""
                title = f"{title_prefix}Spectrum of {channel_name}"
            
            if use_plotly:
                # Use Plotly for plotting
                try:
                    import plotly.graph_objects as go
                    
                    # Get frequency and spectral density
                    f = spec.args
                    S = spec.data
                    
                    # Create figure
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=f, y=S, mode='lines', name='Spectrum'
                    ))
                    
                    # Set layout
                    fig.update_layout(
                        title=title,
                        xaxis_title='Angular Frequency (rad/s)',
                        yaxis_title='Spectral Density',
                        xaxis=dict(range=[w_range[0], min(w_range[1]*1.05, max(f)*1.05)]),
                        yaxis=dict(range=[0, max(S)*1.05])
                    )
                    
                    # Display frequency range information
                    range_text = f"Range: {w_range[0]:.2f}-{w_range[1]:.2f} rad/s"
                    fig.add_annotation(
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        text=range_text,
                        showarrow=False,
                        font=dict(size=10),
                        bgcolor="rgba(255,255,255,0.8)"
                    )
                    
                    # Save or display figure
                    if save_html:
                        fig.write_html(save_html)
                    if show:
                        fig.show()
                except ImportError:
                    logger.warning("Plotly not installed, will use Matplotlib")
                    use_plotly = False
            
            if not use_plotly:
                # Use Matplotlib for plotting
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(spec.args, spec.data, 'b-', linewidth=2)
                ax.set_title(title)
                ax.set_xlabel('Angular Frequency (rad/s)')
                ax.set_ylabel('Spectral Density')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Set x-axis range to specified frequency range
                ax.set_xlim(w_range[0], min(w_range[1]*1.05, max(spec.args)*1.05))
                ax.set_ylim(0, max(spec.data)*1.05)
                
                # Display frequency range information
                range_text = f"Range: {w_range[0]:.2f}-{w_range[1]:.2f} rad/s"
                ax.text(0.02, 0.98, range_text, transform=ax.transAxes, 
                       fontsize=9, va='top', ha='left',
                       bbox=dict(facecolor='white', alpha=0.8, pad=2))
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                if show:
                    plt.show()
                else:
                    plt.close()
        
        return spec
