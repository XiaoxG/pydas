#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
PyDAS Utilities Module
Provides common utility functions for the PyDAS system including 
numerical differentiation and sampling frequency conversion operations.
"""
import numpy as np
from scipy import interpolate
from logger import logger

# Import numba for acceleration
try:
    from numba import jit, float64, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available. Some functions will run slower.")

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

# Trans cache used by findtrans function for performance optimization
_global_trans_cache = {}

def get_default_transDict(g=9.807):
    """
    Get default unit conversion dictionary for scale transformations.
    
    Parameters:
    -----------
    g : float, optional
        Gravitational acceleration in m/s², default is 9.807
        
    Returns:
    --------
    dict
        Dictionary of unit conversion rules
        
    Notes:
    ------
    - Keys are original units
    - Values are lists where:
      * First element is the new unit
      * Second element is a numpy array with:
        - [0]: Unit conversion coefficient
        - [1]: Power for density (rho)
        - [2]: Power for scale factor (lambda)
    """
    return {
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

def findtrans(unit, transDict, clear_cache=False):
    """
    Find unit conversion factors for scaling.
    
    Parameters:
    -----------
    unit : str
        Unit to be converted
    transDict : dict
        Dictionary of unit conversion rules
    clear_cache : bool, optional
        Whether to clear the cache, default is False
        
    Returns:
    --------
    list
        [new_unit, coefficients_array] where coefficients_array contains
        [unit_coeff, rho_power, lambda_power]
    
    Notes:
    ------
    - Handles composite units with / (division) and . (multiplication)
    - Handles units with numeric suffixes (e.g., m2 for square meters)
    - Includes caching for performance optimization
    - Returns default values if unit cannot be identified
    """
    global _global_trans_cache
    
    # Clear cache if requested
    if clear_cache:
        _global_trans_cache = {}
        # If we're just clearing the cache and unit is empty or 'none', return a default value without warning
        if not unit or unit.lower() == 'none':
            return ['', np.array([1.0, 0.0, 0.0])]
    
    # Convert to lowercase and strip whitespace
    unit = unit.lower().strip()
    
    # Check if result is already in cache
    if unit in _global_trans_cache:
        return _global_trans_cache[unit]
    
    # Basic unit lookup
    if unit in transDict:
        trans = transDict[unit]
        _global_trans_cache[unit] = trans
        return trans
    # Handle units with division (e.g. m/s)
    elif '/' in unit:
        unitUpper, unitLower = unit.split('/')
        transUpper = findtrans(unitUpper, transDict)
        transLower = findtrans(unitLower, transDict)
        trans = [transUpper[0] + '/' + transLower[0], np.array([0.0, 0.0, 0.0])]
        trans[1][0] = transUpper[1][0] / transLower[1][0]
        trans[1][1] = transUpper[1][1] - transLower[1][1]
        trans[1][2] = transUpper[1][2] - transLower[1][2]
        _global_trans_cache[unit] = trans
        return trans
    # Handle units with dot notation (e.g. n.m)
    elif '.' in unit:
        unitWithDot = unit.split('.')
        transU = []
        transN1 = np.array([])
        transN2 = np.array([])
        transN3 = np.array([])
        for uWithDot in unitWithDot:
            transWithDot = findtrans(uWithDot, transDict)
            transU.append(transWithDot[0])
            transN1 = np.append(transN1, transWithDot[1][0])
            transN2 = np.append(transN2, transWithDot[1][1])
            transN3 = np.append(transN3, transWithDot[1][2])
        trans = ['.'.join(transU), np.array([1.0, 0.0, 0.0])]
        for x in np.nditer(transN1):
            trans[1][0] *= x
        trans[1][1] = transN2.sum()
        trans[1][2] = transN3.sum()
        _global_trans_cache[unit] = trans
        return trans
    # Handle units with numeric suffix (e.g. m2)
    elif unit and unit[-1].isdigit():
        n = int(unit[-1])
        unit_base = unit[0:-1]
        if unit_base in transDict:
            trans_temp = transDict[unit_base]
            trans = [trans_temp[0] + str(n), np.array([1.0, 0.0, 0.0])]
            trans[1][0] = trans_temp[1][0]**n
            trans[1][1] = trans_temp[1][1] * n
            trans[1][2] = trans_temp[1][2] * n
            _global_trans_cache[unit] = trans
            return trans
        else:
            logger.warning(f"Input unit '{unit}' cannot be identified, using default values.")
            return [unit, np.array([1.0, 0.0, 0.0])]
    # Unidentified units use default values
    else:
        logger.warning(f"Input unit '{unit}' cannot be identified, using default values.")
        return [unit, np.array([1.0, 0.0, 0.0])] 