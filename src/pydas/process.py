#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process module for PyDAS.
Contains functions for data processing and manipulation.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import copy
from .utils import diff1d

from .logger import get_logger
logger = get_logger('pydas.process')

def apply_lowpass_filter(pydas_obj, chName, cutoffull=2, replace=True, returnValue=False, 
                        sseg=0, order=6, plot=False):
    """
    Apply a lowpass filter to a channel.
    
    Parameters:
    -----------
    pydas_obj : PyDAS
        PyDAS object containing the data
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
    if isinstance(chName, str) and chName not in pydas_obj.chInfo['Name'].values:
        logger.error(f"Channel '{chName}' not found.")
        return None
    
    # 模型尺度下调整截止频率
    if pydas_obj.__scale__ == 'model':
        cutoff = cutoffull / 2 / np.pi * np.sqrt(pydas_obj.__lam__)
    else:
        cutoff = cutoffull / 2 / np.pi

    # 处理通道列表
    if isinstance(chName, list):
        results = []
        for ch in chName:
            if ch in pydas_obj.chInfo['Name'].values:
                result = apply_lowpass_filter(pydas_obj, ch, cutoffull, replace, returnValue, sseg, order, plot)
                if returnValue:
                    results.append(result)
            else:
                logger.warning(f"Channel '{ch}' not found, skipping.")
        if returnValue:
            return results
        return None
                
    # 获取数据
    try:
        data = pydas_obj.data[sseg][chName].values
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
        filtered_data = _butter_lowpass_filter(data, cutoff, pydas_obj.__fs__, order)
        
        # 如果需要绘图对比
        if plot:
            try:
                # 为了绘图对比，我们需要创建一个临时通道
                temp_channel_name = f"{chName}_filtered"
                
                # 创建一个临时PyDAS对象的副本，用于比较
                temp_pydas = copy.deepcopy(pydas_obj)
                
                # 添加滤波后的临时通道
                unit = temp_pydas.chInfo.loc[temp_pydas.chInfo['Name'] == chName, 'Unit'].values[0]
                temp_pydas.add_channel(
                    name=temp_channel_name,
                    unit=unit,
                    series=filtered_data,
                    fs=pydas_obj.__fs__,
                    sseg=sseg
                )
                
                # Use Plotly for interactive comparison
                from .plot import plot_channel
                plot_channel(
                    pydas_obj=temp_pydas,
                    ch_name=[chName, temp_channel_name],
                    sseg=sseg,
                    title=f"Lowpass Filter Comparison - {chName} (cutoff={cutoffull} Hz, order={order})",
                    alpha=[0.5, 0.8],  # 原始数据透明度0.5，滤波后数据保持默认0.8
                )
            except Exception as e:
                logger.error(f"Error creating comparison plot: {str(e)}")
        
        # 如果需要替换数据
        if replace:
            pydas_obj.data[sseg][chName] = filtered_data
            logger.info(f'Lowpass for {chName} filter = {cutoffull:3.2f} rad/s in full scale, Lambda = {pydas_obj.__lam__:02d}')
            pydas_obj.updateST(chName=chName)
    except Exception as e:
        logger.error(f"Failed to apply filter to {chName}: {str(e)}")
        return None
            
    # 返回结果（如果需要）
    if returnValue:
        return filtered_data
    
    return None

def apply_highpass_filter(pydas_obj, chName, cutoffull=2, replace=True, returnValue=False, 
                         sseg=0, order=6, plot=False):
    """
    Apply a highpass filter to a channel.
    
    Parameters:
    -----------
    pydas_obj : PyDAS
        PyDAS object containing the data
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
    """
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
            
            # 应用滤波器，减少边缘效应
            y = filtfilt(b, a, data)
            return y
        except Exception as e:
            logger.error(f"Filter error: {str(e)}. Returning original data.")
            return data

    # 检查通道是否存在
    if isinstance(chName, str) and chName not in pydas_obj.chInfo['Name'].values:
        logger.error(f"Channel '{chName}' not found.")
        return None
    
    # 模型尺度下调整截止频率
    if pydas_obj.__scale__ == 'model':
        cutoff = cutoffull / 2 / np.pi * np.sqrt(pydas_obj.__lam__)
    else:
        cutoff = cutoffull / 2 / np.pi

    # 处理通道列表
    if isinstance(chName, list):
        results = []
        for ch in chName:
            if ch in pydas_obj.chInfo['Name'].values:
                result = apply_highpass_filter(pydas_obj, ch, cutoffull, replace, returnValue, sseg, order, plot)
                if returnValue:
                    results.append(result)
            else:
                logger.warning(f"Channel '{ch}' not found, skipping.")
        if returnValue:
            return results
        return None
                
    # 获取数据
    try:
        data = pydas_obj.data[sseg][chName].values
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
        filtered_data = _butter_highpass_filter(data, cutoff, pydas_obj.__fs__, order)
        
        # 如果需要绘图对比
        if plot:
            try:
                # 为了绘图对比，我们需要创建一个临时通道
                temp_channel_name = f"{chName}_filtered"
                
                # 创建一个临时PyDAS对象的副本，用于比较
                temp_pydas = copy.deepcopy(pydas_obj)
                
                # 添加滤波后的临时通道
                unit = temp_pydas.chInfo.loc[temp_pydas.chInfo['Name'] == chName, 'Unit'].values[0]
                temp_pydas.add_channel(
                    name=temp_channel_name,
                    unit=unit,
                    series=filtered_data,
                    fs=pydas_obj.__fs__,
                    sseg=sseg
                )
                
                # Use Plotly for interactive comparison
                from .plot import plot_channel
                plot_channel(
                    pydas_obj=temp_pydas,
                    ch_name=[chName, temp_channel_name],
                    sseg=sseg,
                    title=f"Highpass Filter Comparison - {chName} (cutoff={cutoffull} Hz, order={order})",
                    alpha=[0.5, 0.8],  # 原始数据透明度0.5，滤波后数据保持默认0.8
                )
            except Exception as e:
                logger.error(f"Error creating comparison plot: {str(e)}")
        
        # 如果需要替换数据
        if replace:
            pydas_obj.data[sseg][chName] = filtered_data
            logger.info(f'Highpass for {chName} filter = {cutoffull:3.2f} rad/s in full scale, Lambda = {pydas_obj.__lam__:02d}')
            pydas_obj.updateST(chName=chName)
    except Exception as e:
        logger.error(f"Failed to apply filter to {chName}: {str(e)}")
        return None
            
    # 返回结果（如果需要）
    if returnValue:
        return filtered_data
        
    return None

def remove_mean(pydas_obj, chName, sseg=0):
    """
    Remove the mean value from one or more channels.
    
    Parameters:
    -----------
    pydas_obj : PyDAS
        PyDAS object containing the data
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
            data = pydas_obj.data[sseg][ichName].values
            pydas_obj.data[sseg][ichName] = data - data.mean()
            pydas_obj.updateST(chName=ichName)
        logger.info('remove mean for Channels: ' + ', '.join(chName))
    elif isinstance(chName, str):
        data = pydas_obj.data[sseg][chName].values
        pydas_obj.data[sseg][chName] = data - data.mean()
        pydas_obj.updateST(chName=chName)
        logger.info('remove mean for ' + chName)
    else:
        logger.warning('Unknown type for ChName!')

def add_value(pydas_obj, chName, value2add, sseg=0):
    """
    Add a constant value to one or more channels.
    
    Parameters:
    -----------
    pydas_obj : PyDAS
        PyDAS object containing the data
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
            data = pydas_obj.data[sseg][ichName].values
            pydas_obj.data[sseg][ichName] = data + value2add
            pydas_obj.updateST(chName=ichName)
    elif isinstance(chName, str):
        data = pydas_obj.data[sseg][chName].values
        pydas_obj.data[sseg][chName] = data + value2add
        pydas_obj.updateST(chName=chName)
    else:
        logger.warning('Unknown type for ChName!')  

def multiply_value(pydas_obj, chName, value2mul, sseg=0):
    """
    Multiply one or more channels by a constant value.
    
    Parameters:
    -----------
    pydas_obj : PyDAS
        PyDAS object containing the data
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
            data = pydas_obj.data[sseg][ichName].values
            pydas_obj.data[sseg][ichName] = data * value2mul
            pydas_obj.updateST(chName=ichName)
    elif isinstance(chName, str):
        data = pydas_obj.data[sseg][chName].values
        pydas_obj.data[sseg][chName] = data * value2mul
        pydas_obj.updateST(chName=chName)
    else:
        logger.warning('Unknown type for ChName!')

def move_data(pydas_obj, chName, point_of_move, sseg=0):
    """
    Move data in a channel by a specified number of points.
    
    Parameters:
    -----------
    pydas_obj : PyDAS
        PyDAS object containing the data
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
    if chName in pydas_obj.chInfo['Name'].values:
        data = pydas_obj.data[sseg][chName].values
        n_sample = pydas_obj.segInfo.iloc[sseg]['N sample']
        
        if point_of_move > 0:
            # Move forward (right shift)
            data_new = np.zeros(n_sample)
            data_new[point_of_move:] = data[:n_sample - point_of_move]
        else:
            # Move backward (left shift)
            data_new = np.zeros(n_sample)
            data_new[:n_sample + point_of_move] = data[-point_of_move:]
            
        pydas_obj.data[sseg][chName] = data_new
        pydas_obj.updateST(chName=chName)
        logger.info(f'Moved {chName} by {point_of_move} points')
    else:
        logger.error(f'ERROR! {chName:8s} not found.')

def data_wash(pydas_obj, ChName, method='linear', order=5, threshold=3, sseg=0):
    """
    Clean data by detecting and interpolating outliers.
    
    Parameters:
    -----------
    pydas_obj : PyDAS
        PyDAS object containing the data
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
        if ChName not in pydas_obj.data[sseg].columns:
            logger.error(f"Channel '{ChName}' not found in segment {sseg}")
            raise KeyError(f"Channel '{ChName}' not found")
            
        # 获取数据Series
        data_series = pydas_obj.data[sseg][ChName]
        data_length = len(data_series)
        
        # 为了更高效地处理大数据集，根据数据大小选择不同的处理方法
        if data_length > 1000000:  # 超大数据集
            return _data_wash_large(pydas_obj, ChName, method, order, threshold, sseg)
        
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
                pydas_obj.data[sseg][ChName] = filled_series.values
                
            except Exception as e:
                logger.error(f"Interpolation failed: {str(e)}")
                raise ValueError(f"Interpolation method '{method}' failed: {str(e)}")
        else:
            logger.info(f"No outliers found in channel '{ChName}'")
        
        # 手动更新统计信息，避免列不匹配问题
        series = pydas_obj.data[sseg][ChName].values
        
        # 获取单位
        unit_idx = np.where(pydas_obj.chInfo['Name'].values == ChName)[0][0]
        unit = pydas_obj.chInfo['Unit'].values[unit_idx]
        
        # 手动计算统计量并更新
        pydas_obj.segStatis[sseg].loc[ChName] = [
            np.mean(series), np.std(series), np.amax(series), np.amin(series), unit]
            
        return True
        
    except Exception as e:
        logger.error(f"Data washing failed: {str(e)}")
        return False

def _data_wash_large(pydas_obj, ChName, method='linear', order=5, threshold=3, sseg=0):
    """
    优化的处理大型数据集的数据清洗方法。
    通过分块处理来减少内存占用。
    
    Parameters:
    -----------
    同data_wash方法
    """
    try:
        # 获取数据
        data = pydas_obj.data[sseg][ChName].values
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
                pydas_obj.data[sseg][ChName] = filled_series.values
                
            except Exception as e:
                logger.error(f"Large dataset interpolation failed: {str(e)}")
                raise ValueError(f"Interpolation method '{method}' failed for large dataset: {str(e)}")
        else:
            logger.info(f"No outliers found in channel '{ChName}'")
        
        # 手动更新统计信息，避免列不匹配问题
        cleaned_data = pydas_obj.data[sseg][ChName].values
        
        # 获取单位
        unit_idx = np.where(pydas_obj.chInfo['Name'].values == ChName)[0][0]
        unit = pydas_obj.chInfo['Unit'].values[unit_idx]
        
        # 手动计算统计量并更新
        pydas_obj.segStatis[sseg].loc[ChName] = [
            np.mean(cleaned_data), np.std(cleaned_data), np.amax(cleaned_data), np.amin(cleaned_data), unit]
            
        return True
        
    except Exception as e:
        logger.error(f"Large dataset washing failed: {str(e)}")
        return False

def add_diff1(pydas_obj, name, sseg=0, filter=False, filter_cutoff=2):
    """
    Calculate and add first derivative of a channel.
    
    Parameters:
    -----------
    pydas_obj : PyDAS
        PyDAS object containing the data
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
        if name in pydas_obj.chInfo['Name'].values:
            # Get channel data and unit
            data = pydas_obj.data[sseg][name].values
            unit_idx = np.where(pydas_obj.chInfo['Name'].values == name)[0][0]
            unit = pydas_obj.chInfo['Unit'].values[unit_idx]
            
            # Calculate derivative
            dt = 1.0 / pydas_obj.__fs__
            diff_data = diff1d(data, dt)
            logger.info(f"Calculated first derivative of {name}")
            
            # Apply filter if requested
            if filter:
                diff_data = apply_lowpass_filter(pydas_obj, diff_data, cutoffull=filter_cutoff, 
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
            pydas_obj.add_channel(new_name, new_unit, diff_data, pydas_obj.__fs__, sseg=sseg)
            return True
        else:
            logger.error(f"Channel '{name}' does not exist.")
            return False
    except Exception as e:
        logger.error(f"Failed to add derivative channel: '{name}'")
        logger.error(f"Error in add_diff1: {str(e)}")
        return False

def add_diff2(pydas_obj, name, sseg=0, filter=False, filter_cutoff=2):
    """
    Calculate and add second derivative of a channel.
    
    Parameters:
    -----------
    pydas_obj : PyDAS
        PyDAS object containing the data
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
        if name in pydas_obj.chInfo['Name'].values:
            # Get channel data and unit
            data = pydas_obj.data[sseg][name].values
            unit_idx = np.where(pydas_obj.chInfo['Name'].values == name)[0][0]
            unit = pydas_obj.chInfo['Unit'].values[unit_idx]
            
            # Calculate first derivative
            dt = 1.0 / pydas_obj.__fs__
            diff1_data = diff1d(data, dt)
            logger.info(f"Calculated first derivative of {name}")
            
            # Apply filter if requested
            if filter:
                diff1_data = apply_lowpass_filter(pydas_obj, diff1_data, cutoffull=filter_cutoff, 
                                                     replace=False, returnValue=True)
            
            # Calculate second derivative
            diff2_data = diff1d(diff1_data, dt)
            logger.info(f"Calculated second derivative of {name}")
            
            # Apply filter if requested
            if filter:
                diff2_data = apply_lowpass_filter(pydas_obj, diff2_data, cutoffull=filter_cutoff, 
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
            pydas_obj.add_channel(new_name, new_unit, diff2_data, pydas_obj.__fs__, sseg=sseg)
            logger.info(f"Added second derivative channel {new_name}")
            
            return True
        else:
            logger.error(f"Channel '{name}' does not exist.")
            return False
    except Exception as e:
        logger.error(f"Failed to add derivative channel: '{name}'")
        logger.error(f"Error in add_diff2: {str(e)}")
        return False 