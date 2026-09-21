"""
Process module for PyDAS.
Contains functions for data processing and manipulation.
"""

import logging
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import copy
from .utils import diff1d

logger = logging.getLogger(__name__)

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
        Full-scale cutoff in rad/s, default is 2. In model scale this is
        converted as ``cutoffull / (2*pi) * sqrt(lam)``. To keep the
        existing report/test contract this is **not** Hertz.
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
            # Minimum length for filtfilt is typically 2*order + 1
            min_data_length = 2 * order + 1
            
            # Warn when the series is shorter than the filter needs
            if len(data) < min_data_length:
                logger.warning(f"Data length ({len(data)}) is less than minimum required ({min_data_length}). Filter may not be effective.")
            
            # Nyquist frequency
            nyq = 0.5 * fs
            
            # Normalised cutoff
            normal_cutoff = cutoff / nyq
            
            # Design filter
            b, a = _butter_lowpass(cutoff, fs, order)
            
            # Apply filter
            y = filtfilt(b, a, data)
            return y
        except Exception as e:
            logger.error(f"Filter error: {str(e)}. Returning original data.")
            return data

    # Validate channel name
    if isinstance(chName, str) and chName not in pydas_obj.chInfo['Name'].values:
        logger.error(f"Channel '{chName}' not found.")
        return None
    
    # Convert full-scale rad/s cutoff to the object's current scale
    if pydas_obj.__scale__ == 'model':
        cutoff = cutoffull / 2 / np.pi * np.sqrt(pydas_obj.__lam__)
    else:
        cutoff = cutoffull / 2 / np.pi

    # Recurse when a list of channels is given
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
                
    # Load channel samples
    try:
        data = pydas_obj.data[sseg][chName].values
    except Exception as e:
        logger.error(f"Error accessing data for channel {chName}: {str(e)}")
        return None
        
    # Reject empty series
    if data is None or len(data) <= 0:
        logger.error(f"No data found for channel {chName} in segment {sseg}")
        return None
    
    # Keep a copy for optional comparison plots
    original_data = copy.deepcopy(data)
    
    # Apply filter
    try:
        filtered_data = _butter_lowpass_filter(data, cutoff, pydas_obj.__fs__, order)
        
        # Optional before/after plot
        if plot:
            try:
                # Temporary channel used only for the comparison figure
                temp_channel_name = f"{chName}_filtered"
                
                # Deep-copy so the live object is not mutated for plotting
                temp_pydas = copy.deepcopy(pydas_obj)
                
                # Attach the filtered series as a sibling channel
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
                    alpha=[0.5, 0.8],  # original=0.5, filtered=0.8
                )
            except Exception as e:
                logger.error(f"Error creating comparison plot: {str(e)}")
        
        # Write back when replace=True
        if replace:
            pydas_obj.data[sseg][chName] = filtered_data
            logger.info(f'Lowpass for {chName} filter = {cutoffull:3.2f} rad/s in full scale, Lambda = {pydas_obj.__lam__:.2f}')
            pydas_obj.updateST(chName=chName)
    except Exception as e:
        logger.error(f"Failed to apply filter to {chName}: {str(e)}")
        return None
            
    # Return filtered samples when requested
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
        Full-scale cutoff in rad/s, default is 2. In model scale this is
        converted as ``cutoffull / (2*pi) * sqrt(lam)``. To keep the
        existing report/test contract this is **not** Hertz.
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
            # Minimum length for filtfilt is typically 2*order + 1
            min_data_length = 2 * order + 1
            
            # Warn when the series is shorter than the filter needs
            if len(data) < min_data_length:
                logger.warning(f"Data length ({len(data)}) is less than minimum required ({min_data_length}). Filter may not be effective.")
            
            # Nyquist frequency
            nyq = 0.5 * fs
            
            # Normalised cutoff
            normal_cutoff = cutoff / nyq
            
            # Design filter
            b, a = _butter_highpass(cutoff, fs, order)
            
            # Apply zero-phase filter
            y = filtfilt(b, a, data)
            return y
        except Exception as e:
            logger.error(f"Filter error: {str(e)}. Returning original data.")
            return data

    # Validate channel name
    if isinstance(chName, str) and chName not in pydas_obj.chInfo['Name'].values:
        logger.error(f"Channel '{chName}' not found.")
        return None
    
    # Convert full-scale rad/s cutoff to the object's current scale
    if pydas_obj.__scale__ == 'model':
        cutoff = cutoffull / 2 / np.pi * np.sqrt(pydas_obj.__lam__)
    else:
        cutoff = cutoffull / 2 / np.pi

    # Recurse when a list of channels is given
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
                
    # Load channel samples
    try:
        data = pydas_obj.data[sseg][chName].values
    except Exception as e:
        logger.error(f"Error accessing data for channel {chName}: {str(e)}")
        return None
        
    # Reject empty series
    if data is None or len(data) <= 0:
        logger.error(f"No data found for channel {chName} in segment {sseg}")
        return None
    
    # Keep a copy for optional comparison plots
    original_data = copy.deepcopy(data)
    
    # Apply filter
    try:
        filtered_data = _butter_highpass_filter(data, cutoff, pydas_obj.__fs__, order)
        
        # Optional before/after plot
        if plot:
            try:
                # Temporary channel used only for the comparison figure
                temp_channel_name = f"{chName}_filtered"
                
                # Deep-copy so the live object is not mutated for plotting
                temp_pydas = copy.deepcopy(pydas_obj)
                
                # Attach the filtered series as a sibling channel
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
                    alpha=[0.5, 0.8],  # original=0.5, filtered=0.8
                )
            except Exception as e:
                logger.error(f"Error creating comparison plot: {str(e)}")
        
        # Write back when replace=True
        if replace:
            pydas_obj.data[sseg][chName] = filtered_data
            logger.info(f'Highpass for {chName} filter = {cutoffull:3.2f} rad/s in full scale, Lambda = {pydas_obj.__lam__:.2f}')
            pydas_obj.updateST(chName=chName)
    except Exception as e:
        logger.error(f"Failed to apply filter to {chName}: {str(e)}")
        return None
            
    # Return filtered samples when requested
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
            
        # Load channel samplesSeries
        data_series = pydas_obj.data[sseg][ChName]
        data_length = len(data_series)
        
        # Route huge series to the chunked washer
        if data_length > 1000000:  # huge series
            return _data_wash_large(pydas_obj, ChName, method, order, threshold, sseg)
        
        logger.info(f"Cleaning channel '{ChName}' with {method} interpolation (threshold={threshold}σ)")
        
        # Mean and std via pandas
        arr_mean = data_series.mean()
        arr_std = data_series.std()
        
        # Vectorised outlier mask
        outlier_mask = np.abs(data_series - arr_mean) > threshold * arr_std
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            logger.info(f"Found {outlier_count} outliers in channel '{ChName}'")
            
            # Mask outliers as NaN then interpolate
            cleaned_series = data_series.copy()
            cleaned_series[outlier_mask] = np.nan
            
            # pandas interpolate
            try:
                if method in ['spline', 'polynomial']:
                    # spline/polynomial need an order
                    filled_series = cleaned_series.interpolate(method=method, order=order, limit_direction='both')
                    logger.info(f"Applied {method} interpolation with order {order}")
                else:
                    # linear (and similar) interpolators
                    filled_series = cleaned_series.interpolate(method=method, limit_direction='both')
                    logger.info(f"Applied {method} interpolation")
                
                # Remaining NaNs after interpolate
                remaining_nans = filled_series.isna().sum()
                if remaining_nans > 0:
                    logger.warning(f"{remaining_nans} NaN values could not be interpolated")
                    
                    # Fill leftover edge NaNs
                    filled_series = filled_series.ffill().bfill()
                    
                    # Re-check after ffill/bfill
                    remaining_nans = filled_series.isna().sum()
                    if remaining_nans > 0:
                        logger.error(f"{remaining_nans} NaN values still remain after additional filling")
                    else:
                        logger.info("Remaining NaN values filled with forward/backward fill")
                
                # Write cleaned series back
                pydas_obj.data[sseg][ChName] = filled_series.values
                
            except Exception as e:
                logger.error(f"Interpolation failed: {str(e)}")
                raise ValueError(f"Interpolation method '{method}' failed: {str(e)}")
        else:
            logger.info(f"No outliers found in channel '{ChName}'")
        
        # Update stats in Mean/STD/Max/Min/Unit order
        series = pydas_obj.data[sseg][ChName].values
        
        # Channel unit
        unit_idx = np.where(pydas_obj.chInfo['Name'].values == ChName)[0][0]
        unit = pydas_obj.chInfo['Unit'].values[unit_idx]
        
        # Write mean/std/max/min/unit
        pydas_obj.segStatis[sseg].loc[ChName] = [
            np.mean(series), np.std(series), np.amax(series), np.amin(series), unit]
            
        return True
        
    except Exception as e:
        logger.error(f"Data washing failed: {str(e)}")
        return False

def _data_wash_large(pydas_obj, ChName, method='linear', order=5, threshold=3, sseg=0):
    """
    Chunked outlier cleaning for very long series.

    
    Parameters:
    -----------
    Same parameters as data_wash.
    """
    try:
        # Load channel samples
        data = pydas_obj.data[sseg][ChName].values
        data_length = len(data)
        
        logger.info(f"Using optimized method for large dataset ({data_length} points)")
        
        # Global mean and std for the threshold
        global_mean = np.mean(data)
        global_std = np.std(data)
        
        # Chunk size
        chunk_size = min(100000, data_length // 10)  # at least ~10 chunks
        
        # Working copy
        output_data = np.copy(data)
        total_outliers = 0
        
        # Process chunks
        for start in range(0, data_length, chunk_size):
            end = min(start + chunk_size, data_length)
            chunk = data[start:end]
            
            # Outliers in this chunk
            outlier_mask = np.abs(chunk - global_mean) > threshold * global_std
            outlier_indices = np.where(outlier_mask)[0] + start
            chunk_outlier_count = len(outlier_indices)
            total_outliers += chunk_outlier_count
            
            if chunk_outlier_count > 0:
                # Mark outliers as NaN
                output_data[outlier_indices] = np.nan
        
        if total_outliers > 0:
            logger.info(f"Found {total_outliers} outliers in channel '{ChName}'")
            
            # Interpolate via pandas
            series = pd.Series(output_data)
            
            try:
                if method in ['spline', 'polynomial']:
                    filled_series = series.interpolate(method=method, order=order, limit_direction='both')
                else:
                    filled_series = series.interpolate(method=method, limit_direction='both')
                
                # Fill edge NaNs
                filled_series = filled_series.ffill().bfill()
                
                # Remaining NaNs after interpolate
                remaining_nans = filled_series.isna().sum()
                if remaining_nans > 0:
                    logger.warning(f"{remaining_nans} NaN values could not be filled")
                
                # Write cleaned series back
                pydas_obj.data[sseg][ChName] = filled_series.values
                
            except Exception as e:
                logger.error(f"Large dataset interpolation failed: {str(e)}")
                raise ValueError(f"Interpolation method '{method}' failed for large dataset: {str(e)}")
        else:
            logger.info(f"No outliers found in channel '{ChName}'")
        
        # Update stats in Mean/STD/Max/Min/Unit order
        cleaned_data = pydas_obj.data[sseg][ChName].values
        
        # Channel unit
        unit_idx = np.where(pydas_obj.chInfo['Name'].values == ChName)[0][0]
        unit = pydas_obj.chInfo['Unit'].values[unit_idx]
        
        # Write mean/std/max/min/unit
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

            if 'm' in unit and '/' not in unit:
                new_unit = unit + '/s'
            elif 'deg' in unit and '/' not in unit:
                new_unit = unit + '/s'
            else:
                new_unit = unit + '/s'

            new_name = name + '_d1'
            pydas_obj.add_channel(new_name, new_unit, diff_data, pydas_obj.__fs__, sseg=sseg)
            if filter:
                apply_lowpass_filter(
                    pydas_obj, new_name, cutoffull=filter_cutoff,
                    replace=True, returnValue=False, sseg=sseg,
                )
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
            
            dt = 1.0 / pydas_obj.__fs__
            diff1_data = diff1d(data, dt)
            diff2_data = diff1d(diff1_data, dt)
            logger.info(f"Calculated second derivative of {name}")

            if 'm' in unit and '/' not in unit:
                new_unit = unit + '/s2'
            elif 'deg' in unit and '/' not in unit:
                new_unit = unit + '/s2'
            else:
                new_unit = unit + '/s2'

            new_name = name + '_d2'
            pydas_obj.add_channel(new_name, new_unit, diff2_data, pydas_obj.__fs__, sseg=sseg)
            if filter:
                apply_lowpass_filter(
                    pydas_obj, new_name, cutoffull=filter_cutoff,
                    replace=True, returnValue=False, sseg=sseg,
                )
            logger.info(f"Added second derivative channel {new_name}")
            return True
        else:
            logger.error(f"Channel '{name}' does not exist.")
            return False
    except Exception as e:
        logger.error(f"Failed to add derivative channel: '{name}'")
        logger.error(f"Error in add_diff2: {str(e)}")
        return False 