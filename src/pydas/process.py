"""
Process module for PyDAS.
Contains functions for data processing and manipulation.
"""

import copy
import datetime
import logging

import numpy as np
import pandas as pd
from scipy.signal import butter, correlate, filtfilt

from .core.state import STATS_COLUMNS, froude_scale_factors
from .utils import diff1d, findtrans, get_default_transDict
from .waveModel.objects import TimeSeries

logger = logging.getLogger(__name__)


def _cutoff_hz(pydas_obj, cutoffull):
    """Convert full-scale rad/s cutoff to Hz at the object's current scale."""
    if pydas_obj.__scale__ == "model":
        return cutoffull / 2 / np.pi * np.sqrt(pydas_obj.__lam__)
    return cutoffull / 2 / np.pi


def _apply_butterworth(data, cutoff, fs, order=5, btype="low"):
    """Zero-phase Butterworth filter of type *btype* ('low' or 'high')."""
    try:
        min_data_length = 2 * order + 1
        if len(data) < min_data_length:
            logger.warning(
                "Data length (%s) is less than minimum required (%s). Filter may not be effective.",
                len(data), min_data_length,
            )
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        if normal_cutoff >= 1.0:
            logger.warning(
                "Cutoff frequency (%s Hz) is too high for sampling frequency (%s Hz). "
                "Setting cutoff to 0.99*nyquist.",
                cutoff, fs,
            )
            normal_cutoff = 0.99
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        return filtfilt(b, a, data)
    except Exception as exc:
        logger.error("Filter error: %s. Returning original data.", exc)
        return data


def _apply_filter(
    pydas_obj, chName, cutoffull=2, replace=True, returnValue=False,
    sseg=0, order=6, plot=False, btype="low",
):
    """Shared high/low-pass implementation. Public wrappers keep the old names."""
    label = "Lowpass" if btype == "low" else "Highpass"

    if isinstance(chName, str) and chName not in pydas_obj.chInfo["Name"].values:
        logger.error("Channel '%s' not found.", chName)
        return None

    cutoff = _cutoff_hz(pydas_obj, cutoffull)

    if isinstance(chName, list):
        results = []
        for ch in chName:
            if ch in pydas_obj.chInfo["Name"].values:
                result = _apply_filter(
                    pydas_obj, ch, cutoffull, replace, returnValue, sseg, order, plot, btype,
                )
                if returnValue:
                    results.append(result)
            else:
                logger.warning("Channel '%s' not found, skipping.", ch)
        if returnValue:
            return results
        return None

    try:
        data = pydas_obj.data[sseg][chName].values
    except Exception as exc:
        logger.error("Error accessing data for channel %s: %s", chName, exc)
        return None

    if data is None or len(data) <= 0:
        logger.error("No data found for channel %s in segment %s", chName, sseg)
        return None

    try:
        filtered_data = _apply_butterworth(data, cutoff, pydas_obj.__fs__, order, btype=btype)

        if plot:
            try:
                temp_channel_name = f"{chName}_filtered"
                temp_pydas = copy.deepcopy(pydas_obj)
                unit = temp_pydas.chInfo.loc[temp_pydas.chInfo["Name"] == chName, "Unit"].values[0]
                temp_pydas.add_channel(
                    name=temp_channel_name,
                    unit=unit,
                    series=filtered_data,
                    fs=pydas_obj.__fs__,
                    sseg=sseg,
                )
                from .plot import plot_channel
                plot_channel(
                    pydas_obj=temp_pydas,
                    ch_name=[chName, temp_channel_name],
                    sseg=sseg,
                    title=(
                        f"{label} Filter Comparison - {chName} "
                        f"(cutoff={cutoffull} rad/s full-scale, order={order})"
                    ),
                    alpha=[0.5, 0.8],
                )
            except Exception as exc:
                logger.error("Error creating comparison plot: %s", exc)

        if replace:
            pydas_obj.data[sseg][chName] = filtered_data
            logger.info(
                "%s for %s filter = %3.2f rad/s in full scale, Lambda = %.2f",
                label, chName, cutoffull, pydas_obj.__lam__,
            )
            pydas_obj.updateST(chName=chName)
    except Exception as exc:
        logger.error("Failed to apply filter to %s: %s", chName, exc)
        return None

    if returnValue:
        return filtered_data
    return None


def apply_lowpass_filter(
    pydas_obj, chName, cutoffull=2, replace=True, returnValue=False,
    sseg=0, order=6, plot=False,
):
    """Apply a lowpass filter to a channel.

    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data.
    chName : str or list
        Name of the channel to filter, or list of channel names.
    cutoffull : float, optional
        Full-scale cutoff in rad/s, default is 2. In model scale this is
        converted as ``cutoffull / (2*pi) * sqrt(lam)``. To keep the
        existing report/test contract this is **not** Hertz.
    replace : bool, optional
        Whether to replace original data, default is True.
    returnValue : bool, optional
        Whether to return filtered data, default is False.
    sseg : int, optional
        Segment index, default is 0.
    order : int, optional
        Filter order, default is 6.
    plot : bool, optional
        Whether to plot before/after comparison, default is False.

    Returns
    -------
    numpy.ndarray, optional
        Filtered data if returnValue is True.
    """
    return _apply_filter(
        pydas_obj, chName, cutoffull, replace, returnValue, sseg, order, plot, btype="low",
    )


def apply_highpass_filter(
    pydas_obj, chName, cutoffull=2, replace=True, returnValue=False,
    sseg=0, order=6, plot=False,
):
    """Apply a highpass filter to a channel.

    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data.
    chName : str or list
        Name of the channel to filter, or list of channel names.
    cutoffull : float, optional
        Full-scale cutoff in rad/s, default is 2. In model scale this is
        converted as ``cutoffull / (2*pi) * sqrt(lam)``. To keep the
        existing report/test contract this is **not** Hertz.
    replace : bool, optional
        Whether to replace original data, default is True.
    returnValue : bool, optional
        Whether to return filtered data, default is False.
    sseg : int, optional
        Segment index, default is 0.
    order : int, optional
        Filter order, default is 6.
    plot : bool, optional
        Whether to plot before/after comparison, default is False.

    Returns
    -------
    numpy.ndarray, optional
        Filtered data if returnValue is True.
    """
    return _apply_filter(
        pydas_obj, chName, cutoffull, replace, returnValue, sseg, order, plot, btype="high",
    )

def remove_mean(pydas_obj, chName, sseg=0):
    """
    Remove the mean value from one or more channels.
    
    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data
    chName : str or list
        Name of the channel(s) to process
    sseg : int, optional
        Segment index, default is 0
        
    Raises
    ------
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
    
    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data
    chName : str or list
        Name of the channel(s) to process
    value2add : float
        Value to add to the channel(s)
    sseg : int, optional
        Segment index, default is 0
        
    Raises
    ------
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
    
    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data
    chName : str or list
        Name of the channel(s) to process
    value2mul : float
        Value to multiply the channel(s) by
    sseg : int, optional
        Segment index, default is 0
        
    Raises
    ------
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
    
    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data
    chName : str
        Name of the channel to move
    point_of_move : int
        Number of points to move the data (positive for forward, negative for backward)
    sseg : int, optional
        Segment index, default is 0
        
    Raises
    ------
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
    
    Parameters
    ----------
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
        
    Notes
    -----
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

    
    Parameters
    ----------
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
    
    Parameters
    ----------
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
        
    Notes
    -----
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
    
    Parameters
    ----------
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
        
    Notes
    -----
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


def _correlation_lag(base, reference, n_sample):
    """Return ``(lag, normalized_corr)`` from a demeaned FFT cross-correlation.

    *lag* is ``argmax(corr) - n_sample + 1``, matching the historical
    ``move_ccor`` convention. ``normalized_corr`` is None when the signals
    have zero energy.
    """
    base_rm = np.asarray(base, dtype=np.float64) - np.mean(base)
    ref_rm = np.asarray(reference, dtype=np.float64) - np.mean(reference)
    correlation = correlate(base_rm, ref_rm, method="fft")
    max_corr_idx = int(np.argmax(correlation))
    lag = max_corr_idx - n_sample + 1
    max_corr = correlation[max_corr_idx]
    norm_factor = np.sqrt(np.sum(base_rm ** 2) * np.sum(ref_rm ** 2))
    normalized = float(max_corr / norm_factor) if norm_factor > 0 else None
    return lag, normalized


def move_ccor(pydas_obj, to_move_chName, base_chName, reference_ch, sseg=0):
    """Move channel data using cross-correlation.

    Parameters
    ----------
    pydas_obj : PyDAS
        Object holding the series.
    to_move_chName : str
        Name of channel to move.
    base_chName : str
        Name of base channel.
    reference_ch : str
        Name of reference channel.
    sseg : int, optional
        Segment index to process, default is 0.
    """
    try:
        if to_move_chName not in pydas_obj.data[sseg].columns:
            logger.error("Channel to move '%s' not found in segment %s", to_move_chName, sseg)
            raise KeyError(f"Channel to move '{to_move_chName}' not found")
        if reference_ch not in pydas_obj.data[sseg].columns:
            logger.error("Reference channel '%s' not found in segment %s", reference_ch, sseg)
            raise KeyError(f"Reference channel '{reference_ch}' not found")
        if base_chName not in pydas_obj.data[sseg].columns:
            logger.error("Base channel '%s' not found in segment %s", base_chName, sseg)
            raise KeyError(f"Base channel '{base_chName}' not found")

        try:
            n_sample = pydas_obj.segInfo["N sample"].iloc[sseg]
        except Exception as exc:
            logger.error("Error getting sample count: %s", exc)
            raise ValueError(f"Failed to get sample count: {exc}") from exc

        try:
            lag, normalized = _correlation_lag(
                pydas_obj.data[sseg][base_chName].values,
                pydas_obj.data[sseg][reference_ch].values,
                n_sample,
            )
            if normalized is not None:
                logger.debug(
                    "Maximum correlation between '%s' and '%s': %.4f at lag %s",
                    base_chName, reference_ch, normalized, lag,
                )
            else:
                logger.warning("Could not normalize correlation (division by zero)")
        except Exception as exc:
            logger.error("Error calculating correlation: %s", exc)
            raise ValueError(f"Correlation calculation failed: {exc}") from exc

        try:
            pydas_obj.move_data(to_move_chName, -lag, sseg=sseg)
            logger.info(
                "Moved channel '%s' by %s points based on correlation",
                to_move_chName, -lag,
            )
        except Exception as exc:
            logger.error("Error moving channel '%s': %s", to_move_chName, exc)
            raise ValueError(f"Failed to move channel: {exc}") from exc
    except Exception as exc:
        logger.error("Error in move_ccor: %s", exc)
        raise


def find_move_ccor(pydas_obj, base_chName, reference_ch, sseg=0):
    """Find the number of points to move between channels using cross-correlation.

    Parameters
    ----------
    pydas_obj : PyDAS
        Object holding the series.
    base_chName : str
        Name of base channel.
    reference_ch : str
        Name of reference channel.
    sseg : int, optional
        Segment index to process, default is 0.

    Returns
    -------
    int
        Number of points to pass to :func:`move_data`.
    """
    try:
        if reference_ch not in pydas_obj.data[sseg].columns:
            logger.error("Reference channel '%s' not found in segment %s", reference_ch, sseg)
            raise KeyError(f"Reference channel '{reference_ch}' not found")
        if base_chName not in pydas_obj.data[sseg].columns:
            logger.error("Base channel '%s' not found in segment %s", base_chName, sseg)
            raise KeyError(f"Base channel '{base_chName}' not found")

        try:
            n_sample = pydas_obj.segInfo["N sample"].iloc[sseg]
        except Exception as exc:
            logger.error("Error getting sample count: %s", exc)
            raise ValueError(f"Failed to get sample count: {exc}") from exc

        try:
            lag, normalized = _correlation_lag(
                pydas_obj.data[sseg][base_chName].values,
                pydas_obj.data[sseg][reference_ch].values,
                n_sample,
            )
            lag = -lag
            if normalized is not None:
                logger.info(
                    "Maximum correlation between '%s' and '%s': %.4f at lag %s",
                    base_chName, reference_ch, normalized, lag,
                )
            else:
                logger.warning("Could not normalize correlation (division by zero)")
            return lag
        except Exception as exc:
            logger.error("Error calculating correlation: %s", exc)
            raise ValueError(f"Correlation calculation failed: {exc}") from exc
    except Exception as exc:
        logger.error("Error in find_move_ccor: %s", exc)
        raise


def fix_unit(pydas_obj, chName, newunit, pInfo=False):
    """Fix channel unit.

    Parameters
    ----------
    pydas_obj : PyDAS
        Object holding channel metadata.
    chName : str
        Channel name.
    newunit : str
        New unit to set.
    pInfo : bool, optional
        Whether to print information, default is False.
    """
    if chName not in pydas_obj.chInfo["Name"].values:
        logger.warning("Channel '%s' does not exist.", chName)
        return None
    idx = pydas_obj.chInfo.index[pydas_obj.chInfo["Name"] == chName].tolist()[0]
    pydas_obj.chInfo.loc[idx, "Unit"] = newunit
    logger.info("Channel '%s' unit updated to: %s", chName, newunit)
    if pInfo:
        logger.info("-" * 50)
        logger.info("\n%s", pydas_obj.chInfo.to_string(justify="center"))
        logger.info("-" * 50)
    return None


def to_fullscale(pydas_obj, rho=1.025, g=9.807, pInfo=False):
    """Convert model-scale data to prototype scale.

    Parameters
    ----------
    pydas_obj : PyDAS
        Object to convert in place.
    rho : float, optional
        Water density in kg/m³, default is 1.025.
    g : float, optional
        Gravitational acceleration in m/s², default is 9.807.
    pInfo : bool, optional
        Whether to print information, default is False.

    Notes
    -----
    Uses ``pydas_obj.__lam__`` as the length scale factor.
    """
    if pydas_obj.__scale__ == "prototype":
        logger.warning("The data is already upscaled.")
        return None

    logger.info("Please make sure the channel units are all checked!")
    if pInfo:
        logger.info(pydas_obj.chInfo.to_string(justify="center", columns=["Name", "Unit"]))
    pydas_obj.rho = rho
    pydas_obj.__scale__ = "prototype"

    trans_dict = get_default_transDict(g)
    findtrans("none", trans_dict, clear_cache=True)

    trans_unit = []
    trans_coeff_unit = np.zeros(pydas_obj.__chN__)
    trans_coeff_rho = np.zeros(pydas_obj.__chN__)
    trans_coeff_lam = np.zeros(pydas_obj.__chN__)
    coeffs = np.ones(pydas_obj.__chN__)

    for idx, unit in enumerate(pydas_obj.chInfo["Unit"]):
        if pd.isna(unit):
            trans_unit.append("")
            trans_coeff_unit[idx] = 1.0
            trans_coeff_rho[idx] = 0.0
            trans_coeff_lam[idx] = 0.0
            coeffs[idx] = 1.0
            continue
        new_unit, coeff, c_unit, c_rho, c_lam = froude_scale_factors(
            unit, pydas_obj.__lam__, rho, g
        )
        trans_unit.append(new_unit)
        trans_coeff_unit[idx] = c_unit
        trans_coeff_rho[idx] = c_rho
        trans_coeff_lam[idx] = c_lam
        coeffs[idx] = coeff

    pydas_obj.chInfo["Unit"] = trans_unit
    pydas_obj.chInfo["CoeffUnit"] = trans_coeff_unit
    pydas_obj.chInfo["CoeffRho"] = trans_coeff_rho
    pydas_obj.chInfo["CoeffLam"] = trans_coeff_lam

    pydas_obj.__fs__ = pydas_obj.__fs__ / np.sqrt(pydas_obj.__lam__)
    logger.info("lambda = %.2f", pydas_obj.__lam__)
    if pInfo:
        logger.info(pydas_obj.chInfo.to_string(justify="center"))

    for idx1 in range(pydas_obj.__segN__):
        for idx2, name in enumerate(pydas_obj.chInfo["Name"]):
            pydas_obj.data[idx1][name] *= coeffs[idx2]

    pydas_obj.updateST(sseg=0)
    return None


def channel2fullscale(pydas_obj, channel_name, lam, rho=1.025, g=9.807):
    """Convert a single channel from model scale to prototype scale.

    Parameters
    ----------
    pydas_obj : PyDAS
        Object holding the series (not mutated).
    channel_name : str
        Name of the channel to convert.
    lam : float
        Scale factor.
    rho : float, optional
        Water density in kg/m³, default is 1.025.
    g : float, optional
        Gravitational acceleration in m/s², default is 9.807.

    Returns
    -------
    ts : waveModel.TimeSeries
        TimeSeries of the converted first-segment samples.
    """
    if channel_name not in pydas_obj.chInfo["Name"].values:
        logger.error("Channel %s not found", channel_name)
        return None

    ch_idx = pydas_obj.chInfo[pydas_obj.chInfo["Name"] == channel_name].index[0]
    unit = pydas_obj.chInfo.loc[ch_idx, "Unit"]
    new_unit, coeff, _, _, _ = froude_scale_factors(unit, lam, rho, g)
    logger.debug("Conversion result for unit %s: coeff=%s new_unit=%s", unit, coeff, new_unit)

    fs_scaled = pydas_obj.__fs__ / np.sqrt(lam)
    if pydas_obj.__segN__ > 1:
        logger.info("Multiple segments found. Only converting first segment.")

    try:
        data_raw = pydas_obj.data[0][channel_name]
        data = data_raw.values if hasattr(data_raw, "values") else np.array(data_raw)
        if not np.issubdtype(data.dtype, np.floating):
            logger.debug("Converting data from %s to float64", data.dtype)
            data = data.astype(np.float64)
        else:
            data = data.copy()
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logger.warning("Data contains NaN or Inf values")
        data_scaled = data * coeff
    except Exception as exc:
        logger.error("Error processing data in channel2fullscale: %s", exc)
        return None

    t = np.arange(0, len(data)) / fs_scaled
    try:
        return TimeSeries(data_scaled, t)
    except Exception as exc:
        logger.error("Error creating TimeSeries object: %s", exc)
        return None


def _stats_from_agg(data_frame, ch_info):
    """Build a Mean/STD/Max/Min/Unit table via ``DataFrame.agg``."""
    stats = data_frame.agg(["mean", "std", "max", "min"]).T
    stats.columns = ["Mean", "STD", "Max", "Min"]
    stats["Unit"] = ch_info.set_index("Name")["Unit"]
    return stats[list(STATS_COLUMNS)]


def updateST(pydas_obj, chName="all", sseg=0, engine="pandas"):
    """Update statistical information for channels.

    Parameters
    ----------
    pydas_obj : PyDAS
        Object whose ``segStatis`` will be refreshed.
    chName : str, optional
        Channel name to update, ``'all'`` for all channels.
    sseg : int, optional
        Segment index to process, default is 0.
    engine : {'pandas', 'dask'}, optional
        Statistics backend. Default ``'pandas'`` uses ``DataFrame.agg``.
        ``'dask'`` is optional for very large tables; it falls back to
        pandas if Dask is not installed.
    """
    if chName == "all":
        try:
            data_frame = pydas_obj.data[sseg]
            if engine == "dask":
                try:
                    import dask.dataframe as dd

                    dask_df = dd.from_pandas(
                        data_frame, npartitions=min(32, max(data_frame.shape[1], 1))
                    )
                    stats = pd.DataFrame({
                        "Mean": dask_df.mean().compute(),
                        "STD": dask_df.std().compute(),
                        "Max": dask_df.max().compute(),
                        "Min": dask_df.min().compute(),
                    })
                    stats["Unit"] = pydas_obj.chInfo.set_index("Name")["Unit"]
                    pydas_obj.segStatis[sseg] = stats[list(STATS_COLUMNS)]
                    return None
                except ImportError:
                    logger.info("Dask not available, using pandas aggregation")
            pydas_obj.segStatis[sseg] = _stats_from_agg(data_frame, pydas_obj.chInfo)
        except Exception as exc:
            logger.error("Statistics calculation error: %s", exc)
            data_frame = pydas_obj.data[sseg]
            pydas_obj.segStatis[sseg] = pd.DataFrame({
                "Mean": data_frame.mean(),
                "STD": data_frame.std(),
                "Max": data_frame.max(),
                "Min": data_frame.min(),
                "Unit": pydas_obj.chInfo.set_index("Name")["Unit"],
            })
        return None

    if chName in pydas_obj.chInfo["Name"].values:
        series = pydas_obj.data[sseg][chName]
        stats = series.agg(["mean", "std", "max", "min"])
        unit = pydas_obj.chInfo.loc[pydas_obj.chInfo["Name"].values == chName, "Unit"].values[0]
        pydas_obj.segStatis[sseg].loc[chName] = [
            stats["mean"], stats["std"], stats["max"], stats["min"], unit,
        ]
    else:
        logger.error("ERROR! %-8s not found.", chName)
    return None


def cut_series(pydas_obj, start, stop, sseg=0):
    """Cut a time series to a specified time range in seconds.

    Parameters
    ----------
    pydas_obj : PyDAS
        Object to cut in place.
    start : float
        Start time in seconds (inclusive).
    stop : float
        End time in seconds (exclusive of the sample at ``stop``).
    sseg : int, optional
        Segment index, default is 0.
    """
    def move_timestr(timestr, seconds_float):
        seconds = int(seconds_float)
        milliseconds = int((seconds_float - seconds) * 1000)
        start_time = datetime.datetime.strptime(timestr, "%H:%M:%S.%f")
        start_time_new = (
            start_time + datetime.timedelta(seconds=seconds, milliseconds=milliseconds)
        ).strftime("%H:%M:%S.%f")
        return start_time_new[:-5]

    if sseg < 0 or sseg >= pydas_obj.__segN__:
        raise ValueError(f"Invalid segment index: {sseg}")

    start_idx = int(start * pydas_obj.__fs__)
    stop_idx = int(stop * pydas_obj.__fs__)
    n_rows = len(pydas_obj.data[sseg])
    start_idx = max(0, min(start_idx, n_rows))
    stop_idx = max(start_idx, min(stop_idx, n_rows))
    if stop_idx <= start_idx:
        raise ValueError("cut_series range is empty after applying start/stop.")

    pydas_obj.data[sseg] = pydas_obj.data[sseg].iloc[start_idx:stop_idx].reset_index(drop=True)
    samp_num = pydas_obj.data[sseg].shape[0]
    seg_label = pydas_obj.segInfo.index[sseg]
    orig_start = pydas_obj.segInfo["Start"].iloc[sseg]

    pydas_obj.segInfo.loc[seg_label, "Start"] = move_timestr(orig_start, start)
    pydas_obj.segInfo.loc[seg_label, "Stop"] = move_timestr(orig_start, stop)
    pydas_obj.segInfo.loc[seg_label, "Duration"] = "{0:8.1f}s".format((samp_num - 1) / pydas_obj.__fs__)
    pydas_obj.segInfo.loc[seg_label, "N sample"] = samp_num
    pydas_obj.updateST(sseg=sseg)
    logger.info("Cut time series from {0:5.2f}s to {1:5.2f}s".format(start, stop))
    return None
