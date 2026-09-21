# -*- coding: utf-8 -*-
"""
PyDAS Analysis Module
=====================
This module contains functions for data analysis within the PyDAS package:
- Spectral analysis functions
- Statistical analysis functions
- Visualization tools for analysis results

Author: Xiaoxian Guo
Date: 2026-02-23
Version: 1.1.0
"""

import os
from typing import Union, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import scipy.stats as stats
import logging

logger = logging.getLogger(__name__)
from .waveModel import TimeSeries
from .plot import (
    _plot_statistics_mpl,
    _plot_statistics_plotly,
    _detect_peaks,
    plot_extreme_analysis,
    plot_spectrum,
)

def spectral_analysis(pydas_obj, channel_name: str, method: str = 'cov', L: int = 1024, 
                      plot: bool = False, title: Optional[str] = None, 
                      save_path: Optional[str] = None, plotbackend: Optional[str] = None, 
                      save_html: Optional[str] = None, fullscale: bool = False, 
                      lam: Optional[float] = None, rho: float = 1.025, g: float = 9.807, 
                      freq_range: Tuple[float, float] = (0, 2)):
    """
    Perform spectral analysis on a single channel.
    
    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data.
    channel_name : str
        Name of the channel to analyze.
    method : str, optional
        Spectral analysis method: 'cov' (covariance) or 'psd' (periodogram), default is 'cov'.
    L : int, optional
        Lag window length for spectral analysis, default is 1024.
    plot : bool, optional
        Whether to generate a plot, default is False.
    title : str, optional
        Title for the plot, default is None.
    save_path : str, optional
        Path to save the plot, default is None.
    plotbackend : str, optional
        The plotting backend to use: 'plotly', 'matplotlib', 'seaborn' or None (auto).
    save_html : str, optional
        Path to save interactive HTML plot, default is None.
    fullscale : bool, optional
        Whether to convert data to full scale before analysis, default is False.
    lam : float, optional
        Scale factor lambda, defaults to object's __lam__ attribute.
    rho : float, optional
        Water density [kg/m³], default is 1.025.
    g : float, optional
        Gravitational acceleration [m/s²], default is 9.807.
    freq_range : tuple, optional
        Frequency range in full scale [rad/s], default is (0, 2).
        
    Returns
    -------
    spec : waveModel.SpecData1D
        Spectral data object.
        
    Notes
    -----
    - Spectral analysis is performed using the waveModel toolkit.
    - Freq_range specifies the valid frequency range in full scale.
    """
    # Check if channel exists
    if channel_name not in pydas_obj.chInfo['Name'].values:
        logger.error(f"Channel '{channel_name}' does not exist")
        return None
        
    # Ensure valid scale factor
    if lam is None:
        if hasattr(pydas_obj, '__lam__'):
            lam = pydas_obj.__lam__
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
            ts = pydas_obj.channel2fullscale(channel_name, lam, rho, g)
            if ts is None:
                logger.error(f"Full scale conversion failed for channel: {channel_name}")
                return None
            
            # Log data diagnostics
            logger.debug(f"TimeSeries data type: {type(ts.data)}, shape: {ts.data.shape if hasattr(ts.data, 'shape') else 'unknown'}")
            logger.debug(f"TimeSeries args type: {type(ts.args)}, shape: {ts.args.shape if hasattr(ts.args, 'shape') else 'unknown'}")
            
            # Ensure floating point for analysis
            if hasattr(ts.data, 'dtype') and not np.issubdtype(ts.data.dtype, np.floating):
                logger.warning(f"TimeSeries data is not floating point, converting from {ts.data.dtype}")
                ts.data = np.array(ts.data, dtype=np.float64)
            
            # Calculate spectrum
            try:
                spec = ts.tospecdata(L=L, method=method)
            except TypeError as te:
                logger.error(f"Type error in tospecdata: {str(te)}")
                # Attempt automatic type correction
                logger.debug("Attempting to fix data type issues...")
                if hasattr(ts, 'data'):
                    ts.data = np.array(ts.data, dtype=np.float64)
                if hasattr(ts, 'args'):
                    ts.args = np.array(ts.args, dtype=np.float64)
                # Retry
                spec = ts.tospecdata(L=L, method=method)
            except Exception as e:
                logger.error(f"Error in tospecdata: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Full scale spectral analysis failed: {str(e)}")
            return None
    else:
        # Local analysis on model scale
        sseg = 0
        data = pydas_obj.data[sseg][channel_name].values.copy().astype(np.float64)
        
        # Uniform time vector
        fs = pydas_obj.__fs__
        t = np.arange(0, len(data)) / fs
        
        try:
            # Instantiate TimeSeries object
            ts = TimeSeries(data, t)
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
            # SpecData1D.S is a property alias of .data. Only slice a real
            # stored S attribute so the mask is not applied twice.
            stored_S = spec.__dict__.get('S')
            if stored_S is not None and len(stored_S) == len(idx):
                spec.__dict__['S'] = stored_S[idx]
    except Exception as e:
        logger.warning(f"Error applying frequency range limitation: {str(e)}")
    
    # If plotting is requested
    if plot:
        if title is None:
            title_prefix = "Full Scale " if fullscale else ""
            title = f"{title_prefix}Spectrum of {channel_name}"
        plot_spectrum(
            spec.args,
            spec.data,
            title=title,
            w_range=w_range,
            plotbackend=plotbackend,
            save_path=save_path,
            save_html=save_html,
            show=True,
        )

    return spec

def statistic_analysis(pydas_obj, ch_name, sseg=0, advanced=False, visualization=False, bins=50, 
                       save_fig=False, save_path=None, plotbackend=None, fullscale=False, lam=None, 
                       rho=1.025, g=9.807):
    """
    Perform time-domain statistical analysis on a channel.
    
    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data.
    ch_name : str 
        Channel name to analyze.
    sseg : int, optional
        Segment index to analyze, default is 0.
    advanced : bool, optional
        Compute higher-order statistics (skewness, kurtosis, etc.), default is False.
    visualization : bool, optional
        Show statistical plots, default is False.
    bins : int, optional
        Number of bins for histograms, default is 50.
    save_fig : bool, optional
        Whether to save figures, default is False.
    save_path : str, optional
        Path to save the generated figures, default is None.
    plotbackend : str, optional
        Plotting backend: 'plotly', 'matplotlib', 'seaborn' or None (auto).
    fullscale : bool, optional
        Whether to convert data to prototype scale, default is False.
    lam : float, optional
        Scale factor lambda, required if fullscale=True.
    rho : float, optional
        Water density [kg/m³], default is 1.025.
    g : float, optional
        Gravitational acceleration [m/s²], default is 9.807.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing computed statistics.
    """
    # Ensure ch_name is a string
    if not isinstance(ch_name, str):
        logger.warning("Function only supports single channel analysis. Using first channel.")
        ch_name = ch_name[0] if isinstance(ch_name, list) and len(ch_name) > 0 else ch_name
        
    # Validate channel and segment
    if ch_name not in pydas_obj.chInfo['Name'].values:
        logger.warning(f"Channel '{ch_name}' does not exist.")
        return None
            
    if not isinstance(sseg, int) or sseg >= pydas_obj.__segN__:
        logger.warning(f"Invalid segment index: {sseg}")
        return None
    
    # Handle fullscale conversion
    if fullscale and lam is not None:
        if not isinstance(lam, (int, float)) or lam <= 0:
            logger.warning(f"Invalid scale factor: {lam}.")
            return None
            
        try:
            # Convert to prototype scale
            logger.info(f"Converting channel '{ch_name}' to full scale with λ={lam}")
            ts = pydas_obj.channel2fullscale(ch_name, lam, rho, g)
            
            if ts is None:
                logger.error(f"Failed to convert channel '{ch_name}' to full scale.")
                return None
                
            data = ts.data
            
            # Update metadata
            ch_idx = pydas_obj.chInfo.index[pydas_obj.chInfo['Name'] == ch_name].tolist()[0]
            unit = pydas_obj.chInfo.loc[ch_idx, 'Unit']
            
            from .utils import get_default_transDict, findtrans
            transDict = get_default_transDict(g)
            trans_temp = findtrans(unit, transDict)
            if trans_temp and trans_temp[0]:
                unit = trans_temp[0]
            
            ch_names = [f"{ch_name} (Full Scale)"]
            
        except Exception as e:
            logger.error(f"Error during full scale conversion: {str(e)}")
            return None
    else:
        # Use model scale data
        ch_names = [ch_name]
        data = pydas_obj.data[sseg][ch_name].values
        
        ch_idx = pydas_obj.chInfo.index[pydas_obj.chInfo['Name'] == ch_name].tolist()[0]
        unit = pydas_obj.chInfo.loc[ch_idx, 'Unit']
        
    # Define columns
    columns = ['Mean', 'Std', 'Min', 'Max', 'Median', 'RMS', 
               'Range', 'Peak-to-Peak', 'Zero-Crossings', 'Unit']
    
    if advanced:
        columns.insert(6, 'Skewness')
        columns.insert(7, 'Kurtosis')
        columns.insert(8, '10% Quantile')
        columns.insert(9, '25% Quantile') 
        columns.insert(10, '75% Quantile')
        columns.insert(11, '90% Quantile')
        columns.insert(12, 'Crest Factor')
        columns.insert(13, 'Form Factor')
        
    stats_df = pd.DataFrame(index=ch_names, columns=columns)
    
    # Perform calculations
    name = ch_names[0]
    
    mean = np.mean(data)
    std = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    median = np.median(data)
    rms = np.sqrt(np.mean(np.square(data)))
    range_val = max_val - min_val
    peak_to_peak = max_val - min_val
    
    # Average zero-crossing rate
    zero_crossings = np.sum(np.diff(np.signbit(data - mean))) / len(data)
    
    # Store results
    stats_df.loc[name, 'Mean'] = mean
    stats_df.loc[name, 'Std'] = std
    stats_df.loc[name, 'Min'] = min_val
    stats_df.loc[name, 'Max'] = max_val
    stats_df.loc[name, 'Median'] = median
    stats_df.loc[name, 'RMS'] = rms
    stats_df.loc[name, 'Range'] = range_val
    stats_df.loc[name, 'Peak-to-Peak'] = peak_to_peak
    stats_df.loc[name, 'Zero-Crossings'] = zero_crossings
    stats_df.loc[name, 'Unit'] = unit
    
    # Compute advanced statistics
    if advanced:
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        quantile_10 = np.percentile(data, 10)
        quantile_25 = np.percentile(data, 25)
        quantile_75 = np.percentile(data, 75)
        quantile_90 = np.percentile(data, 90)
        
        abs_data = np.abs(data)
        crest_factor = np.max(abs_data) / rms if rms > 0 else np.nan
        
        form_factor = rms / np.abs(mean) if np.abs(mean) > 0 else np.nan
        
        stats_df.loc[name, 'Skewness'] = skewness
        stats_df.loc[name, 'Kurtosis'] = kurtosis
        stats_df.loc[name, '10% Quantile'] = quantile_10
        stats_df.loc[name, '25% Quantile'] = quantile_25
        stats_df.loc[name, '75% Quantile'] = quantile_75
        stats_df.loc[name, '90% Quantile'] = quantile_90
        stats_df.loc[name, 'Crest Factor'] = crest_factor
        stats_df.loc[name, 'Form Factor'] = form_factor
    
    # Handle visualization
    if visualization:
        logger.info("Visualizing statistical results...")
        try:
            if plotbackend is None:
                try:
                    import plotly
                    use_plotly = True
                except ImportError:
                    use_plotly = False
            else:
                use_plotly = plotbackend.lower() == 'plotly'
            
            if use_plotly:
                try:
                    _plot_statistics_plotly(pydas_obj, ch_names, sseg, stats_df, bins, save_fig, save_path, 
                                           data=data, title_override=f"Full Scale Statistics: {ch_name}" if fullscale else None)
                except Exception as e:
                    logger.warning(f"Plotly error: {e}. Falling back to Matplotlib.")
                    _plot_statistics_mpl(pydas_obj, ch_names, sseg, stats_df, bins, save_fig, save_path, 
                                        data=data, title_override=f"Full Scale Statistics: {ch_name}" if fullscale else None)
            else:
                _plot_statistics_mpl(pydas_obj, ch_names, sseg, stats_df, bins, save_fig, save_path, 
                                   data=data, title_override=f"Full Scale Statistics: {ch_name}" if fullscale else None)
        except Exception as e:
            logger.error(f"Error in statistical visualization: {e}")
    
    # Log results
    scale_info = "Prototype" if fullscale else "Model"
    logger.info(f"Statistical analysis results ({scale_info} scale):")
    logger.info("\n" + stats_df.to_string(float_format=lambda x: f"% .4E" % x))
    
    return stats_df 

def extreme_analysis(pydas_obj, ch_name, sseg=None, visualization=True,
                   bins=30, peak_prominence=1.0, peak_distance=None,
                   visualization_backend='matplotlib', save_path=None, save_html=None,
                   fullscale=True, lam=50, return_period_multipliers=[1, 5, 10],
                   peak_height=None, threshold=None, width=None, wlen=None, rel_height=0.5):
    """
    Perform extreme value analysis on time series data.
    
    Parameters
    ----------
    pydas_obj : PyDAS
        The PyDAS object containing the data to analyze.
    ch_name : str
        The name of the channel to analyze.
    sseg : tuple, optional
        The start and end indices for a segment of the data to analyze.
    visualization : bool, default=True
        Whether to create visualizations of the analysis.
    bins : int, default=30
        The number of bins to use for the histogram.
    peak_prominence : float, default=1.0
        The prominence of peaks to detect.
    peak_distance : int, optional
        The minimum distance between peaks to detect.
    visualization_backend : str, default='matplotlib'
        The backend to use for visualization ('matplotlib' or 'plotly').
    save_path : str, optional
        The path to save the visualization figure (for matplotlib).
    save_html : str, optional
        The path to save the visualization as an HTML file (for plotly).
    fullscale : bool, default=True
        Whether to use full-scale transformation.
    lam : float, default=50
        The lambda parameter for exceedance probability calculation.
    return_period_multipliers : list, default=[1, 5, 10]
        Multipliers for return periods based on data duration.
    peak_height : float or None, default=None
        Required peak height (same as threshold parameter).
    threshold : float or None, default=None
        Required threshold for peaks.
    width : float or None, default=None
        Required width of peaks in samples.
    wlen : int or None, default=None
        Window length for peak detection.
    rel_height : float, default=0.5
        Relative height for peak width calculation.
    
    Returns
    -------
    dict
        A dictionary containing the analysis results:
        - 'peaks_positive': Positive peaks in the data
        - 'peaks_negative': Negative peaks in the data
        - 'all_peaks': All peaks (absolute values)
        - 'duration_seconds': Duration of the data in seconds
        - 'duration_hours': Duration of the data in hours
        - 'peak_statistics': Statistics of the peaks
        - 'exceedance_table': Table of exceedance probabilities
        - 'extreme_value_model': Parameters of the fitted extreme value model
        - 'return_values': Return values for specified return periods
        - 'return_value_confidence_intervals': Confidence intervals for return values
        - 'visualization': The visualization figure (if visualization=True)
    """
    try:
        from scipy import optimize
        from scipy.signal import find_peaks
        
        # Get data
        if isinstance(sseg, tuple) and len(sseg) == 2 and sseg[0] >= 0 and sseg[1] < len(pydas_obj.data):
            # Valid segment
            if ch_name not in pydas_obj.data[sseg[0]].columns:
                logger.error(f"Channel '{ch_name}' not found in segment {sseg}")
                return None
            
            # Get data
            data = pydas_obj.data[sseg[0]][ch_name].copy()
            
            # Handle full scale conversion if needed
            if fullscale and lam is not None:
                data *= lam
        elif ch_name in pydas_obj.data[0].columns:
            # Use all segments combined
            data = []
            iterable = pydas_obj.data.values() if isinstance(pydas_obj.data, dict) else pydas_obj.data
            for seg in iterable:
                if ch_name in seg.columns:
                    data.append(seg[ch_name])
            
            data = pd.concat(data, ignore_index=True)
            
            # Handle full scale conversion if needed
            if fullscale and lam is not None:
                data *= lam
        else:
            logger.error(f"Channel '{ch_name}' not found in data")
            return None
        
        # Ensure data is a numpy array
        data_array = np.array(data)
        
        # Data duration
        dt = 1.0  # default sample interval [s]
        
        # Prefer dt from the PyDAS object
        if hasattr(pydas_obj, 'dt') and pydas_obj.dt is not None:
            dt = pydas_obj.dt
        # Else estimate dt from a time vector
        elif hasattr(pydas_obj, 'time') and len(pydas_obj.time) > 1:
            dt = (pydas_obj.time[-1] - pydas_obj.time[0]) / (len(pydas_obj.time) - 1)
        
        # Duration in seconds
        data_duration_seconds = len(data_array) * dt
        
        # Convert to hours
        data_duration_hours = data_duration_seconds / 3600
        
        # Peak indices via scipy.signal.find_peaks
        # Positive peaks
        pos_peaks_idx, _ = find_peaks(data_array, height=peak_height, threshold=threshold, 
                                 distance=peak_distance, prominence=peak_prominence, 
                                 width=width, wlen=wlen, rel_height=rel_height)
        
        # Negative peaks
        neg_peaks_idx, _ = find_peaks(-data_array, height=peak_height, threshold=threshold, 
                                 distance=peak_distance, prominence=peak_prominence, 
                                 width=width, wlen=wlen, rel_height=rel_height)
        
        # Peak values from indices
        peaks_positive = data_array[pos_peaks_idx] if len(pos_peaks_idx) > 0 else np.array([])
        peaks_negative = -data_array[neg_peaks_idx] if len(neg_peaks_idx) > 0 else np.array([])
        
        # Absolute values of all peaks
        all_peaks = np.concatenate([np.abs(peaks_positive), np.abs(peaks_negative)])
        
        if len(all_peaks) == 0:
            logger.warning("No peaks detected. Try adjusting peak detection parameters.")
            return {
                'peaks_positive': peaks_positive,
                'peaks_negative': peaks_negative,
                'all_peaks': all_peaks,
                'duration_seconds': data_duration_seconds,
                'duration_hours': data_duration_hours,
                'message': "No peaks detected. Try adjusting peak detection parameters."
            }

        # Peak summary statistics
        peak_stats = {
            'mean': np.mean(all_peaks),
            'median': np.median(all_peaks),
            'std': np.std(all_peaks),
            'min': np.min(all_peaks),
            'max': np.max(all_peaks),
            'count': len(all_peaks)
        }
        
        # Empirical exceedance
        sorted_peaks = np.sort(all_peaks)[::-1]  # Sort in descending order
        n = len(sorted_peaks)
        ranks = np.arange(1, n+1)
        
        # Calculate exceedance probabilities using Weibull formula
        exceedance_prob = ranks / (n + 1)
        
        # Return periods from the record length
        # Hours as the time unit
        # Return periods [h]
        return_periods = [data_duration_hours * multiplier for multiplier in return_period_multipliers]
        
        # Human-readable period labels
        return_period_labels = []
        for period in return_periods:
            if period < 24:  # < 1 day
                return_period_labels.append(f"{period:.1f} hours")
            elif period < 24*30:  # < ~1 month
                return_period_labels.append(f"{period/24:.1f} days")
            elif period < 24*365:  # < 1 year
                return_period_labels.append(f"{period/(24*30):.1f} months")
            else:  # >= 1 year
                return_period_labels.append(f"{period/(24*365.25):.1f} years")

        # Fit GEV and Gumbel
        # GEV first
        try:
            # Fit GEV distribution to peaks
            gev_params = stats.genextreme.fit(all_peaks)
            
            # GEV AIC
            gev_nll = -np.sum(stats.genextreme.logpdf(all_peaks, *gev_params))
            gev_k = len(gev_params)  # parameter count
            gev_aic = 2 * gev_k + 2 * gev_nll
            
            # Shape parameter
            shape = gev_params[0]
            
            # Return levels at the requested periods
            # GEV return level: ξ≠0
            # GEV return level: ξ=0 (Gumbel limit)
            gev_return_values = {}
            gev_confidence_intervals = {}
            
            # Bootstrap confidence intervals
            n_bootstrap = 1000
            bootstrap_return_values = {label: [] for label in return_period_labels}
            
            # Bootstrap samples
            rng = np.random.RandomState(42)  # reproducible bootstrap
            for _ in range(n_bootstrap):
                # Sample peaks with replacement
                bootstrap_sample = rng.choice(all_peaks, size=len(all_peaks), replace=True)
                try:
                    # Fit GEV
                    bootstrap_params = stats.genextreme.fit(bootstrap_sample)
                    bootstrap_shape = bootstrap_params[0]
                    
                    # Return levels for this sample
                    for i, T in enumerate(return_periods):
                        if abs(bootstrap_shape) < 1e-6:  # Shape parameter close to zero
                            return_val = bootstrap_params[1] - bootstrap_params[2] * np.log(-np.log(1 - 1/T))
                        else:
                            return_val = bootstrap_params[1] - (bootstrap_params[2] / bootstrap_shape) * (1 - (-np.log(1 - 1/T)) ** (-bootstrap_shape))
                        bootstrap_return_values[return_period_labels[i]].append(return_val)
                except:
                    # Skip failed fits
                    continue
            
            # Return levels and confidence intervals for this sample
            for i, T in enumerate(return_periods):
                label = return_period_labels[i]
                if abs(shape) < 1e-6:  # Shape parameter close to zero
                    return_val = gev_params[1] - gev_params[2] * np.log(-np.log(1 - 1/T))
                else:
                    return_val = gev_params[1] - (gev_params[2] / shape) * (1 - (-np.log(1 - 1/T)) ** (-shape))
                gev_return_values[label] = return_val
                
                # 95% CI when enough samples
                bootstrap_values = bootstrap_return_values[label]
                if len(bootstrap_values) > 50:  # enough bootstrap samples
                    lower_ci = np.percentile(bootstrap_values, 2.5)
                    upper_ci = np.percentile(bootstrap_values, 97.5)
                    gev_confidence_intervals[label] = (lower_ci, upper_ci)
                else:
                    gev_confidence_intervals[label] = (None, None)
            
            # Store GEV params, not the frozen dist (pickling)
            # Recreate the dist from params when needed
            gev_model = {
                'distribution': 'GEV',
                'shape': gev_params[0],
                'loc': gev_params[1],
                'scale': gev_params[2],
                'aic': gev_aic
            }
        except Exception as e:
            logger.warning(f"Error fitting GEV distribution: {e}")
            gev_model = None
            gev_return_values = {}
            gev_confidence_intervals = {}
            gev_aic = float('inf')
        
        # Gumbel (GEV with ξ=0)
        try:
            # Fit Gumbel distribution to peaks
            gumbel_params = stats.gumbel_r.fit(all_peaks)
            
            # Gumbel AIC
            gumbel_nll = -np.sum(stats.gumbel_r.logpdf(all_peaks, *gumbel_params))
            gumbel_k = len(gumbel_params)  # parameter count
            gumbel_aic = 2 * gumbel_k + 2 * gumbel_nll
            
            # Return levels at the requested periods
            # Gumbel return level
            gumbel_return_values = {}
            gumbel_confidence_intervals = {}
            
            # Bootstrap confidence intervals
            n_bootstrap = 1000
            bootstrap_return_values = {label: [] for label in return_period_labels}
            
            # Bootstrap samples
            rng = np.random.RandomState(42)  # reproducible bootstrap
            for _ in range(n_bootstrap):
                # Sample peaks with replacement
                bootstrap_sample = rng.choice(all_peaks, size=len(all_peaks), replace=True)
                try:
                    # Fit Gumbel
                    bootstrap_params = stats.gumbel_r.fit(bootstrap_sample)
                    
                    # Return levels for this sample
                    for i, T in enumerate(return_periods):
                        return_val = bootstrap_params[0] + bootstrap_params[1] * (-np.log(-np.log(1 - 1/T)))
                        bootstrap_return_values[return_period_labels[i]].append(return_val)
                except:
                    # Skip failed fits
                    continue
            
            # Return levels and confidence intervals for this sample
            for i, T in enumerate(return_periods):
                label = return_period_labels[i]
                return_val = gumbel_params[0] + gumbel_params[1] * (-np.log(-np.log(1 - 1/T)))
                gumbel_return_values[label] = return_val
                
                # 95% CI when enough samples
                bootstrap_values = bootstrap_return_values[label]
                if len(bootstrap_values) > 50:  # enough bootstrap samples
                    lower_ci = np.percentile(bootstrap_values, 2.5)
                    upper_ci = np.percentile(bootstrap_values, 97.5)
                    gumbel_confidence_intervals[label] = (lower_ci, upper_ci)
                else:
                    gumbel_confidence_intervals[label] = (None, None)
            
            # Store Gumbel params, not the frozen dist
            # Recreate the dist from params when needed
            gumbel_model = {
                'distribution': 'Gumbel',
                'loc': gumbel_params[0],
                'scale': gumbel_params[1],
                'aic': gumbel_aic
            }
        except Exception as e:
            logger.warning(f"Error fitting Gumbel distribution: {e}")
            gumbel_model = None
            gumbel_return_values = {}
            gumbel_confidence_intervals = {}
            gumbel_aic = float('inf')
        
        # Pick the model with lower AIC
        if gev_aic < gumbel_aic and gev_model is not None:
            best_model = gev_model
            return_values = gev_return_values
            return_value_confidence_intervals = gev_confidence_intervals
            logger.info("GEV distribution selected as best fit")
        elif gumbel_model is not None:
            best_model = gumbel_model
            return_values = gumbel_return_values
            return_value_confidence_intervals = gumbel_confidence_intervals
            logger.info("Gumbel distribution selected as best fit")
        else:
            best_model = None
            return_values = {}
            return_value_confidence_intervals = {}
            logger.warning("No valid distribution model could be fitted")
        
        # Exceedance table
        exceedance_table = pd.DataFrame({
            'Peak Value': sorted_peaks,
            'Exceedance Probability': exceedance_prob,
            'Return Period (hours)': 1 / exceedance_prob * data_duration_hours / n
        })
        
        # Result dict
        results = {
            'peaks_positive': peaks_positive,
            'peaks_negative': peaks_negative,
            'all_peaks': all_peaks,
            'peak_indices': {'positive': pos_peaks_idx, 'negative': neg_peaks_idx},
            'duration_seconds': data_duration_seconds,
            'duration_hours': data_duration_hours,
            'peak_statistics': peak_stats,
            'exceedance_table': exceedance_table,
            'extreme_value_model': best_model,
            'return_values': return_values,
            'return_value_confidence_intervals': return_value_confidence_intervals,
            'return_periods': {'periods': return_periods, 'labels': return_period_labels}
        }
        
        # Create visualization if requested
        if visualization:
            # Delegate plotting to pydas.plot
            figure = plot_extreme_analysis(
                results=results, 
                visualization_backend=visualization_backend,
                save_path=save_path,
                save_html=save_html,
                visualization=visualization,
                title=None,  # auto title
                ch_name=ch_name,
                pydas_obj=pydas_obj,
                bins=bins,
                fullscale=fullscale,
                return_periods=return_periods,
                return_period_labels=return_period_labels
            )
            
            # Attach figure to the result
            results['figure'] = figure
        
        return results
    
    except Exception as e:
        logger.error(f"Error in extreme_analysis: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None 
