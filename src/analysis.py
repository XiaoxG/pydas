# -*- coding: utf-8 -*-
"""
PyDAS Analysis Module

This module contains functions for data analysis within the PyDAS package:
- Spectral analysis functions
- Statistical analysis functions
- Visualization tools for analysis results

These functions operate on PyDAS objects to provide specialized analysis capabilities.

Functions are designed to be imported into the main PyDAS class for seamless integration.

Author: Xiaoxian Guo
Date: 2025-04-12
Version: 1.0.3
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from logger import logger
from waveModel.timeseries import TimeSeries
from plot import _plot_statistics_mpl, _plot_statistics_plotly, _detect_peaks, plot_extreme_analysis

def spectral_analysis(pydas_obj, channel_name, method='cov', L=1024, plot=False, title=None, 
                      save_path=None, plotbackend=None, save_html=None,
                      fullscale=False, lam=None, rho=1.025, g=9.807, freq_range=(0, 2)):
    """
    Perform spectral analysis on a single channel and return a spectral data object
    
    Parameters:
    -----------
    pydas_obj : PyDAS
        PyDAS object containing the data
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
    plotbackend : str, optional
        The plotting backend to use: 'plotly', 'matplotlib', 'seaborn' or None (auto), default is None
    save_html : str, optional
        Path to save interactive HTML plot, default is None
    fullscale : bool, optional
        Whether to convert data to full scale before analysis, default is False
    lam : float, optional
        Scale factor, used only when fullscale=True, default uses object's __lam__ attribute
    rho : float, optional
        Water density (kg/m3), default is 1.025
    g : float, optional
        Gravitational acceleration (m/s2), default is 9.807
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
            
            # 检查TimeSeries对象数据，确认其类型
            logger.debug(f"TimeSeries data type: {type(ts.data)}, shape: {ts.data.shape if hasattr(ts.data, 'shape') else 'unknown'}")
            logger.debug(f"TimeSeries args type: {type(ts.args)}, shape: {ts.args.shape if hasattr(ts.args, 'shape') else 'unknown'}")
            
            # 检查数据是否为浮点数
            if hasattr(ts.data, 'dtype') and not np.issubdtype(ts.data.dtype, np.floating):
                logger.warning(f"TimeSeries data is not floating point, converting from {ts.data.dtype}")
                ts.data = np.array(ts.data, dtype=np.float64)
            
            # Calculate spectrum
            try:
                spec = ts.tospecdata(L=L, method=method)
            except TypeError as te:
                logger.error(f"Type error in tospecdata: {str(te)}")
                # 尝试修复数据类型问题
                logger.debug("Attempting to fix data type issues...")
                if hasattr(ts, 'data'):
                    ts.data = np.array(ts.data, dtype=np.float64)
                if hasattr(ts, 'args'):
                    ts.args = np.array(ts.args, dtype=np.float64)
                # 再次尝试
                spec = ts.tospecdata(L=L, method=method)
            except Exception as e:
                logger.error(f"Error in tospecdata: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Full scale spectral analysis failed: {str(e)}")
            return None
    else:
        # Default using the first data segment
        sseg = 0
        
        # Get channel data
        data = pydas_obj.data[sseg][channel_name].values.copy().astype(np.float64)
        
        # Create time vector (assuming equal sampling intervals)
        fs = pydas_obj.__fs__
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
        
        if plotbackend is None:
            # Auto-detect: use Plotly if available, else Matplotlib
            try:
                import plotly
                use_plotly = True
            except ImportError:
                use_plotly = False
        else:
            # Use specified backend
            use_plotly = plotbackend.lower() == 'plotly'
        
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
                    yaxis=dict(range=[0, max(S)*1.05]),
                    legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99)
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
            plt.show()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')   
            else:
                plt.close()
    
    return spec

def statistic_analysis(pydas_obj, ch_name, sseg=0, advanced=False, visualization=False, bins=50, 
                       save_fig=False, save_path=None, plotbackend=None, fullscale=False, lam=None, 
                       rho=1.025, g=9.807):
    """
    对通道进行时域统计分析
    
    Parameters:
    -----------
    pydas_obj : PyDAS
        PyDAS object containing the data
    ch_name : str 
        要分析的通道名称
    sseg : int, optional
        要分析的数据段索引，默认为0
    advanced : bool, optional
        是否计算高级统计量（偏度、峰度、分位数等），默认为False
    visualization : bool, optional
        是否显示统计量可视化，默认为False
    bins : int, optional
        直方图的箱数，默认为50
    save_fig : bool, optional
        是否保存图形，默认为False
    save_path : str, optional
        图形保存路径，默认为None（当前目录）
    plotbackend : str, optional
        The plotting backend to use: 'plotly', 'matplotlib', 'seaborn' or None (auto), default is None
    fullscale : bool, optional
        是否转换为原型尺度，默认为False
    lam : float, optional
        尺度系数，仅在fullscale=True时使用，默认为None
    rho : float, optional
        水密度(kg/m³)，仅在fullscale=True时使用，默认为1.025
    g : float, optional
        重力加速度(m/s²)，仅在fullscale=True时使用，默认为9.807
        
    Returns:
    --------
    pandas.DataFrame
        包含各种统计量的DataFrame
        
    Notes:
    ------
    - 基本统计量：均值、标准差、最大值、最小值、中位数、均方根(RMS)
    - 高级统计量：偏度、峰度、分位数(10%,25%,75%,90%)、峰值因子、波形因子、过零率
    - 可视化包括：概率密度函数(PDF)、累积分布函数(CDF)和Q-Q图
    - 当fullscale=True时，使用channel2fullscale函数转换通道数据为原型尺度
    """
    # 确保ch_name是单个字符串（单通道）
    if not isinstance(ch_name, str):
        logger.warning("Function only supports single channel analysis. Using first channel from the list.")
        ch_name = ch_name[0] if isinstance(ch_name, list) and len(ch_name) > 0 else ch_name
        
    # 验证通道名和段索引
    if ch_name not in pydas_obj.chInfo['Name'].values:
        logger.warning(f"Channel '{ch_name}' does not exist.")
        return None
            
    if not isinstance(sseg, int) or sseg >= pydas_obj.__segN__:
        logger.warning(f"Invalid segment index: {sseg}")
        return None
    
    # 处理fullscale转换
    if fullscale and lam is not None:
        # 验证lam值是否有效
        if not isinstance(lam, (int, float)) or lam <= 0:
            logger.warning(f"Invalid scale factor: {lam}. Must be a positive number.")
            return None
            
        try:
            # 使用channel2fullscale转换为原型尺度
            logger.info(f"Converting channel '{ch_name}' to full scale with λ={lam}")
            ts = pydas_obj.channel2fullscale(ch_name, lam, rho, g)
            
            if ts is None:
                logger.error(f"Failed to convert channel '{ch_name}' to full scale.")
                return None
                
            # 获取转换后的数据
            data = ts.data
            
            # 获取通道单位（从转换后的TimeSeries获取或保持原始）
            ch_idx = pydas_obj.chInfo.index[pydas_obj.chInfo['Name'] == ch_name].tolist()[0]
            unit = pydas_obj.chInfo.loc[ch_idx, 'Unit']
            
            # 获取单位转换字典以确定转换后的单位
            from utils import get_default_transDict, findtrans
            transDict = get_default_transDict(g)
            trans_temp = findtrans(unit, transDict)
            if trans_temp and trans_temp[0]:
                unit = trans_temp[0]
            
            # 创建结果DataFrame的索引
            ch_names = [f"{ch_name} (Full Scale)"]
            
        except Exception as e:
            logger.error(f"Error during full scale conversion: {str(e)}")
            return None
    else:
        # 使用原始模型尺度数据
        ch_names = [ch_name]
        # 获取通道数据
        data = pydas_obj.data[sseg][ch_name].values
        
        # 获取通道单位
        ch_idx = pydas_obj.chInfo.index[pydas_obj.chInfo['Name'] == ch_name].tolist()[0]
        unit = pydas_obj.chInfo.loc[ch_idx, 'Unit']
        
    # 创建结果DataFrame
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
    
    # 计算通道的统计量
    name = ch_names[0]
    
    # 基本统计量
    mean = np.mean(data)
    std = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    median = np.median(data)
    rms = np.sqrt(np.mean(np.square(data)))
    range_val = max_val - min_val
    peak_to_peak = max_val - min_val  # 同range，保留为更专业的术语
    
    # 过零率计算 (均值过零率)
    zero_crossings = np.sum(np.diff(np.signbit(data - mean))) / len(data)
    
    # 填充基本统计量
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
    
    # 高级统计量
    if advanced:
        # 计算偏度和峰度
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        # 计算分位数
        quantile_10 = np.percentile(data, 10)
        quantile_25 = np.percentile(data, 25)
        quantile_75 = np.percentile(data, 75)
        quantile_90 = np.percentile(data, 90)
        
        # 计算峰值因子 (Crest Factor) = |x_peak| / x_rms
        abs_data = np.abs(data)
        crest_factor = np.max(abs_data) / rms if rms > 0 else np.nan
        
        # 计算波形因子 (Form Factor) = x_rms / |x_mean|
        form_factor = rms / np.abs(mean) if np.abs(mean) > 0 else np.nan
        
        # 填充高级统计量
        stats_df.loc[name, 'Skewness'] = skewness
        stats_df.loc[name, 'Kurtosis'] = kurtosis
        stats_df.loc[name, '10% Quantile'] = quantile_10
        stats_df.loc[name, '25% Quantile'] = quantile_25
        stats_df.loc[name, '75% Quantile'] = quantile_75
        stats_df.loc[name, '90% Quantile'] = quantile_90
        stats_df.loc[name, 'Crest Factor'] = crest_factor
        stats_df.loc[name, 'Form Factor'] = form_factor
    
    # 可视化
    if visualization:
        logger.info("Visualizing statistical results...")
        try:
            # Determine which backend to use
            if plotbackend is None:
                # Auto-detect: use Plotly if available, else Matplotlib
                try:
                    import plotly
                    use_plotly = True
                except ImportError:
                    use_plotly = False
            else:
                # Use specified backend
                use_plotly = plotbackend.lower() == 'plotly'
            
            # Call appropriate plotting function
            if use_plotly:
                try:
                    _plot_statistics_plotly(pydas_obj, ch_names, sseg, stats_df, bins, save_fig, save_path, 
                                           data=data, title_override=f"Full Scale Statistical Analysis for {ch_name}" if fullscale else None)
                except Exception as e:
                    logger.warning(f"Error using Plotly for visualization: {e}. Falling back to Matplotlib.")
                    _plot_statistics_mpl(pydas_obj, ch_names, sseg, stats_df, bins, save_fig, save_path, 
                                        data=data, title_override=f"Full Scale Statistical Analysis for {ch_name}" if fullscale else None)
            else:
                _plot_statistics_mpl(pydas_obj, ch_names, sseg, stats_df, bins, save_fig, save_path, 
                                   data=data, title_override=f"Full Scale Statistical Analysis for {ch_name}" if fullscale else None)
        except Exception as e:
            logger.error(f"Error in statistical visualization: {e}")
    
    # 输出统计结果
    scale_info = "原型尺度" if fullscale else "模型尺度"
    logger.info(f"Statistical analysis results ({scale_info}):")
    logger.info("\n" + stats_df.to_string(float_format=lambda x: f"% .4E" % x))
    
    return None 

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
        # Import necessary libraries
        import numpy as np
        import pandas as pd
        import scipy.stats as stats
        from scipy import optimize
        from scipy.signal import find_peaks
        import logging

        logger = logging.getLogger(__name__)
        
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
            for seg in pydas_obj.data:
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
        
        # 计算数据持续时间
        dt = 1.0  # 默认采样间隔为1秒
        
        # 尝试从pydas对象获取dt
        if hasattr(pydas_obj, 'dt') and pydas_obj.dt is not None:
            dt = pydas_obj.dt
        # 如果有时间数组，尝试从中估计dt
        elif hasattr(pydas_obj, 'time') and len(pydas_obj.time) > 1:
            dt = (pydas_obj.time[-1] - pydas_obj.time[0]) / (len(pydas_obj.time) - 1)
        
        # 计算数据持续时间（秒）
        data_duration_seconds = len(data_array) * dt
        
        # 转换为小时
        data_duration_hours = data_duration_seconds / 3600
        
        # 直接使用scipy.signal.find_peaks进行峰值检测以获得精确的索引
        # 检测正峰值
        pos_peaks_idx, _ = find_peaks(data_array, height=peak_height, threshold=threshold, 
                                 distance=peak_distance, prominence=peak_prominence, 
                                 width=width, wlen=wlen, rel_height=rel_height)
        
        # 检测负峰值 
        neg_peaks_idx, _ = find_peaks(-data_array, height=peak_height, threshold=threshold, 
                                 distance=peak_distance, prominence=peak_prominence, 
                                 width=width, wlen=wlen, rel_height=rel_height)
        
        # 根据索引获取峰值
        peaks_positive = data_array[pos_peaks_idx] if len(pos_peaks_idx) > 0 else np.array([])
        peaks_negative = -data_array[neg_peaks_idx] if len(neg_peaks_idx) > 0 else np.array([])
        
        # 合并所有峰值的绝对值
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

        # 计算峰值的基本统计信息
        peak_stats = {
            'mean': np.mean(all_peaks),
            'median': np.median(all_peaks),
            'std': np.std(all_peaks),
            'min': np.min(all_peaks),
            'max': np.max(all_peaks),
            'count': len(all_peaks)
        }
        
        # 计算超越概率和经验分布
        sorted_peaks = np.sort(all_peaks)[::-1]  # Sort in descending order
        n = len(sorted_peaks)
        ranks = np.arange(1, n+1)
        
        # Calculate exceedance probabilities using Weibull formula
        exceedance_prob = ranks / (n + 1)
        
        # 设置回归周期基于数据持续时间
        # 在这里，我们使用小时作为时间单位
        # 计算回归周期（小时）
        return_periods = [data_duration_hours * multiplier for multiplier in return_period_multipliers]
        
        # 创建适当的标签
        return_period_labels = []
        for period in return_periods:
            if period < 24:  # 小于一天
                return_period_labels.append(f"{period:.1f} hours")
            elif period < 24*30:  # 小于一个月（近似）
                return_period_labels.append(f"{period/24:.1f} days")
            elif period < 24*365:  # 小于一年
                return_period_labels.append(f"{period/(24*30):.1f} months")
            else:  # 一年或以上
                return_period_labels.append(f"{period/(24*365.25):.1f} years")

        # 拟合极值分布（GEV 和 Gumbel）
        # 首先尝试GEV分布
        try:
            # Fit GEV distribution to peaks
            gev_params = stats.genextreme.fit(all_peaks)
            
            # 为GEV计算AIC（Akaike信息准则）
            gev_nll = -np.sum(stats.genextreme.logpdf(all_peaks, *gev_params))
            gev_k = len(gev_params)  # 参数数量
            gev_aic = 2 * gev_k + 2 * gev_nll
            
            # 检查形状参数（shape parameter）
            shape = gev_params[0]
            
            # 计算给定回归周期的回归值
            # 对于GEV分布，回归值R(T) = μ - σ/ξ * [1 - (-ln(1-1/T))^(-ξ)] 当 ξ≠0
            # 当 ξ=0 时，R(T) = μ - σ * ln(-ln(1-1/T))
            gev_return_values = {}
            gev_confidence_intervals = {}
            
            # 使用bootstrap方法计算置信区间
            n_bootstrap = 1000
            bootstrap_return_values = {label: [] for label in return_period_labels}
            
            # 创建bootstrap样本
            rng = np.random.RandomState(42)  # 固定随机种子以获得可重复结果
            for _ in range(n_bootstrap):
                # 从峰值中有放回抽样
                bootstrap_sample = rng.choice(all_peaks, size=len(all_peaks), replace=True)
                try:
                    # 拟合GEV分布
                    bootstrap_params = stats.genextreme.fit(bootstrap_sample)
                    bootstrap_shape = bootstrap_params[0]
                    
                    # 计算各回归周期的回归值
                    for i, T in enumerate(return_periods):
                        if abs(bootstrap_shape) < 1e-6:  # Shape parameter close to zero
                            return_val = bootstrap_params[1] - bootstrap_params[2] * np.log(-np.log(1 - 1/T))
                        else:
                            return_val = bootstrap_params[1] - (bootstrap_params[2] / bootstrap_shape) * (1 - (-np.log(1 - 1/T)) ** (-bootstrap_shape))
                        bootstrap_return_values[return_period_labels[i]].append(return_val)
                except:
                    # 如果拟合失败，忽略这个bootstrap样本
                    continue
            
            # 计算各回归周期的回归值和置信区间
            for i, T in enumerate(return_periods):
                label = return_period_labels[i]
                if abs(shape) < 1e-6:  # Shape parameter close to zero
                    return_val = gev_params[1] - gev_params[2] * np.log(-np.log(1 - 1/T))
                else:
                    return_val = gev_params[1] - (gev_params[2] / shape) * (1 - (-np.log(1 - 1/T)) ** (-shape))
                gev_return_values[label] = return_val
                
                # 计算95%置信区间（如果bootstrap样本足够）
                bootstrap_values = bootstrap_return_values[label]
                if len(bootstrap_values) > 50:  # 确保有足够的bootstrap样本
                    lower_ci = np.percentile(bootstrap_values, 2.5)
                    upper_ci = np.percentile(bootstrap_values, 97.5)
                    gev_confidence_intervals[label] = (lower_ci, upper_ci)
                else:
                    gev_confidence_intervals[label] = (None, None)
            
            # GEV分布参数 - 不直接存储分布对象以避免序列化问题
            # 而是保存参数，在需要时重新创建分布对象
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
        
        # 拟合Gumbel分布（Generalized Extreme Value分布的特例，形状参数为0）
        try:
            # Fit Gumbel distribution to peaks
            gumbel_params = stats.gumbel_r.fit(all_peaks)
            
            # 为Gumbel计算AIC
            gumbel_nll = -np.sum(stats.gumbel_r.logpdf(all_peaks, *gumbel_params))
            gumbel_k = len(gumbel_params)  # 参数数量
            gumbel_aic = 2 * gumbel_k + 2 * gumbel_nll
            
            # 计算给定回归周期的回归值
            # 对于Gumbel分布，回归值R(T) = μ - σ * ln(-ln(1-1/T))
            gumbel_return_values = {}
            gumbel_confidence_intervals = {}
            
            # 使用bootstrap方法计算置信区间
            n_bootstrap = 1000
            bootstrap_return_values = {label: [] for label in return_period_labels}
            
            # 创建bootstrap样本
            rng = np.random.RandomState(42)  # 固定随机种子以获得可重复结果
            for _ in range(n_bootstrap):
                # 从峰值中有放回抽样
                bootstrap_sample = rng.choice(all_peaks, size=len(all_peaks), replace=True)
                try:
                    # 拟合Gumbel分布
                    bootstrap_params = stats.gumbel_r.fit(bootstrap_sample)
                    
                    # 计算各回归周期的回归值
                    for i, T in enumerate(return_periods):
                        return_val = bootstrap_params[0] + bootstrap_params[1] * (-np.log(-np.log(1 - 1/T)))
                        bootstrap_return_values[return_period_labels[i]].append(return_val)
                except:
                    # 如果拟合失败，忽略这个bootstrap样本
                    continue
            
            # 计算各回归周期的回归值和置信区间
            for i, T in enumerate(return_periods):
                label = return_period_labels[i]
                return_val = gumbel_params[0] + gumbel_params[1] * (-np.log(-np.log(1 - 1/T)))
                gumbel_return_values[label] = return_val
                
                # 计算95%置信区间（如果bootstrap样本足够）
                bootstrap_values = bootstrap_return_values[label]
                if len(bootstrap_values) > 50:  # 确保有足够的bootstrap样本
                    lower_ci = np.percentile(bootstrap_values, 2.5)
                    upper_ci = np.percentile(bootstrap_values, 97.5)
                    gumbel_confidence_intervals[label] = (lower_ci, upper_ci)
                else:
                    gumbel_confidence_intervals[label] = (None, None)
            
            # Gumbel分布参数 - 不直接存储分布对象以避免序列化问题
            # 而是保存参数，在需要时重新创建分布对象
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
        
        # 选择最佳模型（基于AIC值）
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
        
        # 创建超越概率表
        exceedance_table = pd.DataFrame({
            'Peak Value': sorted_peaks,
            'Exceedance Probability': exceedance_prob,
            'Return Period (hours)': 1 / exceedance_prob * data_duration_hours / n
        })
        
        # 准备结果字典
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
            # 调用plot.py中的可视化函数
            figure = plot_extreme_analysis(
                results=results, 
                visualization_backend=visualization_backend,
                save_path=save_path,
                save_html=save_html,
                visualization=visualization,
                title=None,  # 让函数自动生成标题
                ch_name=ch_name,
                pydas_obj=pydas_obj,
                bins=bins,
                fullscale=fullscale,
                return_periods=return_periods,
                return_period_labels=return_period_labels
            )
            
            # 将图形对象添加到结果中
            results['figure'] = figure
        
        return results
    
    except Exception as e:
        logger.error(f"Error in extreme_analysis: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None 