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

def spectral_analysis(pydas_obj, channel_name, method='cov', L=1024, plot=False, title=None, 
                      show=True, save_path=None, use_plotly=True, save_html=None,
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

def statistic_analysis(pydas_obj, ch_name, sseg=0, advanced=False, visualization=False, bins=50, 
                       save_fig=False, save_path=None, use_plotly=False, fullscale=False, lam=None, 
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
    use_plotly : bool, optional
        是否使用plotly进行可视化，默认为False
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
        title_prefix = "Full Scale " if fullscale else ""
        title = f"{title_prefix}Statistical Analysis for {ch_name}"
        
        if use_plotly:
            _plot_statistics_plotly(pydas_obj, ch_names, sseg, stats_df, bins, save_fig, save_path, 
                                   data=data, title_override=title)
        else:
            _plot_statistics_mpl(pydas_obj, ch_names, sseg, stats_df, bins, save_fig, save_path, 
                                data=data, title_override=title)
    
    # 输出统计结果
    scale_info = "原型尺度" if fullscale else "模型尺度"
    logger.info(f"统计分析结果 ({scale_info}):")
    logger.info("\n" + stats_df.to_string(float_format=lambda x: f"% .4E" % x))
    
    return stats_df

def _plot_statistics_mpl(pydas_obj, ch_names, sseg, stats_df, bins, save_fig, save_path, data=None, title_override=None):
    """使用matplotlib绘制统计分析图"""
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import numpy as np
    
    for name in ch_names:
        # 创建一个2x2的子图布局
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # 设置标题，如果有自定义标题则使用它
        if title_override:
            plt.suptitle(title_override, fontsize=16)
        else:
            plt.suptitle(f'Statistical Analysis for Channel: {name} (Segment {sseg})', fontsize=16)
        
        # 获取数据
        if data is None:
            data = pydas_obj.data[sseg][name].values
        
        # 1. 时间序列图
        axs[0, 0].plot(data)
        axs[0, 0].set_title('Time Series')
        axs[0, 0].set_xlabel('Sample')
        axs[0, 0].set_ylabel(f'{name} [{stats_df.loc[name, "Unit"]}]')
        axs[0, 0].grid(True)
        
        # 添加统计信息文本框
        stats_text = (f"Mean: {stats_df.loc[name, 'Mean']:.4E}\n"
                     f"Std: {stats_df.loc[name, 'Std']:.4E}\n"
                     f"RMS: {stats_df.loc[name, 'RMS']:.4E}\n"
                     f"Range: {stats_df.loc[name, 'Range']:.4E}")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axs[0, 0].text(0.05, 0.95, stats_text, transform=axs[0, 0].transAxes, 
                       fontsize=9, verticalalignment='top', bbox=props)
        
        # 2. 直方图和PDF
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        axs[0, 1].bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0], 
                     alpha=0.6, color='skyblue', label='Histogram')
        
        # 拟合正态分布
        mu, sigma = stats.norm.fit(data)
        x = np.linspace(min(data), max(data), 100)
        pdf = stats.norm.pdf(x, mu, sigma)
        axs[0, 1].plot(x, pdf, 'r-', lw=2, label=f'Normal PDF\n(μ={mu:.2E}, σ={sigma:.2E})')
        
        axs[0, 1].set_title('Histogram and PDF')
        axs[0, 1].set_xlabel(f'{name} [{stats_df.loc[name, "Unit"]}]')
        axs[0, 1].set_ylabel('Density')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # 3. 经验累积分布函数（ECDF）
        sorted_data = np.sort(data)
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        axs[1, 0].step(sorted_data, ecdf, where='post', label='ECDF')
        
        # 理论CDF
        cdf = stats.norm.cdf(x, mu, sigma)
        axs[1, 0].plot(x, cdf, 'r-', lw=2, label='Normal CDF')
        
        axs[1, 0].set_title('Empirical CDF')
        axs[1, 0].set_xlabel(f'{name} [{stats_df.loc[name, "Unit"]}]')
        axs[1, 0].set_ylabel('Probability')
        axs[1, 0].grid(True)
        axs[1, 0].legend()
        
        # 4. Q-Q图
        stats.probplot(data, dist="norm", plot=axs[1, 1])
        axs[1, 1].set_title('Q-Q Plot (Normal Distribution)')
        axs[1, 1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_fig:
            if save_path is None:
                save_path = os.getcwd()
            plt.savefig(f"{save_path}/{name}_seg{sseg}_stats.png", dpi=300, bbox_inches='tight')
        
        plt.show()

def _plot_statistics_plotly(pydas_obj, ch_names, sseg, stats_df, bins, save_fig, save_path, data=None, title_override=None):
    """使用plotly绘制统计分析图"""
    try:
        import plotly.graph_objects as go
        import plotly.subplots as sp
        import plotly.figure_factory as ff
        import numpy as np
        import scipy.stats as stats
        
        for name in ch_names:
            # 创建子图
            fig = sp.make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Time Series', 
                    'Histogram and PDF', 
                    'Empirical CDF', 
                    'Q-Q Plot (Normal Distribution)'
                ),
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                      [{'type': 'scatter'}, {'type': 'scatter'}]],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # 获取数据
            if data is None:
                data = pydas_obj.data[sseg][name].values
            
            # 1. 时间序列图
            fig.add_trace(
                go.Scatter(
                    y=data,
                    mode='lines',
                    name='Time Series'
                ),
                row=1, col=1
            )
            
            # 添加统计信息注释
            stats_text = (f"Mean: {stats_df.loc[name, 'Mean']:.4E}<br>"
                         f"Std: {stats_df.loc[name, 'Std']:.4E}<br>"
                         f"RMS: {stats_df.loc[name, 'RMS']:.4E}<br>"
                         f"Range: {stats_df.loc[name, 'Range']:.4E}")
            
            fig.add_annotation(
                xref="x domain", yref="y domain",
                x=0.05, y=0.95,
                text=stats_text,
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1,
                borderpad=4,
                font=dict(size=10),
                row=1, col=1
            )
            
            # 2. 直方图和PDF
            hist, bin_edges = np.histogram(data, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # 直方图
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=hist,
                    name='Histogram',
                    marker_color='skyblue',
                    opacity=0.6
                ),
                row=1, col=2
            )
            
            # 拟合正态分布
            mu, sigma = stats.norm.fit(data)
            x = np.linspace(min(data), max(data), 100)
            pdf = stats.norm.pdf(x, mu, sigma)
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=pdf,
                    mode='lines',
                    name=f'Normal PDF (μ={mu:.2E}, σ={sigma:.2E})',
                    line=dict(color='red', width=2)
                ),
                row=1, col=2
            )
            
            # 3. 经验累积分布函数（ECDF）
            sorted_data = np.sort(data)
            ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            fig.add_trace(
                go.Scatter(
                    x=sorted_data,
                    y=ecdf,
                    mode='lines',
                    line=dict(shape='hv'),
                    name='ECDF'
                ),
                row=2, col=1
            )
            
            # 理论CDF
            cdf = stats.norm.cdf(x, mu, sigma)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=cdf,
                    mode='lines',
                    name='Normal CDF',
                    line=dict(color='red', width=2)
                ),
                row=2, col=1
            )
            
            # 4. Q-Q图
            # 计算理论分位数
            theoretical_quantiles = np.random.normal(0, 1, len(data))
            theoretical_quantiles.sort()
            
            # 计算样本分位数
            sample_quantiles = np.sort(data)
            
            # 添加Q-Q线
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode='markers',
                    name='Q-Q Plot',
                    marker=dict(size=5)
                ),
                row=2, col=2
            )
            
            # 理论Q-Q线
            min_val = min(theoretical_quantiles)
            max_val = max(theoretical_quantiles)
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val * sigma + mu, max_val * sigma + mu],
                    mode='lines',
                    name='Theoretical Q-Q Line',
                    line=dict(color='red', width=2)
                ),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_layout(
                title=title_override if title_override else f'Statistical Analysis for Channel: {name} (Segment {sseg})',
                height=800,
                width=1000,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            
            # 更新x轴标题
            fig.update_xaxes(title_text="Sample", row=1, col=1)
            fig.update_xaxes(title_text=f"{name} [{stats_df.loc[name, 'Unit']}]", row=1, col=2)
            fig.update_xaxes(title_text=f"{name} [{stats_df.loc[name, 'Unit']}]", row=2, col=1)
            fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
            
            # 更新y轴标题
            fig.update_yaxes(title_text=f"{name} [{stats_df.loc[name, 'Unit']}]", row=1, col=1)
            fig.update_yaxes(title_text="Density", row=1, col=2)
            fig.update_yaxes(title_text="Probability", row=2, col=1)
            fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
            
            # 保存和显示
            if save_fig:
                if save_path is None:
                    save_path = os.getcwd()
                
                try:
                    # 先尝试保存为HTML
                    fig.write_html(f"{save_path}/{name}_seg{sseg}_stats.html")
                    logger.info(f"Saved interactive plot to: {save_path}/{name}_seg{sseg}_stats.html")
                    
                    # 如果plotly.io可用，也可以保存为图像
                    import plotly.io as pio
                    pio.write_image(fig, f"{save_path}/{name}_seg{sseg}_stats.png")
                    logger.info(f"Saved static plot to: {save_path}/{name}_seg{sseg}_stats.png")
                except Exception as e:
                    logger.warning(f"Could not save image: {str(e)}")
                    logger.warning("Try installing the required packages: pip install -U kaleido")
            
            # 显示图形
            fig.show()
            
    except ImportError as e:
        logger.warning(f"Could not use plotly for visualization: {str(e)}")
        logger.warning("Using matplotlib as fallback...")
        _plot_statistics_mpl(pydas_obj, ch_names, sseg, stats_df, bins, save_fig, save_path, 
                            data=data, title_override=title_override) 