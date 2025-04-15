"""
PyDAS Reporting Module
======================
This module contains reporting functions for PyDAS data.
- Excel report generation
- Statistical summary reports
- Channel analysis reports

Author: Xiaoxian Guo
Date: 2025-04-12
"""

import os
import numpy as np
import pandas as pd
import logging
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils import get_column_letter
import scipy.signal as signal
import copy
from scipy import stats as spstats
from openpyxl.worksheet.page import PageMargins, PrintPageSetup
import matplotlib.pyplot as plt
from waveModel.specmodels import Jonswap
import matplotlib.gridspec as gridspec
from scipy.stats import norm

# Set up logging
logger = logging.getLogger('pydas.reporting')

def analyze_channel_data(data_scaled, mean_val=None, std_val=None, zerocrossing_analysis=True, 
                      amplitude_analysis=True, significant_percentile=33.0, n_hr_forecast=3,
                      data_duration_hours=None, dt=None):
    """
    Analyze channel data and return statistical results
    
    Parameters
    ----------
    data_scaled : numpy.ndarray
        The scaled data to analyze
    mean_val : float, optional
        Pre-calculated mean value
    std_val : float, optional
        Pre-calculated standard deviation
    zerocrossing_analysis : bool, default=True
        Whether to perform zero-crossing analysis
    amplitude_analysis : bool, default=True
        Whether to perform amplitude analysis
    significant_percentile : float, default=33.0
        Percentile for significant value calculation
    n_hr_forecast : int, default=3
        Number of hours for extreme value forecast
    data_duration_hours : float, optional
        Duration of data in hours
    dt : float, optional
        Time step
        
    Returns
    -------
    dict
        Dictionary containing all statistical results
    """
    # Calculate basic statistics if not provided
    if mean_val is None:
        mean_val = np.mean(data_scaled)
    if std_val is None:
        std_val = np.std(data_scaled)
    max_val = np.max(data_scaled)
    min_val = np.min(data_scaled)
    
    # Calculate skewness and kurtosis
    skewness = spstats.skew(data_scaled)
    kurtosis = spstats.kurtosis(data_scaled)
    
    # Initialize results
    results = {
        'maximum': max_val,
        'minimum': min_val,
        'mean': mean_val,
        'STD': std_val,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'maximum_double_amplitude': 0,
        'sign_double_amplitude': 0,
        'mean_zerocross_period': 0,
        'zero_upcross': 0,
        'estimated_max': 0,
        'estimated_min': 0
    }
    
    if zerocrossing_analysis:
        # Calculate mean crossings
        data_centered = data_scaled - mean_val
        zero_crossings = np.where(np.diff(np.signbit(data_centered)))[0]
        upcrossings = [i for i in zero_crossings if data_centered[i+1] > data_centered[i]]
        results['zero_upcross'] = len(upcrossings)
        
        # Calculate mean period
        if results['zero_upcross'] > 1:
            periods = np.diff(upcrossings) * dt
            results['mean_zerocross_period'] = np.mean(periods)
        
        # Find peaks and troughs
        peaks, _ = signal.find_peaks(data_scaled)
        troughs, _ = signal.find_peaks(-data_scaled)
        
        if len(peaks) > 0 and len(troughs) > 0:
            # Sort peaks and troughs
            peaks = np.sort(peaks)
            troughs = np.sort(troughs)
            
            # Calculate double amplitudes
            double_amplitudes = []
            
            # Method 1: For each peak, find the largest amplitude with adjacent troughs
            for peak_idx in peaks:
                peak_val = data_scaled[peak_idx]
                
                # Find previous trough
                prev_troughs = troughs[troughs < peak_idx]
                prev_amp = 0
                if len(prev_troughs) > 0:
                    prev_trough_idx = prev_troughs[-1]
                    prev_amp = peak_val - data_scaled[prev_trough_idx]
                
                # Find next trough
                next_troughs = troughs[troughs > peak_idx]
                next_amp = 0
                if len(next_troughs) > 0:
                    next_trough_idx = next_troughs[0]
                    next_amp = peak_val - data_scaled[next_trough_idx]
                
                # Select larger amplitude
                double_amp = max(prev_amp, next_amp, 0)
                if double_amp > 0:
                    double_amplitudes.append(double_amp)
            
            # Method 2: Use zero-crossing waves if available
            if zerocrossing_analysis and len(upcrossings) > 1:
                wave_amplitudes = []
                for i in range(len(upcrossings) - 1):
                    start_idx = upcrossings[i]
                    end_idx = upcrossings[i+1]
                    wave_segment = data_scaled[start_idx:end_idx+1]
                    if len(wave_segment) > 2:
                        wave_amp = np.max(wave_segment) - np.min(wave_segment)
                        wave_amplitudes.append(wave_amp)
                
                if len(wave_amplitudes) > len(double_amplitudes):
                    logger.info(f"Using zero-crossing method for wave amplitude calculation")
                    double_amplitudes = wave_amplitudes
                elif len(double_amplitudes) > 0:
                    logger.info(f"Using peak-trough matching method for wave amplitude calculation")
            
            if len(double_amplitudes) > 0:
                results['maximum_double_amplitude'] = np.max(double_amplitudes)
                
                # Calculate significant double amplitude
                n_waves = len(double_amplitudes)
                sorted_amps = np.sort(double_amplitudes)[::-1]
                n_significant = max(1, int(n_waves * significant_percentile / 100))
                results['sign_double_amplitude'] = np.mean(sorted_amps[:n_significant])
                
                logger.info(f"Calculated significant double amplitude from {n_significant} highest waves")
    
    # Extreme value estimation
    if n_hr_forecast > 0 and data_duration_hours > 0:
        try:
            data_centered = data_scaled - mean_val
            
            if zerocrossing_analysis and 'peaks' in locals() and len(peaks) > 10:
                # Use peak statistics method
                centered_peaks, _ = signal.find_peaks(data_centered)
                centered_troughs, _ = signal.find_peaks(-data_centered)
                
                if len(centered_peaks) > 0 and len(centered_troughs) > 0:
                    # Process peaks
                    peak_values = data_centered[centered_peaks]
                    if len(peak_values) > 0:
                        sorted_peaks = np.sort(peak_values)[::-1]
                        peaks_per_hour = len(peak_values) / data_duration_hours
                        expected_peaks_in_forecast = peaks_per_hour * n_hr_forecast
                        
                        try:
                            num_fitting_peaks = max(10, int(len(sorted_peaks) * 0.1))
                            fitting_peaks = sorted_peaks[:num_fitting_peaks]
                            c, loc, scale = spstats.weibull_min.fit(fitting_peaks, floc=0)
                            p = 1 - 1/expected_peaks_in_forecast
                            peak_extreme = spstats.weibull_min.ppf(p, c, loc, scale)
                            results['estimated_max'] = peak_extreme + mean_val
                        except:
                            logger.warning("Weibull fitting failed for peaks, using normal distribution")
                            forecast_factor = np.sqrt(n_hr_forecast / data_duration_hours)
                            extreme_factor = 3.5 + 0.5 * np.log(n_hr_forecast / data_duration_hours)
                            extreme_std = std_val * forecast_factor
                            results['estimated_max'] = mean_val + extreme_factor * extreme_std
                    
                    # Process troughs
                    trough_values = data_centered[centered_troughs]
                    if len(trough_values) > 0:
                        sorted_troughs = np.sort(trough_values)
                        troughs_per_hour = len(trough_values) / data_duration_hours
                        expected_troughs_in_forecast = troughs_per_hour * n_hr_forecast
                        
                        try:
                            num_fitting_troughs = max(10, int(len(sorted_troughs) * 0.1))
                            fitting_troughs = -sorted_troughs[:num_fitting_troughs]
                            c, loc, scale = spstats.weibull_min.fit(fitting_troughs, floc=0)
                            p = 1 - 1/expected_troughs_in_forecast
                            trough_extreme = -spstats.weibull_min.ppf(p, c, loc, scale)
                            results['estimated_min'] = trough_extreme + mean_val
                        except:
                            logger.warning("Weibull fitting failed for troughs, using normal distribution")
                            forecast_factor = np.sqrt(n_hr_forecast / data_duration_hours)
                            extreme_factor = 3.5 + 0.5 * np.log(n_hr_forecast / data_duration_hours)
                            extreme_std = std_val * forecast_factor
                            results['estimated_min'] = mean_val - extreme_factor * extreme_std
            else:
                # Use normal distribution method
                logger.info("Using normal distribution method for extreme value estimation")
                forecast_factor = np.sqrt(n_hr_forecast / data_duration_hours)
                extreme_factor = 3.5 + 0.5 * np.log(n_hr_forecast / data_duration_hours)
                extreme_std = std_val * forecast_factor
                results['estimated_max'] = mean_val + extreme_factor * extreme_std
                results['estimated_min'] = mean_val - extreme_factor * extreme_std
        except Exception as e:
            logger.warning(f"Error in extreme value estimation: {str(e)}")
            forecast_factor = np.sqrt(n_hr_forecast / data_duration_hours)
            extreme_factor = 3.5 + 0.5 * np.log(n_hr_forecast / data_duration_hours)
            extreme_std = std_val * forecast_factor
            results['estimated_max'] = mean_val + extreme_factor * extreme_std
            results['estimated_min'] = mean_val - extreme_factor * extreme_std
    
    return results

def channel_report(pydas_obj, output_file='channel_report.xlsx', sseg=0, fullscale=True, 
                  lam=None, rho=1.025, g=9.807, header_text=None, include_charts=False, 
                  significant_percentile=33.0, wave_analysis=True, format_sheet=True, 
                  zerocrossing_analysis=True, amplitude_analysis=True, 
                  n_hr_forecast=3, cutoffperiod=15.0):
    """
    为PyDAS对象的所有通道生成详细的Excel分析报告，包括高低频分离分析
    
    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS对象，包含要分析的数据
    output_file : str, default='channel_report.xlsx'
        输出Excel文件的路径
    sseg : int, default=0
        要分析的数据段索引
    fullscale : bool, default=True
        是否使用实际尺度（原型尺度）值
    lam : float, optional
        尺度因子，仅在fullscale=True且PyDAS对象未设置__lam__属性时使用
    rho : float, default=1.025
        水密度 (kg/m³)，仅用于fullscale=True时
    g : float, default=9.807
        重力加速度 (m/s²)，仅用于fullscale=True时
    header_text : str, optional
        报告中的标题文本
    include_charts : bool, default=False
        是否在报告中包含图表
    significant_percentile : float, default=33.0
        计算显著值的百分位数
    wave_analysis : bool, default=True
        是否进行波浪分析
    format_sheet : bool, default=True
        是否设置Excel格式
    zerocrossing_analysis : bool, default=True
        是否进行过零分析
    amplitude_analysis : bool, default=True
        是否进行振幅分析
    n_hr_forecast : int, default=3
        极值估计的预测小时数
    cutoffperiod : float, default=15.0
        高低频分离的截止周期（秒），用于分离高频和低频成分
        
    Returns
    -------
    pandas.DataFrame
        包含所有通道统计数据的DataFrame
        
    Notes
    -----
    - 生成一个包含所有通道统计分析的Excel报告
    - 报告包括基本统计值、过零分析、振幅分析和极值估计
    - 可以选择是否使用实际尺度值（原型尺度）
    - 分析可能需要一些时间，特别是对于大型数据集
    """
    # 检查输入
    if sseg >= pydas_obj.__segN__:
        logger.error(f"段索引 {sseg} 超出最大段数 ({pydas_obj.__segN__-1})")
        return None

    if fullscale:
        # 确定尺度因子
        if lam is None:
            if hasattr(pydas_obj, '__lam__'):
                lam = pydas_obj.__lam__
            else:
                logger.warning("Scale factor lam not provided and object has no __lam__ attribute")
                return None
        
        logger.info(f"Creating data copy and converting to full scale (λ={lam}, ρ={rho}, g={g})...")
        pydas_analysis = copy.deepcopy(pydas_obj)
        pydas_analysis.to_fullscale(rho=rho, g=g, pInfo=False)
        scale_text = "Full Scale"
    else:
        pydas_analysis = pydas_obj
        scale_text = "Model Scale"
    
    if header_text is None:
        if hasattr(pydas_obj, 'filename'):
            filename = os.path.basename(pydas_obj.filename)
            header_text = f"Wave only [{filename}, {scale_text}]"
        else:
            header_text = f"Wave Analysis Report [{scale_text}]"
    
    # 创建结果DataFrame的列
    columns = [
        'channel\nID', 'Name', 'unit', 'number\nof zero\nupcross', 
        'maximum', 'minimum', 'mean', 'STD',
        'maximum\ndouble\namplitude', 'sign.\ndouble\namplitude', 
        'skewness', 'kurtosis',
        'mean\nzerocro.\nperiod', 'estimated\n3hr\nmaximum', 'estimated\n3hr\nminimum'
    ]
    
    # 创建三个结果DataFrame
    results_total = pd.DataFrame(columns=columns)
    results_low = pd.DataFrame(columns=columns)
    results_high = pd.DataFrame(columns=columns)
    
    # 获取通道信息
    ch_info = pydas_analysis.chInfo
    
    # 计算分析周期
    dt = 1.0 / pydas_analysis.__fs__
    data_duration_seconds = len(pydas_analysis.data[sseg]) * dt
    data_duration_hours = data_duration_seconds / 3600
    
    # 计算截止频率（从周期转换为rad/s）
    cutoff_freq = 2 * np.pi / cutoffperiod
    
    # 对每个通道进行分析
    for ch_idx, (_, row) in enumerate(ch_info.iterrows(), 1):
        ch_name = row['Name']
        ch_unit = row['Unit']
        
        # 获取原始数据
        data_scaled = pydas_analysis.data[sseg][ch_name].values
        
        # 分离高低频
        data_low = pydas_analysis.apply_lowpass_filter(ch_name, cutoff_freq, returnValue=True)
        data_high = pydas_analysis.apply_highpass_filter(ch_name, cutoff_freq, returnValue=True)
        
        # 分析三组数据
        results_total_ch = analyze_channel_data(
            data_scaled, zerocrossing_analysis=zerocrossing_analysis,
            amplitude_analysis=amplitude_analysis, significant_percentile=significant_percentile,
            n_hr_forecast=n_hr_forecast, data_duration_hours=data_duration_hours, dt=dt
        )
        
        results_low_ch = analyze_channel_data(
            data_low, zerocrossing_analysis=zerocrossing_analysis,
            amplitude_analysis=amplitude_analysis, significant_percentile=significant_percentile,
            n_hr_forecast=n_hr_forecast, data_duration_hours=data_duration_hours, dt=dt
        )
        
        results_high_ch = analyze_channel_data(
            data_high, zerocrossing_analysis=zerocrossing_analysis,
            amplitude_analysis=amplitude_analysis, significant_percentile=significant_percentile,
            n_hr_forecast=n_hr_forecast, data_duration_hours=data_duration_hours, dt=dt
        )
        
        # 将结果添加到相应的DataFrame
        results_total.loc[ch_idx] = [
            ch_idx, ch_name, ch_unit, results_total_ch['zero_upcross'],
            results_total_ch['maximum'], results_total_ch['minimum'],
            results_total_ch['mean'], results_total_ch['STD'],
            results_total_ch['maximum_double_amplitude'],
            results_total_ch['sign_double_amplitude'],
            results_total_ch['skewness'], results_total_ch['kurtosis'],
            results_total_ch['mean_zerocross_period'],
            results_total_ch['estimated_max'], results_total_ch['estimated_min']
        ]
        
        results_low.loc[ch_idx] = [
            ch_idx, ch_name, ch_unit, results_low_ch['zero_upcross'],
            results_low_ch['maximum'], results_low_ch['minimum'],
            results_low_ch['mean'], results_low_ch['STD'],
            results_low_ch['maximum_double_amplitude'],
            results_low_ch['sign_double_amplitude'],
            results_low_ch['skewness'], results_low_ch['kurtosis'],
            results_low_ch['mean_zerocross_period'],
            results_low_ch['estimated_max'], results_low_ch['estimated_min']
        ]
        
        results_high.loc[ch_idx] = [
            ch_idx, ch_name, ch_unit, results_high_ch['zero_upcross'],
            results_high_ch['maximum'], results_high_ch['minimum'],
            results_high_ch['mean'], results_high_ch['STD'],
            results_high_ch['maximum_double_amplitude'],
            results_high_ch['sign_double_amplitude'],
            results_high_ch['skewness'], results_high_ch['kurtosis'],
            results_high_ch['mean_zerocross_period'],
            results_high_ch['estimated_max'], results_high_ch['estimated_min']
        ]

    # 创建Excel文件
    if output_file:
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 写入三个表
                results_total.to_excel(writer, sheet_name='Total Statistics', index=False)
                results_low.to_excel(writer, sheet_name=f'Low Freq (T>{cutoffperiod}s)', index=False)
                results_high.to_excel(writer, sheet_name=f'High Freq (T<{cutoffperiod}s)', index=False)
                
                if format_sheet:
                    # 格式化三个表
                    for sheet_name in ['Total Statistics', f'Low Freq (T>{cutoffperiod}s)', f'High Freq (T<{cutoffperiod}s)']:
                        ws = writer.sheets[sheet_name]
                        
                        # 添加标题行
                        ws.insert_rows(0, 2)
                        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(columns))
                        title_cell = ws.cell(row=1, column=1, value=f"{header_text} - {sheet_name}")
                        
                        # 设置标题行格式
                        title_cell.font = Font(bold=True, size=14)
                        title_cell.alignment = Alignment(horizontal='center', vertical='center')
                        
                        # 设置列宽
                        for i, column in enumerate(columns, 1):
                            col_width = max(len(str(c)) for c in results_total[column].astype(str)) * 1.2
                            col_width = max(col_width, len(column) * 1.2)
                            ws.column_dimensions[get_column_letter(i)].width = min(col_width, 20)
                        
                        # 添加粗边框样式
                        thick_border = Border(
                            left=Side(style='thin'), 
                            right=Side(style='thin'),
                            top=Side(style='thin'),
                            bottom=Side(style='thin')
                        )
                        
                        # 设置数据行格式
                        for row in range(3, ws.max_row + 1):
                            for col in range(1, ws.max_column + 1):
                                cell = ws.cell(row=row, column=col)
                                
                                # 应用边框
                                cell.border = thick_border
                                
                                # 对齐
                                if col == 1:  # 通道编号
                                    cell.alignment = Alignment(horizontal='center')
                                elif col in [2]:  # 名称
                                    cell.alignment = Alignment(horizontal='left')
                                else:  # 数值
                                    cell.alignment = Alignment(horizontal='center')
                                    
                                    # 格式化数值
                                    if isinstance(cell.value, (int, float)) and col >= 4:
                                        if abs(cell.value) < 0.001:
                                            # 对于接近0的值直接显示0.000
                                            cell.value = "0.000"
                                        else:
                                            # 普通数字，3位小数
                                            cell.value = f"{cell.value:.3f}"
                        
                        # 设置标题行格式
                        header_row = 3
                        header_fill = PatternFill(start_color='E9E9E9', end_color='E9E9E9', fill_type='solid')
                        
                        for col in range(1, ws.max_column + 1):
                            cell = ws.cell(row=header_row, column=col)
                            cell.font = Font(bold=True, size=10)
                            cell.fill = header_fill
                            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
                            cell.border = thick_border
                        
                        # 设置页面布局
                        ws.page_setup.orientation = 'landscape'
                        ws.page_setup.paperSize = 9  # A4
                        ws.page_margins = PageMargins(left=0.5, right=0.5, top=0.5, bottom=0.5)
                        ws.page_setup.fitToWidth = 1
                        ws.page_setup.fitToHeight = 0
                        ws.print_area = f'A1:{get_column_letter(ws.max_column)}{ws.max_row}'
                        ws.print_title_rows = '1:3'
                        
                        # 设置列宽
                        col_width_map = {
                            1: 6,   # 通道编号
                            2: 20,  # 通道名称
                            3: 8    # 单位
                        }
                        
                        default_width = 12
                        special_widths = {
                            4: 10,  # 零上穿数
                            9: 15,  # 最大双振幅
                            10: 15, # 显著双振幅
                            11: 15, # 偏度
                            12: 15, # 峰度
                            13: 12, # 平均零上穿周期
                            14: 14, # 预估3小时最大值
                            15: 14  # 预估3小时最小值
                        }
                        
                        for i in range(1, ws.max_column + 1):
                            col_letter = get_column_letter(i)
                            if i in col_width_map:
                                ws.column_dimensions[col_letter].width = col_width_map[i]
                            elif i in special_widths:
                                ws.column_dimensions[col_letter].width = special_widths[i]
                            else:
                                ws.column_dimensions[col_letter].width = default_width
                            
                            ws.column_dimensions[col_letter].bestFit = True
                        
                        ws.sheet_properties.pageSetUpPr.fitToPage = True
            
            logger.info(f"Report successfully exported to {output_file}")
        
        except Exception as e:
            logger.error(f"Error exporting Excel file: {str(e)}")
    
    # 返回所有结果DataFrame
    return results_total, results_low, results_high 

def wave_report(pydas_obj, ch_name, sseg=0, save_path=None, title=None, L=1024,
                Hs=None, Tp=None, gamma=None, bins=50, fullscale=False, lam=None, 
                rho=1.025, g=9.807):
    """
    生成波浪分析报告，包括时间序列、谱分析和峰值统计
    
    Parameters
    ----------
    pydas_obj : PyDAS object
        PyDAS对象
    ch_name : str
        要分析的通道名称
    sseg : int, optional
        数据段索引，默认为0
    save_path : str, optional
        保存图片的路径，默认为None
    title : str, optional
        图表标题，默认为None
    L : int, optional
        谱分析的数据块长度，默认为1024
    Hs : float, optional
        JONSWAP谱的有效波高，默认为None
    Tp : float, optional
        JONSWAP谱的峰值周期，默认为None
    gamma : float, optional
        JONSWAP谱的峰值增强因子，默认为None
    bins : int, optional
        直方图的bin数量，默认为50
    fullscale : bool, optional
        是否使用实际尺度数据，默认为False
    lam : float, optional
        尺度因子，默认为None
    rho : float, optional
        水密度 (kg/m3)，默认为1.025
    g : float, optional
        重力加速度 (m/s2)，默认为9.807
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        生成的图表对象
    """
    # 检查通道是否存在
    if ch_name not in pydas_obj.data[sseg]:
        raise ValueError(f"Channel {ch_name} not found in segment {sseg}")
    
    # 获取数据
    data = pydas_obj.data[sseg][ch_name].values
    unit = 'cm'
    T = np.arange(len(data)) / pydas_obj.__fs__

    # 如果需要转换为实际尺度
    if fullscale:
        if lam is None:
            lam = pydas_obj.__lam__
        ts = pydas_obj.channel2fullscale(ch_name, lam, rho, g)
        data = ts.data
        T = ts.args
        unit = 'm'  # 更新单位
    
    # 创建图表
    fig = plt.figure(figsize=(10, 11))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
    
    # 1. 时间序列图（第一行，全宽）
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(T/3600, data)
    ax1.set_xlabel('Time (hr)')
    ax1.set_ylabel(f'Amplitude ({unit})')
    ax1.grid(True, axis='x')
    ax1.set_title('Time Series')
    ax1.set_xlim(T[0]/3600, T[-1]/3600)
    if title:
        fig.suptitle(title, y=0.95)
    
    # 2. 谱分析（第二行，左）
    ax2 = fig.add_subplot(gs[1, 0])
    spec = pydas_obj.spectral_analysis(ch_name, method='cov', L=L, plot=False, 
                                     fullscale=fullscale, lam=lam, rho=rho, g=g)
    
    # 获取谱数据
    freq = spec.args
    psd = spec.data
    
    # 绘制测量谱
    ax2.plot(freq, psd, label='Measured')
    
    # 如果提供了JONSWAP参数，绘制理论谱
    if Hs is not None and Tp is not None:
        from waveModel.specmodels import Jonswap
        jonswap_spec = Jonswap(Hs, Tp, gamma=gamma if gamma is not None else 3.3)
        ax2.plot(jonswap_spec.args, jonswap_spec.data, 'r--', label='JONSWAP')

    ax2.set_xlim(0, 2)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel('Frequency (rad/s)')
    ax2.set_ylabel(f'PSD ({unit}$^2$ s/rad)')
    # ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Wave Spectrum')
    
    # 计算并显示谱特征
    m0 = np.trapz(psd, freq)  # 使用数值积分计算零阶矩
    Hm0 = 4.0 * np.sqrt(m0)
    text = f'Hm0 = {Hm0:.2f} {unit}'
    ax2.text(0.05, 0.95, text, transform=ax2.transAxes, verticalalignment='top')
    
    # 3. 直方图和正态拟合（第二行，右）
    ax3 = fig.add_subplot(gs[1, 1])
    n, bins, patches = ax3.hist(data, bins=bins, density=True, alpha=0.6)
    
    # 拟合正态分布
    mu, std = np.mean(data), np.std(data)
    x = np.linspace(min(data), max(data), 100)
    p = norm.pdf(x, mu, std)
    ax3.plot(x, p, 'r-', linewidth=2, label=f'Normal (μ={mu:.2f}, σ={std:.2f})')
    
    ax3.set_xlabel(f'Amplitude ({unit})')
    ax3.set_ylabel('Probability Density')
    ax3.grid(True)
    ax3.set_title('Amplitude Distribution')
    ax3.legend()
    
    # 4. 峰值统计（第三行）
    # 检测峰值
    peaks, _ = signal.find_peaks(data, prominence=1.0)
    peak_values = data[peaks]
    
    # 峰值直方图（第三行，左）
    ax4 = fig.add_subplot(gs[2, 0])
    n_peaks, bins_peaks, patches_peaks = ax4.hist(peak_values, bins=bins, density=True, alpha=0.6)
    
    # 拟合Weibull分布
    shape, loc, scale = spstats.weibull_min.fit(peak_values, floc=0)
    x_peaks = np.linspace(0, max(peak_values), 100)
    p_peaks = spstats.weibull_min.pdf(x_peaks, shape, loc, scale)
    ax4.plot(x_peaks, p_peaks, 'r-', linewidth=2, 
             label=f'Weibull (k={shape:.2f}, λ={scale:.2f})')
    
    ax4.set_xlabel(f'Peak Amplitude ({unit})')
    ax4.set_ylabel('Probability Density')
    ax4.grid(True)
    ax4.set_title('Peak Distribution')
    ax4.set_xlim(left=0)
    ax4.legend()
    
    # 显示峰值统计
    text = f'Peak Count: {len(peaks)}\nMean: {np.mean(peak_values):.2f}\nMax: {max(peak_values):.2f}\nMin: {min(peak_values):.2f}'
    ax4.text(0.05, 0.95, text, transform=ax4.transAxes, verticalalignment='top')
    
    # Q-Q图（第三行，右）
    ax5 = fig.add_subplot(gs[2, 1])
    spstats.probplot(peak_values, dist="norm", plot=ax5)
    ax5.set_title('Q-Q Plot of Peaks')
    ax5.grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig