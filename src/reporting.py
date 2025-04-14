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
from openpyxl.chart import LineChart, Reference
import scipy.signal as signal
import copy
from scipy import stats as spstats
from openpyxl.worksheet.page import PageMargins, PrintPageSetup

# Set up logging
logger = logging.getLogger('pydas.reporting')

def channel_report(pydas_obj, output_file='channel_report.xlsx', sseg=0, fullscale=True, 
                  lam=None, rho=1.025, g=9.807, header_text=None, include_charts=False, 
                  significant_percentile=33.0, wave_analysis=True, format_sheet=True, 
                  zerocrossing_analysis=True, amplitude_analysis=True, 
                  n_hr_forecast=3):
    """
    为PyDAS对象的所有通道生成详细的Excel分析报告
    
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
                logger.warning("Scale factor lam not provided and object has no __lam__ attribute. fullscale=True requires a scale factor.")
                return None
        
        # 创建PyDAS对象的深拷贝进行全尺度转换
        logger.info(f"Creating data copy and converting to full scale (λ={lam}, ρ={rho}, g={g})...")
        pydas_analysis = copy.deepcopy(pydas_obj)
        pydas_analysis.to_fullscale(rho=rho, g=g, pInfo=False)
        scale_text = "Full Scale"
    else:
        # 使用原始模型尺度
        pydas_analysis = pydas_obj
        scale_text = "Model Scale"
    
    # 如果提供了标题文本，则使用它，否则使用默认的
    if header_text is None:
        if hasattr(pydas_obj, 'filename'):
            filename = os.path.basename(pydas_obj.filename)
            header_text = f"Wave only [{filename}, {scale_text}]"
        else:
            header_text = f"Wave Analysis Report [{scale_text}]"
    
    # 创建结果DataFrame
    columns = [
        'channel\nnumber', 'title', 'unit', 'number\nof zero\nupcross', 
        'maximum\nvalue', 'minimum\nvalue', 'mean\nvalue', 'standard\ndeviation',
        'maximum\ndouble\namplitude', 'sign.\ndouble\namplitude', 
        'sign.\npositive\nsin. ampl.', 'sign.\nnegative\nsin. ampl.',
        'mean\nzerocro.\nperiod', 'estimated\n3hr\nmaximum', 'estimated\n3hr\nminimum'
    ]
    
    results_df = pd.DataFrame(columns=columns)
    
    # 获取通道信息
    ch_info = pydas_analysis.chInfo
    
    # 计算分析周期
    dt = 1.0 / pydas_analysis.__fs__
    data_duration_seconds = len(pydas_analysis.data[sseg]) * dt
    data_duration_hours = data_duration_seconds / 3600
    
    # 对每个通道进行分析
    for ch_idx, (_, row) in enumerate(ch_info.iterrows(), 1):
        ch_name = row['Name']
        ch_unit = row['Unit']
        
        # 获取数据 (已经是缩放后的数据，如果使用了to_fullscale)
        data_scaled = pydas_analysis.data[sseg][ch_name].values
        
        # 基本统计量计算
        mean_val = np.mean(data_scaled)
        std_val = np.std(data_scaled)
        max_val = np.max(data_scaled)
        min_val = np.min(data_scaled)
        
        # 过零分析
        zero_upcross = 0
        mean_period = 0
        max_double_amp = 0
        sign_double_amp = 0
        sign_pos_amp = 0
        sign_neg_amp = 0
        estimated_max = 0
        estimated_min = 0
        
        # 初始化双振幅数组
        double_amplitudes = []
        
        if zerocrossing_analysis:
            # 计算均值过零点
            mean_level = mean_val
            data_centered = data_scaled - mean_level
            
            # 找出上穿零点
            zero_crossings = np.where(np.diff(np.signbit(data_centered)))[0]
            upcrossings = [i for i in zero_crossings if data_centered[i+1] > data_centered[i]]
            zero_upcross = len(upcrossings)
            
            # 计算过零周期
            if zero_upcross > 1:
                periods = np.diff(upcrossings) * dt
                mean_period = np.mean(periods)
            
            # 计算波峰和波谷 (简化版，使用scipy.signal)
            peaks, _ = signal.find_peaks(data_scaled)
            troughs, _ = signal.find_peaks(-data_scaled)
            
            # 确保有足够的峰值和谷值
            if len(peaks) > 0 and len(troughs) > 0:
                # 排序确保波峰和波谷按时间顺序
                peaks = np.sort(peaks)
                troughs = np.sort(troughs)
                
                # 方法1: 对每个波峰，找到它之前和之后的波谷，选择振幅较大的一个
                for peak_idx in peaks:
                    peak_val = data_scaled[peak_idx]
                    
                    # 找到峰值之前的波谷
                    prev_troughs = troughs[troughs < peak_idx]
                    if len(prev_troughs) > 0:
                        prev_trough_idx = prev_troughs[-1]  # 取最近的前一个波谷
                        prev_trough_val = data_scaled[prev_trough_idx]
                        prev_amp = peak_val - prev_trough_val
                    else:
                        prev_amp = 0
                    
                    # 找到峰值之后的波谷
                    next_troughs = troughs[troughs > peak_idx]
                    if len(next_troughs) > 0:
                        next_trough_idx = next_troughs[0]  # 取最近的后一个波谷
                        next_trough_val = data_scaled[next_trough_idx]
                        next_amp = peak_val - next_trough_val
                    else:
                        next_amp = 0
                    
                    # 选择较大的振幅
                    if prev_amp > 0 and next_amp > 0:
                        double_amp = max(prev_amp, next_amp)
                    else:
                        double_amp = max(prev_amp, next_amp, 0)
                    
                    if double_amp > 0:  # 只添加有效的振幅
                        double_amplitudes.append(double_amp)
                
                # 方法2: 使用零上穿点计算波浪的波峰到波谷振幅
                # 如果已经计算了零上穿点，使用这些点来定义波浪
                if zerocrossing_analysis and 'upcrossings' in locals() and len(upcrossings) > 1:
                    wave_amplitudes = []
                    
                    # 对每两个连续的零上穿点之间的数据
                    for i in range(len(upcrossings) - 1):
                        start_idx = upcrossings[i]
                        end_idx = upcrossings[i+1]
                        
                        # 提取这段波浪数据
                        wave_segment = data_scaled[start_idx:end_idx+1]
                        
                        # 找到这段数据中的最大值和最小值
                        if len(wave_segment) > 2:  # 确保有足够的点
                            wave_max = np.max(wave_segment)
                            wave_min = np.min(wave_segment)
                            wave_amp = wave_max - wave_min
                            wave_amplitudes.append(wave_amp)
                    
                    # 如果波浪法得到的振幅更多，使用波浪法结果
                    if len(wave_amplitudes) > len(double_amplitudes):
                        logger.info(f"Using zero-crossing method for wave amplitude calculation, identified {len(wave_amplitudes)} waves")
                        double_amplitudes = wave_amplitudes
                    elif len(double_amplitudes) > 0:
                        logger.info(f"Using peak-trough matching method for wave amplitude calculation, identified {len(double_amplitudes)} waves")
            
            if len(double_amplitudes) > 0:
                # 最大双振幅
                max_double_amp = np.max(double_amplitudes)
                
                # 显著双振幅（最高三分之一的平均值）
                # 计算显著波高/振幅 (H1/3 or Hs)
                # 标准定义：将所有波高按从大到小排序，取前1/3的平均值
                # 注意：significant_percentile应为33.33(%)以符合传统定义
                n_waves = len(double_amplitudes)
                if n_waves > 0:
                    # 从大到小排序所有双振幅
                    sorted_amps = np.sort(double_amplitudes)[::-1]
                    
                    # 计算要取平均的波浪数量
                    n_significant = max(1, int(n_waves * significant_percentile / 100))
                    
                    # 计算显著双振幅 (前n_significant个最大值的平均值)
                    sign_double_amp = np.mean(sorted_amps[:n_significant])
                    
                    # 记录使用了多少波浪计算显著值
                    logger.info(f"Calculated significant double amplitude from {n_significant} highest waves out of {n_waves} total waves (top {significant_percentile:.1f}%)")
                    
                    # 显著正负单向振幅 - 完全重写这部分计算
                    if amplitude_analysis:
                        # 计算显著单向振幅的几种方法：
                        # 1. 直接取显著双振幅的一半（默认方法，适用于对称波）
                        # 2. 使用所有波峰/波谷的统计（适用于有足够数据点的情况）
                        
                        # 首先尝试使用波峰和波谷的统计
                        all_peaks = []
                        all_troughs = []
                        
                        # 获取所有有效的波峰和波谷值
                        if len(peaks) > 0 and len(troughs) > 0:
                            # 波峰值（相对于均值的正偏差）
                            for p in peaks:
                                peak_val = data_scaled[p]
                                peak_deviation = peak_val - mean_val
                                if peak_deviation > 0:  # 只考虑高于均值的波峰
                                    all_peaks.append(peak_deviation)
                            
                            # 波谷值（相对于均值的负偏差的绝对值）
                            for t in troughs:
                                trough_val = data_scaled[t]
                                trough_deviation = mean_val - trough_val
                                if trough_deviation > 0:  # 只考虑低于均值的波谷
                                    all_troughs.append(trough_deviation)
                        
                        # 根据数据点数量决定使用哪种方法
                        if len(all_peaks) >= n_significant and len(all_troughs) >= n_significant:
                            # 方法2：有足够的数据点，使用波峰/波谷统计
                            # 对波峰偏差值进行排序并取最高三分之一
                            sorted_peaks = np.sort(all_peaks)[::-1]
                            sign_pos_amp = np.mean(sorted_peaks[:n_significant])
                            
                            # 对波谷偏差值进行排序并取最高三分之一
                            sorted_troughs = np.sort(all_troughs)[::-1]
                            sign_neg_amp = np.mean(sorted_troughs[:n_significant])
                            
                            logger.info(f"Using peak/trough statistics for significant single amplitudes (n_sig={n_significant})")
                        else:
                            # 方法1：数据点不足，使用显著双振幅推导
                            # 使用标准差作为参考，估计波形的非对称性
                            asymmetry = 0.0
                            
                            # 如果有足够的波峰和波谷，计算不对称性
                            if len(all_peaks) > 0 and len(all_troughs) > 0:
                                mean_peak = np.mean(all_peaks)
                                mean_trough = np.mean(all_troughs)
                                total_amp = mean_peak + mean_trough
                                if total_amp > 0:
                                    # 计算不对称比 (-1到1之间)
                                    asymmetry = (mean_peak - mean_trough) / total_amp
                            
                            # 基于标准差和不对称性分配显著双振幅
                            # 对于完全对称波，pos_amp = neg_amp = 0.5 * sign_double_amp
                            base_amp = 0.5 * sign_double_amp
                            asymmetry_factor = min(max(asymmetry, -0.5), 0.5)  # 限制在±0.5范围内
                            
                            sign_pos_amp = base_amp * (1 + asymmetry_factor)
                            sign_neg_amp = base_amp * (1 - asymmetry_factor)
                            
                            logger.info(f"Using significant double amplitude and asymmetry factor ({asymmetry:.3f}) to estimate single amplitudes")
                        
                        # 确保显著振幅不小于标准差（合理性检查）
                        min_amp = std_val * 1.5  # 使用标准差的1.5倍作为最小值
                        if sign_pos_amp < min_amp:
                            sign_pos_amp = min_amp
                            logger.info(f"Adjusted positive significant amplitude to minimum value ({min_amp:.3f})")
                        
                        if sign_neg_amp < min_amp:
                            sign_neg_amp = min_amp
                            logger.info(f"Adjusted negative significant amplitude to minimum value ({min_amp:.3f})")
                        
                        # 验证结果的合理性
                        total_single_amp = sign_pos_amp + sign_neg_amp
                        amp_ratio = sign_double_amp / total_single_amp if total_single_amp > 0 else 0
                        
                        logger.info(f"Significant amplitudes - double: {sign_double_amp:.3f}, positive: {sign_pos_amp:.3f}, negative: {sign_neg_amp:.3f}, ratio: {amp_ratio:.3f}")
                        
                        # 如果比例明显偏离1.0，进行调整
                        if amp_ratio > 0 and (amp_ratio < 0.8 or amp_ratio > 1.2):
                            # 调整单向振幅使比例接近1.0，同时保持不对称性
                            adjustment = sign_double_amp / total_single_amp
                            sign_pos_amp *= adjustment
                            sign_neg_amp *= adjustment
                            logger.info(f"Adjusted single amplitudes to match double amplitude, new values - positive: {sign_pos_amp:.3f}, negative: {sign_neg_amp:.3f}")
        
        if n_hr_forecast > 0:
            # 使用极值统计方法进行极值估计
            try:
                # 如果数据长度接近3小时（测试数据通常为3小时）
                actual_duration_hours = data_duration_hours
                
                if abs(actual_duration_hours - n_hr_forecast) < 0.5:
                    # 数据长度已经接近3小时，使用直接方法
                    logger.info(f"Data duration ({actual_duration_hours:.2f} hours) is close to forecast duration ({n_hr_forecast} hours), using direct method for extreme value estimation")
                    # 使用数据的直接最大/最小值作为基础，加上小幅调整
                    estimated_max = max_val * 1.05  # 增加5%作为保守估计
                    estimated_min = min_val * 1.05 if min_val < 0 else min_val * 0.95  # 负值增加5%，正值减少5%
                else:
                    # 首先去除均值，以便更准确地进行极值分析
                    data_centered = data_scaled - mean_val
                    
                    # 如果数据有足够的峰值，采用峰值统计的方法
                    if zerocrossing_analysis and 'peaks' in locals() and len(peaks) > 10:
                        # 重新计算波峰和波谷 (对去均值后的数据)
                        centered_peaks, _ = signal.find_peaks(data_centered)
                        centered_troughs, _ = signal.find_peaks(-data_centered)
                        
                        if len(centered_peaks) > 0 and len(centered_troughs) > 0:
                            # 提取所有正峰值
                            peak_values = data_centered[centered_peaks]
                            # 提取所有负谷值 
                            trough_values = data_centered[centered_troughs]
                            
                            # 构造Weibull分布函数估计极值
                            # 对正峰值进行估计
                            if len(peak_values) > 0:
                                # 按降序排列峰值
                                sorted_peaks = np.sort(peak_values)[::-1]
                                # 估计正极值 - 使用Weibull分布
                                # 计算每小时的峰值数量
                                peaks_per_hour = len(peak_values) / actual_duration_hours
                                # 3小时预期峰值数量
                                expected_peaks_in_3hr = peaks_per_hour * n_hr_forecast
                                
                                # 使用前10%的峰值拟合分布
                                num_fitting_peaks = max(10, int(len(sorted_peaks) * 0.1))
                                fitting_peaks = sorted_peaks[:num_fitting_peaks]
                                
                                # 拟合Weibull分布
                                try:
                                    c, loc, scale = spstats.weibull_min.fit(fitting_peaks, floc=0)
                                    # 计算3小时内可能的最大峰值
                                    p = 1 - 1/expected_peaks_in_3hr
                                    peak_extreme = spstats.weibull_min.ppf(p, c, loc, scale)
                                    # 加回均值得到最终极值
                                    estimated_max = peak_extreme + mean_val
                                    logger.info(f"Using Weibull distribution method to estimate positive extreme value: {estimated_max:.3f}")
                                except:
                                    # 拟合失败时使用最大峰值加上调整
                                    estimated_max = sorted_peaks[0] * 1.1 + mean_val
                                    logger.warning(f"Weibull fitting failed, using simplified method to estimate positive extreme value: {estimated_max:.3f}")
                            
                            # 对负谷值进行估计
                            if len(trough_values) > 0:
                                # 按升序排列谷值（找最小的负值）
                                sorted_troughs = np.sort(trough_values)
                                # 估计负极值 - 使用Weibull分布
                                # 计算每小时的谷值数量
                                troughs_per_hour = len(trough_values) / actual_duration_hours
                                # 3小时预期谷值数量
                                expected_troughs_in_3hr = troughs_per_hour * n_hr_forecast
                                
                                # 使用前10%的谷值拟合分布
                                num_fitting_troughs = max(10, int(len(sorted_troughs) * 0.1))
                                fitting_troughs = -sorted_troughs[:num_fitting_troughs]  # 取反使得最负的值变为最大的正值
                                
                                # 拟合Weibull分布
                                try:
                                    c, loc, scale = spstats.weibull_min.fit(fitting_troughs, floc=0)
                                    # 计算3小时内可能的最小谷值
                                    p = 1 - 1/expected_troughs_in_3hr
                                    trough_extreme = -spstats.weibull_min.ppf(p, c, loc, scale)  # 取反转回负值
                                    # 加回均值得到最终极值
                                    estimated_min = trough_extreme + mean_val
                                    logger.info(f"Using Weibull distribution method to estimate negative extreme value: {estimated_min:.3f}")
                                except:
                                    # 拟合失败时使用最小谷值加上调整
                                    estimated_min = sorted_troughs[0] * 1.1 + mean_val
                                    logger.warning(f"Weibull fitting failed, using simplified method to estimate negative extreme value: {estimated_min:.3f}")
                        else:
                            # 如果找不到足够的峰谷，使用正态极值估计
                            logger.info("Insufficient peaks/troughs in centered data, using normal distribution method")
                            # 基于观测时长和预测时长的比例进行缩放
                            forecast_factor = np.sqrt(n_hr_forecast / actual_duration_hours)
                            # 使用极值因子，基于3-sigma规则但更为保守
                            extreme_factor = 3.5 + 0.5 * np.log(n_hr_forecast / actual_duration_hours)
                            extreme_std = std_val * forecast_factor
                            estimated_max = mean_val + extreme_factor * extreme_std
                            estimated_min = mean_val - extreme_factor * extreme_std
                    else:
                        # 如果无法使用峰值统计，使用正态极值估计方法
                        # 基于观测时长和预测时长的比例进行缩放
                        logger.info("Using normal extreme value estimation method (observation duration less than 3 hours)")
                        forecast_factor = np.sqrt(n_hr_forecast / actual_duration_hours)
                        
                        # 使用极值因子，基于3-sigma规则但更为保守
                        extreme_factor = 3.5 + 0.5 * np.log(n_hr_forecast / actual_duration_hours)
                        extreme_std = std_val * forecast_factor
                        estimated_max = mean_val + extreme_factor * extreme_std
                        estimated_min = mean_val - extreme_factor * extreme_std
            except Exception as e:
                # 出错时使用原始简单方法
                logger.warning(f"Error in extreme value estimation: {str(e)}, using simplified method")
                forecast_factor = np.sqrt(n_hr_forecast / data_duration_hours) if data_duration_hours > 0 else 1.0
                extreme_std = std_val * forecast_factor
                estimated_max = mean_val + 3.5 * extreme_std  # 3.5是一个经验系数
                estimated_min = mean_val - 3.5 * extreme_std
        
        # 将结果添加到DataFrame
        results_df.loc[ch_idx] = [
            ch_idx, ch_name, ch_unit, zero_upcross, 
            max_val, min_val, mean_val, std_val,
            max_double_amp, sign_double_amp, 
            sign_pos_amp, sign_neg_amp,
            mean_period, estimated_max, estimated_min
        ]
    
    # 创建Excel文件
    if output_file:
        try:
            # 使用pandas将DataFrame导出到Excel
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # 写入主数据
                results_df.to_excel(writer, sheet_name='Channel Statistics', index=False)
                
                # 如果需要格式化
                if format_sheet:
                    wb = writer.book
                    ws = writer.sheets['Channel Statistics']
                    
                    # 添加标题行
                    ws.insert_rows(0, 2)
                    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(columns))
                    title_cell = ws.cell(row=1, column=1, value=header_text)
                    
                    # 设置标题行格式
                    title_cell.font = Font(bold=True, size=14)
                    title_cell.alignment = Alignment(horizontal='center', vertical='center')
                    
                    # 设置列宽
                    for i, column in enumerate(columns, 1):
                        col_width = max(len(str(c)) for c in results_df[column].astype(str)) * 1.2
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
                            elif col in [2, 3]:  # 名称、单位
                                cell.alignment = Alignment(horizontal='left')
                            else:  # 数值
                                cell.alignment = Alignment(horizontal='right')
                                
                                # 格式化数值
                                if isinstance(cell.value, (int, float)) and col >= 4:
                                    if abs(cell.value) < 0.001 and cell.value != 0:
                                        # 科学计数法
                                        cell.value = f"{cell.value:.4E}"
                                    else:
                                        # 普通数字，3位小数
                                        cell.value = f"{cell.value:.3f}"
                    
                    # 设置标题行格式
                    header_row = 3  # 标题所在行
                    header_fill = PatternFill(start_color='E9E9E9', end_color='E9E9E9', fill_type='solid')
                    
                    for col in range(1, ws.max_column + 1):
                        cell = ws.cell(row=header_row, column=col)
                        cell.font = Font(bold=True, size=10)
                        cell.fill = header_fill
                        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
                        cell.border = thick_border
                    
                    # 设置页面布局为横向A4并调整表格宽度适应打印
                    # 设置页面方向为横向，纸张大小为A4
                    ws.page_setup.orientation = 'landscape'
                    ws.page_setup.paperSize = 9  # A4纸张大小
                    
                    # 设置合适的页边距（单位：英寸）
                    ws.page_margins = PageMargins(left=0.5, right=0.5, top=0.5, bottom=0.5)
                    
                    # 设置自动调整为一页宽
                    ws.page_setup.fitToWidth = 1
                    ws.page_setup.fitToHeight = 0  # 0表示根据内容自动确定页数
                    
                    # 设置打印区域
                    ws.print_area = f'A1:{get_column_letter(ws.max_column)}{ws.max_row}'
                    
                    # 设置打印标题行
                    ws.print_title_rows = '1:3'  # 重复打印前3行作为标题
                    
                    # 调整列宽 - 使用更大的默认宽度
                    # 为不同类型的列设置合适的宽度
                    col_width_map = {
                        1: 6,   # 通道编号
                        2: 20,  # 通道名称
                        3: 8    # 单位
                    }
                    
                    # 设置默认宽度
                    default_width = 12  # 数值列的默认宽度
                    
                    # 特殊列的宽度（例如较长的标题列）
                    special_widths = {
                        4: 10,  # 零上穿数
                        9: 15,  # 最大双振幅
                        10: 15, # 显著双振幅
                        11: 15, # 显著正向振幅
                        12: 15, # 显著负向振幅
                        13: 12, # 平均零上穿周期
                        14: 14, # 预估3小时最大值
                        15: 14  # 预估3小时最小值
                    }
                    
                    # 应用列宽设置
                    for i in range(1, ws.max_column + 1):
                        col_letter = get_column_letter(i)
                        if i in col_width_map:
                            ws.column_dimensions[col_letter].width = col_width_map[i]
                        elif i in special_widths:
                            ws.column_dimensions[col_letter].width = special_widths[i]
                        else:
                            ws.column_dimensions[col_letter].width = default_width
                        
                        # 启用最佳宽度适应
                        ws.column_dimensions[col_letter].bestFit = True
                    
                    # 设置调整到单页宽度但同时保持可读性
                    ws.sheet_properties.pageSetUpPr.fitToPage = True
                
                # 如果需要包含图表（已禁用）
                # 用户反馈不需要时间序列图表页
                '''
                if include_charts:
                    # 创建一个新的工作表用于图表
                    wb.create_sheet("Charts")
                    charts_ws = wb["Charts"]
                    
                    # 对于每个通道创建一个时间序列图表
                    chart_row = 1
                    chart_col = 1
                    max_charts_per_row = 2
                    
                    for ch_idx, (_, row) in enumerate(ch_info.iterrows(), 1):
                        ch_name = row['Name']
                        
                        # 提取时间和数据
                        time_data = np.arange(0, len(pydas_analysis.data[sseg])) / pydas_analysis.__fs__
                        channel_data_scaled = pydas_analysis.data[sseg][ch_name].values
                        
                        # 对于大数据集，进行下采样
                        max_points = 1000
                        if len(time_data) > max_points:
                            step = len(time_data) // max_points
                            time_data = time_data[::step]
                            channel_data_scaled = channel_data_scaled[::step]
                        
                        # 创建一个新的数据表用于图表
                        chart_data_start_row = charts_ws.max_row + 2
                        
                        # 写入标题
                        charts_ws.cell(row=chart_data_start_row, column=1, value='Time (s)')
                        charts_ws.cell(row=chart_data_start_row, column=2, value=ch_name)
                        
                        # 写入数据
                        for i, (t, val) in enumerate(zip(time_data, channel_data_scaled), 1):
                            charts_ws.cell(row=chart_data_start_row + i, column=1, value=t)
                            charts_ws.cell(row=chart_data_start_row + i, column=2, value=val)
                        
                        # 创建图表
                        chart = LineChart()
                        chart.title = f"{ch_name} Time Series"
                        chart.x_axis.title = "Time (s)"
                        chart.y_axis.title = ch_unit if ch_unit else "Value"
                        
                        # 设置数据范围
                        data = Reference(
                            charts_ws, 
                            min_col=2, 
                            min_row=chart_data_start_row, 
                            max_row=chart_data_start_row + len(time_data)
                        )
                        
                        cats = Reference(
                            charts_ws, 
                            min_col=1, 
                            min_row=chart_data_start_row + 1, 
                            max_row=chart_data_start_row + len(time_data)
                        )
                        
                        chart.add_data(data, titles_from_data=True)
                        chart.set_categories(cats)
                        
                        # 设置图表样式
                        s1 = chart.series[0]
                        s1.graphicalProperties.line.width = 20000  # 增加线条粗细
                        s1.graphicalProperties.line.solidFill = "4472C4"  # 蓝色
                        
                        # 添加图表到工作表
                        chart_pos = f"{get_column_letter(chart_col)}{chart_row}"
                        charts_ws.add_chart(chart, chart_pos)
                        
                        # 更新图表位置
                        chart_col += 1
                        if chart_col > max_charts_per_row:
                            chart_col = 1
                            chart_row += 15  # 每个图表的高度
                '''
            
            logger.info(f"Report successfully exported to {output_file}")
        
        except Exception as e:
            logger.error(f"Error exporting Excel file: {str(e)}")
    
    # 返回结果DataFrame
    return results_df 