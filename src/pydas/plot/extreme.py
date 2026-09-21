"""
PyDAS Plot - Extreme Value Analysis
====================================
Contains peak detection and extreme value analysis visualization functions.
"""

import logging
import numpy as np
import os

from . import PLOT_CONFIG, get_plot_backend, apply_style

logger = logging.getLogger('pydas.plot.extreme')


# 添加峰值检测函数

def _detect_peaks(data, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5):
    """
    A wrapper for scipy.signal.find_peaks to detect peaks in data
    
    Parameters:
    -----------
    data : numpy.ndarray
        The data to detect peaks in
    height : float or None, optional
        Required height of peaks
    threshold : float or None, optional
        Required threshold of peaks
    distance : int or None, optional
        Required minimum horizontal distance between neighboring peaks
    prominence : float or None, optional
        Required prominence of peaks
    width : float or None, optional
        Required width of peaks
    wlen : int or None, optional
        Use at most this many samples in prominence computation
    rel_height : float, optional
        Used to calculate peak width as percentage of its prominence
        
    Returns:
    --------
    tuple
        (peaks, properties) where peaks is indices of peaks and properties is a dict
        with properties of the peaks
    """
    from scipy.signal import find_peaks
    
    return find_peaks(data, height=height, threshold=threshold, distance=distance,
                      prominence=prominence, width=width, wlen=wlen, rel_height=rel_height)

def plot_extreme_analysis(results, visualization_backend='matplotlib', save_path=None, save_html=None, 
                         visualization=True, title=None, ch_name=None, pydas_obj=None, bins=30,
                         fullscale=True, return_periods=None, return_period_labels=None, unit=None):
    """
    可视化极值分析结果
    
    Parameters
    ----------
    results : dict
        极值分析结果字典，包含以下键：
        - 'peaks_positive': 正峰值
        - 'peaks_negative': 负峰值
        - 'all_peaks': 所有峰值（绝对值）
        - 'peak_indices': 峰值索引 {'positive': pos_indices, 'negative': neg_indices}
        - 'duration_seconds': 数据时长（秒）
        - 'exceedance_table': 超越概率表
        - 'extreme_value_model': 极值模型参数
        - 'return_values': 回归值
        - 'return_value_confidence_intervals': 回归值置信区间
    visualization_backend : str, default='matplotlib'
        可视化后端 ('matplotlib' 或 'plotly')
    save_path : str, optional
        图表保存路径（用于matplotlib）
    save_html : str, optional
        交互式图表保存路径（用于plotly）
    visualization : bool, default=True
        是否显示可视化结果
    title : str, optional
        图表标题
    ch_name : str, optional
        通道名称
    pydas_obj : PyDAS object, optional
        PyDAS对象，用于获取通道信息和数据
    bins : int, default=30
        直方图的箱数
    fullscale : bool, default=True
        是否使用原型尺度
    return_periods : array-like, optional
        回归周期
    return_period_labels : list, optional
        回归周期标签
    unit : str, optional
        数据单位
        
    Returns
    -------
    object
        matplotlib.figure.Figure 或 plotly.graph_objects.Figure 对象
    """
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    
    # 检查结果字典是否包含必要的键
    required_keys = ['peaks_positive', 'peaks_negative', 'all_peaks', 'peak_indices',
                    'duration_seconds', 'exceedance_table']
    for key in required_keys:
        if key not in results:
            logger.error(f"结果字典缺少必要的键: {key}")
            return None
    
    # 获取数据
    peaks_positive = results['peaks_positive']
    peaks_negative = results['peaks_negative']
    all_peaks = results['all_peaks']
    pos_peaks_idx = results['peak_indices']['positive']
    neg_peaks_idx = results['peak_indices']['negative']
    exceedance = results['exceedance_table']
    
    # 从pydas_obj获取其他必要信息
    data_array = None
    fs = 1.0
    
    if pydas_obj is not None:
        # 尝试获取通道单位
        if unit is None and ch_name is not None:
            channel_info = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == ch_name]
            unit = "" if channel_info.empty else channel_info['Unit'].values[0]
        # 获取采样率
        if hasattr(pydas_obj, '__fs__'):
            fs = pydas_obj.__fs__
        # 获取全部数据
        if ch_name is not None and hasattr(pydas_obj, 'data') and len(pydas_obj.data) > 0:
            if ch_name in pydas_obj.data[0].columns:
                # 使用首个段中的数据
                data_array = pydas_obj.data[0][ch_name].values
    
    # 如果需要使用回归周期但未提供，则使用结果中的
    if return_periods is None and 'return_periods' in results:
        return_periods = results['return_periods']['periods']
        return_period_labels = results['return_periods']['labels']
    
    # 设置单位字符串
    unit_str = f" [{unit}]" if unit else ""
    
    # 根据后端创建可视化
    backend = visualization_backend.lower()
    
    # Plotly后端
    if backend == 'plotly':
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # 创建2x2子图
            fig = make_subplots(rows=2, cols=2, 
                                subplot_titles=("Original Data with Detected Peaks", 
                                              "Peak Value Histogram", 
                                              "Empirical Exceedance Probability", 
                                              "Return Period Plot"),
                                specs=[[{}, {}], 
                                      [{}, {}]])
            
            # 图1：原始数据和检测到的峰值
            if data_array is not None:
                time = np.arange(len(data_array)) / fs
                
                # 对大数据集进行下采样
                if len(data_array) > 50000:
                    step = len(data_array) // 50000 + 1
                    plot_time = time[::step]
                    plot_data = data_array[::step]
                else:
                    plot_time = time
                    plot_data = data_array
                
                # 添加原始数据
                fig.add_trace(
                    go.Scatter(x=plot_time, y=plot_data, 
                             mode='lines', name='Original Data',
                             line=dict(color='rgba(0,0,255,0.5)', width=1)),
                    row=1, col=1
                )
            
            # 添加正峰值
            if len(pos_peaks_idx) > 0:
                pos_peak_times = pos_peaks_idx / fs
                
                fig.add_trace(
                    go.Scatter(x=pos_peak_times, y=peaks_positive, 
                             mode='markers', name='Positive Peaks',
                             marker=dict(color='red', size=8, symbol='circle')),
                    row=1, col=1
                )
            
            # 添加负峰值
            if len(neg_peaks_idx) > 0:
                neg_peak_times = neg_peaks_idx / fs
                
                fig.add_trace(
                    go.Scatter(x=neg_peak_times, y=peaks_negative, 
                             mode='markers', name='Negative Peaks',
                             marker=dict(color='green', size=8, symbol='circle')),
                    row=1, col=1
                )
            
            # 图2：峰值直方图
            if len(all_peaks) > 0:
                # 创建直方图
                fig.add_trace(
                    go.Histogram(x=all_peaks, nbinsx=bins, 
                               name='Peak Histogram',
                               marker=dict(color='rgba(0,0,255,0.7)')),
                    row=1, col=2
                )
                
                # 添加极值分布拟合曲线
                if 'extreme_value_model' in results:
                    model = results['extreme_value_model']
                    x = np.linspace(min(all_peaks), max(all_peaks), 100)
                    
                    # 为已知分布类型绘制曲线
                    if model['distribution'] == 'GEV':
                        shape = model['shape']
                        loc = model['loc']
                        scale = model['scale']
                        y = stats.genextreme.pdf(x, shape, loc, scale)
                        distrib_name = f"GEV (ξ={shape:.3f}, μ={loc:.3f}, σ={scale:.3f})"
                        
                        # 缩放PDF以匹配直方图比例
                        bin_width = (max(all_peaks) - min(all_peaks)) / bins
                        y = y * len(all_peaks) * bin_width
                        
                        # 添加分布曲线
                        fig.add_trace(
                            go.Scatter(x=x, y=y, mode='lines', name=distrib_name,
                                     line=dict(color='red', width=2)),
                            row=1, col=2
                        )
                    elif model['distribution'] == 'Gumbel':
                        loc = model['loc']
                        scale = model['scale']
                        y = stats.gumbel_r.pdf(x, loc, scale)
                        distrib_name = f"Gumbel (μ={loc:.3f}, σ={scale:.3f})"
                        
                        # 缩放PDF以匹配直方图比例
                        bin_width = (max(all_peaks) - min(all_peaks)) / bins
                        y = y * len(all_peaks) * bin_width
                        
                        # 添加分布曲线
                        fig.add_trace(
                            go.Scatter(x=x, y=y, mode='lines', name=distrib_name,
                                     line=dict(color='red', width=2)),
                            row=1, col=2
                        )
            
            # 图3：经验超越概率
            fig.add_trace(
                go.Scatter(x=exceedance['Exceedance Probability'], 
                         y=exceedance['Peak Value'],
                         mode='markers', name='Empirical Exceedance',
                         marker=dict(color='blue', size=8)),
                row=2, col=1
            )
            
            # 添加极值分布拟合曲线
            if 'extreme_value_model' in results:
                model = results['extreme_value_model']
                x = np.logspace(-3, np.log10(0.9), 100)  # 0.001到0.9的概率
                
                if model['distribution'] == 'GEV':
                    shape = model['shape']
                    loc = model['loc']
                    scale = model['scale']
                    y = stats.genextreme.ppf(1-x, shape, loc, scale)
                    line_name = 'GEV Model'
                elif model['distribution'] == 'Gumbel':
                    loc = model['loc']
                    scale = model['scale']
                    y = stats.gumbel_r.ppf(1-x, loc, scale)
                    line_name = 'Gumbel Model'
                else:
                    line_name = 'Fitted Model'
                    
                # 添加分布曲线
                fig.add_trace(
                    go.Scatter(x=x, y=y, mode='lines', name=line_name,
                             line=dict(color='red', width=2)),
                    row=2, col=1
                )
                
                # 设置x轴对数刻度
                fig.update_xaxes(type='log', row=2, col=1)
            
            # 图4：回归周期图
            # 转换为年单位用于绘图
            return_period_years_data = exceedance['Return Period (hours)'] / (24 * 365.25)
            
            fig.add_trace(
                go.Scatter(x=return_period_years_data, 
                         y=exceedance['Peak Value'],
                         mode='markers', name='Empirical Return Period',
                         marker=dict(color='blue', size=8)),
                row=2, col=2
            )
            
            # 添加理论回归周期和回归值
            if 'extreme_value_model' in results and 'return_values' in results and return_periods is not None:
                # 绘制理论回归周期
                rps = np.array(return_periods) / (24 * 365.25)  # 转换为年
                rv_list = [results['return_values'][label] for label in return_period_labels]
                
                fig.add_trace(
                    go.Scatter(x=rps, y=rv_list, mode='lines+markers', 
                             name='Model Return Values',
                             line=dict(color='red', width=2),
                             marker=dict(color='red', size=10)),
                    row=2, col=2
                )
                
                # 添加置信区间
                if 'return_value_confidence_intervals' in results:
                    # 为关注的回归周期添加点
                    last_label = return_period_labels[-1]
                    
                    if last_label in results['return_value_confidence_intervals']:
                        ci = results['return_value_confidence_intervals'][last_label]
                        ci_lower = ci[0] if isinstance(ci, tuple) else ci.get('lower_95', 0)
                        ci_upper = ci[1] if isinstance(ci, tuple) else ci.get('upper_95', 0)
                        
                        # 添加CI信息到图表
                        fig.add_trace(
                            go.Scatter(x=[rps[-1]], 
                                     y=[results['return_values'][last_label]],
                                     error_y=dict(
                                         type='data',
                                         symmetric=False,
                                         array=[ci_upper - results['return_values'][last_label]],
                                         arrayminus=[results['return_values'][last_label] - ci_lower],
                                         visible=True,
                                         color='red',
                                         width=3
                                     ),
                                     mode='markers',
                                     name=f'{last_label} (95% CI)',
                                     marker=dict(color='darkred', size=12, symbol='diamond')),
                            row=2, col=2
                        )
            
            # 创建图表标题
            if title is None:
                if ch_name is not None:
                    scale_str = "Full Scale" if fullscale else "Model Scale"
                    title = f"Extreme Value Analysis for {ch_name}{unit_str} ({scale_str})"
                else:
                    title = "Extreme Value Analysis"
            
            # 更新布局
            fig.update_layout(
                title=title,
                width=1300,  # 增加宽度为图例留出空间
                height=900,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.2),
                margin=dict(r=150)  # 增加右侧边距为图例腾出空间
            )
            
            # 更新坐标轴标签
            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig.update_yaxes(title_text=f"Value{unit_str}", row=1, col=1)
            
            fig.update_xaxes(title_text="Peak Value", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            
            fig.update_xaxes(title_text="Exceedance Probability", row=2, col=1)
            fig.update_yaxes(title_text=f"Peak Value{unit_str}", row=2, col=1)
            
            fig.update_xaxes(title_text="Return Period (years)", row=2, col=2)
            fig.update_yaxes(title_text=f"Peak Value{unit_str}", row=2, col=2)
            
            # 保存或显示图表
            if save_html is not None:
                fig.write_html(save_html)
                logger.info(f"Interactive plot saved to {save_html}")
            
            if save_path is not None:
                fig.write_image(save_path)
                logger.info(f"Plot saved to {save_path}")
            
            if visualization:
                fig.show()
            
            return fig
        
        except ImportError:
            logger.warning("Plotly不可用，回退到matplotlib")
            backend = 'matplotlib'
        except Exception as e:
            logger.error(f"创建Plotly可视化时出错: {str(e)}")
            backend = 'matplotlib'
    
    # Matplotlib后端
    if backend in ['matplotlib', 'seaborn']:
        try:
            import matplotlib.pyplot as plt
            
            # 创建2x2子图
            fig, axs = plt.subplots(2, 2, figsize=(15, 12))
            
            # 图1：原始数据和检测到的峰值
            if data_array is not None:
                time = np.arange(len(data_array)) / fs
                
                # 大数据集下采样
                if len(data_array) > 10000:
                    step = len(data_array) // 10000 + 1
                    plot_time = time[::step]
                    plot_data = data_array[::step]
                else:
                    plot_time = time
                    plot_data = data_array
                
                # 绘制数据
                axs[0, 0].plot(plot_time, plot_data, 'b-', alpha=0.5, linewidth=1, label='Data')
            
            # 添加正峰值
            if len(pos_peaks_idx) > 0:
                pos_peak_times = pos_peaks_idx / fs
                axs[0, 0].plot(pos_peak_times, peaks_positive, 'ro', label='Positive Peaks')
            
            # 添加负峰值
            if len(neg_peaks_idx) > 0:
                neg_peak_times = neg_peaks_idx / fs
                axs[0, 0].plot(neg_peak_times, peaks_negative, 'go', label='Negative Peaks')
            
            axs[0, 0].set_title('Original Data with Detected Peaks')
            axs[0, 0].set_xlabel('Time (s)')
            axs[0, 0].set_ylabel(f'Value{unit_str}')
            axs[0, 0].legend()
            
            # 图2：峰值直方图
            if len(all_peaks) > 0:
                axs[0, 1].hist(all_peaks, bins=bins, alpha=0.7, color='blue', label='Peaks')
                
                # 添加极值分布拟合曲线
                if 'extreme_value_model' in results:
                    model = results['extreme_value_model']
                    x = np.linspace(min(all_peaks), max(all_peaks), 100)
                    
                    if model['distribution'] == 'GEV':
                        shape = model['shape']
                        loc = model['loc']
                        scale = model['scale']
                        y = stats.genextreme.pdf(x, shape, loc, scale)
                        distrib_name = f"GEV (ξ={shape:.3f}, μ={loc:.3f}, σ={scale:.3f})"
                        
                        # 缩放PDF以匹配直方图比例
                        bin_width = (max(all_peaks) - min(all_peaks)) / bins
                        y = y * len(all_peaks) * bin_width
                        
                        # 添加分布曲线
                        axs[0, 1].plot(x, y, 'r-', linewidth=2, label=distrib_name)
                        axs[0, 1].legend()
                    elif model['distribution'] == 'Gumbel':
                        loc = model['loc']
                        scale = model['scale']
                        y = stats.gumbel_r.pdf(x, loc, scale)
                        distrib_name = f"Gumbel (μ={loc:.3f}, σ={scale:.3f})"
                        
                        # 缩放PDF以匹配直方图比例
                        bin_width = (max(all_peaks) - min(all_peaks)) / bins
                        y = y * len(all_peaks) * bin_width
                        
                        # 添加分布曲线
                        axs[0, 1].plot(x, y, 'r-', linewidth=2, label=distrib_name)
                        axs[0, 1].legend()
            
            axs[0, 1].set_title('Peak Value Histogram')
            axs[0, 1].set_xlabel('Peak Value')
            axs[0, 1].set_ylabel('Count')
            
            # 图3：经验超越概率
            axs[1, 0].loglog(exceedance['Exceedance Probability'], 
                          exceedance['Peak Value'], 'bo', markersize=6,
                          label='Empirical Exceedance')
            
            # 添加极值分布拟合曲线
            if 'extreme_value_model' in results:
                model = results['extreme_value_model']
                x = np.logspace(-3, np.log10(0.9), 100)  # 0.001到0.9的概率
                
                if model['distribution'] == 'GEV':
                    shape = model['shape']
                    loc = model['loc']
                    scale = model['scale']
                    y = stats.genextreme.ppf(1-x, shape, loc, scale)
                    line_name = 'GEV Model'
                elif model['distribution'] == 'Gumbel':
                    loc = model['loc']
                    scale = model['scale']
                    y = stats.gumbel_r.ppf(1-x, loc, scale)
                    line_name = 'Gumbel Model'
                else:
                    line_name = 'Fitted Model'
                    
                axs[1, 0].loglog(x, y, 'r-', linewidth=2, label=line_name)
                axs[1, 0].legend()
            
            axs[1, 0].set_title('Empirical Exceedance Probability')
            axs[1, 0].set_xlabel('Exceedance Probability')
            axs[1, 0].set_ylabel(f'Peak Value{unit_str}')
            axs[1, 0].grid(True, which='both', ls='-', alpha=0.3)
            
            # 图4：回归周期图
            # 转换为年单位用于绘图
            return_period_years_data = exceedance['Return Period (hours)'] / (24 * 365.25)
            
            axs[1, 1].loglog(return_period_years_data, exceedance['Peak Value'], 'bo', 
                          markersize=6, label='Empirical Return Period')
            
            # 添加理论回归周期和回归值
            if 'extreme_value_model' in results and 'return_values' in results and return_periods is not None:
                # 转换为年单位
                rps = np.array(return_periods) / (24 * 365.25)
                rv_list = [results['return_values'][label] for label in return_period_labels]
                
                axs[1, 1].loglog(rps, rv_list, 'ro-', linewidth=2, markersize=8,
                             label='Model Return Values')
                
                # 添加置信区间
                if 'return_value_confidence_intervals' in results:
                    last_label = return_period_labels[-1]
                    
                    if last_label in results['return_value_confidence_intervals']:
                        ci = results['return_value_confidence_intervals'][last_label]
                        ci_lower = ci[0] if isinstance(ci, tuple) else ci.get('lower_95', 0)
                        ci_upper = ci[1] if isinstance(ci, tuple) else ci.get('upper_95', 0)
                        
                        # 添加CI到图表
                        rv_value = results['return_values'][last_label]
                        axs[1, 1].errorbar(rps[-1], rv_value,
                                        yerr=[[rv_value - ci_lower], 
                                              [ci_upper - rv_value]],
                                        fmt='rD', markersize=10, capsize=8, linewidth=2,
                                        label=f'{last_label} (95% CI)')
            
            axs[1, 1].set_title('Return Period Plot')
            axs[1, 1].set_xlabel('Return Period (years)')
            axs[1, 1].set_ylabel(f'Peak Value{unit_str}')
            axs[1, 1].grid(True, which='both', ls='-', alpha=0.3)
            axs[1, 1].legend()
            
            # 创建图表标题
            if title is None:
                if ch_name is not None:
                    scale_str = "Full Scale" if fullscale else "Model Scale"
                    title = f"Extreme Value Analysis for {ch_name}{unit_str} ({scale_str})"
                else:
                    title = "Extreme Value Analysis"
            
            fig.suptitle(title, fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            
            # 保存图表
            if save_path is not None:
                plt.savefig(save_path, dpi=300)
                logger.info(f"Plot saved to {save_path}")
            
            # 显示图表
            if visualization:
                plt.show()
            else:
                plt.close(fig)
            
            return fig
            
        except ImportError:
            logger.error("Matplotlib不可用")
            return None
        except Exception as e:
            logger.error(f"创建Matplotlib可视化时出错: {str(e)}")
            return None
            
    # 如果到达这里，说明所有后端都失败了
    logger.error("所有可视化后端都失败了")
    return None
