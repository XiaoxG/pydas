"""
PyDAS Plot Module
================
This module contains plotting functions for PyDAS data.
"""

import logging
import numpy as np
import pandas as pd
import os

# Set up logging
logger = logging.getLogger('pydas.plot')

# 检查plotly-resampler库是否可用
try:
    import plotly_resampler
    HAS_PLOTLY_RESAMPLER = True
    logger.info("plotly-resampler库已加载，将启用大数据集优化功能")
except ImportError:
    HAS_PLOTLY_RESAMPLER = False
    logger.info("未检测到plotly-resampler，建议安装以优化大数据集: pip install plotly-resampler")

def use_webgl_rendering(fig, data_length=None, threshold=10000):
    """
    将plotly图表转换为使用WebGL渲染以提高大数据集的性能
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        要优化的plotly图表对象
    data_length : int, optional
        数据点数量，如果不提供则从图表数据中估计
    threshold : int, optional
        触发WebGL的数据点阈值，默认为10000
        
    Returns:
    --------
    plotly.graph_objects.Figure
        优化后的图表对象
    """
    try:
        import plotly.graph_objects as go
        
        # 估计数据点数量(如果未提供)
        if data_length is None:
            data_length = 0
            for trace in fig.data:
                if hasattr(trace, 'x') and trace.x is not None:
                    data_length = max(data_length, len(trace.x))
                    
        # 如果数据量小于阈值，则不需要优化
        if data_length < threshold:
            return fig
            
        # 转换所有散点图为WebGL模式
        for i, trace in enumerate(fig.data):
            if hasattr(trace, 'type') and trace.type == 'scatter':
                # 获取当前trace的所有属性
                trace_dict = trace.to_plotly_json()
                # 修改类型为scattergl
                trace_dict['type'] = 'scattergl'
                # 替换原trace
                fig.data[i] = trace_dict
                
        # 其他WebGL优化设置
        fig.update_layout(
            uirevision='constant',  # 保持UI状态
            hovermode='closest',    # 优化悬停性能
        )
        
        logger.info(f"已启用WebGL渲染加速 ({data_length} 数据点)")
        return fig
    except Exception as e:
        logger.warning(f"启用WebGL渲染失败: {e}")
        return fig  # 返回原始图表

def create_resampable_plot(x, y, name=None, title=None, n_shown_samples=5000):
    """
    创建可动态重采样的图表，适用于非常大的时间序列数据集
    
    Parameters:
    -----------
    x : numpy.ndarray
        x轴数据
    y : numpy.ndarray
        y轴数据
    name : str, optional
        数据系列名称
    title : str, optional
        图表标题
    n_shown_samples : int, optional
        初始显示的数据点数，默认5000
        
    Returns:
    --------
    FigureResampler or None
        可重采样图表对象，如果库不可用则返回None
    """
    if not HAS_PLOTLY_RESAMPLER:
        logger.warning("未安装plotly_resampler库，无法使用动态重采样功能")
        return None
        
    try:
        from plotly_resampler import FigureResampler
        import plotly.graph_objects as go
        
        # 创建基础图表
        fig = go.Figure()
        
        # 添加数据
        trace_name = name if name else "数据"
        fig.add_trace(go.Scatter(x=x, y=y, name=trace_name))
        
        # 设置布局
        if title:
            fig.update_layout(title=title)
            
        # 创建可重采样的图表
        fig_resampler = FigureResampler(
            fig, 
            default_n_shown_samples=n_shown_samples,
            resampled_trace_prefix_suffix=(None, " (重采样)")
        )
        
        logger.info(f"已创建可动态重采样图表 (数据点: {len(x)}, 显示点数: {n_shown_samples})")
        return fig_resampler
    except Exception as e:
        logger.warning(f"创建可重采样图表失败: {e}")
        return None

def lttb_downsample(x, y, n_out):
    """
    使用LTTB (Largest-Triangle-Three-Buckets) 算法进行下采样
    保留数据的视觉特征
    
    Parameters:
    -----------
    x : numpy.ndarray
        x轴数据
    y : numpy.ndarray
        y轴数据
    n_out : int
        输出点数
        
    Returns:
    --------
    tuple
        (x_sampled, y_sampled) 下采样后的数据点
    """
    n = len(x)
    if n <= n_out:
        return x, y
        
    # 始终保留第一点和最后一点
    sampled_x = np.zeros(n_out)
    sampled_y = np.zeros(n_out)
    sampled_x[0] = x[0]
    sampled_y[0] = y[0]
    sampled_x[n_out-1] = x[n-1]
    sampled_y[n_out-1] = y[n-1]
    
    # 计算桶大小
    bucket_size = (n - 2) / (n_out - 2)
    
    # 对每个输出点
    for i in range(1, n_out-1):
        # 计算三个桶的范围
        a = int((i - 1) * bucket_size) + 1
        b = int(i * bucket_size) + 1
        c = int((i + 1) * bucket_size) + 1 if i < n_out-2 else n-1
        
        # 当前点a
        point_a_x = sampled_x[i-1]
        point_a_y = sampled_y[i-1]
        
        # 计算下一个点c
        point_c_x = x[c-1]
        point_c_y = y[c-1]
        
        # 在中间桶b中寻找形成最大面积的点
        max_area = -1
        max_idx = b
        
        for j in range(a, b):
            area = abs(
                (point_a_x - point_c_x) * (y[j] - point_a_y) - 
                (point_a_x - x[j]) * (point_c_y - point_a_y)
            ) * 0.5
            if area > max_area:
                max_area = area
                max_idx = j
        
        # 保存最佳点
        sampled_x[i] = x[max_idx]
        sampled_y[i] = y[max_idx]
    
    return sampled_x, sampled_y

# Global plot configuration
PLOT_CONFIG = {
    # 通用尺寸配置
    'figsize': {
        'small': (8, 6),
        'medium': (12, 8),
        'large': (16, 10),
        'wide': (12, 4),
        'square': (8, 8),
        'tall': (6, 8),
    },
    
    # 通用字体配置
    'font': {
        'family': 'Arial, sans-serif',
        'size': {
            'small': 8,
            'medium': 10,
            'large': 12,
            'title': 14,
            'label': 10,
            'tick': 8,
            'legend': 9,
            'annotation': 9,
        },
        'weight': 'normal',
    },
    
    # 图表样式
    'style': {
        'matplotlib': {
            'default': 'seaborn-v0_8-whitegrid',
            'light': 'seaborn-v0_8-whitegrid',
            'dark': 'seaborn-v0_8-dark',
            'paper': 'seaborn-v0_8-paper',
            'talk': 'seaborn-v0_8-talk',
            'colorblind': 'seaborn-v0_8-colorblind',
        },
        'plotly': {
            'default': 'plotly_white',
            'light': 'plotly_white',
            'dark': 'plotly_dark',
            'paper': 'ggplot2',
        },
        'seaborn': {
            'default': 'whitegrid',
            'light': 'whitegrid',
            'dark': 'darkgrid',
            'paper': 'ticks',
            'talk': 'whitegrid',
        },
    },
    
    # 默认颜色
    'colors': {
        'default': 'tab10',  # matplotlib colormap名称
        'sequential': 'viridis',
        'diverging': 'coolwarm',
        'qualitative': 'tab10',
        'single': 'blue',
        'fit': 'red',
        'background': 'white',
        'grid': '#CCCCCC',
        'annotation': 'gray',
    },
    
    # 图表元素设置
    'elements': {
        'line_width': 1.5,
        'marker_size': 5,
        'alpha': 0.8,
        'grid': True,
        'dpi': 300,
        'edge_color': '#000000',
    },
    
    # 统计表配置
    'stats': {
        'table_width': 0.3,
        'table_height': 0.2,
        'table_font_size': 10,
        'header_color': '#EEEEEE',
    },
}

# 导出常用配置供外部使用
DEFAULT_FIGSIZE = PLOT_CONFIG['figsize']['medium']
DEFAULT_FONT_SIZE = PLOT_CONFIG['font']['size']['medium']
DEFAULT_DPI = PLOT_CONFIG['elements']['dpi']

def get_plot_backend(backend=None):
    """
    获取指定的绘图后端，如果指定的后端不可用，则尝试其他后端
    
    Parameters:
    -----------
    backend : str or None
        要使用的后端: 'plotly', 'matplotlib', 'seaborn', 或 None (自动选择)
        
    Returns:
    --------
    str
        实际使用的后端名称
    """
    if backend is None:
        # 按优先级尝试后端
        try:
            import plotly
            return 'plotly'
        except ImportError:
            try:
                import seaborn
                return 'seaborn'
            except ImportError:
                try:
                    import matplotlib
                    return 'matplotlib'
                except ImportError:
                    logger.error("No available plotting backend found. Install plotly, seaborn, or matplotlib.")
                    return None
    
    # 检查指定的后端是否可用
    if backend.lower() == 'plotly':
        try:
            import plotly
            return 'plotly'
        except ImportError:
            logger.warning("Plotly not available. Trying alternative backends.")
            return get_plot_backend(None)
    
    elif backend.lower() == 'seaborn':
        try:
            import seaborn
            return 'seaborn'
        except ImportError:
            logger.warning("Seaborn not available. Trying alternative backends.")
            return get_plot_backend(None)
    
    elif backend.lower() == 'matplotlib':
        try:
            import matplotlib
            return 'matplotlib'
        except ImportError:
            logger.warning("Matplotlib not available. Trying alternative backends.")
            return get_plot_backend(None)
    
    else:
        logger.warning(f"Unknown backend '{backend}'. Using default.")
        return get_plot_backend(None)

def apply_style(backend, style=None):
    """
    应用指定的绘图样式到指定的后端
    
    Parameters:
    -----------
    backend : str
        绘图后端: 'plotly', 'matplotlib', 或 'seaborn'
    style : str or None
        样式名称，如果为None则使用默认样式
        
    Returns:
    --------
    None
    """
    if backend is None:
        return
    
    # 如果未指定样式，使用默认样式
    if style is None:
        style = 'default'
    
    if backend.lower() == 'matplotlib':
        try:
            import matplotlib.pyplot as plt
            plt.style.use(PLOT_CONFIG['style']['matplotlib'].get(style.lower(), 
                                                              PLOT_CONFIG['style']['matplotlib']['default']))
        except Exception as e:
            logger.warning(f"Failed to apply matplotlib style: {e}")
    
    elif backend.lower() == 'seaborn':
        try:
            import seaborn as sns
            sns.set_theme(style=PLOT_CONFIG['style']['seaborn'].get(style.lower(), 
                                                               PLOT_CONFIG['style']['seaborn']['default']))
        except Exception as e:
            logger.warning(f"Failed to apply seaborn style: {e}")
    
    # Plotly样式在创建图表时应用

def validate_channel(pydas_obj, ch_idx):
    """
    Validate and convert channel index or name to channel name.
    
    Parameters:
    -----------
    pydas_obj : PyDAS
        PyDAS object containing the data
    ch_idx : int, str, or list
        Channel index, name, or list of indices/names
        
    Returns:
    --------
    str or list or None
        Channel name(s), or None if invalid
    """
    if isinstance(ch_idx, int):
        if ch_idx < len(pydas_obj.chInfo):
            return pydas_obj.chInfo.iloc[ch_idx]['Name']
        else:
            logger.error(f"Channel index {ch_idx} out of bounds.")
            return None
    elif isinstance(ch_idx, str):
        if ch_idx in pydas_obj.chInfo['Name'].values:
            return ch_idx
        else:
            logger.error(f"Channel '{ch_idx}' not found.")
            return None
    elif isinstance(ch_idx, list):
        # Process channel list - support both index lists and name lists
        result = []
        
        # If list of strings (channel names), validate each name
        if all(isinstance(item, str) for item in ch_idx):
            for name in ch_idx:
                if name in pydas_obj.chInfo['Name'].values:
                    result.append(name)
                else:
                    logger.warning(f"Channel '{name}' not found, skipping.")
            
            if not result:
                logger.error("No valid channels to plot.")
                return None
            return result
            
        # If list of integers (channel indices), convert to names
        elif all(isinstance(item, int) for item in ch_idx):
            for idx in ch_idx:
                if idx < len(pydas_obj.chInfo):
                    result.append(pydas_obj.chInfo.iloc[idx]['Name'])
                else:
                    logger.warning(f"Channel index {idx} out of bounds, skipping.")
            
            if not result:
                logger.error("No valid channels to plot.")
                return None
            return result
        else:
            logger.error("Channel list must contain all strings or all integers.")
            return None
    else:
        logger.error("Channel identifier must be an integer, string, or list.")
        return None


# Re-export from submodules for backward compatibility
from .timeseries import plot_channel
from .statistics import plot_histogram, _plot_statistics_mpl, _plot_statistics_plotly, boxplot_channel
from .scatter import plot_xy, _lttb_downsample
from .extreme import _detect_peaks, plot_extreme_analysis

__all__ = [
    # Config
    "PLOT_CONFIG", "DEFAULT_FIGSIZE", "DEFAULT_FONT_SIZE", "DEFAULT_DPI",
    # Utilities
    "use_webgl_rendering", "create_resampable_plot", "lttb_downsample",
    "get_plot_backend", "apply_style", "validate_channel",
    "HAS_PLOTLY_RESAMPLER",
    # Submodule functions
    "plot_channel",
    "plot_histogram", "_plot_statistics_mpl", "_plot_statistics_plotly", "boxplot_channel",
    "plot_xy", "_lttb_downsample",
    "_detect_peaks", "plot_extreme_analysis",
]
