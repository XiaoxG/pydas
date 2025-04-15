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

def plot_channel(pydas_obj, ch_name, sseg=0, title=None, xlabel='Time (s)', ylabel=None, 
              xlim=None, ylim=None, grid=True, show=True, save_path=None, 
              plotbackend=None, style=None, downsampling=False, max_points=40000, save_html=None,
              dpi=None, width=None, height=None, color=None, alpha=None, linewidth=None, 
              figsize=None, stats=True, table_width=None, column_widths=None,
              use_dask=True, use_webgl=True, use_resampler=False, n_shown_samples=5000,
              chunk_size=10000, data_decimation='auto', fullscale=False, lam=None, 
              rho=1.025, g=9.807):
    """
    Plot a channel from a PyDAS object, with options for interactive web-based plotting.
    
    Parameters:
        pydas_obj (PyDAS): The PyDAS object containing channel data
        ch_name (str or list): Channel name or list of channel names to plot
        sseg (int): Segment index to plot (default: 0)
        title (str): Plot title (default: None, auto-generated)
        xlabel (str): X-axis label (default: 'Time (s)')
        ylabel (str): Y-axis label (default: None, auto-generated)
        xlim (tuple): X-axis limits as (min, max) (default: None)
        ylim (tuple): Y-axis limits as (min, max) (default: None)
        grid (bool): Whether to show grid (default: True)
        show (bool): Whether to display the plot (default: True)
        save_path (str): Path to save the plot (default: None)
        plotbackend (str): Plotting backend to use ('plotly', 'matplotlib', 'seaborn', or None for auto) (default: None)
        style (str): Plot style to use (default: None, uses backend's default style)
        downsampling (bool): Whether to downsample large datasets (default: False)
        max_points (int): Maximum number of points to plot before downsampling (default: 40000)
        save_html (str): Path to save as interactive HTML (default: None)
        dpi (int): DPI for saved image (default: None, uses CONFIG default)
        width (int): Width in pixels for plot (default: None)
        height (int): Height in pixels for plot (default: None)
        color (str): Line color (default: None, auto-generated)
        alpha (float): Line transparency (default: None, uses CONFIG default)
        linewidth (float): Line width (default: None, uses CONFIG default)
        figsize (tuple): Figure size in inches (default: None, uses CONFIG default)
        stats (bool): Whether to include statistics (default: True)
        table_width (float): Width of the statistics table (default: None, uses CONFIG default)
        column_widths (list): Column widths for statistics table (default: None)
        use_dask (bool): Use Dask for large data processing (default: True)
        use_webgl (bool): Use WebGL for Plotly rendering for better performance (default: True)
        use_resampler (bool): Use plotly-resampler for dynamic downsampling (default: False)
        n_shown_samples (int): Number of samples to show initially (default: 5000)
        chunk_size (int): Chunk size for Dask processing (default: 10000)
        data_decimation (str or int): Decimation method for large datasets ('auto', 'lttb', or an integer for step) (default: 'auto')
        fullscale (bool): Whether to use full scale for plotting (default: False)
        lam (float): Lambda parameter for full scale (default: None)
        rho (float): Density parameter for full scale (default: 1.025)
        g (float): Gravitational acceleration for full scale (default: 9.807)
    
    Returns:
        Figure object (matplotlib.figure.Figure or plotly.graph_objects.Figure)
    """
    # 使用配置默认值（如果未指定）
    if dpi is None:
        dpi = PLOT_CONFIG['elements']['dpi']
    if alpha is None:
        alpha = PLOT_CONFIG['elements']['alpha']
    if linewidth is None:
        linewidth = PLOT_CONFIG['elements']['line_width']
    if table_width is None:
        table_width = PLOT_CONFIG['stats']['table_width']
    if figsize is None:
        figsize = PLOT_CONFIG['figsize']['wide']

    try:
        # Check if PyDAS object is valid
        if not hasattr(pydas_obj, 'chInfo') or not hasattr(pydas_obj, 'data'):
            logger.error("Invalid PyDAS object - missing required attributes")
            return None
        
        # Check if the segment index is valid
        if sseg < 0 or sseg >= len(pydas_obj.data):
            logger.error(f"Invalid segment index {sseg}, must be between 0 and {len(pydas_obj.data)-1}")
            return None
            
        # Convert single channel name to list for uniform processing
        if isinstance(ch_name, str):
            channel_list = [ch_name]
            is_list = False
        else:
            channel_list = ch_name
            is_list = True

        # 获取实际可用的绘图后端
        backend = get_plot_backend(plotbackend)
        if backend is None:
            logger.error("No available plotting backend found")
            return None
            
        # 应用样式
        apply_style(backend, style)

        # Flag to track if we've successfully created a plot
        plot_created = False
        fig = None
        plt = None  # Initialize plt as None, import later as needed
        
        # If backend is plotly, try to use Plotly for interactive web-based plotting
        if backend == 'plotly':
            try:
                # Import Plotly modules
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Create figure with secondary y-axis for multiple channels with different units
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Setup for statistical data
                stats_data = {}
                
                # Create color palette for multiple channels
                if is_list and color is None:
                    import matplotlib.pyplot as plt
                    from matplotlib import cm
                    colors = cm.get_cmap(PLOT_CONFIG['colors']['qualitative'], len(channel_list))
                    color_list = []
                    for i in range(len(channel_list)):
                        rgba = colors(i)
                        color_list.append(f'rgb({int(255*rgba[0])},{int(255*rgba[1])},{int(255*rgba[2])})')
                elif not is_list and color is None:
                    color_list = [PLOT_CONFIG['colors']['single']]
                elif isinstance(color, list):
                    color_list = color
                else:
                    color_list = [color] * len(channel_list)
                
                # Process each channel
                for i, channel in enumerate(channel_list):
                    # Check if channel exists
                    if channel not in pydas_obj.data[sseg].columns:
                        logger.warning(f"Channel '{channel}' not found in segment {sseg}, skipping.")
                        continue
                    
                    # Get data
                    y_data = pydas_obj.data[sseg][channel]
                    data_length = len(y_data)
                    
                    # Get X-axis data (time)
                    x_data = np.arange(data_length) / pydas_obj.__fs__
                    
                    # Get channel unit
                    unit = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel]['Unit'].values[0]
                    
                    # Apply full scale conversion if requested
                    if fullscale:
                        # Make sure we have a valid scale factor
                        if lam is None:
                            if hasattr(pydas_obj, '__lam__'):
                                lam = pydas_obj.__lam__
                            else:
                                logger.warning(f"fullscale=True but no lambda parameter provided and no __lam__ attribute found. Using model scale.")
                                # Continue with model scale
                        else:
                            try:
                                logger.info(f"Converting channel '{channel}' to full scale with λ={lam}")
                                # Get TimeSeries object in full scale
                                ts = pydas_obj.channel2fullscale(channel, lam, rho, g)
                                
                                if ts is None:
                                    logger.warning(f"Full scale conversion failed for channel '{channel}', using model scale.")
                                else:
                                    # Use the full scale data
                                    y_data = ts.data
                                    # Time might also be scaled, use the TimeSeries args if available
                                    if hasattr(ts, 'args') and ts.args is not None:
                                        x_data = ts.args
                                    
                                    # Check for unit conversion
                                    from utils import get_default_transDict, findtrans
                                    try:
                                        transDict = get_default_transDict(g)
                                        trans_temp = findtrans(unit, transDict)
                                        if trans_temp and trans_temp[0]:
                                            unit = trans_temp[0]
                                            logger.info(f"Unit converted from model scale to full scale: {unit}")
                                    except Exception as e:
                                        logger.warning(f"Unit conversion failed: {e}")
                                        
                                    # Update title to indicate full scale
                                    if title is None:
                                        title = f"Full Scale Time Series Plot - {channel}"
                                    elif "Full Scale" not in title:
                                        title = f"Full Scale: {title}"
                            except Exception as e:
                                logger.warning(f"Error in full scale conversion: {e}")
                                # Continue with model scale
                    
                    # Store data for statistics calculation
                    stats_data[channel] = {
                        'unit': unit
                    }
                    
                    # Check if we should use plotly-resampler for this large dataset
                    if use_resampler and not is_list and HAS_PLOTLY_RESAMPLER and data_length > 50000:
                        logger.info(f"Using plotly-resampler for channel '{channel}' ({data_length} points)")
                        
                        # Get channel unit for y-axis label
                        unit = stats_data[channel]['unit']
                        
                        # Create resampable plot
                        resampler_title = title
                        if resampler_title is None:
                            if fullscale:
                                resampler_title = f"Full Scale Time Series Plot - {channel}"
                            else:
                                resampler_title = f"Time Series Plot - {channel}"
                                
                        fr = create_resampable_plot(
                            x=x_data, 
                            y=y_data, 
                            name=f"{channel} ({unit})",
                            title=resampler_title,
                            n_shown_samples=n_shown_samples
                        )
                        
                        if fr is not None:
                            # Calculate stats if requested
                            if stats:
                                stats_data[channel]['mean'] = y_data.mean()
                                stats_data[channel]['min'] = y_data.min()
                                stats_data[channel]['max'] = y_data.max()
                                stats_data[channel]['std'] = y_data.std()
                            
                            # Save if requested
                            if save_path:
                                try:
                                    # Save as PNG
                                    png_path = save_path + '.png' if not save_path.endswith('.png') else save_path
                                    fr.write_image(png_path, width=width or 1200, height=height or 800)
                                    logger.info(f"Saved plot to {png_path}")
                                except Exception as e:
                                    logger.warning(f"Failed to save image: {e}")
                            
                            if save_html:
                                try:
                                    html_path = save_html if save_html.endswith('.html') else save_html + '.html'
                                    fr.write_html(html_path)
                                    logger.info(f"Saved interactive HTML to {html_path}")
                                except Exception as e:
                                    logger.warning(f"Failed to save HTML: {e}")
                            
                            # Show the interactive plot
                            if show:
                                fr.show_dash()
                            
                            # Return the resampler figure
                            return fr
                    
                    # 使用配置的统一处理逻辑后的代码
                    # ... [保留原有代码中的数据处理逻辑，如Dask处理、下采样等]
                    # Use Dask for large datasets if enabled
                    very_large_data = data_length > 100000
                    extremely_large_data = data_length > 1000000
                    
                    if use_dask and very_large_data:
                        try:
                            import dask.dataframe as dd
                            import dask.array as da
                            
                            # If the data is extremely large, calculate stats with Dask
                            if stats:
                                # Create dask series for efficient computation
                                ds = dd.from_pandas(y_data, chunksize=chunk_size)
                                stats_data[channel]['mean'] = ds.mean().compute()
                                stats_data[channel]['min'] = ds.min().compute()
                                stats_data[channel]['max'] = ds.max().compute()
                                stats_data[channel]['std'] = ds.std().compute()
                            
                            # Handle decimation for extremely large datasets
                            if extremely_large_data or downsampling:
                                if data_decimation == 'auto':
                                    # Use LTTB algorithm for large datasets
                                    if data_length > max_points:
                                        logger.info(f"Using LTTB downsampling for channel '{channel}' from {data_length} to {max_points} points.")
                                        x_down, y_down = _lttb_downsample(x_data, y_data.values, max_points)
                                        x_data, y_data = x_down, y_down
                                elif data_decimation == 'lttb':
                                    # Force LTTB algorithm
                                    logger.info(f"Using LTTB downsampling for channel '{channel}' from {data_length} to {max_points} points.")
                                    x_down, y_down = _lttb_downsample(x_data, y_data.values, max_points)
                                    x_data, y_data = x_down, y_down
                                elif isinstance(data_decimation, int):
                                    # Use step-based decimation with specific step
                                    step = data_decimation
                                    logger.info(f"Using step-based downsampling for channel '{channel}' with step {step}.")
                                    x_data = x_data[::step]
                                    y_data = y_data.iloc[::step]
                                else:
                                    # Default to standard downsampling if enabled
                                    if downsampling and data_length > max_points:
                                        step = int(data_length / max_points)
                                        logger.info(f"Using uniform downsampling for channel '{channel}' from {data_length} to ~{data_length//step} points.")
                                        x_data = x_data[::step]
                                        y_data = y_data.iloc[::step]
                            
                        except ImportError:
                            logger.warning("Dask not available. Falling back to pandas.")
                            if downsampling and data_length > max_points:
                                # Standard downsampling
                                step = int(data_length / max_points)
                                logger.info(f"Downsampling channel '{channel}' from {data_length} to ~{data_length//step} points.")
                                x_data = x_data[::step]
                                y_data = y_data.iloc[::step]
                            
                            # Calculate stats with pandas
                            if stats:
                                stats_data[channel]['mean'] = y_data.mean()
                                stats_data[channel]['min'] = y_data.min()
                                stats_data[channel]['max'] = y_data.max()
                                stats_data[channel]['std'] = y_data.std()
                    else:
                        # Standard approach for smaller datasets
                        if downsampling and data_length > max_points:
                            # Standard downsampling
                            step = int(data_length / max_points)
                            logger.info(f"Downsampling channel '{channel}' from {data_length} to ~{data_length//step} points.")
                            x_data = x_data[::step]
                            y_data = y_data.iloc[::step]
                        
                        # Calculate stats with pandas
                        if stats:
                            stats_data[channel]['mean'] = y_data.mean()
                            stats_data[channel]['min'] = y_data.min()
                            stats_data[channel]['max'] = y_data.max()
                            stats_data[channel]['std'] = y_data.std()
                    
                    # Get channel unit for y-axis label
                    unit = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel]['Unit'].values[0]
                    
                    # Add trace to the figure - use ScatterGL for better performance with large datasets
                    if use_webgl and len(x_data) > 5000:
                        import plotly.graph_objects as go
                        scatter_type = go.Scattergl
                    else:
                        scatter_type = go.Scatter
                    
                    # Create trace name with full scale indication if needed
                    trace_name = f"{channel} ({unit})"
                    if fullscale:
                        trace_name = f"{channel} (Full Scale, {unit})"
                    
                    fig.add_trace(
                        scatter_type(
                            x=x_data,
                            y=y_data,
                            name=trace_name,
                            mode='lines',
                            line=dict(color=color_list[i], width=linewidth),
                            opacity=alpha
                        ),
                        secondary_y=(i > 0 and is_list)  # Use secondary y-axis for additional channels
                    )
                
                # If no channels were plotted successfully
                if len(fig.data) == 0:
                    logger.error("No valid channels to plot.")
                    return None
                
                # Add statistics table if requested
                if stats:
                    # Prepare table data
                    ch_names = []
                    mean_values = []
                    max_values = []
                    min_values = []
                    std_values = []
                    units = []
                    
                    for channel in stats_data:
                        ch_names.append(channel)
                        mean_values.append(f"{stats_data[channel].get('mean', 'N/A'):.2g}")
                        max_values.append(f"{stats_data[channel].get('max', 'N/A'):.2g}")
                        min_values.append(f"{stats_data[channel].get('min', 'N/A'):.2g}")
                        std_values.append(f"{stats_data[channel].get('std', 'N/A'):.2g}")
                        units.append(stats_data[channel]['unit'])
                    
                    # Create table
                    fig.add_trace(
                        go.Table(
                            header=dict(
                                values=["Ch.", "Mean", "Max", "Min", "Std", "Unit"],
                                font=dict(size=PLOT_CONFIG['stats']['table_font_size']),
                                align="center",
                                fill=dict(color=PLOT_CONFIG['stats']['header_color'])
                            ),
                            cells=dict(
                                values=[ch_names, mean_values, max_values, min_values, std_values, units],
                                font=dict(size=PLOT_CONFIG['font']['size']['small']),
                                align="center"
                            ),
                            domain=dict(x=[0, table_width], y=[0, 0.2])
                        )
                    )
                
                # Create a settings dict to be applied to each plot
                plot_settings = {
                    "scrollZoom": True,  # Enable mouse scroll for zooming
                    "modeBarButtonsToAdd": ["drawopenpath", "eraseshape"],  # Add drawing tools
                    "modeBarButtonsToRemove": ["lasso2d"]  # Remove lasso selection
                }
                
                # Set plot title
                if title is None:
                    title = "Time Series Plot"
                    if fullscale:
                        title = "Full Scale Time Series Plot"
                    if is_list:
                        title += f" (Multiple Channels)"
                    else:
                        title += f" - {channel_list[0]}"
                
                # Set y-axis label if not provided
                if ylabel is None:
                    if not is_list:
                        unit = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel_list[0]]['Unit'].values[0]
                        ylabel = f"{channel_list[0]} ({unit})"
                
                # Update layout with config settings
                fig.update_layout(
                    title=title,
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=PLOT_CONFIG['font']['size']['legend'])
                    ),
                    hovermode="closest",
                    template=PLOT_CONFIG['style']['plotly'].get(style, PLOT_CONFIG['style']['plotly']['default']),
                    width=width,
                    height=height,
                    grid=dict(rows=1, columns=1, pattern="independent"),
                    margin=dict(l=50, r=50, t=50, b=50),
                    font=dict(
                        family=PLOT_CONFIG['font']['family'],
                        size=PLOT_CONFIG['font']['size']['medium']
                    ),
                    # 优化性能设置
                    uirevision='constant'  # 维持缩放级别
                )
                
                # Update axes
                fig.update_xaxes(showgrid=grid, zeroline=grid)
                fig.update_yaxes(showgrid=grid, zeroline=grid)
                
                # Set axis limits if provided
                if xlim is not None:
                    fig.update_xaxes(range=xlim)
                if ylim is not None:
                    fig.update_yaxes(range=ylim)
                
                # Save as HTML if requested
                if save_html is not None:
                    fig.write_html(save_html, config=plot_settings)
                    logger.info(f"Interactive plot saved to {save_html}")
                
                # Save as image if requested
                if save_path is not None:
                    fig.write_image(save_path, width=width or 1200, height=height or 800, scale=2)
                    logger.info(f"Plot saved to {save_path}")
                
                # Show plot if requested
                if show:
                    fig.show(config=plot_settings)
                
                plot_created = True
                
            except ImportError:
                logger.warning("Plotly not available. Falling back to matplotlib.")
                backend = 'matplotlib'
            except Exception as e:
                logger.warning(f"Error using Plotly: {str(e)}. Falling back to matplotlib.")
                backend = 'matplotlib'
        
        # If backend is matplotlib/seaborn or plotly failed
        if backend in ['matplotlib', 'seaborn'] or not plot_created:
            try:
                import matplotlib.pyplot as plt
                if backend == 'seaborn':
                    import seaborn as sns
                
                # Create figure and axis
                fig, ax = plt.subplots(figsize=figsize)
                
                # Create color palette for multiple channels
                if is_list and color is None:
                    colors = [plt.cm.get_cmap(PLOT_CONFIG['colors']['qualitative'])(i % 10) for i in range(len(channel_list))]
                elif not is_list and color is None:
                    colors = [PLOT_CONFIG['colors']['single']]
                elif isinstance(color, list):
                    colors = color
                else:
                    colors = [color] * len(channel_list)
                
                # Process each channel
                for i, channel in enumerate(channel_list):
                    # Check if channel exists
                    if channel not in pydas_obj.data[sseg].columns:
                        logger.warning(f"Channel '{channel}' not found in segment {sseg}, skipping.")
                        continue
                    
                    # Get data
                    y_data = pydas_obj.data[sseg][channel]
                    data_length = len(y_data)
                    
                    # Get X-axis data (time)
                    x_data = np.arange(data_length) / pydas_obj.__fs__
                    
                    # 使用配置的统一处理逻辑后的代码
                    # ... [保留原有代码的数据处理逻辑，如下采样、Dask等]
                    # Use Dask for large datasets if enabled
                    very_large_data = data_length > 100000
                    extremely_large_data = data_length > 1000000
                    
                    if use_dask and very_large_data:
                        try:
                            import dask.dataframe as dd
                            import dask.array as da
                            
                            # Handle decimation for extremely large datasets
                            if extremely_large_data or downsampling:
                                if data_decimation == 'auto':
                                    # Use LTTB algorithm for large datasets
                                    if data_length > max_points:
                                        logger.info(f"Using LTTB downsampling for channel '{channel}' from {data_length} to {max_points} points.")
                                        x_down, y_down = _lttb_downsample(x_data, y_data.values, max_points)
                                        x_data, y_data = x_down, y_down
                                elif data_decimation == 'lttb':
                                    # Force LTTB algorithm
                                    logger.info(f"Using LTTB downsampling for channel '{channel}' from {data_length} to {max_points} points.")
                                    x_down, y_down = _lttb_downsample(x_data, y_data.values, max_points)
                                    x_data, y_data = x_down, y_down
                                elif isinstance(data_decimation, int):
                                    # Use step-based decimation with specific step
                                    step = data_decimation
                                    logger.info(f"Using step-based downsampling for channel '{channel}' with step {step}.")
                                    x_data = x_data[::step]
                                    y_data = y_data.iloc[::step]
                                else:
                                    # Default to standard downsampling if enabled
                                    if downsampling and data_length > max_points:
                                        step = int(data_length / max_points)
                                        logger.info(f"Using uniform downsampling for channel '{channel}' from {data_length} to ~{data_length//step} points.")
                                        x_data = x_data[::step]
                                        y_data = y_data.iloc[::step]
                            
                        except ImportError:
                            logger.warning("Dask not available. Falling back to pandas.")
                            if downsampling and data_length > max_points:
                                # Standard downsampling
                                step = int(data_length / max_points)
                                logger.info(f"Downsampling channel '{channel}' from {data_length} to ~{data_length//step} points.")
                                x_data = x_data[::step]
                                y_data = y_data.iloc[::step]
                            
                            # Calculate stats with pandas
                            if stats:
                                stats_data[channel]['mean'] = y_data.mean()
                                stats_data[channel]['min'] = y_data.min()
                                stats_data[channel]['max'] = y_data.max()
                                stats_data[channel]['std'] = y_data.std()
                    else:
                        # Standard approach for smaller datasets
                        if downsampling and data_length > max_points:
                            # Simple uniform downsampling
                            step = int(data_length / max_points)
                            logger.info(f"Downsampling channel '{channel}' from {data_length} to ~{data_length//step} points.")
                            x_data = x_data[::step]
                            y_data = y_data.iloc[::step]
                    
                    # Get channel unit for label
                    unit = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel]['Unit'].values[0]
                    
                    # Plot data
                    ax.plot(x_data, y_data, label=f"{channel} ({unit})", 
                           color=colors[i], linewidth=linewidth, alpha=alpha)
                
                # Set plot title
                if title is None:
                    title = "Time Series Plot"
                    if fullscale:
                        title = "Full Scale Time Series Plot"
                    if is_list:
                        title += f" (Multiple Channels)"
                    else:
                        title += f" - {channel_list[0]}"
                ax.set_title(title, fontsize=PLOT_CONFIG['font']['size']['title'])
                
                # Set axis labels
                ax.set_xlabel(xlabel, fontsize=PLOT_CONFIG['font']['size']['label'])
                
                # Set y-axis label if not provided
                if ylabel is None:
                    if not is_list:
                        unit = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel_list[0]]['Unit'].values[0]
                        ylabel = f"{channel_list[0]} ({unit})"
                ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG['font']['size']['label'])
                
                # 设置刻度字体大小
                ax.tick_params(axis='both', which='major', labelsize=PLOT_CONFIG['font']['size']['tick'])
                
                # Set grid
                ax.grid(grid)
                
                # Set axis limits if provided
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                
                # 如果需要显示图例
                if is_list:
                    ax.legend(fontsize=PLOT_CONFIG['font']['size']['legend'])
                
                # Add statistical information if requested
                if stats:
                    # 计算并显示统计信息
                    if is_list:
                        stats_text = ""
                        for i, channel in enumerate(channel_list):
                            if channel in pydas_obj.data[sseg].columns:
                                data = pydas_obj.data[sseg][channel]
                                stats_text += (f"{channel}: μ={np.mean(data):.4g}, σ={np.std(data):.4g}, "
                                             f"min={np.min(data):.4g}, max={np.max(data):.4g}\n")
                    else:
                        channel = channel_list[0]
                        if channel in pydas_obj.data[sseg].columns:
                            data = pydas_obj.data[sseg][channel]
                            stats_text = (f"{channel}: μ={np.mean(data):.4g}, σ={np.std(data):.4g}, "
                                        f"min={np.min(data):.4g}, max={np.max(data):.4g}")
                    
                    # 在图中添加统计信息文本
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                           fontsize=PLOT_CONFIG['font']['size']['annotation'])
                
                # Adjust layout
                plt.tight_layout()
                
                # Save figure if requested
                if save_path is not None:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                    logger.info(f"Plot saved to {save_path}")
                
                # Show plot if requested
                if show:
                    plt.show()
                else:
                    plt.close(fig)
                
                plot_created = True
                
            except ImportError:
                logger.error("No available plotting libraries found (matplotlib, seaborn, plotly).")
                return None
        
        # Return fig object if not showing or return None if showing
        return None if show else fig
        
    except Exception as e:
        logger.error(f"Error in plot_channel: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def plot_histogram(pydas_obj, ch_name, sseg=0, title=None, xlabel=None, ylabel='Count', 
                bins=50, xlim=None, ylim=None, grid=True, show=True, save_path=None, 
                plotbackend=None, style=None, save_html=None, dpi=300, width=None, height=None, 
                color=None, alpha=0.6, figsize=(12, 6), fit_gaussian=True, fit_color='red'):
    """
    Plot a histogram of a channel from a PyDAS object.
    
    Parameters:
        pydas_obj (PyDAS): The PyDAS object containing channel data
        ch_name (str or list): Channel name or list of channel names
        sseg (int): Segment index to plot (default: 0)
        title (str): Plot title (default: None, auto-generated)
        xlabel (str): X-axis label (default: None, auto-generated)
        ylabel (str): Y-axis label (default: 'Count')
        bins (int): Number of histogram bins (default: 50)
        xlim (tuple): X-axis limits as (min, max) (default: None)
        ylim (tuple): Y-axis limits as (min, max) (default: None)
        grid (bool): Whether to show grid (default: True)
        show (bool): Whether to display the plot (default: True)
        save_path (str): Path to save the plot (default: None)
        plotbackend (str): Plotting backend to use ('plotly', 'matplotlib', 'seaborn', or None for auto) (default: None)
        style (str): Plot style to use (default: None, uses backend's default style)
        save_html (str): Path to save as interactive HTML (default: None)
        dpi (int): DPI for saved image (default: 300)
        width (int): Width in pixels for Plotly plot (default: None)
        height (int): Height in pixels for Plotly plot (default: None)
        color (str or list): Histogram color or list of colors (default: None, auto-generated)
        alpha (float): Histogram transparency (default: 0.6)
        figsize (tuple): Figure size for matplotlib in inches (default: (12, 6))
        fit_gaussian (bool): Whether to fit a Gaussian distribution (default: True)
        fit_color (str or list): Color of Gaussian fit curve (default: 'red')
        
    Returns:
        Figure object (matplotlib.figure.Figure or plotly.graph_objects.Figure)
    """
    try:
        # Check if PyDAS object is valid
        if not hasattr(pydas_obj, 'chInfo') or not hasattr(pydas_obj, 'data'):
            logger.error("Invalid PyDAS object - missing required attributes")
            return None
        
        # Check if the segment index is valid
        if sseg < 0 or sseg >= len(pydas_obj.data):
            logger.error(f"Invalid segment index {sseg}, must be between 0 and {len(pydas_obj.data)-1}")
            return None
            
        # Convert single channel name to list for uniform processing
        if isinstance(ch_name, str):
            channel_list = [ch_name]
            is_list = False
        else:
            channel_list = ch_name
            is_list = True

        # 获取实际可用的绘图后端
        backend = get_plot_backend(plotbackend)
        if backend is None:
            logger.error("No available plotting backend found")
            return None
            
        # 应用样式
        apply_style(backend, style)

        # Flag to track if we've successfully created a plot
        plot_created = False
        fig = None
        plt = None  # Initialize plt as None, import later as needed
        
        # If backend is plotly, try to use Plotly for interactive web-based plotting
        if backend == 'plotly':
            try:
                # Import Plotly modules
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                import plotly.figure_factory as ff
                
                # Determine subplot layout based on number of channels
                n_channels = len(channel_list)
                if n_channels <= 1:
                    rows, cols = 1, 1
                elif n_channels <= 2:
                    rows, cols = 1, 2
                elif n_channels <= 4:
                    rows, cols = 2, 2
                elif n_channels <= 6:
                    rows, cols = 2, 3
                else:
                    rows, cols = 3, 3
                
                # Create figure with subplots
                fig = make_subplots(rows=rows, cols=cols, subplot_titles=[])
                
                # Create color palette for multiple channels
                if is_list and color is None:
                    import matplotlib.pyplot as plt
                    from matplotlib import cm
                    colors = cm.get_cmap('tab10', n_channels)
                    color_list = []
                    for i in range(n_channels):
                        rgba = colors(i)
                        color_list.append(f'rgb({int(255*rgba[0])},{int(255*rgba[1])},{int(255*rgba[2])})')
                elif not is_list and color is None:
                    color_list = ['blue']
                elif isinstance(color, list):
                    color_list = color
                else:
                    color_list = [color] * n_channels
                
                # Create fit color list
                if isinstance(fit_color, list):
                    fit_color_list = fit_color
                else:
                    fit_color_list = [fit_color] * n_channels
                
                # Process each channel
                for i, channel in enumerate(channel_list):
                    # Calculate row and column indices
                    row = (i // cols) + 1
                    col = (i % cols) + 1
                    
                    # Check if channel exists
                    if channel not in pydas_obj.data[sseg].columns:
                        logger.warning(f"Channel '{channel}' not found in segment {sseg}, skipping.")
                        continue
                    
                    # Get data
                    data = pydas_obj.data[sseg][channel]
                    
                    # Get channel unit for label
                    unit = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel]['Unit'].values[0]
                    
                    # Create histogram trace
                    hist_trace = go.Histogram(
                        x=data,
                        nbinsx=bins,
                        name=channel,
                        marker=dict(color=color_list[i], opacity=alpha),
                        showlegend=False
                    )
                    
                    # Add trace to the figure
                    fig.add_trace(hist_trace, row=row, col=col)
                    
                    # Fit Gaussian distribution if requested
                    if fit_gaussian:
                        import scipy.stats as stats
                        
                        # Calculate histogram values
                        hist_vals, bin_edges = np.histogram(data, bins=bins)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        
                        # Fit normal distribution
                        mu, sigma = stats.norm.fit(data)
                        
                        # Create x values for the fitted line
                        x_min, x_max = data.min(), data.max()
                        x = np.linspace(x_min, x_max, 100)
                        
                        # Calculate PDF values
                        pdf = stats.norm.pdf(x, mu, sigma)
                        
                        # Scale PDF to match histogram height
                        bin_width = bin_edges[1] - bin_edges[0]
                        pdf_scaled = pdf * len(data) * bin_width
                        
                        # Add fit line
                        fit_trace = go.Scatter(
                            x=x,
                            y=pdf_scaled,
                            mode='lines',
                            name=f'Normal Fit (μ={mu:.4g}, σ={sigma:.4g})',
                            line=dict(color=fit_color_list[i], width=2),
                            showlegend=False
                        )
                        
                        fig.add_trace(fit_trace, row=row, col=col)
                        
                        # Add fit parameters as annotation
                        fit_text = f"μ = {mu:.4g}<br>σ = {sigma:.4g}"
                        fig.add_annotation(
                            x=0.95, y=0.95,
                            xref="paper", yref="paper",
                            xanchor="right", yanchor="top",
                            text=fit_text,
                            showarrow=False,
                            bgcolor="rgba(255, 255, 255, 0.7)",
                            bordercolor="gray",
                            borderwidth=1,
                            font=dict(size=10),
                            align="right"
                        )
                    
                    # Set subplot title
                    fig.update_xaxes(title_text=f"{channel} ({unit})", row=row, col=col)
                    if col == 1:  # First column
                        fig.update_yaxes(title_text=ylabel, row=row, col=col)
                    
                    # Apply grid settings
                    fig.update_xaxes(showgrid=grid, row=row, col=col)
                    fig.update_yaxes(showgrid=grid, row=row, col=col)
                    
                    # Set axis limits if provided
                    if xlim is not None:
                        fig.update_xaxes(range=xlim, row=row, col=col)
                    if ylim is not None:
                        fig.update_yaxes(range=ylim, row=row, col=col)
                
                # Set global title
                if title is None:
                    title = "Histogram Analysis"
                    if not is_list:
                        title += f" - {channel_list[0]}"
                
                # Update layout
                fig.update_layout(
                    title=title,
                    showlegend=False,
                    template="plotly_white",
                    width=width or 1200,
                    height=height or 800,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                # Save as HTML if requested
                if save_html is not None:
                    fig.write_html(save_html)
                    logger.info(f"Interactive histogram saved to {save_html}")
                
                # Save as image if requested
                if save_path is not None:
                    fig.write_image(save_path, width=width or 1200, height=height or 800, scale=2)
                    logger.info(f"Histogram saved to {save_path}")
                
                # Show plot if requested
                if show:
                    fig.show()
                
                plot_created = True
                
            except ImportError:
                logger.warning("Plotly not available. Falling back to matplotlib.")
                backend = 'matplotlib'
            except Exception as e:
                logger.warning(f"Error using Plotly: {str(e)}. Falling back to matplotlib.")
                backend = 'matplotlib'
        
        # If backend is matplotlib/seaborn or plotly failed
        if backend in ['matplotlib', 'seaborn'] or not plot_created:
            try:
                import matplotlib.pyplot as plt
                from matplotlib import gridspec
                
                # Determine subplot layout based on number of channels
                n_channels = len(channel_list)
                if n_channels <= 1:
                    rows, cols = 1, 1
                elif n_channels <= 2:
                    rows, cols = 1, 2
                elif n_channels <= 4:
                    rows, cols = 2, 2
                elif n_channels <= 6:
                    rows, cols = 2, 3
                else:
                    rows, cols = 3, 3
                
                # Create figure and subplots
                fig, axes = plt.subplots(rows, cols, figsize=figsize)
                
                # Make axes iterable even for a single subplot
                if n_channels == 1:
                    axes = np.array([axes])
                axes = axes.flatten()
                
                # Create color palette for multiple channels
                if is_list and color is None:
                    colors = plt.cm.tab10(np.linspace(0, 1, n_channels))
                    color_list = colors.tolist()
                elif not is_list and color is None:
                    color_list = ['blue']
                elif isinstance(color, list):
                    color_list = color
                else:
                    color_list = [color] * n_channels
                
                # Create fit color list
                if isinstance(fit_color, list):
                    fit_color_list = fit_color
                else:
                    fit_color_list = [fit_color] * n_channels
                
                # Process each channel
                for i, channel in enumerate(channel_list):
                    # Get current axis
                    ax = axes[i]
                    
                    # Check if channel exists
                    if channel not in pydas_obj.data[sseg].columns:
                        logger.warning(f"Channel '{channel}' not found in segment {sseg}, skipping.")
                        continue
                    
                    # Get data
                    data = pydas_obj.data[sseg][channel]
                    
                    # Get channel unit for label
                    unit = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel]['Unit'].values[0]
                    
                    # Plot histogram
                    n, bins, patches = ax.hist(data, bins=bins, color=color_list[i], alpha=alpha, edgecolor='black', linewidth=0.5)
                    
                    # Fit Gaussian distribution if requested
                    if fit_gaussian:
                        import scipy.stats as stats
                        
                        # Fit normal distribution
                        mu, sigma = stats.norm.fit(data)
                        
                        # Create x values for the fitted line
                        x_min, x_max = data.min(), data.max()
                        x = np.linspace(x_min, x_max, 100)
                        
                        # Calculate PDF values
                        pdf = stats.norm.pdf(x, mu, sigma)
                        
                        # Scale PDF to match histogram height
                        bin_width = bins[1] - bins[0]
                        pdf_scaled = pdf * len(data) * bin_width
                        
                        # Plot fit line
                        ax.plot(x, pdf_scaled, linewidth=2, color=fit_color_list[i])
                        
                        # Add fit parameters as text
                        fit_text = f"μ = {mu:.4g}\nσ = {sigma:.4g}"
                        ax.text(0.95, 0.95, fit_text, transform=ax.transAxes,
                               verticalalignment='top', horizontalalignment='right',
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
                    
                    # Set labels and title
                    ax.set_xlabel(f"{channel} ({unit})" if xlabel is None else xlabel)
                    ax.set_ylabel(ylabel)
                    ax.set_title(channel)
                    
                    # Apply grid setting
                    ax.grid(grid)
                    
                    # Set axis limits if provided
                    if xlim is not None:
                        ax.set_xlim(xlim)
                    if ylim is not None:
                        ax.set_ylim(ylim)
                
                # Set global title
                if title is None:
                    fig_title = "Histogram Analysis"
                    if not is_list:
                        fig_title += f" - {channel_list[0]}"
                else:
                    fig_title = title
                
                fig.suptitle(fig_title)
                plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for title
                
                # Save as image if requested
                if save_path is not None:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                    logger.info(f"Histogram saved to {save_path}")
                
                # Show plot if requested
                if show:
                    plt.show()
                
                plot_created = True
                
            except ImportError:
                logger.error("Matplotlib not available. Cannot create histogram plot.")
                return None
            except Exception as e:
                logger.error(f"Error creating matplotlib histogram: {str(e)}")
                return None
        
        # Return figure object
        return fig
        
    except Exception as e:
        logger.error(f"Error in plot_histogram: {str(e)}")
        return None

def plot_xy(pydas_obj, x_ch_name, y_ch_name, sseg=0, title=None, 
         xlabel=None, ylabel=None, xlim=None, ylim=None, grid=True, 
         show=True, save_path=None, plotbackend=None, style=None, save_html=None,
         dpi=None, width=None, height=None, color=None, alpha=None, 
         marker_size=None, figsize=None, line=False, fit_line=False,
         fit_color=None, fit_line_width=None, fit_alpha=None,
         show_stats=False, downsampling=True, max_points=10000,
         density_plot=False, density_colorscale=None, 
         density_opacity=None, use_webgl=True, adaptive_sampling=False,
         datashade=False, contour_levels=20, sampling_algorithm='lttb',
         memory_efficient=True, bin_size=None, sns_style=None, 
         sns_bins=50, sns_pthresh=0.1, sns_cmap=None,
         sns_contour_levels=5, sns_contour_color=None, sns_linewidths=None,
         use_resampler=False, n_shown_samples=5000):
    """
    Create an XY scatter plot with one channel on the X-axis and another on the Y-axis.
    
    Parameters:
        pydas_obj (PyDAS): The PyDAS object containing channel data
        x_ch_name (str): Channel name for the X-axis
        y_ch_name (str): Channel name for the Y-axis
        sseg (int): Segment index to plot (default: 0)
        title (str): Plot title (default: None, auto-generated)
        xlabel (str): X-axis label (default: None, auto-generated)
        ylabel (str): Y-axis label (default: None, auto-generated)
        xlim (tuple): X-axis limits as (min, max) (default: None)
        ylim (tuple): Y-axis limits as (min, max) (default: None)
        grid (bool): Whether to show grid (default: True)
        show (bool): Whether to display the plot (default: True)
        save_path (str): Path to save the plot (default: None)
        plotbackend (str): Plotting backend to use ('plotly', 'matplotlib', 'seaborn', or None for auto) (default: None)
        style (str): Plot style to use (default: None, uses backend's default style)
        save_html (str): Path to save as interactive HTML (default: None)
        dpi (int): DPI for saved image (default: None, uses CONFIG default)
        width (int): Width in pixels for plot (default: None)
        height (int): Height in pixels for plot (default: None)
        color (str): Color for scatter points (default: None, auto-generated)
        alpha (float): Transparency for scatter points (default: None, uses CONFIG default)
        marker_size (float): Size of scatter points (default: None, uses CONFIG default)
        figsize (tuple): Figure size in inches (default: None, uses CONFIG default)
        line (bool): Connect points with lines (default: False)
        fit_line (bool): Show linear regression fit line (default: False)
        fit_color (str): Color for fit line (default: None, uses CONFIG default)
        fit_line_width (float): Width of fit line (default: None, uses CONFIG default)
        fit_alpha (float): Transparency of fit line (default: None, uses CONFIG default)
        show_stats (bool): Show statistical information on the plot (default: False)
        downsampling (bool): Apply downsampling for large datasets (default: True)
        max_points (int): Maximum number of points to show before downsampling (default: 10000)
        density_plot (bool): Show density contour plot for large datasets (default: False)
        density_colorscale (str): Colorscale for density plot (default: None, uses CONFIG default)
        density_opacity (float): Opacity for density contours (default: None, uses CONFIG default)
        use_webgl (bool): Use WebGL rendering for better performance (default: True)
        adaptive_sampling (bool): Use adaptive sampling to preserve signal features (default: False)
        datashade (bool): Use datashading for very large datasets (default: False)
        contour_levels (int): Number of contour levels for density plot (default: 20)
        sampling_algorithm (str): Algorithm for downsampling ('lttb', 'uniform', 'peak') (default: 'lttb')
        memory_efficient (bool): Use memory-efficient methods for very large datasets (default: True)
        bin_size (tuple): Bin size for 2D histogram (x_bins, y_bins) (default: None, auto)
        sns_style (str): Seaborn style theme (default: None, uses CONFIG default)
        sns_bins (int): Number of bins for Seaborn histplot (default: 50)
        sns_pthresh (float): Threshold for Seaborn histplot (default: 0.1)
        sns_cmap (str): Colormap for Seaborn histplot (default: None, uses CONFIG default)
        sns_contour_levels (int): Number of levels for Seaborn kdeplot (default: 5)
        sns_contour_color (str): Color of contour lines for Seaborn kdeplot (default: None, uses CONFIG default)
        sns_linewidths (float): Line width for Seaborn kdeplot (default: None, uses CONFIG default)
        use_resampler (bool): Use plotly-resampler for dynamic downsampling (default: False)
        n_shown_samples (int): Number of samples to show initially (default: 5000)
        
    Returns:
        tuple: (pandas.DataFrame with x and y data, figure object)
    """
    # 使用配置默认值（如果未指定）
    if dpi is None:
        dpi = PLOT_CONFIG['elements']['dpi']
    if alpha is None:
        alpha = PLOT_CONFIG['elements']['alpha']
    if marker_size is None:
        marker_size = PLOT_CONFIG['elements']['marker_size']
    if figsize is None:
        figsize = PLOT_CONFIG['figsize']['square']
    if fit_color is None:
        fit_color = PLOT_CONFIG['colors']['fit']
    if fit_line_width is None:
        fit_line_width = PLOT_CONFIG['elements']['line_width']
    if fit_alpha is None:
        fit_alpha = PLOT_CONFIG['elements']['alpha']
    if density_colorscale is None:
        density_colorscale = PLOT_CONFIG['colors']['sequential']
    if density_opacity is None:
        density_opacity = 0.7
    if sns_style is None:
        sns_style = PLOT_CONFIG['style']['seaborn']['default']
    if sns_cmap is None:
        sns_cmap = 'mako'
    if sns_contour_color is None:
        sns_contour_color = 'w'
    if sns_linewidths is None:
        sns_linewidths = 1.0

    try:
        # Check if PyDAS object is valid
        if not hasattr(pydas_obj, 'chInfo') or not hasattr(pydas_obj, 'data'):
            logger.error("Invalid PyDAS object - missing required attributes")
            return None
        
        # Check if the segment index is valid
        if sseg < 0 or sseg >= len(pydas_obj.data):
            logger.error(f"Invalid segment index {sseg}, must be between 0 and {len(pydas_obj.data)-1}")
            return None
            
        # Check if channels exist
        if x_ch_name not in pydas_obj.data[sseg].columns:
            logger.error(f"X-axis channel '{x_ch_name}' not found in segment {sseg}")
            return None
        
        if y_ch_name not in pydas_obj.data[sseg].columns:
            logger.error(f"Y-axis channel '{y_ch_name}' not found in segment {sseg}")
            return None
        
        # Get data
        x_data = pydas_obj.data[sseg][x_ch_name]
        y_data = pydas_obj.data[sseg][y_ch_name]
        
        # Get channel units for labels
        x_unit = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == x_ch_name]['Unit'].values[0]
        y_unit = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == y_ch_name]['Unit'].values[0]
        
        # Create a DataFrame for the data
        df = pd.DataFrame({x_ch_name: x_data, y_ch_name: y_data})
        
        # Apply downsampling if needed and enabled
        if downsampling and len(df) > max_points:
            logger.info(f"Downsampling from {len(df)} to {max_points} points for plotting.")
            
            if sampling_algorithm == 'lttb' and len(df) > max_points:
                # Largest Triangle Three Buckets algorithm
                df_downsampled = _lttb_downsample(df[x_ch_name].values, df[y_ch_name].values, max_points)
                df = pd.DataFrame({x_ch_name: df_downsampled[0], y_ch_name: df_downsampled[1]})
            else:
                # Simple uniform downsampling
                step = int(len(df) / max_points)
                df = df.iloc[::step]
        
        # 获取实际可用的绘图后端
        backend = get_plot_backend(plotbackend)
        if backend is None:
            logger.error("No available plotting backend found")
            return None
            
        # 应用样式
        apply_style(backend, style)
        
        # Flag to track if we've successfully created a plot
        plot_created = False
        fig = None
        plt = None  # Initialize plt as None, import later as needed

        # If backend is seaborn, use Seaborn for layered bivariate plots
        if backend == 'seaborn':
            try:
                # Import required modules
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Create a figure
                f, ax = plt.subplots(figsize=figsize)
                
                # Create a layered bivariate plot
                # 1. Scatter plot (points)
                sns.scatterplot(
                    x=df[x_ch_name], 
                    y=df[y_ch_name], 
                    s=marker_size, 
                    color=color if color is not None else ".15",  # Default to dark gray like in example
                    alpha=alpha,
                    ax=ax
                )
                
                # 2. 2D histogram/heatmap (density of points)
                if density_plot:
                    sns.histplot(
                        x=df[x_ch_name], 
                        y=df[y_ch_name], 
                        bins=sns_bins, 
                        pthresh=sns_pthresh, 
                        cmap=sns_cmap,
                        ax=ax
                    )
                
                # 3. Contour plot (density contours)
                sns.kdeplot(
                    x=df[x_ch_name], 
                    y=df[y_ch_name], 
                    levels=sns_contour_levels, 
                    color=sns_contour_color, 
                    linewidths=sns_linewidths,
                    ax=ax
                )
                
                # Add linear regression fit line if requested
                if fit_line:
                    try:
                        # Calculate linear regression
                        import numpy as np
                        from scipy import stats
                        
                        # Remove NaN values
                        df_clean = df.dropna()
                        x_fit = df_clean[x_ch_name].values
                        y_fit = df_clean[y_ch_name].values
                        
                        if len(x_fit) > 1:  # Need at least 2 points for regression
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x_fit, y_fit)
                            
                            # Create fit line
                            x_range = np.linspace(df[x_ch_name].min(), df[x_ch_name].max(), 100)
                            y_fit_line = intercept + slope * x_range
                            
                            # Add fit line to plot
                            ax.plot(
                                x_range, 
                                y_fit_line, 
                                color=fit_color, 
                                linewidth=fit_line_width, 
                                alpha=fit_alpha,
                                label=f'Linear fit (y = {slope:.4g}x + {intercept:.4g}, r² = {r_value**2:.4g})'
                            )
                            
                            # Add legend
                            ax.legend(fontsize=PLOT_CONFIG['font']['size']['legend'])
                            
                            # Add fit statistics as text annotation
                            fit_text = f"y = {slope:.4g}x + {intercept:.4g}\nr² = {r_value**2:.4g}"
                            ax.text(
                                0.05, 0.95, fit_text, 
                                transform=ax.transAxes,
                                verticalalignment='top',
                                horizontalalignment='left',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                                fontsize=PLOT_CONFIG['font']['size']['annotation']
                            )
                    except Exception as e:
                        logger.warning(f"Error adding fit line: {e}")
                
                # Add statistical information if requested
                if show_stats:
                    # Calculate statistics
                    x_mean = np.mean(df[x_ch_name])
                    y_mean = np.mean(df[y_ch_name])
                    x_std = np.std(df[x_ch_name])
                    y_std = np.std(df[y_ch_name])
                    x_min = np.min(df[x_ch_name])
                    y_min = np.min(df[y_ch_name])
                    x_max = np.max(df[x_ch_name])
                    y_max = np.max(df[y_ch_name])
                    corr = df[x_ch_name].corr(df[y_ch_name])
                    
                    # Create stats string
                    stats_text = (
                        f"{x_ch_name}: μ={x_mean:.4g}, σ={x_std:.4g}, min={x_min:.4g}, max={x_max:.4g}\n"
                        f"{y_ch_name}: μ={y_mean:.4g}, σ={y_std:.4g}, min={y_min:.4g}, max={y_max:.4g}\n"
                        f"Correlation: {corr:.4g}"
                    )
                    
                    # Position text to not overlap with other elements
                    ax.text(
                        0.05, 0.05, stats_text, 
                        transform=ax.transAxes,
                        verticalalignment='bottom',
                        horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                        fontsize=PLOT_CONFIG['font']['size']['annotation']
                    )
                
                # Set plot title
                if title is None:
                    title = f"Bivariate Plot: {y_ch_name} vs {x_ch_name}"
                ax.set_title(title, fontsize=PLOT_CONFIG['font']['size']['title'])
                
                # Set axis labels
                if xlabel is None:
                    xlabel = f"{x_ch_name} ({x_unit})"
                if ylabel is None:
                    ylabel = f"{y_ch_name} ({y_unit})"
                ax.set_xlabel(xlabel, fontsize=PLOT_CONFIG['font']['size']['label'])
                ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG['font']['size']['label'])
                
                # 设置刻度字体大小
                ax.tick_params(axis='both', which='major', labelsize=PLOT_CONFIG['font']['size']['tick'])
                
                # Set grid
                ax.grid(grid)
                
                # Set axis limits if provided
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save figure if requested
                if save_path is not None:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                    logger.info(f"Plot saved to {save_path}")
                
                # Show plot if requested
                if show:
                    plt.show()
                else:
                    plt.close(f)
                
                plot_created = True
                fig = f  # Store figure for return
                
            except ImportError:
                logger.warning("Seaborn not available. Trying alternative backends.")
                backend = get_plot_backend('plotly')
            except Exception as e:
                logger.warning(f"Error using Seaborn: {str(e)}. Trying alternative backends.")
                backend = get_plot_backend('plotly')
        
        # If backend is plotly and Seaborn wasn't used or failed
        if backend == 'plotly' and not plot_created:
            try:
                # Import Plotly modules
                import plotly.graph_objects as go
                import plotly.express as px
                from plotly.subplots import make_subplots
                
                # 首先检查是否使用plotly-resampler
                if use_resampler and HAS_PLOTLY_RESAMPLER and len(df) > 10000:
                    logger.info(f"使用plotly-resampler处理大数据集XY图 ({len(df)} 点)")
                    
                    # 创建标题（如果未提供）
                    if title is None:
                        title = f"XY Plot: {y_ch_name} vs {x_ch_name}"
                    
                    # 创建可重采样图表
                    fr = create_resampable_plot(
                        x=df[x_ch_name],
                        y=df[y_ch_name],
                        name=f"{y_ch_name} vs {x_ch_name}",
                        title=title,
                        n_shown_samples=n_shown_samples
                    )
                    
                    if fr is not None:
                        # 配置布局
                        x_axis_label = xlabel if xlabel else f"{x_ch_name} ({x_unit})"
                        y_axis_label = ylabel if ylabel else f"{y_ch_name} ({y_unit})"
                        
                        fr.update_layout(
                            xaxis_title=x_axis_label,
                            yaxis_title=y_axis_label,
                            hovermode='closest',
                            width=width,
                            height=height
                        )
                        
                        # 设置坐标轴范围（如果提供）
                        if xlim:
                            fr.update_xaxes(range=xlim)
                        if ylim:
                            fr.update_yaxes(range=ylim)
                        
                        # 如果需要，添加回归线
                        if fit_line:
                            try:
                                # 计算线性回归
                                from scipy import stats as scipy_stats
                                
                                # 移除NaN值
                                df_clean = df.dropna()
                                x_fit = df_clean[x_ch_name].values
                                y_fit = df_clean[y_ch_name].values
                                
                                if len(x_fit) > 1:  # 至少需要2个点进行回归
                                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_fit, y_fit)
                                    
                                    # 创建拟合线
                                    x_range = np.linspace(df[x_ch_name].min(), df[x_ch_name].max(), 100)
                                    y_fit_line = intercept + slope * x_range
                                    
                                    # 添加拟合线到图表
                                    fr.add_trace(
                                        go.Scatter(
                                            x=x_range,
                                            y=y_fit_line,
                                            mode='lines',
                                            name=f'拟合线 (y = {slope:.4g}x + {intercept:.4g})',
                                            line=dict(color=fit_color, width=fit_line_width),
                                            opacity=fit_alpha
                                        )
                                    )
                            except Exception as e:
                                logger.warning(f"添加拟合线时出错: {e}")
                        
                        # 保存图表（如果需要）
                        if save_path:
                            try:
                                # 保存为PNG
                                png_path = save_path if save_path.endswith('.png') else save_path + '.png'
                                fr.write_image(png_path, width=width or 1200, height=height or 800)
                                logger.info(f"图表已保存至 {png_path}")
                            except Exception as e:
                                logger.warning(f"保存图像失败: {e}")
                        
                        if save_html:
                            try:
                                html_path = save_html if save_html.endswith('.html') else save_html + '.html'
                                fr.write_html(html_path)
                                logger.info(f"交互式HTML已保存至 {html_path}")
                            except Exception as e:
                                logger.warning(f"保存HTML失败: {e}")
                        
                        # 显示交互式图表
                        if show:
                            fr.show_dash()
                        
                        # 返回数据和图表
                        result_df = df.copy()
                        return result_df, fr
                
                # Create figure
                fig = go.Figure()
                
                # Create scatter plot
                if density_plot and len(df) > 1000:
                    # Create density scatter plot for large datasets
                    try:
                        # Use Plotly Express for density scatter
                        fig = px.density_contour(
                            df, x=x_ch_name, y=y_ch_name,
                            nbinsx=50, nbinsy=50,
                            color_scale=density_colorscale,
                            marginal_x="histogram", marginal_y="histogram"
                        )
                        
                        # Add scatter plot on top
                        fig.add_trace(
                            go.Scattergl(
                                x=df[x_ch_name],
                                y=df[y_ch_name],
                                mode='markers',
                                marker=dict(color=color if color is not None else 'blue', size=marker_size, opacity=alpha/2),
                                name='Data points',
                                showlegend=False
                            ) if use_webgl else go.Scatter(
                                x=df[x_ch_name],
                                y=df[y_ch_name],
                                mode='markers',
                                marker=dict(color=color if color is not None else 'blue', size=marker_size, opacity=alpha/2),
                                name='Data points',
                                showlegend=False
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Error creating density plot, falling back to regular scatter: {e}")
                        # Fall back to regular scatter plot
                        fig = go.Figure(
                            go.Scattergl(
                                x=df[x_ch_name],
                                y=df[y_ch_name],
                                mode='markers',
                                marker=dict(color=color if color is not None else 'blue', size=marker_size, opacity=alpha),
                                name='Data points'
                            ) if use_webgl else go.Scatter(
                                x=df[x_ch_name],
                                y=df[y_ch_name],
                                mode='markers',
                                marker=dict(color=color if color is not None else 'blue', size=marker_size, opacity=alpha),
                                name='Data points'
                            )
                        )
                else:
                    # Regular scatter plot
                    scatter_mode = 'markers+lines' if line else 'markers'
                    fig.add_trace(
                        go.Scattergl(
                            x=df[x_ch_name],
                            y=df[y_ch_name],
                            mode=scatter_mode,
                            marker=dict(color=color if color is not None else 'blue', size=marker_size, opacity=alpha),
                            name='Data points'
                        ) if use_webgl else go.Scatter(
                            x=df[x_ch_name],
                            y=df[y_ch_name],
                            mode=scatter_mode,
                            marker=dict(color=color if color is not None else 'blue', size=marker_size, opacity=alpha),
                            name='Data points'
                        )
                    )
                
                # Add linear regression fit line if requested
                if fit_line:
                    try:
                        # Calculate linear regression
                        import numpy as np
                        from scipy import stats
                        
                        # Remove NaN values
                        df_clean = df.dropna()
                        x_fit = df_clean[x_ch_name].values
                        y_fit = df_clean[y_ch_name].values
                        
                        if len(x_fit) > 1:  # Need at least 2 points for regression
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x_fit, y_fit)
                            
                            # Create fit line
                            x_range = np.linspace(df[x_ch_name].min(), df[x_ch_name].max(), 100)
                            y_fit_line = intercept + slope * x_range
                            
                            # Add fit line to plot
                            fig.add_trace(
                                go.Scatter(
                                    x=x_range,
                                    y=y_fit_line,
                                    mode='lines',
                                    name=f'Linear fit (y = {slope:.4g}x + {intercept:.4g})',
                                    line=dict(color=fit_color, width=fit_line_width),
                                    opacity=fit_alpha
                                )
                            )
                            
                            # Add fit statistics as annotation
                            fit_text = f"y = {slope:.4g}x + {intercept:.4g}<br>r² = {r_value**2:.4g}"
                            fig.add_annotation(
                                x=0.05, y=0.95,
                                xref="paper", yref="paper",
                                text=fit_text,
                                showarrow=False,
                                bgcolor="rgba(255, 255, 255, 0.7)",
                                bordercolor="gray",
                                borderwidth=1,
                                font=dict(size=PLOT_CONFIG['font']['size']['annotation']),
                                align="left"
                            )
                    except Exception as e:
                        logger.warning(f"Error adding fit line: {e}")
                
                # Set plot title
                if title is None:
                    title = f"XY Plot: {y_ch_name} vs {x_ch_name}"
                
                # Set axis labels
                if xlabel is None:
                    xlabel = f"{x_ch_name} ({x_unit})"
                if ylabel is None:
                    ylabel = f"{y_ch_name} ({y_unit})"
                
                # Update layout with config settings
                fig.update_layout(
                    title=title,
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=PLOT_CONFIG['font']['size']['legend'])
                    ),
                    hovermode="closest",
                    template=PLOT_CONFIG['style']['plotly'].get(style, PLOT_CONFIG['style']['plotly']['default']),
                    width=width,
                    height=height,
                    showlegend=True,
                    font=dict(
                        family=PLOT_CONFIG['font']['family'],
                        size=PLOT_CONFIG['font']['size']['medium']
                    ),
                    # 优化性能设置
                    uirevision='constant'  # 维持缩放级别
                )
                
                # Update axes
                fig.update_xaxes(showgrid=grid, zeroline=grid)
                fig.update_yaxes(showgrid=grid, zeroline=grid)
                
                # Set axis limits if provided
                if xlim is not None:
                    fig.update_xaxes(range=xlim)
                if ylim is not None:
                    fig.update_yaxes(range=ylim)
                
                # Add statistical information if requested
                if show_stats:
                    # Calculate statistics
                    x_mean = np.mean(df[x_ch_name])
                    y_mean = np.mean(df[y_ch_name])
                    x_std = np.std(df[x_ch_name])
                    y_std = np.std(df[y_ch_name])
                    x_min = np.min(df[x_ch_name])
                    y_min = np.min(df[y_ch_name])
                    x_max = np.max(df[x_ch_name])
                    y_max = np.max(df[y_ch_name])
                    corr = df[x_ch_name].corr(df[y_ch_name])
                    
                    # Create stats table
                    stats_table = go.Table(
                        header=dict(
                            values=["Statistic", x_ch_name, y_ch_name],
                            font=dict(size=PLOT_CONFIG['stats']['table_font_size']),
                            align="left",
                            fill=dict(color=PLOT_CONFIG['stats']['header_color'])
                        ),
                        cells=dict(
                            values=[
                                ["Mean", "Std", "Min", "Max", "Correlation"],
                                [f"{x_mean:.4g}", f"{x_std:.4g}", f"{x_min:.4g}", f"{x_max:.4g}", f"{corr:.4g}"],
                                [f"{y_mean:.4g}", f"{y_std:.4g}", f"{y_min:.4g}", f"{y_max:.4g}", ""]
                            ],
                            font=dict(size=PLOT_CONFIG['font']['size']['small']),
                            align="left"
                        ),
                        domain=dict(x=[0.7, 1], y=[0, PLOT_CONFIG['stats']['table_height']])
                    )
                    
                    fig.add_trace(stats_table)
                
                # 创建一个应用于每个图的设置字典
                plot_settings = {
                    "scrollZoom": True,  # 启用鼠标滚轮缩放
                    "modeBarButtonsToAdd": ["drawopenpath", "eraseshape"],  # 添加绘图工具
                    "modeBarButtonsToRemove": ["lasso2d"]  # 移除套索选择
                }
                
                # Save as HTML if requested
                if save_html is not None:
                    fig.write_html(save_html, config=plot_settings)
                    logger.info(f"Interactive plot saved to {save_html}")
                
                # Save as image if requested
                if save_path is not None:
                    fig.write_image(save_path, width=width or 1000, height=height or 1000, scale=2)
                    logger.info(f"Plot saved to {save_path}")
                
                # Show plot if requested
                if show:
                    fig.show(config=plot_settings)
                
                plot_created = True
                
            except ImportError:
                logger.warning("Plotly not available. Falling back to matplotlib.")
                backend = 'matplotlib'
            except Exception as e:
                logger.warning(f"Error using Plotly: {str(e)}. Falling back to matplotlib.")
                backend = 'matplotlib'
        
        # If backend is matplotlib or all other backends failed
        if backend == 'matplotlib' or not plot_created:
            try:
                import matplotlib.pyplot as plt
                from matplotlib.colors import LogNorm
                
                # Create figure and axis
                fig, ax = plt.subplots(figsize=figsize)
                
                # Create scatter plot
                if density_plot and len(df) > 1000:
                    try:
                        # Create 2D histogram first
                        h, xedges, yedges = np.histogram2d(
                            df[x_ch_name], 
                            df[y_ch_name], 
                            bins=50 if bin_size is None else bin_size
                        )
                        
                        # Create contour plot
                        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
                        contour = ax.contourf(
                            X, Y, h.T, 
                            levels=contour_levels,
                            cmap=density_colorscale,
                            alpha=density_opacity
                        )
                        plt.colorbar(contour, ax=ax, label='Density')
                        
                        # Add scatter plot with reduced opacity
                        ax.scatter(df[x_ch_name], df[y_ch_name], 
                                  s=marker_size, color=color if color is not None else 'blue', alpha=alpha/2)
                    except Exception as e:
                        logger.warning(f"Error creating density plot, falling back to regular scatter: {e}")
                        # Fall back to regular scatter plot
                        ax.scatter(df[x_ch_name], df[y_ch_name], 
                                 s=marker_size, color=color if color is not None else 'blue', alpha=alpha)
                else:
                    # Regular scatter plot
                    if line:
                        ax.plot(df[x_ch_name], df[y_ch_name], 
                               marker='o', markersize=marker_size, color=color if color is not None else 'blue',
                               alpha=alpha, linestyle='-', linewidth=PLOT_CONFIG['elements']['line_width'])
                    else:
                        ax.scatter(df[x_ch_name], df[y_ch_name], 
                                  s=marker_size, color=color if color is not None else 'blue', alpha=alpha)
                
                # Add linear regression fit line if requested
                if fit_line:
                    try:
                        # Calculate linear regression
                        from scipy import stats
                        
                        # Remove NaN values
                        df_clean = df.dropna()
                        x_fit = df_clean[x_ch_name].values
                        y_fit = df_clean[y_ch_name].values
                        
                        if len(x_fit) > 1:  # Need at least 2 points for regression
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x_fit, y_fit)
                            
                            # Create fit line
                            x_range = np.linspace(df[x_ch_name].min(), df[x_ch_name].max(), 100)
                            y_fit_line = intercept + slope * x_range
                            
                            # Add fit line to plot
                            ax.plot(x_range, y_fit_line, color=fit_color, 
                                   linewidth=fit_line_width, alpha=fit_alpha,
                                   label=f'Linear fit (y = {slope:.4g}x + {intercept:.4g}, r² = {r_value**2:.4g})')
                            
                            # Add legend
                            ax.legend(fontsize=PLOT_CONFIG['font']['size']['legend'])
                    except Exception as e:
                        logger.warning(f"Error adding fit line: {e}")
                
                # Set plot title
                if title is None:
                    title = f"XY Plot: {y_ch_name} vs {x_ch_name}"
                ax.set_title(title, fontsize=PLOT_CONFIG['font']['size']['title'])
                
                # Set axis labels
                if xlabel is None:
                    xlabel = f"{x_ch_name} ({x_unit})"
                if ylabel is None:
                    ylabel = f"{y_ch_name} ({y_unit})"
                ax.set_xlabel(xlabel, fontsize=PLOT_CONFIG['font']['size']['label'])
                ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG['font']['size']['label'])
                
                # 设置刻度字体大小
                ax.tick_params(axis='both', which='major', labelsize=PLOT_CONFIG['font']['size']['tick'])
                
                # Set grid
                ax.grid(grid)
                
                # Set axis limits if provided
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                
                # Add statistical information if requested
                if show_stats:
                    # Calculate statistics
                    x_mean = np.mean(df[x_ch_name])
                    y_mean = np.mean(df[y_ch_name])
                    x_std = np.std(df[x_ch_name])
                    y_std = np.std(df[y_ch_name])
                    x_min = np.min(df[x_ch_name])
                    y_min = np.min(df[y_ch_name])
                    x_max = np.max(df[x_ch_name])
                    y_max = np.max(df[y_ch_name])
                    corr = df[x_ch_name].corr(df[y_ch_name])
                    
                    # Create stats string
                    stats_text = (
                        f"{x_ch_name}: μ={x_mean:.4g}, σ={x_std:.4g}, min={x_min:.4g}, max={x_max:.4g}\n"
                        f"{y_ch_name}: μ={y_mean:.4g}, σ={y_std:.4g}, min={y_min:.4g}, max={y_max:.4g}\n"
                        f"Correlation: {corr:.4g}"
                    )
                    
                    # Add stats text to plot
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                           fontsize=PLOT_CONFIG['font']['size']['annotation'])
                
                # Adjust layout
                plt.tight_layout()
                
                # Save figure if requested
                if save_path is not None:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                    logger.info(f"Plot saved to {save_path}")
                
                # Show plot if requested
                if show:
                    plt.show()
                else:
                    plt.close(fig)
                
                plot_created = True
                
            except ImportError:
                logger.error("No available plotting libraries found (matplotlib, seaborn, plotly).")
                return None
        
        # Return data and fig object if not showing or just data if showing
        return None
        
    except Exception as e:
        logger.error(f"Error in plot_xy: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def _lttb_downsample(data_x, data_y, n_out):
    """
    Downsample data using the Largest Triangle Three Buckets algorithm.
    This algorithm preserves the visual characteristics of the data.
    
    Parameters:
    -----------
    data_x : array-like
        X-coordinates of the data points
    data_y : array-like
        Y-coordinates of the data points
    n_out : int
        Number of output points
        
    Returns:
    --------
    tuple
        (x_downsampled, y_downsampled) - downsampled data points
    """
    n = len(data_x)
    if n <= n_out:
        return data_x, data_y
    
    # Convert to numpy arrays if not already
    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    
    # Create output arrays for downsampled data
    out_x = np.zeros(n_out)
    out_y = np.zeros(n_out)
    
    # Always include the first point
    out_x[0] = data_x[0]
    out_y[0] = data_y[0]
    
    # Always include the last point
    out_x[n_out-1] = data_x[n-1]
    out_y[n_out-1] = data_y[n-1]
    
    # If output size is 2, we're done
    if n_out == 2:
        return out_x, out_y
    
    # Bucket size
    bucket_size = (n - 2) / (n_out - 2)
    
    # Process all other output points
    for i in range(1, n_out-1):
        # Calculate bucket range
        bucket_start = int((i - 1) * bucket_size) + 1
        bucket_end = int(i * bucket_size) + 1
        
        # Ensure bucket_end doesn't exceed array bounds
        if bucket_end >= n:
            bucket_end = n - 1
        
        # Point from the previous bucket
        prev_x = out_x[i-1]
        prev_y = out_y[i-1]
        
        # Calculate areas of triangles formed by the point from the previous bucket,
        # the point from the next bucket, and each point in the current bucket
        max_area = -1
        max_area_idx = bucket_start
        
        # Find the point in the bucket with the largest triangle area
        for j in range(bucket_start, bucket_end):
            # For the last point in the output, compare with the actual last point of the input
            if i == n_out - 2:
                next_x = data_x[n-1]
                next_y = data_y[n-1]
            else:
                # Otherwise, compare with a representative point from the next bucket
                next_x = data_x[bucket_end]
                next_y = data_y[bucket_end]
            
            # Calculate triangle area
            area = abs((prev_x - next_x) * (data_y[j] - prev_y) - 
                       (prev_x - data_x[j]) * (next_y - prev_y)) * 0.5
            
            # Update if this is the largest area so far
            if area > max_area:
                max_area = area
                max_area_idx = j
        
        # Save the point with the largest area
        out_x[i] = data_x[max_area_idx]
        out_y[i] = data_y[max_area_idx]
    
    return out_x, out_y

def _plot_statistics_mpl(pydas_obj, ch_names, sseg, stats_df, bins, save_fig, save_path, data=None, title_override=None):
    """Using matplotlib to plot statistical analysis graph"""
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import scipy.signal as signal
    import numpy as np
    import matplotlib.gridspec as gridspec
    
    for name in ch_names:
        # Create a figure with grid specification
        fig = plt.figure(figsize=(14, 12))
        # Use GridSpec to create custom layout
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
        
        # Set title, use custom title if provided
        if title_override:
            plt.suptitle(title_override, fontsize=16)
        else:
            plt.suptitle(f'Statistical Analysis for Channel: {name} (Segment {sseg})', fontsize=16)
        
        # Get data
        if data is None:
            data = pydas_obj.data[sseg][name].values
            
        # Calculate time axis in seconds
        dt = 1.0 / pydas_obj.__fs__  # Time step in seconds
        time_axis = np.arange(0, len(data) * dt, dt)
            
        # Detect peaks (calculate before plotting to mark in time series)
        data_abs = np.abs(data)  # Consider both positive and negative peaks
        peaks, _ = signal.find_peaks(data_abs, height=np.mean(data_abs) + 0.5 * np.std(data_abs))
        peak_values = data_abs[peaks]
        
        # If too few peaks found, lower threshold and redetect
        if len(peak_values) < bins / 5:
            peaks, _ = signal.find_peaks(data_abs, height=np.mean(data_abs))
            peak_values = data_abs[peaks]
        
        # Get channel unit
        unit = stats_df.loc[name, 'Unit']
        
        # 1. Time series plot (full width)
        ax1 = plt.subplot(gs[0, :])  # Span the first row with two columns
        ax1.plot(time_axis, data)
        # Mark peak positions in time series
        if len(peaks) > 0:
            ax1.plot(time_axis[peaks], data[peaks], 'ro', markersize=3, alpha=0.6)
        ax1.set_title('Time Series')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(f'{name} [{unit}]')
        ax1.grid(True)
        
        # 2. Histogram and PDF (second row, left)
        ax2 = plt.subplot(gs[1, 0])
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        ax2.bar(bin_centers, hist, width=bin_centers[1]-bin_centers[0], 
                     alpha=0.6, color='skyblue', label='Histogram')
        
        # Fit normal distribution
        mu, sigma = stats.norm.fit(data)
        x = np.linspace(min(data), max(data), 100)
        pdf = stats.norm.pdf(x, mu, sigma)
        ax2.plot(x, pdf, 'r-', lw=2, label=f'Normal PDF\n(μ={mu:.2E}, σ={sigma:.2E})')
        
        ax2.set_title('Histogram and PDF')
        ax2.set_xlabel(f'{name} [{unit}]')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Empirical cumulative distribution function (ECDF) (second row, right)
        ax3 = plt.subplot(gs[1, 1])
        sorted_data = np.sort(data)
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        ax3.step(sorted_data, ecdf, where='post', label='ECDF')
        
        # Theoretical CDF
        cdf = stats.norm.cdf(x, mu, sigma)
        ax3.plot(x, cdf, 'r-', lw=2, label='Normal CDF')
        
        ax3.set_title('Empirical CDF')
        ax3.set_xlabel(f'{name} [{unit}]')
        ax3.set_ylabel('Probability')
        ax3.grid(True)
        ax3.legend()
        
        # 4. Q-Q plot (third row, left)
        ax4 = plt.subplot(gs[2, 0])
        stats.probplot(data, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Normal Distribution)')
        ax4.grid(True)
        
        # 5. Peak value probability density function (third row, right)
        ax5 = plt.subplot(gs[2, 1])
        
        # Plot probability density function of peaks
        if len(peak_values) > 1:
            # Peak histogram
            hist_peaks, bin_edges_peaks = np.histogram(peak_values, bins=min(bins, len(peak_values)//2 + 5), density=True)
            bin_centers_peaks = (bin_edges_peaks[:-1] + bin_edges_peaks[1:]) / 2
            
            ax5.bar(bin_centers_peaks, hist_peaks, 
                          width=bin_centers_peaks[1]-bin_centers_peaks[0] if len(bin_centers_peaks) > 1 else 0.1,
                          alpha=0.6, color='salmon', label='Peak Histogram')
            
            # Try to fit normal distribution
            try:
                mu_peaks, sigma_peaks = stats.norm.fit(peak_values)
                x_peaks = np.linspace(min(peak_values), max(peak_values), 100)
                pdf_peaks = stats.norm.pdf(x_peaks, mu_peaks, sigma_peaks)
                ax5.plot(x_peaks, pdf_peaks, 'g-', lw=2, 
                              label=f'Peak PDF\n(μ={mu_peaks:.2E}, σ={sigma_peaks:.2E})')
            except:
                # Fitting failed, ignore
                pass
            
            ax5.set_title('Peak Value PDF')
            ax5.set_xlabel(f'Peak Magnitude [{unit}]')
            ax5.set_ylabel('Density')
            ax5.legend()
            ax5.grid(True)
            
            # Add peak statistics info
            peak_stats_text = (f"Peak Count: {len(peak_values)}\n"
                              f"Mean: {np.mean(peak_values):.4E}\n"
                              f"Max: {np.max(peak_values):.4E}\n"
                              f"Min: {np.min(peak_values):.4E}")
            ax5.text(0.05, 0.95, peak_stats_text, transform=ax5.transAxes, 
                          fontsize=9, verticalalignment='top', 
                          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        else:
            # Too few peaks, cannot plot probability density function
            ax5.text(0.5, 0.5, "Insufficient peaks detected for analysis", 
                         ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Peak Value PDF')
            ax5.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_fig:
            if save_path is None:
                save_path = os.getcwd()
            plt.savefig(f"{save_path}/{name}_seg{sseg}_stats.png", dpi=300, bbox_inches='tight')
        
        plt.show()

def _plot_statistics_plotly(pydas_obj, ch_names, sseg, stats_df, bins, save_fig, save_path, data=None, title_override=None, use_webgl=True, max_points=50000):
    """Using plotly to plot statistical analysis graph"""
    try:
        import plotly.graph_objects as go
        import plotly.subplots as sp
        import plotly.figure_factory as ff
        import numpy as np
        import scipy.stats as stats
        import scipy.signal as signal
        
        for name in ch_names:
            # Create subplots with new layout
            fig = sp.make_subplots(
                rows=3, cols=2,
                # First row is full-width time series, others maintain
                column_widths=[0.5, 0.5],
                row_heights=[0.33, 0.33, 0.33],
                subplot_titles=(
                    'Time Series', '',  # First title spans entire row
                    'Histogram and PDF', 'Empirical CDF',
                    'Q-Q Plot (Normal Distribution)', 'Peak Value PDF'
                ),
                specs=[
                    [{"colspan": 2}, None],  # First row: Full-width time series
                    [{"type": "scatter"}, {"type": "scatter"}],  # Second row: Histogram|CDF
                    [{"type": "scatter"}, {"type": "scatter"}]   # Third row: Q-Q plot|Peak PDF
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Get data
            if data is None:
                data = pydas_obj.data[sseg][name].values
            
            # Check data size and downsample if needed
            data_length = len(data)
            downsample = data_length > max_points
            
            # Calculate time axis in seconds
            dt = 1.0 / pydas_obj.__fs__  # Time step in seconds
            time_axis = np.arange(0, data_length * dt, dt)
            
            if downsample:
                logger.info(f"Downsampling data from {data_length} to {max_points} points for plotting")
                # Calculate downsample step
                step = int(data_length / max_points)
                # Basic uniform downsampling for visualization
                indices = np.arange(0, data_length, step)
                plot_data = data[indices]
                plot_time = time_axis[indices]
            else:
                # Use original data
                plot_data = data
                plot_time = time_axis
                
            # Detect peaks
            data_abs = np.abs(data)  # Consider both positive and negative peaks
            peaks, _ = signal.find_peaks(data_abs, height=np.mean(data_abs) + 0.5 * np.std(data_abs))
            peak_values = data_abs[peaks]
            
            # If too few peaks found, lower threshold and redetect
            if len(peak_values) < bins / 5:
                peaks, _ = signal.find_peaks(data_abs, height=np.mean(data_abs))
                peak_values = data_abs[peaks]
            
            # Use WebGL for better performance with large datasets if requested
            scatter_type = go.Scattergl if use_webgl else go.Scatter
            
            # 1. Time series plot (full width)
            fig.add_trace(
                scatter_type(
                    x=plot_time,
                    y=plot_data,
                    mode='lines',
                    name='Time Series'
                ),
                row=1, col=1  # Place in first row, it will span both columns due to colspan=2
            )
            
            # Mark peak positions in time series
            if len(peaks) > 0:
                # If data was downsampled, we need to filter peaks to only show those in the plot
                if downsample:
                    # Find peaks that are included in the downsampled indices
                    mask = np.isin(peaks, indices)
                    visible_peaks = peaks[mask] if any(mask) else []
                    visible_peak_values = data[visible_peaks] if len(visible_peaks) > 0 else []
                    
                    peak_times = time_axis[visible_peaks]
                    peak_data = visible_peak_values
                else:
                    peak_times = time_axis[peaks]
                    peak_data = data[peaks]
                
                if len(peak_times) > 0:
                    fig.add_trace(
                        scatter_type(
                            x=peak_times,
                            y=peak_data,
                            mode='markers',
                            name='Peaks',
                            marker=dict(color='red', size=6),
                            showlegend=True
                        ),
                        row=1, col=1
                    )
            
            # Update x-axis label for time series
            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            
            # Update y-axis label with unit
            unit = stats_df.loc[name, 'Unit']
            fig.update_yaxes(title_text=f"Value ({unit})", row=1, col=1)
            
            # 2. Histogram and PDF
            hist_data = ff.create_distplot(
                [data],
                [name],
                bin_size=(np.max(data) - np.min(data)) / bins,
                show_curve=True,
                show_rug=False
            )
            
            fig.add_trace(
                hist_data['data'][0],  # Histogram
                row=2, col=1
            )
            
            fig.add_trace(
                hist_data['data'][1],  # PDF
                row=2, col=1
            )
            
            # Update x-axis label for histogram
            fig.update_xaxes(title_text=f"Value ({unit})", row=2, col=1)
            fig.update_yaxes(title_text="Density", row=2, col=1)
            
            # 3. Empirical CDF
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            fig.add_trace(
                scatter_type(
                    x=sorted_data,
                    y=cdf,
                    mode='lines',
                    name='Empirical CDF'
                ),
                row=2, col=2
            )
            
            # Update x-axis label for CDF
            fig.update_xaxes(title_text=f"Value ({unit})", row=2, col=2)
            fig.update_yaxes(title_text="Cumulative Probability", row=2, col=2)
            
            # 4. Q-Q Plot
            qq = stats.probplot(data, dist="norm")
            theoretical_quantiles = qq[0][0]
            sample_quantiles = qq[0][1]
            
            fig.add_trace(
                scatter_type(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode='markers',
                    name='Q-Q Plot'
                ),
                row=3, col=1
            )
            
            # Add reference line
            min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
            max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
            fig.add_trace(
                scatter_type(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Reference Line',
                    line=dict(color='red', dash='dash')
                ),
                row=3, col=1
            )
            
            # Update x-axis label for Q-Q plot
            fig.update_xaxes(title_text="Theoretical Quantiles", row=3, col=1)
            fig.update_yaxes(title_text="Sample Quantiles", row=3, col=1)
            
            # 5. Peak Value PDF
            if len(peak_values) > 0:
                peak_hist = ff.create_distplot(
                    [peak_values],
                    ['Peak Values'],
                    bin_size=(np.max(peak_values) - np.min(peak_values)) / bins,
                    show_curve=True,
                    show_rug=False
                )
                    
                fig.add_trace(
                    peak_hist['data'][0],  # Histogram
                        row=3, col=2
                    )
                
                fig.add_trace(
                    peak_hist['data'][1],  # PDF
                    row=3, col=2
                )
                
                # Update x-axis label for peak PDF
                fig.update_xaxes(title_text=f"Peak Value ({unit})", row=3, col=2)
                fig.update_yaxes(title_text="Density", row=3, col=2)
            
            # Update layout
            fig.update_layout(
                title=title_override if title_override else f"Statistical Analysis for {name}",
                height=1200,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Save figure if requested
            if save_fig:
                if save_path is None:
                    save_path = f"statistics_{name}.html"
                fig.write_html(save_path)
            
            # Show figure
            fig.show()
            
    except Exception as e:
        logger.error(f"Error in plotly statistical visualization: {e}")
        raise

def boxplot_channel(pydas_obj, ch_name, sseg=0, title=None, xlabel=None, ylabel=None, 
                  xlim=None, ylim=None, grid=True, show=True, save_path=None, 
                  plotbackend=None, style=None, save_html=None, dpi=None, width=None, height=None, 
                  color=None, alpha=0.8, figsize=None, notch=False, vert=True, showfliers=True,
                  showmeans=False, meanline=False, boxprops=None, whiskerprops=None, 
                  capprops=None, flierprops=None, medianprops=None, meanprops=None,
                  pointpos=0, jitter=0.3, boxpoints='outliers', quartilemethod='linear',
                  boxwidth=0.5, orientation=None, use_peaks=False, separate_pos_neg_peaks=False,
                  peak_height=None, peak_threshold=None, peak_distance=None, peak_prominence=1.0,
                  peak_width=None, peak_wlen=None, peak_rel_height=0.5):
    """
    Plot a boxplot of one or multiple channels from a PyDAS object.
    
    Parameters:
        pydas_obj (PyDAS): The PyDAS object containing channel data
        ch_name (str or list): Channel name or list of channel names
        sseg (int): Segment index to plot (default: 0)
        title (str): Plot title (default: None, auto-generated)
        xlabel (str): X-axis label (default: None, auto-generated)
        ylabel (str): Y-axis label (default: None, auto-generated)
        xlim (tuple): X-axis limits as (min, max) (default: None)
        ylim (tuple): Y-axis limits as (min, max) (default: None)
        grid (bool): Whether to show grid (default: True)
        show (bool): Whether to display the plot (default: True)
        save_path (str): Path to save the plot (default: None)
        plotbackend (str): Plotting backend to use ('plotly', 'matplotlib', 'seaborn', or None for auto) (default: None)
        style (str): Plot style to use (default: None, uses backend's default style)
        save_html (str): Path to save as interactive HTML (default: None, only works with plotly backend)
        dpi (int): DPI for saved image (default: None, uses CONFIG default)
        width (int): Width in pixels for plot (default: None)
        height (int): Height in pixels for plot (default: None)
        color (str or list): Box color(s) (default: None, auto-generated)
        alpha (float): Transparency level (default: 0.8)
        figsize (tuple): Figure size in inches (default: None, uses CONFIG default)
        notch (bool): Whether to create notched boxes (default: False)
        vert (bool): For matplotlib, if True, boxes are drawn vertical (default: True)
        showfliers (bool): Whether to show outliers (default: True)
        showmeans (bool): Whether to show mean line (default: False)
        meanline (bool): Whether to show the mean as a line instead of a point (default: False)
        boxprops (dict): Properties for the box (matplotlib only) (default: None)
        whiskerprops (dict): Properties for the whiskers (matplotlib only) (default: None)
        capprops (dict): Properties for the caps (matplotlib only) (default: None)
        flierprops (dict): Properties for the fliers (matplotlib only) (default: None)
        medianprops (dict): Properties for the median (matplotlib only) (default: None)
        meanprops (dict): Properties for the mean (matplotlib only) (default: None)
        pointpos (float): Position of points in boxplot - 0 means points are placed over the center of the box, negative/positive values offset the points (plotly only) (default: 0)
        jitter (float): Jitter amount for points (plotly only) (default: 0.3)
        boxpoints (str): Display mode for points ('all', 'outliers', 'suspectedoutliers', False) (plotly only) (default: 'outliers')
        quartilemethod (str): Method for computing quartiles (plotly only) (default: 'linear')
        boxwidth (float): Width of boxes (default: 0.5)
        orientation (str): 'v' for vertical, 'h' for horizontal (default: None)
        use_peaks (bool): Whether to use peak values for boxplot instead of all data (default: False)
        separate_pos_neg_peaks (bool): Whether to separate positive and negative peaks into different boxes (default: False)
        peak_height (float or tuple): Required height of peaks (default: None)
        peak_threshold (float or tuple): Required threshold of peaks (default: None)
        peak_distance (int): Required minimal horizontal distance between peaks (default: None) 
        peak_prominence (float or tuple): Required prominence of peaks (default: 1.0)
        peak_width (float or tuple): Required width of peaks (default: None)
        peak_wlen (int): Window length for peak prominence calculation (default: None)
        peak_rel_height (float): Relative height for peak width calculation (default: 0.5)
        
    Returns:
        object: Figure object or None
    """
    try:
        # Input validation
        if pydas_obj is None:
            logger.error("PyDAS object cannot be None")
            return None
        
        if not hasattr(pydas_obj, 'data') or not hasattr(pydas_obj, 'chInfo'):
            logger.error("Invalid PyDAS object")
            return None
        
        # Validate and convert channel name/index
        ch_names = validate_channel(pydas_obj, ch_name)
        if ch_names is None:
            return None
        
        # Determine if single channel or multiple channels
        # 显式处理单通道情况，避免DataFrame歧义
        if isinstance(ch_names, list):
            channel_list = ch_names
            is_list = True
        else:
            # 如果不是列表，那么是单个通道名（字符串）
            channel_list = [ch_names]
            is_list = False
        
        # Get plot backend
        backend = get_plot_backend(plotbackend)
        if backend is None:
            logger.error("No available plotting backend found.")
            return None
        
        # 确保channel_list中的每个元素都是字符串
        for i, ch in enumerate(channel_list):
            if not isinstance(ch, str):
                logger.warning(f"Channel at index {i} is not a string. Converting to string.")
                channel_list[i] = str(ch)
        
        # Apply style
        apply_style(backend, style)
        
        # Set default figsize if not provided
        if figsize is None:
            figsize = PLOT_CONFIG['figsize']['medium']
        
        # Set default dpi if not provided
        if dpi is None:
            dpi = PLOT_CONFIG['elements']['dpi']
        
        # Set default alpha if not provided
        if alpha is None:
            alpha = PLOT_CONFIG['elements']['alpha']
        
        # Initialize flag to track if plot was created
        plot_created = False
        
        # If backend is plotly
        if backend == 'plotly':
            try:
                import matplotlib.pyplot as plt  # 导入这里需要的plt
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                import numpy as np
                
                # Create figure with appropriate subplots
                n_channels = len(channel_list)
                
                # Initialize figure
                fig = go.Figure()
                
                # Process each channel
                data_list = []
                names_list = []
                
                for channel in channel_list:
                    # 确保channel是字符串
                    channel = str(channel)
                    
                    # Check if channel exists
                    if channel not in pydas_obj.data[sseg].columns:
                        logger.warning(f"Channel '{channel}' not found in segment {sseg}, skipping.")
                        continue
                    
                    # Get data
                    raw_data = pydas_obj.data[sseg][channel].dropna()
                    
                    # 确保数据非空
                    if raw_data.empty:
                        logger.warning(f"Channel '{channel}' contains no valid data after dropping NaN values, skipping.")
                        continue
                    
                    # 获取通道单位
                    channel_info = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel]
                    if channel_info.empty:
                        unit = ""
                        logger.warning(f"Could not find unit information for channel '{channel}'")
                    else:
                        unit = channel_info['Unit'].values[0]
                    
                    # 判断是否使用峰值分析
                    if use_peaks:
                        # 检测峰值
                        pos_peaks, neg_peaks = _detect_peaks(
                            raw_data, 
                            height=peak_height, 
                            threshold=peak_threshold,
                            distance=peak_distance, 
                            prominence=peak_prominence,
                            width=peak_width, 
                            wlen=peak_wlen, 
                            rel_height=peak_rel_height
                        )
                        
                        # 分离正负峰值处理
                        if separate_pos_neg_peaks:
                            # 如果有正峰值，添加到数据列表
                            if len(pos_peaks) > 0:
                                data_list.append(pos_peaks)
                                names_list.append(f"{channel} (+) ({unit})")
                            else:
                                logger.warning(f"No positive peaks found for channel '{channel}'")
                            
                            # 如果有负峰值，添加到数据列表
                            if len(neg_peaks) > 0:
                                data_list.append(neg_peaks)
                                names_list.append(f"{channel} (-) ({unit})")
                            else:
                                logger.warning(f"No negative peaks found for channel '{channel}'")
                        else:
                            # 合并所有峰值
                            all_peaks = np.concatenate([pos_peaks, neg_peaks])
                            if len(all_peaks) > 0:
                                data_list.append(all_peaks)
                                names_list.append(f"{channel} (Peaks) ({unit})")
                            else:
                                logger.warning(f"No peaks found for channel '{channel}'")
                    else:
                        # 使用全部数据
                        data_list.append(raw_data)
                        names_list.append(f"{channel} ({unit})")
                
                # 如果没有有效数据，返回None
                if not data_list:
                    logger.error("No valid data to plot")
                    return None
                
                # Create color palette for multiple channels
                if color is None:
                    colorscale = PLOT_CONFIG['colors']['qualitative']
                    colors = [f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})" 
                             for r, g, b, _ in plt.cm.get_cmap(colorscale)(np.linspace(0, 1, n_channels))]
                elif isinstance(color, list):
                    colors = color
                else:
                    colors = [color] * n_channels
                
                # Determine orientation
                plot_orientation = orientation or ('v' if vert else 'h')
                
                # Add boxplot traces
                for i, (data, name) in enumerate(zip(data_list, names_list)):
                    fig.add_trace(go.Box(
                        y=data if plot_orientation == 'v' else None,
                        x=data if plot_orientation == 'h' else None,
                        name=name,
                        marker_color=colors[i] if i < len(colors) else None,
                        boxmean=showmeans,
                        notched=notch,
                        boxpoints=boxpoints,
                        jitter=jitter,
                        pointpos=pointpos,
                        quartilemethod=quartilemethod,
                        width=boxwidth,  # 在Plotly中使用width而不是boxwidth
                        orientation=plot_orientation
                    ))
                
                # Set plot title
                if title is None:
                    title = "Boxplot Analysis"
                    if use_peaks:
                        title += " (Peak Values)"
                    if not is_list and channel_list:  # 确保channel_list非空
                        title += f" - {channel_list[0]}"
                
                # Set axis labels
                x_title = xlabel
                y_title = ylabel
                
                if x_title is None and plot_orientation == 'v':
                    x_title = "Channel"
                elif y_title is None and plot_orientation == 'h':
                    y_title = "Channel"
                
                if y_title is None and plot_orientation == 'v':
                    y_title = "Value"
                elif x_title is None and plot_orientation == 'h':
                    x_title = "Value"
                
                # Update layout
                fig.update_layout(
                    title=title,
                    xaxis_title=x_title,
                    yaxis_title=y_title,
                    boxmode='group',
                    template=PLOT_CONFIG['style']['plotly'].get(style, PLOT_CONFIG['style']['plotly']['default']),
                    width=width or 1200,
                    height=height or 800,
                    margin=dict(l=50, r=50, t=50, b=50),
                    font=dict(
                        family=PLOT_CONFIG['font']['family'],
                        size=PLOT_CONFIG['font']['size']['medium']
                    ),
                )
                
                # Update axes
                fig.update_xaxes(showgrid=grid, zeroline=grid)
                fig.update_yaxes(showgrid=grid, zeroline=grid)
                
                # Set axis limits if provided
                if xlim is not None:
                    fig.update_xaxes(range=xlim)
                if ylim is not None:
                    fig.update_yaxes(range=ylim)
                
                # Save as HTML if requested
                if save_html is not None:
                    fig.write_html(save_html)
                    logger.info(f"Interactive boxplot saved to {save_html}")
                
                # Save as image if requested
                if save_path is not None:
                    fig.write_image(save_path, width=width or 1200, height=height or 800, scale=2)
                    logger.info(f"Boxplot saved to {save_path}")
                
                # Show plot if requested
                if show:
                    plot_settings = {
                        "scrollZoom": True,
                        "modeBarButtonsToAdd": ["drawopenpath", "eraseshape"],
                        "modeBarButtonsToRemove": ["lasso2d"]
                    }
                    fig.show(config=plot_settings)
                
                plot_created = True
                
            except ImportError:
                logger.warning("Plotly not available. Falling back to matplotlib.")
                backend = 'matplotlib'
            except Exception as e:
                logger.warning(f"Error using Plotly: {str(e)}. Falling back to matplotlib.")
                backend = 'matplotlib'
        
        # If backend is seaborn or matplotlib failed
        if backend == 'seaborn' or (backend == 'matplotlib' and not plot_created):
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                import numpy as np
                
                # Create figure
                fig, ax = plt.subplots(figsize=figsize)
                
                # Prepare data for boxplot
                data_list = []
                labels = []
                
                # Process each channel
                for channel in channel_list:
                    # 确保channel是字符串
                    channel = str(channel)
                    
                    # Check if channel exists
                    if channel not in pydas_obj.data[sseg].columns:
                        logger.warning(f"Channel '{channel}' not found in segment {sseg}, skipping.")
                        continue
                    
                    # Get data
                    raw_data = pydas_obj.data[sseg][channel].dropna()
                    
                    # 确保数据非空
                    if raw_data.empty:
                        logger.warning(f"Channel '{channel}' contains no valid data after dropping NaN values, skipping.")
                        continue
                    
                    # 获取通道单位
                    channel_info = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel]
                    if channel_info.empty:
                        unit = ""
                        logger.warning(f"Could not find unit information for channel '{channel}'")
                    else:
                        unit = channel_info['Unit'].values[0]
                    
                    # 判断是否使用峰值分析
                    if use_peaks:
                        # 检测峰值
                        pos_peaks, neg_peaks = _detect_peaks(
                            raw_data, 
                            height=peak_height, 
                            threshold=peak_threshold,
                            distance=peak_distance, 
                            prominence=peak_prominence,
                            width=peak_width, 
                            wlen=peak_wlen, 
                            rel_height=peak_rel_height
                        )
                        
                        # 分离正负峰值处理
                        if separate_pos_neg_peaks:
                            # 如果有正峰值，添加到数据列表
                            if len(pos_peaks) > 0:
                                data_list.append(pos_peaks)
                                labels.append(f"{channel} (+) ({unit})")
                            else:
                                logger.warning(f"No positive peaks found for channel '{channel}'")
                            
                            # 如果有负峰值，添加到数据列表
                            if len(neg_peaks) > 0:
                                data_list.append(neg_peaks)
                                labels.append(f"{channel} (-) ({unit})")
                            else:
                                logger.warning(f"No negative peaks found for channel '{channel}'")
                        else:
                            # 合并所有峰值
                            all_peaks = np.concatenate([pos_peaks, neg_peaks])
                            if len(all_peaks) > 0:
                                data_list.append(all_peaks)
                                labels.append(f"{channel} (Peaks) ({unit})")
                            else:
                                logger.warning(f"No peaks found for channel '{channel}'")
                    else:
                        # 使用全部数据
                        data_list.append(raw_data)
                        labels.append(f"{channel} ({unit})")
                
                # 如果没有有效数据，返回None
                if not data_list:
                    logger.error("No valid data to plot")
                    return None
                
                # Create color palette for multiple channels
                if color is None:
                    colors = sns.color_palette(PLOT_CONFIG['colors']['qualitative'], n_colors=len(data_list))
                elif isinstance(color, list):
                    colors = color
                else:
                    colors = [color] * len(data_list)
                
                # Create seaborn boxplot
                props = {}
                if boxprops is not None: props['boxprops'] = boxprops
                if whiskerprops is not None: props['whiskerprops'] = whiskerprops
                if capprops is not None: props['capprops'] = capprops
                if flierprops is not None: props['flierprops'] = flierprops
                if medianprops is not None: props['medianprops'] = medianprops
                if meanprops is not None: props['meanprops'] = meanprops
                
                # Create the boxplot with adjusted properties
                palette = colors[:len(data_list)]
                
                # Determine orientation parameters for seaborn
                orient = "v" if vert else "h"
                
                # Create the boxplot
                sns.boxplot(
                    data=data_list,
                    orient=orient,
                    notch=notch,
                    showfliers=showfliers,
                    showmeans=showmeans,
                    meanline=meanline,
                    width=boxwidth,
                    palette=palette,
                    ax=ax,
                    **props
                )
                
                # Set xticks and labels
                # 先设置ticks，再设置ticklabels
                positions = np.arange(len(labels))
                if vert:
                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels)
                else:
                    ax.set_yticks(positions)
                    ax.set_yticklabels(labels)
                
                # Set plot title
                if title is None:
                    title = "Boxplot Analysis"
                    if use_peaks:
                        title += " (Peak Values)"
                    if not is_list and channel_list:  # 确保channel_list非空
                        title += f" - {channel_list[0]}"
                ax.set_title(title, fontsize=PLOT_CONFIG['font']['size']['title'])
                
                # Set axis labels
                if xlabel is None:
                    xlabel = "Channel" if vert else "Value"
                if ylabel is None:
                    ylabel = "Value" if vert else "Channel"
                    
                ax.set_xlabel(xlabel, fontsize=PLOT_CONFIG['font']['size']['label'])
                ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG['font']['size']['label'])
                
                # Apply grid setting
                ax.grid(grid)
                
                # Set axis limits if provided
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save as image if requested
                if save_path is not None:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                    logger.info(f"Boxplot saved to {save_path}")
                
                # Show plot if requested
                if show:
                    plt.show()
                else:
                    plt.close(fig)
                
                plot_created = True
                
            except ImportError:
                logger.warning("Seaborn not available. Falling back to matplotlib.")
                backend = 'matplotlib'
            except Exception as e:
                logger.warning(f"Error using Seaborn: {str(e)}. Falling back to matplotlib.")
                backend = 'matplotlib'
        
        # If backend is matplotlib or seaborn failed
        if backend == 'matplotlib' and not plot_created:
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Create figure
                fig, ax = plt.subplots(figsize=figsize)
                
                # Prepare data for boxplot
                data_list = []
                labels = []
                
                # Process each channel
                for channel in channel_list:
                    # 确保channel是字符串
                    channel = str(channel)
                    
                    # Check if channel exists
                    if channel not in pydas_obj.data[sseg].columns:
                        logger.warning(f"Channel '{channel}' not found in segment {sseg}, skipping.")
                        continue
                    
                    # Get data
                    raw_data = pydas_obj.data[sseg][channel].dropna()
                    
                    # 确保数据非空
                    if raw_data.empty:
                        logger.warning(f"Channel '{channel}' contains no valid data after dropping NaN values, skipping.")
                        continue
                    
                    # 获取通道单位
                    channel_info = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel]
                    if channel_info.empty:
                        unit = ""
                        logger.warning(f"Could not find unit information for channel '{channel}'")
                    else:
                        unit = channel_info['Unit'].values[0]
                    
                    # 判断是否使用峰值分析
                    if use_peaks:
                        # 检测峰值
                        pos_peaks, neg_peaks = _detect_peaks(
                            raw_data, 
                            height=peak_height, 
                            threshold=peak_threshold,
                            distance=peak_distance, 
                            prominence=peak_prominence,
                            width=peak_width, 
                            wlen=peak_wlen, 
                            rel_height=peak_rel_height
                        )
                        
                        # 分离正负峰值处理
                        if separate_pos_neg_peaks:
                            # 如果有正峰值，添加到数据列表
                            if len(pos_peaks) > 0:
                                data_list.append(pos_peaks)
                                labels.append(f"{channel} (+) ({unit})")
                            else:
                                logger.warning(f"No positive peaks found for channel '{channel}'")
                            
                            # 如果有负峰值，添加到数据列表
                            if len(neg_peaks) > 0:
                                data_list.append(neg_peaks)
                                labels.append(f"{channel} (-) ({unit})")
                            else:
                                logger.warning(f"No negative peaks found for channel '{channel}'")
                        else:
                            # 合并所有峰值
                            all_peaks = np.concatenate([pos_peaks, neg_peaks])
                            if len(all_peaks) > 0:
                                data_list.append(all_peaks)
                                labels.append(f"{channel} (Peaks) ({unit})")
                            else:
                                logger.warning(f"No peaks found for channel '{channel}'")
                    else:
                        # 使用全部数据
                        data_list.append(raw_data)
                        labels.append(f"{channel} ({unit})")
                
                # 如果没有有效数据，返回None
                if not data_list:
                    logger.error("No valid data to plot")
                    return None
                
                # Create color palette for multiple channels
                if color is None:
                    colors = [plt.cm.get_cmap(PLOT_CONFIG['colors']['qualitative'])(i/10) for i in range(len(data_list))]
                elif isinstance(color, list):
                    colors = color
                else:
                    colors = [color] * len(data_list)
                
                # Create the boxplot
                box_props = {
                    'notch': notch,
                    'vert': vert,
                    'showfliers': showfliers,
                    'showmeans': showmeans,
                    'meanline': meanline,
                    'patch_artist': True,
                    'widths': boxwidth,
                }
                
                # Add optional properties if provided
                if boxprops is not None: box_props['boxprops'] = boxprops
                if whiskerprops is not None: box_props['whiskerprops'] = whiskerprops
                if capprops is not None: box_props['capprops'] = capprops
                if flierprops is not None: box_props['flierprops'] = flierprops
                if medianprops is not None: box_props['medianprops'] = medianprops
                if meanprops is not None: box_props['meanprops'] = meanprops
                
                # Create the boxplot - 不要在这里设置labels，而是在后面单独设置
                bplot = ax.boxplot(data_list, **box_props)
                
                # Set colors for boxes
                for patch, color in zip(bplot['boxes'], colors[:len(data_list)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(alpha)
                
                # Set plot title
                if title is None:
                    title = "Boxplot Analysis"
                    if use_peaks:
                        title += " (Peak Values)"
                    if not is_list and channel_list:  # 确保channel_list非空
                        title += f" - {channel_list[0]}"
                ax.set_title(title, fontsize=PLOT_CONFIG['font']['size']['title'])
                
                # Set axis labels
                if xlabel is None:
                    xlabel = "Channel" if vert else "Value"
                if ylabel is None:
                    ylabel = "Value" if vert else "Channel"
                    
                ax.set_xlabel(xlabel, fontsize=PLOT_CONFIG['font']['size']['label'])
                ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG['font']['size']['label'])
                
                # Apply grid setting
                ax.grid(grid)
                
                # Set axis limits if provided
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                
                # 正确设置ticks和ticklabels
                positions = range(1, len(labels) + 1)  # boxplot positions从1开始
                if vert:
                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels, rotation=45 if len(labels) > 3 else 0, 
                                     ha='right' if len(labels) > 3 else 'center')
                else:
                    ax.set_yticks(positions)
                    ax.set_yticklabels(labels)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save as image if requested
                if save_path is not None:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                    logger.info(f"Boxplot saved to {save_path}")
                
                # Show plot if requested
                if show:
                    plt.show()
                else:
                    plt.close(fig)
                
                plot_created = True
                
            except ImportError:
                logger.error("Matplotlib not available. Cannot create boxplot.")
                return None
            except Exception as e:
                logger.error(f"Error creating matplotlib boxplot: {str(e)}")
                return None
        
        # Return figure object
        return fig if not show else None
    
    except Exception as e:
        logger.error(f"Error in boxplot_channel: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

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
