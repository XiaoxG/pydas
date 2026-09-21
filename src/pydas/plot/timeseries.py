"""
PyDAS Plot - Time Series
========================
Contains the plot_channel function for time series visualization.
"""

import logging
import numpy as np
import os

from . import (
    PLOT_CONFIG, get_plot_backend, apply_style,
    use_webgl_rendering, create_resampable_plot, HAS_PLOTLY_RESAMPLER,
    lttb_downsample,
)
from .scatter import _lttb_downsample

logger = logging.getLogger('pydas.plot.timeseries')

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
                                    from ..utils import get_default_transDict, findtrans
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
