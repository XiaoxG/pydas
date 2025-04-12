"""
PyDAS Plot Module
================
This module contains plotting functions for PyDAS data.
"""

import logging
import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger('pydas.plot')

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
              use_plotly=True, downsampling=False, max_points=40000, save_html=None,
              dpi=300, width=None, height=None, color=None, alpha=0.8, linewidth=1, 
              figsize=(12, 4), stats=True, table_width=0.3, column_widths=None,
              use_dask=True, use_webgl=True, chunk_size=10000, data_decimation='auto'):
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
        use_plotly (bool): Use Plotly for interactive web-based plotting (default: True)
        downsampling (bool): Whether to downsample large datasets (default: False)
        max_points (int): Maximum number of points to plot before downsampling (default: 40000)
        save_html (str): Path to save as interactive HTML (default: None)
        dpi (int): DPI for saved image (default: 300)
        width (int): Width in pixels for Plotly plot (default: None)
        height (int): Height in pixels for Plotly plot (default: None)
        color (str): Line color (default: None, auto-generated)
        alpha (float): Line transparency (default: 0.8)
        linewidth (float): Line width (default: 1)
        figsize (tuple): Figure size for matplotlib in inches (default: (12, 4))
        stats (bool): Whether to include statistics (default: True)
        table_width (float): Width of the statistics table (default: 0.3)
        column_widths (list): Column widths for statistics table (default: None)
        use_dask (bool): Use Dask for large data processing (default: True)
        use_webgl (bool): Use WebGL for Plotly rendering for better performance (default: True)
        chunk_size (int): Chunk size for Dask processing (default: 10000)
        data_decimation (str or int): Decimation method for large datasets ('auto', 'lttb', or an integer for step) (default: 'auto')
    
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

        # Flag to track if we've successfully created a plot
        plot_created = False
        fig = None
        plt = None  # Initialize plt as None, import later as needed
        
        # If use_plotly is True, try to use Plotly for interactive web-based plotting
        if use_plotly:
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
                    colors = cm.get_cmap('tab10', len(channel_list))
                    color_list = []
                    for i in range(len(channel_list)):
                        rgba = colors(i)
                        color_list.append(f'rgb({int(255*rgba[0])},{int(255*rgba[1])},{int(255*rgba[2])})')
                elif not is_list and color is None:
                    color_list = ['blue']
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
                    
                    # Store data for statistics calculation
                    stats_data[channel] = {
                        'unit': pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel]['Unit'].values[0]
                    }
                    
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
                    
                    fig.add_trace(
                        scatter_type(
                            x=x_data,
                            y=y_data,
                            name=f"{channel} ({unit})",
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
                                font=dict(size=12),
                                align="center"
                            ),
                            cells=dict(
                                values=[ch_names, mean_values, max_values, min_values, std_values, units],
                                font=dict(size=11),
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
                    if is_list:
                        title += f" (Multiple Channels)"
                    else:
                        title += f" - {channel_list[0]}"
                
                # Set y-axis label if not provided
                if ylabel is None:
                    if not is_list:
                        unit = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel_list[0]]['Unit'].values[0]
                        ylabel = f"{channel_list[0]} ({unit})"
                
                # Update layout
                fig.update_layout(
                    title=title,
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode="closest",
                    template="plotly_white",
                    width=width,
                    height=height,
                    grid=dict(rows=1, columns=1, pattern="independent"),
                    margin=dict(l=50, r=50, t=50, b=50),
                    # Optimize for performance
                    uirevision='constant'  # Maintain zoom level on updates
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
                use_plotly = False
            except Exception as e:
                logger.warning(f"Error using Plotly: {str(e)}. Falling back to matplotlib.")
                use_plotly = False
        
        # If Plotly is not used or not available, use matplotlib
        if not use_plotly or not plot_created:
            try:
                import matplotlib.pyplot as plt
                
                # Create figure and axis
                fig, ax = plt.subplots(figsize=figsize)
                
                # Create color palette for multiple channels
                if is_list and color is None:
                    colors = [plt.cm.tab10(i % 10) for i in range(len(channel_list))]
                elif not is_list and color is None:
                    colors = ['blue']
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
                    if is_list:
                        title += f" (Multiple Channels)"
                    else:
                        title += f" - {channel_list[0]}"
                ax.set_title(title)
                
                # Set axis labels
                ax.set_xlabel(xlabel)
                
                # Set y-axis label if not provided
                if ylabel is None:
                    if not is_list:
                        unit = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel_list[0]]['Unit'].values[0]
                        ylabel = f"{channel_list[0]} ({unit})"
                ax.set_ylabel(ylabel)
                
                # Set grid
                ax.grid(grid)
                
                # Set axis limits if provided
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                
                # Add statistical information if requested
                if stats:
                    # Calculate statistics
                    x_mean = np.mean(x_data)
                    y_mean = np.mean(y_data)
                    x_std = np.std(x_data)
                    y_std = np.std(y_data)
                    x_min = np.min(x_data)
                    y_min = np.min(y_data)
                    x_max = np.max(x_data)
                    y_max = np.max(y_data)
                    corr = np.corrcoef(x_data, y_data)[0, 1]
                    
                    # Create stats string
                    stats_text = (
                        f"{x_data.name}: μ={x_mean:.4g}, σ={x_std:.4g}, min={x_min:.4g}, max={x_max:.4g}\n"
                        f"{y_data.name}: μ={y_mean:.4g}, σ={y_std:.4g}, min={y_min:.4g}, max={y_max:.4g}\n"
                        f"Correlation: {corr:.4g}"
                    )
                    
                    # Add stats text to plot
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
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
                logger.error("Neither Plotly nor Matplotlib is available for plotting.")
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
                use_plotly=True, save_html=None, dpi=300, width=None, height=None, 
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
        use_plotly (bool): Use Plotly for interactive web-based plotting (default: True)
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

        # Flag to track if we've successfully created a plot
        plot_created = False
        fig = None
        plt = None  # Initialize plt as None, import later as needed
        
        # If use_plotly is True, try to use Plotly for interactive web-based plotting
        if use_plotly:
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
                use_plotly = False
            except Exception as e:
                logger.warning(f"Error using Plotly: {str(e)}. Falling back to matplotlib.")
                use_plotly = False
        
        # If Plotly is not used or not available, use matplotlib
        if not use_plotly or not plot_created:
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
                
                # Create figure with subplots
                fig = plt.figure(figsize=figsize)
                gs = gridspec.GridSpec(rows, cols)
                
                # Create color palette for multiple channels
                if is_list and color is None:
                    colors = plt.cm.tab10(np.linspace(0, 1, n_channels))
                elif not is_list and color is None:
                    colors = ['blue']
                elif isinstance(color, list):
                    colors = color
                else:
                    colors = [color] * n_channels
                
                # Create fit color list
                if isinstance(fit_color, list):
                    fit_color_list = fit_color
                else:
                    fit_color_list = [fit_color] * n_channels
                
                # Process each channel
                for i, channel in enumerate(channel_list):
                    # Calculate row and column indices
                    row = i // cols
                    col = i % cols
                    
                    # Create subplot
                    ax = plt.subplot(gs[row, col])
                    
                    # Check if channel exists
                    if channel not in pydas_obj.data[sseg].columns:
                        logger.warning(f"Channel '{channel}' not found in segment {sseg}, skipping.")
                        continue
                    
                    # Get data
                    data = pydas_obj.data[sseg][channel]
                    
                    # Get channel unit for label
                    unit = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel]['Unit'].values[0]
                    
                    # Plot histogram
                    n, bins, patches = ax.hist(data, bins=bins, alpha=alpha, color=colors[i])
                    
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
                        ax.plot(x, pdf_scaled, color=fit_color_list[i], linewidth=2)
                        
                        # Add fit parameters as text
                        fit_text = f"μ = {mu:.4g}\nσ = {sigma:.4g}"
                        ax.text(0.95, 0.95, fit_text, transform=ax.transAxes,
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                    
                    # Set labels
                    ax.set_title(f"{channel} ({unit})")
                    if col == 0:  # First column
                        ax.set_ylabel(ylabel)
                    if row == rows - 1:  # Last row
                        if xlabel is None:
                            xlabel = f"{channel} ({unit})"
                        ax.set_xlabel(xlabel)
                    
                    # Set grid
                    ax.grid(grid)
                    
                    # Set axis limits if provided
                    if xlim is not None:
                        ax.set_xlim(xlim)
                    if ylim is not None:
                        ax.set_ylim(ylim)
                
                # Set global title
                if title is None:
                    title = "Histogram Analysis"
                    if not is_list:
                        title += f" - {channel_list[0]}"
                fig.suptitle(title)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save figure if requested
                if save_path is not None:
                    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
                    logger.info(f"Histogram saved to {save_path}")
                
                # Show plot if requested
                if show:
                    plt.show()
                else:
                    plt.close(fig)
                
                plot_created = True
                
            except ImportError:
                logger.error("Neither Plotly nor Matplotlib is available for plotting.")
                return None
        
        # Return fig object if not showing or return None if showing
        return None if show else fig
        
    except Exception as e:
        logger.error(f"Error in plot_histogram: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def plot_xy(pydas_obj, x_ch_name, y_ch_name, sseg=0, title=None, 
         xlabel=None, ylabel=None, xlim=None, ylim=None, grid=True, 
         show=True, save_path=None, use_plotly=True, save_html=None,
         dpi=300, width=None, height=None, color='blue', alpha=0.8, 
         marker_size=5, figsize=(8, 8), line=False, fit_line=False,
         fit_color='red', fit_line_width=2, fit_alpha=0.8,
         show_stats=False, downsampling=True, max_points=10000,
         density_plot=False, density_colorscale='Viridis', 
         density_opacity=0.7, use_webgl=True, adaptive_sampling=False,
         datashade=False, contour_levels=20, sampling_algorithm='lttb',
         memory_efficient=True, bin_size=None):
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
        use_plotly (bool): Use Plotly for interactive web-based plotting (default: True)
        save_html (str): Path to save as interactive HTML (default: None)
        dpi (int): DPI for saved image (default: 300)
        width (int): Width in pixels for Plotly plot (default: None)
        height (int): Height in pixels for Plotly plot (default: None)
        color (str): Color for scatter points (default: 'blue')
        alpha (float): Transparency for scatter points (default: 0.8)
        marker_size (float): Size of scatter points (default: 5)
        figsize (tuple): Figure size for matplotlib in inches (default: (8, 8))
        line (bool): Connect points with lines (default: False)
        fit_line (bool): Show linear regression fit line (default: False)
        fit_color (str): Color for fit line (default: 'red')
        fit_line_width (float): Width of fit line (default: 2)
        fit_alpha (float): Transparency of fit line (default: 0.8)
        show_stats (bool): Show statistical information on the plot (default: False)
        downsampling (bool): Apply downsampling for large datasets (default: True)
        max_points (int): Maximum number of points to show before downsampling (default: 10000)
        density_plot (bool): Show density contour plot for large datasets (default: False)
        density_colorscale (str): Colorscale for density plot (default: 'Viridis')
        density_opacity (float): Opacity for density contours (default: 0.7)
        use_webgl (bool): Use WebGL rendering for better performance (default: True)
        adaptive_sampling (bool): Use adaptive sampling to preserve signal features (default: False)
        datashade (bool): Use datashading for very large datasets (default: False)
        contour_levels (int): Number of contour levels for density plot (default: 20)
        sampling_algorithm (str): Algorithm for downsampling ('lttb', 'uniform', 'peak') (default: 'lttb')
        memory_efficient (bool): Use memory-efficient methods for very large datasets (default: True)
        bin_size (tuple): Bin size for 2D histogram (x_bins, y_bins) (default: None, auto)
        
    Returns:
        tuple: (pandas.DataFrame with x and y data, figure object)
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
        
        # Flag to track if we've successfully created a plot
        plot_created = False
        fig = None
        plt = None  # Initialize plt as None, import later as needed
        
        # If use_plotly is True, try to use Plotly for interactive web-based plotting
        if use_plotly:
            try:
                # Import Plotly modules
                import plotly.graph_objects as go
                import plotly.express as px
                from plotly.subplots import make_subplots
                
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
                                marker=dict(color=color, size=marker_size, opacity=alpha/2),
                                name='Data points',
                                showlegend=False
                            ) if use_webgl else go.Scatter(
                                x=df[x_ch_name],
                                y=df[y_ch_name],
                                mode='markers',
                                marker=dict(color=color, size=marker_size, opacity=alpha/2),
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
                                marker=dict(color=color, size=marker_size, opacity=alpha),
                                name='Data points'
                            ) if use_webgl else go.Scatter(
                                x=df[x_ch_name],
                                y=df[y_ch_name],
                                mode='markers',
                                marker=dict(color=color, size=marker_size, opacity=alpha),
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
                            marker=dict(color=color, size=marker_size, opacity=alpha),
                            name='Data points'
                        ) if use_webgl else go.Scatter(
                            x=df[x_ch_name],
                            y=df[y_ch_name],
                            mode=scatter_mode,
                            marker=dict(color=color, size=marker_size, opacity=alpha),
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
                                font=dict(size=12),
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
                
                # Update layout
                fig.update_layout(
                    title=title,
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode="closest",
                    template="plotly_white",
                    width=width,
                    height=height,
                    showlegend=True
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
                            font=dict(size=12),
                            align="left"
                        ),
                        cells=dict(
                            values=[
                                ["Mean", "Std", "Min", "Max", "Correlation"],
                                [f"{x_mean:.4g}", f"{x_std:.4g}", f"{x_min:.4g}", f"{x_max:.4g}", f"{corr:.4g}"],
                                [f"{y_mean:.4g}", f"{y_std:.4g}", f"{y_min:.4g}", f"{y_max:.4g}", ""]
                            ],
                            font=dict(size=11),
                            align="left"
                        ),
                        domain=dict(x=[0.7, 1], y=[0, 0.2])
                    )
                    
                    fig.add_trace(stats_table)
                
                # Save as HTML if requested
                if save_html is not None:
                    fig.write_html(save_html)
                    logger.info(f"Interactive plot saved to {save_html}")
                
                # Save as image if requested
                if save_path is not None:
                    fig.write_image(save_path, width=width or 1000, height=height or 1000, scale=2)
                    logger.info(f"Plot saved to {save_path}")
                
                # Show plot if requested
                if show:
                    fig.show()
                
                plot_created = True
                
            except ImportError:
                logger.warning("Plotly not available. Falling back to matplotlib.")
                use_plotly = False
            except Exception as e:
                logger.warning(f"Error using Plotly: {str(e)}. Falling back to matplotlib.")
                use_plotly = False
        
        # If Plotly is not used or not available, use matplotlib
        if not use_plotly or not plot_created:
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
                                  s=marker_size, color=color, alpha=alpha/2)
                    except Exception as e:
                        logger.warning(f"Error creating density plot, falling back to regular scatter: {e}")
                        # Fall back to regular scatter plot
                        ax.scatter(df[x_ch_name], df[y_ch_name], 
                                 s=marker_size, color=color, alpha=alpha)
                else:
                    # Regular scatter plot
                    if line:
                        ax.plot(df[x_ch_name], df[y_ch_name], 
                               marker='o', markersize=marker_size, color=color, alpha=alpha, 
                               linestyle='-', linewidth=1)
                    else:
                        ax.scatter(df[x_ch_name], df[y_ch_name], 
                                  s=marker_size, color=color, alpha=alpha)
                
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
                            ax.legend()
                    except Exception as e:
                        logger.warning(f"Error adding fit line: {e}")
                
                # Set plot title
                if title is None:
                    title = f"XY Plot: {y_ch_name} vs {x_ch_name}"
                ax.set_title(title)
                
                # Set axis labels
                if xlabel is None:
                    xlabel = f"{x_ch_name} ({x_unit})"
                if ylabel is None:
                    ylabel = f"{y_ch_name} ({y_unit})"
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                
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
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
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
                logger.error("Neither Plotly nor Matplotlib is available for plotting.")
                return None
        
        # Return data and fig object if not showing or just data if showing
        return df if show else (df, fig)
        
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

# Function declarations will be added below 