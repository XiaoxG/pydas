"""
PyDAS Plot - Statistics
=======================
Contains statistical visualization functions: histogram, boxplot,
and statistics plot helpers for matplotlib and plotly backends.
"""

import logging
import numpy as np
import pandas as pd
import os

from . import PLOT_CONFIG, get_plot_backend, apply_style, lttb_downsample

logger = logging.getLogger(__name__)

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

        # Resolve the available plotting backend
        backend = get_plot_backend(plotbackend)
        if backend is None:
            logger.error("No available plotting backend found")
            return None
            
        # Apply style
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
        # Handle a single channel explicitly to avoid DataFrame ambiguity
        if isinstance(ch_names, list):
            channel_list = ch_names
            is_list = True
        else:
            # A non-list argument is a single channel name
            channel_list = [ch_names]
            is_list = False
        
        # Get plot backend
        backend = get_plot_backend(plotbackend)
        if backend is None:
            logger.error("No available plotting backend found.")
            return None
        
        # Require every channel_list entry to be a string
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
                import matplotlib.pyplot as plt  # local import used by this backend
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
                    # Channel name must be a string
                    channel = str(channel)
                    
                    # Check if channel exists
                    if channel not in pydas_obj.data[sseg].columns:
                        logger.warning(f"Channel '{channel}' not found in segment {sseg}, skipping.")
                        continue
                    
                    # Get data
                    raw_data = pydas_obj.data[sseg][channel].dropna()
                    
                    # Skip empty series
                    if raw_data.empty:
                        logger.warning(f"Channel '{channel}' contains no valid data after dropping NaN values, skipping.")
                        continue
                    
                    # Channel unit
                    channel_info = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel]
                    if channel_info.empty:
                        unit = ""
                        logger.warning(f"Could not find unit information for channel '{channel}'")
                    else:
                        unit = channel_info['Unit'].values[0]
                    
                    # Peak analysis vs full series
                    if use_peaks:
                        # Detect peaks
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
                        
                        # Split positive and negative peaks
                        if separate_pos_neg_peaks:
                            # Append positive peaks
                            if len(pos_peaks) > 0:
                                data_list.append(pos_peaks)
                                names_list.append(f"{channel} (+) ({unit})")
                            else:
                                logger.warning(f"No positive peaks found for channel '{channel}'")
                            
                            # Append negative peaks
                            if len(neg_peaks) > 0:
                                data_list.append(neg_peaks)
                                names_list.append(f"{channel} (-) ({unit})")
                            else:
                                logger.warning(f"No negative peaks found for channel '{channel}'")
                        else:
                            # Combine all peaks
                            all_peaks = np.concatenate([pos_peaks, neg_peaks])
                            if len(all_peaks) > 0:
                                data_list.append(all_peaks)
                                names_list.append(f"{channel} (Peaks) ({unit})")
                            else:
                                logger.warning(f"No peaks found for channel '{channel}'")
                    else:
                        # Use the full series
                        data_list.append(raw_data)
                        names_list.append(f"{channel} ({unit})")
                
                # Nothing to plot
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
                        width=boxwidth,  # Plotly uses width, not boxwidth
                        orientation=plot_orientation
                    ))
                
                # Set plot title
                if title is None:
                    title = "Boxplot Analysis"
                    if use_peaks:
                        title += " (Peak Values)"
                    if not is_list and channel_list:  # channel_list must be non-empty
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
                
                # Set equal aspect ratio if requested
                if equal_aspect:
                    fig.update_layout(
                        yaxis=dict(
                            scaleanchor="x",
                            scaleratio=1,
                        )
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
                    # Channel name must be a string
                    channel = str(channel)
                    
                    # Check if channel exists
                    if channel not in pydas_obj.data[sseg].columns:
                        logger.warning(f"Channel '{channel}' not found in segment {sseg}, skipping.")
                        continue
                    
                    # Get data
                    raw_data = pydas_obj.data[sseg][channel].dropna()
                    
                    # Skip empty series
                    if raw_data.empty:
                        logger.warning(f"Channel '{channel}' contains no valid data after dropping NaN values, skipping.")
                        continue
                    
                    # Channel unit
                    channel_info = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel]
                    if channel_info.empty:
                        unit = ""
                        logger.warning(f"Could not find unit information for channel '{channel}'")
                    else:
                        unit = channel_info['Unit'].values[0]
                    
                    # Peak analysis vs full series
                    if use_peaks:
                        # Detect peaks
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
                        
                        # Split positive and negative peaks
                        if separate_pos_neg_peaks:
                            # Append positive peaks
                            if len(pos_peaks) > 0:
                                data_list.append(pos_peaks)
                                labels.append(f"{channel} (+) ({unit})")
                            else:
                                logger.warning(f"No positive peaks found for channel '{channel}'")
                            
                            # Append negative peaks
                            if len(neg_peaks) > 0:
                                data_list.append(neg_peaks)
                                labels.append(f"{channel} (-) ({unit})")
                            else:
                                logger.warning(f"No negative peaks found for channel '{channel}'")
                        else:
                            # Combine all peaks
                            all_peaks = np.concatenate([pos_peaks, neg_peaks])
                            if len(all_peaks) > 0:
                                data_list.append(all_peaks)
                                labels.append(f"{channel} (Peaks) ({unit})")
                            else:
                                logger.warning(f"No peaks found for channel '{channel}'")
                    else:
                        # Use the full series
                        data_list.append(raw_data)
                        labels.append(f"{channel} ({unit})")
                
                # Nothing to plot
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
                # Set ticks before tick labels
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
                    if not is_list and channel_list:  # channel_list must be non-empty
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
                    # Channel name must be a string
                    channel = str(channel)
                    
                    # Check if channel exists
                    if channel not in pydas_obj.data[sseg].columns:
                        logger.warning(f"Channel '{channel}' not found in segment {sseg}, skipping.")
                        continue
                    
                    # Get data
                    raw_data = pydas_obj.data[sseg][channel].dropna()
                    
                    # Skip empty series
                    if raw_data.empty:
                        logger.warning(f"Channel '{channel}' contains no valid data after dropping NaN values, skipping.")
                        continue
                    
                    # Channel unit
                    channel_info = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == channel]
                    if channel_info.empty:
                        unit = ""
                        logger.warning(f"Could not find unit information for channel '{channel}'")
                    else:
                        unit = channel_info['Unit'].values[0]
                    
                    # Peak analysis vs full series
                    if use_peaks:
                        # Detect peaks
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
                        
                        # Split positive and negative peaks
                        if separate_pos_neg_peaks:
                            # Append positive peaks
                            if len(pos_peaks) > 0:
                                data_list.append(pos_peaks)
                                labels.append(f"{channel} (+) ({unit})")
                            else:
                                logger.warning(f"No positive peaks found for channel '{channel}'")
                            
                            # Append negative peaks
                            if len(neg_peaks) > 0:
                                data_list.append(neg_peaks)
                                labels.append(f"{channel} (-) ({unit})")
                            else:
                                logger.warning(f"No negative peaks found for channel '{channel}'")
                        else:
                            # Combine all peaks
                            all_peaks = np.concatenate([pos_peaks, neg_peaks])
                            if len(all_peaks) > 0:
                                data_list.append(all_peaks)
                                labels.append(f"{channel} (Peaks) ({unit})")
                            else:
                                logger.warning(f"No peaks found for channel '{channel}'")
                    else:
                        # Use the full series
                        data_list.append(raw_data)
                        labels.append(f"{channel} ({unit})")
                
                # Nothing to plot
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
                
                # Create the boxplot; set labels afterwards
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
                    if not is_list and channel_list:  # channel_list must be non-empty
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
                
                # Set ticks and tick labels
                positions = range(1, len(labels) + 1)  # matplotlib boxplot positions are 1-based
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
