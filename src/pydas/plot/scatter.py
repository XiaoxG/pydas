"""
PyDAS Plot - Scatter
====================
Contains scatter plot and XY plot functions with LTTB downsampling support.
"""

import logging
import numpy as np
import os

from . import PLOT_CONFIG, get_plot_backend, apply_style, lttb_downsample

logger = logging.getLogger(__name__)

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
         use_resampler=False, n_shown_samples=5000, equal_aspect=True, square_plot=True):
    """
    Create a scatter plot of two channels.
    
    Parameters:
    -----------
    pydas_obj : PyDAS object
        PyDAS object
    x_ch_name : str
        Channel name for x-axis
    y_ch_name : str
        Channel name for y-axis
    sseg : int, optional
        Segment index, default is 0
    title : str, optional
        Plot title
    xlabel, ylabel : str, optional
        Axis labels
    xlim, ylim : tuple, optional
        Axis limits as (min, max)
    grid : bool, optional
        Whether to show grid
    show : bool, optional
        Whether to show plot
    save_path : str, optional
        Path to save plot
    plotbackend : str, optional
        Plotting backend ('matplotlib', 'plotly', 'seaborn', 'datashader')
    style : str, optional
        Plot style
    save_html : str, optional
        Path to save interactive HTML (Plotly only)
    dpi : int, optional
        DPI for saved plot
    width, height : int, optional
        Width and height of plot in pixels
    color : str or tuple, optional
        Color for markers
    alpha : float, optional
        Transparency of markers (0-1)
    marker_size : float, optional
        Size of markers
    figsize : tuple, optional
        Figure size as (width, height) in inches
    line : bool, optional
        Whether to connect points with a line
    fit_line : bool, optional
        Whether to add a linear regression line
    fit_color : str, optional
        Color for regression line
    fit_line_width : float, optional
        Line width for regression line
    fit_alpha : float, optional
        Transparency for regression line
    show_stats : bool, optional
        Whether to show statistics on plot
    downsampling : bool, optional
        Whether to downsample large datasets
    max_points : int, optional
        Maximum number of points to display
    density_plot : bool, optional
        Whether to create a density plot
    density_colorscale : str, optional
        Colorscale for density plot
    density_opacity : float, optional
        Opacity for density plot
    use_webgl : bool, optional
        Whether to use WebGL for better performance (Plotly only)
    adaptive_sampling : bool, optional
        Whether to use adaptive sampling for large datasets
    datashade : bool, optional
        Whether to use datashading for large datasets
    contour_levels : int, optional
        Number of contour levels for density plot
    sampling_algorithm : str, optional
        Algorithm for data downsampling ('lttb', 'minmax', 'uniform')
    memory_efficient : bool, optional
        Whether to optimize for memory usage
    bin_size : int or tuple, optional
        Bin size for density plot
    sns_style : str, optional
        Seaborn style
    sns_bins : int, optional
        Number of bins for seaborn KDE
    sns_pthresh : float, optional
        Threshold for seaborn contour plot
    sns_cmap : str, optional
        Colormap for seaborn plot
    sns_contour_levels : int, optional
        Number of contour levels for seaborn
    sns_contour_color : str, optional
        Color for seaborn contour lines
    sns_linewidths : float, optional
        Line width for seaborn contour lines
    use_resampler : bool, optional
        Whether to use plotly-resampler
    n_shown_samples : int, optional
        Number of samples to show with resampler
    equal_aspect : bool, optional
        Whether to keep x and y axes with equal scale (1:1 aspect ratio), ensuring that 
        equal distances in data space are visually equal in both directions. This is 
        important for accurate representation of physical quantities. Default is True.
    square_plot : bool, optional
        Whether to ensure plot is square by adjusting axis limits based on the larger range. 
        Only applies when equal_aspect is True. Default is True.
        
    Returns:
    --------
    object
        matplotlib.figure.Figure or plotly.graph_objects.Figure or None
    """
    # Use configured defaults when not specified
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
        
        # Calculate axis limits for square plot if requested
        if equal_aspect and square_plot and xlim is None and ylim is None:
            x_min, x_max = df[x_ch_name].min(), df[x_ch_name].max()
            y_min, y_max = df[y_ch_name].min(), df[y_ch_name].max()
            
            # Add 5% padding to the ranges
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_pad = 0.05 * x_range
            y_pad = 0.05 * y_range
            
            # Calculate padded bounds
            x_min_padded = x_min - x_pad
            x_max_padded = x_max + x_pad
            y_min_padded = y_min - y_pad
            y_max_padded = y_max + y_pad
            
            # Recalculate ranges with padding
            x_range_padded = x_max_padded - x_min_padded
            y_range_padded = y_max_padded - y_min_padded
            
            # Find the maximum range to create a square plot
            max_range = max(x_range_padded, y_range_padded)
            
            # Calculate centers
            x_center = (x_min_padded + x_max_padded) / 2
            y_center = (y_min_padded + y_max_padded) / 2
            
            # Set square limits based on max range
            xlim = (x_center - max_range/2, x_center + max_range/2)
            ylim = (y_center - max_range/2, y_center + max_range/2)
            
            logger.info(f"Setting square axis limits: xlim={xlim}, ylim={ylim}")
        
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
                
                # Tick font size
                ax.tick_params(axis='both', which='major', labelsize=PLOT_CONFIG['font']['size']['tick'])
                
                # Set grid
                ax.grid(grid)
                
                # Set equal aspect ratio if requested
                if equal_aspect:
                    ax.set_aspect('equal')
                
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
                
                # Prefer plotly-resampler when requested
                if use_resampler and HAS_PLOTLY_RESAMPLER and len(df) > 10000:
                    logger.info(f"Using plotly-resampler for large XY plot ({len(df)} points)")
                    
                    # Build a title when none was given
                    if title is None:
                        title = f"XY Plot: {y_ch_name} vs {x_ch_name}"
                    
                    # Build a resampable figure
                    fr = create_resampable_plot(
                        x=df[x_ch_name],
                        y=df[y_ch_name],
                        name=f"{y_ch_name} vs {x_ch_name}",
                        title=title,
                        n_shown_samples=n_shown_samples
                    )
                    
                    if fr is not None:
                        # Layout
                        x_axis_label = xlabel if xlabel else f"{x_ch_name} ({x_unit})"
                        y_axis_label = ylabel if ylabel else f"{y_ch_name} ({y_unit})"
                        
                        fr.update_layout(
                            xaxis_title=x_axis_label,
                            yaxis_title=y_axis_label,
                            hovermode='closest',
                            width=width,
                            height=height
                        )
                        
                        # Axis limits when provided
                        if xlim:
                            fr.update_xaxes(range=xlim)
                        if ylim:
                            fr.update_yaxes(range=ylim)
                        
                        # Optional regression line
                        if fit_line:
                            try:
                                # Linear regression
                                from scipy import stats as scipy_stats
                                
                                # Drop NaN values
                                df_clean = df.dropna()
                                x_fit = df_clean[x_ch_name].values
                                y_fit = df_clean[y_ch_name].values
                                
                                if len(x_fit) > 1:  # need at least two points
                                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_fit, y_fit)
                                    
                                    # Fitted line
                                    x_range = np.linspace(df[x_ch_name].min(), df[x_ch_name].max(), 100)
                                    y_fit_line = intercept + slope * x_range
                                    
                                    # Add the fitted line
                                    fr.add_trace(
                                        go.Scatter(
                                            x=x_range,
                                            y=y_fit_line,
                                            mode='lines',
                                            name=f'fit (y = {slope:.4g}x + {intercept:.4g})',
                                            line=dict(color=fit_color, width=fit_line_width),
                                            opacity=fit_alpha
                                        )
                                    )
                            except Exception as e:
                                logger.warning(f"Failed to add fitted line: {e}")
                        
                        # Save the figure when requested
                        if save_path:
                            try:
                                # Save PNG
                                png_path = save_path if save_path.endswith('.png') else save_path + '.png'
                                fr.write_image(png_path, width=width or 1200, height=height or 800)
                                logger.info(f"Figure saved to {png_path}")
                            except Exception as e:
                                logger.warning(f"Failed to save image: {e}")
                        
                        if save_html:
                            try:
                                html_path = save_html if save_html.endswith('.html') else save_html + '.html'
                                fr.write_html(html_path)
                                logger.info(f"Interactive HTML saved to {html_path}")
                            except Exception as e:
                                logger.warning(f"Failed to save HTML: {e}")
                        
                        # Show the interactive figure
                        if show:
                            fr.show_dash()
                        
                        # Return data and figure
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
                    # Performance settings
                    uirevision='constant'  # keep zoom level
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
                
                # Config applied to each figure
                plot_settings = {
                    "scrollZoom": True,  # enable scroll-wheel zoom
                    "modeBarButtonsToAdd": ["drawopenpath", "eraseshape"],  # drawing tools
                    "modeBarButtonsToRemove": ["lasso2d"]  # drop lasso select
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
                
                # Tick font size
                ax.tick_params(axis='both', which='major', labelsize=PLOT_CONFIG['font']['size']['tick'])
                
                # Set grid
                ax.grid(grid)
                
                # Set equal aspect ratio if requested
                if equal_aspect:
                    ax.set_aspect('equal')
                
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
