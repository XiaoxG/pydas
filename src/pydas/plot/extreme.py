"""
PyDAS Plot - Extreme Value Analysis
====================================
Contains peak detection and extreme value analysis visualization functions.
"""

import logging
import numpy as np
import os

from . import PLOT_CONFIG, get_plot_backend, apply_style

logger = logging.getLogger(__name__)


# Peak-detection helper

def _detect_peaks(data, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5):
    """
    A wrapper for scipy.signal.find_peaks to detect peaks in data
    
    Parameters
    ----------
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
        
    Returns
    -------
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
    """Visualise extreme-value analysis results.

    Parameters
    ----------
    results : dict
        Extreme-value analysis result dictionary with the following keys:
        - 'peaks_positive': positive peaks
        - 'peaks_negative': negative peaks
        - 'all_peaks': all peaks (absolute values)
        - 'peak_indices': peak indices {'positive': pos_indices, 'negative': neg_indices}
        - 'duration_seconds': data duration in seconds
        - 'exceedance_table': exceedance probability table
        - 'extreme_value_model': extreme-value model parameters
        - 'return_values': return values
        - 'return_value_confidence_intervals': return-value confidence intervals
    visualization_backend : str, default='matplotlib'
        Visualisation backend ('matplotlib' or 'plotly')
    save_path : str, optional
        Figure save path (matplotlib)
    save_html : str, optional
        Interactive figure save path (plotly)
    visualization : bool, default=True
        Whether to display the figure
    title : str, optional
        Figure title
    ch_name : str, optional
        Channel name
    pydas_obj : PyDAS object, optional
        PyDAS object used to obtain channel information and data
    bins : int, default=30
        Number of histogram bins
    fullscale : bool, default=True
        Whether to use prototype (full) scale
    return_periods : array-like, optional
        Return periods
    return_period_labels : list, optional
        Return-period labels
    unit : str, optional
        Data unit

    Returns
    -------
    object
        matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    
    # Require the keys needed to draw the figure
    required_keys = ['peaks_positive', 'peaks_negative', 'all_peaks', 'peak_indices',
                    'duration_seconds', 'exceedance_table']
    for key in required_keys:
        if key not in results:
            logger.error(f"Result dictionary is missing required key: {key}")
            return None
    
    # Load series data
    peaks_positive = results['peaks_positive']
    peaks_negative = results['peaks_negative']
    all_peaks = results['all_peaks']
    pos_peaks_idx = results['peak_indices']['positive']
    neg_peaks_idx = results['peak_indices']['negative']
    exceedance = results['exceedance_table']
    
    # Pull remaining metadata from the PyDAS object
    data_array = None
    fs = 1.0
    
    if pydas_obj is not None:
        # Try to read the channel unit
        if unit is None and ch_name is not None:
            channel_info = pydas_obj.chInfo[pydas_obj.chInfo['Name'] == ch_name]
            unit = "" if channel_info.empty else channel_info['Unit'].values[0]
        # Sampling rate
        if hasattr(pydas_obj, '__fs__'):
            fs = pydas_obj.__fs__
        # Full-channel series
        if ch_name is not None and hasattr(pydas_obj, 'data') and len(pydas_obj.data) > 0:
            if ch_name in pydas_obj.data[0].columns:
                # Use the first segment
                data_array = pydas_obj.data[0][ch_name].values
    
    # Fall back to return periods stored on the result
    if return_periods is None and 'return_periods' in results:
        return_periods = results['return_periods']['periods']
        return_period_labels = results['return_periods']['labels']
    
    # Unit annotation
    unit_str = f" [{unit}]" if unit else ""
    
    # Dispatch by visualisation backend
    backend = visualization_backend.lower()
    
    # Plotly backend
    if backend == 'plotly':
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # 2x2 subplot layout
            fig = make_subplots(rows=2, cols=2, 
                                subplot_titles=("Original Data with Detected Peaks", 
                                              "Peak Value Histogram", 
                                              "Empirical Exceedance Probability", 
                                              "Return Period Plot"),
                                specs=[[{}, {}], 
                                      [{}, {}]])
            
            # Panel 1: raw series and detected peaks
            if data_array is not None:
                time = np.arange(len(data_array)) / fs
                
                # Downsample large series
                if len(data_array) > 50000:
                    step = len(data_array) // 50000 + 1
                    plot_time = time[::step]
                    plot_data = data_array[::step]
                else:
                    plot_time = time
                    plot_data = data_array
                
                # Raw series
                fig.add_trace(
                    go.Scatter(x=plot_time, y=plot_data, 
                             mode='lines', name='Original Data',
                             line=dict(color='rgba(0,0,255,0.5)', width=1)),
                    row=1, col=1
                )
            
            # Positive peaks
            if len(pos_peaks_idx) > 0:
                pos_peak_times = pos_peaks_idx / fs
                
                fig.add_trace(
                    go.Scatter(x=pos_peak_times, y=peaks_positive, 
                             mode='markers', name='Positive Peaks',
                             marker=dict(color='red', size=8, symbol='circle')),
                    row=1, col=1
                )
            
            # Negative peaks
            if len(neg_peaks_idx) > 0:
                neg_peak_times = neg_peaks_idx / fs
                
                fig.add_trace(
                    go.Scatter(x=neg_peak_times, y=peaks_negative, 
                             mode='markers', name='Negative Peaks',
                             marker=dict(color='green', size=8, symbol='circle')),
                    row=1, col=1
                )
            
            # Panel 2: peak histogram
            if len(all_peaks) > 0:
                # Histogram
                fig.add_trace(
                    go.Histogram(x=all_peaks, nbinsx=bins, 
                               name='Peak Histogram',
                               marker=dict(color='rgba(0,0,255,0.7)')),
                    row=1, col=2
                )
                
                # Fitted extreme-value distribution
                if 'extreme_value_model' in results:
                    model = results['extreme_value_model']
                    x = np.linspace(min(all_peaks), max(all_peaks), 100)
                    
                    # Draw PDF for known distribution types
                    if model['distribution'] == 'GEV':
                        shape = model['shape']
                        loc = model['loc']
                        scale = model['scale']
                        y = stats.genextreme.pdf(x, shape, loc, scale)
                        distrib_name = f"GEV (ξ={shape:.3f}, μ={loc:.3f}, σ={scale:.3f})"
                        
                        # Scale PDF to histogram counts
                        bin_width = (max(all_peaks) - min(all_peaks)) / bins
                        y = y * len(all_peaks) * bin_width
                        
                        # Distribution curve
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
                        
                        # Scale PDF to histogram counts
                        bin_width = (max(all_peaks) - min(all_peaks)) / bins
                        y = y * len(all_peaks) * bin_width
                        
                        # Distribution curve
                        fig.add_trace(
                            go.Scatter(x=x, y=y, mode='lines', name=distrib_name,
                                     line=dict(color='red', width=2)),
                            row=1, col=2
                        )
            
            # Panel 3: empirical exceedance
            fig.add_trace(
                go.Scatter(x=exceedance['Exceedance Probability'], 
                         y=exceedance['Peak Value'],
                         mode='markers', name='Empirical Exceedance',
                         marker=dict(color='blue', size=8)),
                row=2, col=1
            )
            
            # Fitted extreme-value distribution
            if 'extreme_value_model' in results:
                model = results['extreme_value_model']
                x = np.logspace(-3, np.log10(0.9), 100)  # probabilities from 0.001 to 0.9
                
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
                    
                # Distribution curve
                fig.add_trace(
                    go.Scatter(x=x, y=y, mode='lines', name=line_name,
                             line=dict(color='red', width=2)),
                    row=2, col=1
                )
                
                # Logarithmic x-axis
                fig.update_xaxes(type='log', row=2, col=1)
            
            # Panel 4: return-period plot
            # Convert to years for plotting
            return_period_years_data = exceedance['Return Period (hours)'] / (24 * 365.25)
            
            fig.add_trace(
                go.Scatter(x=return_period_years_data, 
                         y=exceedance['Peak Value'],
                         mode='markers', name='Empirical Return Period',
                         marker=dict(color='blue', size=8)),
                row=2, col=2
            )
            
            # Theoretical return periods and return values
            if 'extreme_value_model' in results and 'return_values' in results and return_periods is not None:
                # Theoretical return-period curve
                rps = np.array(return_periods) / (24 * 365.25)  # convert to years
                rv_list = [results['return_values'][label] for label in return_period_labels]
                
                fig.add_trace(
                    go.Scatter(x=rps, y=rv_list, mode='lines+markers', 
                             name='Model Return Values',
                             line=dict(color='red', width=2),
                             marker=dict(color='red', size=10)),
                    row=2, col=2
                )
                
                # Confidence intervals
                if 'return_value_confidence_intervals' in results:
                    # Highlight requested return periods
                    last_label = return_period_labels[-1]
                    
                    if last_label in results['return_value_confidence_intervals']:
                        ci = results['return_value_confidence_intervals'][last_label]
                        ci_lower = ci[0] if isinstance(ci, tuple) else ci.get('lower_95', 0)
                        ci_upper = ci[1] if isinstance(ci, tuple) else ci.get('upper_95', 0)
                        
                        # Attach CI annotations
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
            
            # Figure title
            if title is None:
                if ch_name is not None:
                    scale_str = "Full Scale" if fullscale else "Model Scale"
                    title = f"Extreme Value Analysis for {ch_name}{unit_str} ({scale_str})"
                else:
                    title = "Extreme Value Analysis"
            
            # Layout
            fig.update_layout(
                title=title,
                width=1300,  # extra width for the legend
                height=900,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.2),
                margin=dict(r=150)  # extra right margin for the legend
            )
            
            # Axis labels
            fig.update_xaxes(title_text="Time (s)", row=1, col=1)
            fig.update_yaxes(title_text=f"Value{unit_str}", row=1, col=1)
            
            fig.update_xaxes(title_text="Peak Value", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            
            fig.update_xaxes(title_text="Exceedance Probability", row=2, col=1)
            fig.update_yaxes(title_text=f"Peak Value{unit_str}", row=2, col=1)
            
            fig.update_xaxes(title_text="Return Period (years)", row=2, col=2)
            fig.update_yaxes(title_text=f"Peak Value{unit_str}", row=2, col=2)
            
            # Save or show the figure
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
            logger.warning("Plotly is unavailable; falling back to matplotlib")
            backend = 'matplotlib'
        except Exception as e:
            logger.error(f"Failed to create Plotly visualisation: {str(e)}")
            backend = 'matplotlib'
    
    # Matplotlib backend
    if backend in ['matplotlib', 'seaborn']:
        try:
            import matplotlib.pyplot as plt
            
            # 2x2 subplot layout
            fig, axs = plt.subplots(2, 2, figsize=(15, 12))
            
            # Panel 1: raw series and detected peaks
            if data_array is not None:
                time = np.arange(len(data_array)) / fs
                
                # Downsample large series
                if len(data_array) > 10000:
                    step = len(data_array) // 10000 + 1
                    plot_time = time[::step]
                    plot_data = data_array[::step]
                else:
                    plot_time = time
                    plot_data = data_array
                
                # Plot the series
                axs[0, 0].plot(plot_time, plot_data, 'b-', alpha=0.5, linewidth=1, label='Data')
            
            # Positive peaks
            if len(pos_peaks_idx) > 0:
                pos_peak_times = pos_peaks_idx / fs
                axs[0, 0].plot(pos_peak_times, peaks_positive, 'ro', label='Positive Peaks')
            
            # Negative peaks
            if len(neg_peaks_idx) > 0:
                neg_peak_times = neg_peaks_idx / fs
                axs[0, 0].plot(neg_peak_times, peaks_negative, 'go', label='Negative Peaks')
            
            axs[0, 0].set_title('Original Data with Detected Peaks')
            axs[0, 0].set_xlabel('Time (s)')
            axs[0, 0].set_ylabel(f'Value{unit_str}')
            axs[0, 0].legend()
            
            # Panel 2: peak histogram
            if len(all_peaks) > 0:
                axs[0, 1].hist(all_peaks, bins=bins, alpha=0.7, color='blue', label='Peaks')
                
                # Fitted extreme-value distribution
                if 'extreme_value_model' in results:
                    model = results['extreme_value_model']
                    x = np.linspace(min(all_peaks), max(all_peaks), 100)
                    
                    if model['distribution'] == 'GEV':
                        shape = model['shape']
                        loc = model['loc']
                        scale = model['scale']
                        y = stats.genextreme.pdf(x, shape, loc, scale)
                        distrib_name = f"GEV (ξ={shape:.3f}, μ={loc:.3f}, σ={scale:.3f})"
                        
                        # Scale PDF to histogram counts
                        bin_width = (max(all_peaks) - min(all_peaks)) / bins
                        y = y * len(all_peaks) * bin_width
                        
                        # Distribution curve
                        axs[0, 1].plot(x, y, 'r-', linewidth=2, label=distrib_name)
                        axs[0, 1].legend()
                    elif model['distribution'] == 'Gumbel':
                        loc = model['loc']
                        scale = model['scale']
                        y = stats.gumbel_r.pdf(x, loc, scale)
                        distrib_name = f"Gumbel (μ={loc:.3f}, σ={scale:.3f})"
                        
                        # Scale PDF to histogram counts
                        bin_width = (max(all_peaks) - min(all_peaks)) / bins
                        y = y * len(all_peaks) * bin_width
                        
                        # Distribution curve
                        axs[0, 1].plot(x, y, 'r-', linewidth=2, label=distrib_name)
                        axs[0, 1].legend()
            
            axs[0, 1].set_title('Peak Value Histogram')
            axs[0, 1].set_xlabel('Peak Value')
            axs[0, 1].set_ylabel('Count')
            
            # Panel 3: empirical exceedance
            axs[1, 0].loglog(exceedance['Exceedance Probability'], 
                          exceedance['Peak Value'], 'bo', markersize=6,
                          label='Empirical Exceedance')
            
            # Fitted extreme-value distribution
            if 'extreme_value_model' in results:
                model = results['extreme_value_model']
                x = np.logspace(-3, np.log10(0.9), 100)  # probabilities from 0.001 to 0.9
                
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
            
            # Panel 4: return-period plot
            # Convert to years for plotting
            return_period_years_data = exceedance['Return Period (hours)'] / (24 * 365.25)
            
            axs[1, 1].loglog(return_period_years_data, exceedance['Peak Value'], 'bo', 
                          markersize=6, label='Empirical Return Period')
            
            # Theoretical return periods and return values
            if 'extreme_value_model' in results and 'return_values' in results and return_periods is not None:
                # Convert to years
                rps = np.array(return_periods) / (24 * 365.25)
                rv_list = [results['return_values'][label] for label in return_period_labels]
                
                axs[1, 1].loglog(rps, rv_list, 'ro-', linewidth=2, markersize=8,
                             label='Model Return Values')
                
                # Confidence intervals
                if 'return_value_confidence_intervals' in results:
                    last_label = return_period_labels[-1]
                    
                    if last_label in results['return_value_confidence_intervals']:
                        ci = results['return_value_confidence_intervals'][last_label]
                        ci_lower = ci[0] if isinstance(ci, tuple) else ci.get('lower_95', 0)
                        ci_upper = ci[1] if isinstance(ci, tuple) else ci.get('upper_95', 0)
                        
                        # Attach CI annotations
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
            
            # Figure title
            if title is None:
                if ch_name is not None:
                    scale_str = "Full Scale" if fullscale else "Model Scale"
                    title = f"Extreme Value Analysis for {ch_name}{unit_str} ({scale_str})"
                else:
                    title = "Extreme Value Analysis"
            
            fig.suptitle(title, fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            
            # Save figure
            if save_path is not None:
                plt.savefig(save_path, dpi=300)
                logger.info(f"Plot saved to {save_path}")
            
            # Show figure
            if visualization:
                plt.show()
            else:
                plt.close(fig)
            
            return fig
            
        except ImportError:
            logger.error("Matplotlib is unavailable")
            return None
        except Exception as e:
            logger.error(f"Failed to create Matplotlib visualisation: {str(e)}")
            return None
            
    # Reached if every backend failed
    logger.error("All visualisation backends failed")
    return None
