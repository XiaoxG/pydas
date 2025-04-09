"""
PyDAS Plot Module
================
This module contains plotting functions for PyDAS data.
"""

import logging
import numpy as np
import json

# Set up logging
logger = logging.getLogger('pydas_plot')

def plot_channel(pydas_obj, ch_name, sseg=0, title=None, xlabel='Time (s)', ylabel=None, 
              xlim=None, ylim=None, grid=True, show=True, save_path=None, 
              use_plotly=True, downsampling=True, max_points=40000, save_html=None,
              dpi=300, width=None, height=None, color=None, alpha=0.8, linewidth=1, 
              figsize=(12, 4), stats=True, table_width=0.3, column_widths=None):
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
        downsampling (bool): Whether to downsample large datasets (default: True)
        max_points (int): Maximum number of points to plot before downsampling (default: 20000)
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
    
    Returns:
        Figure object (matplotlib.figure.Figure or plotly.graph_objects.Figure)
    """
    try:
        logger = logging.getLogger('pydas')
        
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
        plt = None  # 初始化plt为None，稍后按需导入
        
        # If use_plotly is True, try to use Plotly for interactive web-based plotting
        if use_plotly:
            try:
                # 尝试使用Plotly创建图形
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                import json
                
                # HTML template for interactive plots - 定义在最前面，确保后续代码可以访问
                html_template = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyDAS Channel Plot</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }
        #plotDiv {
            width: 100%;
            height: 100vh;
        }
        .loading {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .loading-text {
            font-size: 24px;
            font-family: Arial, sans-serif;
        }
    </style>
</head>
<body>
    <div id="loadingDiv" class="loading">
        <div class="loading-text">Loading, please wait...</div>
    </div>
    <div id="plotDiv"></div>
    <script>
        // 完整数据用于统计计算
        var statsData = {stats_data};
        
        var plotData = {plot_data};
        
        // Use WebGL rendering for better performance
        var plot = Plotly.newPlot('plotDiv', plotData.data, plotData.layout, {
            responsive: true,
            displayModeBar: true,
            scrollZoom: true,
            showTips: false
        }).then(function() {
            // Hide loading indicator
            document.getElementById('loadingDiv').style.display = 'none';
            
            // 添加事件监听器用于缩放后更新统计信息
            var plotDiv = document.getElementById('plotDiv');
            plotDiv.on('plotly_relayout', function(eventData) {
                updateStatistics(eventData);
            });
        });
        
        // 根据当前可见范围更新统计信息
        function updateStatistics(eventData) {
            // 检查是否有范围更新
            if (!eventData) return;
            
            // 获取x轴范围
            var xMin, xMax;
            if (eventData['xaxis.range[0]'] !== undefined) {
                xMin = eventData['xaxis.range[0]'];
                xMax = eventData['xaxis.range[1]'];
            } else if (eventData['xaxis.range']) {
                xMin = eventData['xaxis.range'][0];
                xMax = eventData['xaxis.range'][1];
            } else {
                return; // 没有有效的范围数据
            }
            
            // 为每个通道计算新的统计信息
            var channelNames = [];
            var meanValues = [];
            var maxValues = [];
            var minValues = [];
            var stdValues = [];
            var unitValues = [];
            
            Object.keys(statsData).forEach(function(channel) {
                var xData = statsData[channel].x;
                var yData = statsData[channel].y;
                var unit = statsData[channel].unit;
                
                // 过滤可见范围内的数据
                var visibleData = [];
                for (var i = 0; i < xData.length; i++) {
                    if (xData[i] >= xMin && xData[i] <= xMax) {
                        visibleData.push(yData[i]);
                    }
                }
                
                // 如果有可见数据，计算统计量
                if (visibleData.length > 0) {
                    channelNames.push(channel);
                    unitValues.push(unit);
                    
                    // 计算平均值
                    var mean = visibleData.reduce(function(a, b) { return a + b; }, 0) / visibleData.length;
                    meanValues.push(mean.toPrecision(4));
                    
                    // 计算最大值和最小值
                    var max = Math.max.apply(null, visibleData);
                    var min = Math.min.apply(null, visibleData);
                    maxValues.push(max.toPrecision(4));
                    minValues.push(min.toPrecision(4));
                    
                    // 计算标准差
                    var variance = 0;
                    for (var i = 0; i < visibleData.length; i++) {
                        variance += Math.pow(visibleData[i] - mean, 2);
                    }
                    variance /= visibleData.length;
                    var stdDev = Math.sqrt(variance);
                    stdValues.push(stdDev.toPrecision(4));
                }
            });
            
            // 更新表格数据
            var tableUpdate = {
                cells: {
                    values: [channelNames, meanValues, maxValues, minValues, stdValues, unitValues]
                }
            };
            
            // 找到表格的索引
            var tableIndex = -1;
            for (var i = 0; i < plotData.data.length; i++) {
                if (plotData.data[i].type === 'table') {
                    tableIndex = i;
                    break;
                }
            }
            
            if (tableIndex >= 0) {
                // 更新表格
                Plotly.update('plotDiv', tableUpdate, {}, [tableIndex]);
            }
        }
        
        // Adjust plot size to maintain 3:1 aspect ratio
        function resizePlot() {
            var width = document.getElementById('plotDiv').offsetWidth;
            var height = width / 3;
            Plotly.relayout('plotDiv', {
                width: width,
                height: height
            });
        }
        
        // Resize on page load
        window.addEventListener('load', resizePlot);
        
        // Resize on window resize
        window.addEventListener('resize', resizePlot);
        
        // Performance optimization function
        function optimizePerformance() {
            // Limit update frequency
            var throttleTimeout;
            var plotDiv = document.getElementById('plotDiv');
            
            plotDiv.addEventListener('mousemove', function(e) {
                if (!throttleTimeout) {
                    throttleTimeout = setTimeout(function() {
                        throttleTimeout = null;
                    }, 30); // Limit to once every 30ms
                } else {
                    e.stopPropagation();
                }
            }, true);
        }
        
        // Call performance optimization
        optimizePerformance();
    </script>
</body>
</html>
'''
                
                # Get time vector
                n_sample = pydas_obj.segInfo.iloc[sseg]['N sample']
                
                # Filter out non-existent channels
                valid_channels = []
                for name in channel_list:
                    if name in pydas_obj.chInfo['Name'].values:
                        valid_channels.append(name)
                    else:
                        logger.warning(f"Channel '{name}' not found.")
                
                if not valid_channels:
                    logger.warning("No valid channels to plot.")
                    return None
                
                # Calculate downsampling factor
                if downsampling and n_sample > max_points:
                    downsample_factor = int(np.ceil(n_sample / max_points))
                    logger.debug(f"Data points ({n_sample}) exceed threshold ({max_points}), applying 1:{downsample_factor} downsampling")
                else:
                    downsample_factor = 1
                
                # Create downsampled time vector
                time = np.arange(0, n_sample, downsample_factor) / pydas_obj.__fs__
                
                # 创建带侧边表格的布局
                # 计算图表和表格的宽度比例
                plot_width_fraction = 1 - table_width
                
                # 创建具有2列、1行的子图布局
                fig = make_subplots(
                    rows=1, 
                    cols=2,
                    column_widths=[plot_width_fraction, table_width],
                    specs=[[{"type": "scatter"}, {"type": "table"}]],
                    horizontal_spacing=0.02
                )
                
                # Add traces for each channel to the first column (data plot)
                for name in valid_channels:
                    # Get channel unit
                    unit = pydas_obj.chInfo.loc[pydas_obj.chInfo['Name'] == name, 'Unit'].values[0]
                    
                    # Get channel data and downsample
                    data = pydas_obj.data[sseg][name].values[::downsample_factor]
                    
                    # Use WebGL rendering for better performance
                    fig.add_trace(
                        go.Scattergl(
                            x=time,
                            y=data,
                            mode='lines',
                            name=name,
                            hovertemplate='Time: %{x:.3f}s<br>Value: %{y:.6f} ' + unit + '<extra></extra>'
                        ),
                        row=1, col=1
                    )
                
                # Calculate dimensions if not provided
                if width is None:
                    width = 1200  # Default width is 1200 pixels
                
                if height is None:
                    height = width // 3  # Maintain 3:1 aspect ratio
                
                # Set title if not provided
                if title is None:
                    if len(valid_channels) == 1:
                        title = f'Channel: {valid_channels[0]} - Segment {sseg}'
                    else:
                        title = f'Channel Plot - Segment {sseg}'
                
                # 准备统计表格数据 - 单位只在行尾显示一次
                header_values = ["Channel", "Mean", "Max", "Min", "Std", "Unit"]
                cell_values = [[], [], [], [], [], []]
                
                for name in valid_channels:
                    # 获取通道单位
                    unit = pydas_obj.chInfo.loc[pydas_obj.chInfo['Name'] == name, 'Unit'].values[0]
                    
                    # 获取数据
                    data = pydas_obj.data[sseg][name].values[::downsample_factor]
                    
                    # 计算统计量
                    mean_val = np.mean(data)
                    max_val = np.max(data)
                    min_val = np.min(data)
                    std_val = np.std(data)
                    
                    # 添加到表格数据 - 数值不带单位
                    cell_values[0].append(name)
                    cell_values[1].append(f"{mean_val:.4g}")
                    cell_values[2].append(f"{max_val:.4g}")
                    cell_values[3].append(f"{min_val:.4g}")
                    cell_values[4].append(f"{std_val:.4g}")
                    cell_values[5].append(unit)  # 单位单独放在最后一列
                
                # 如果没有提供列宽，设置默认列宽
                if column_widths is None:
                    column_widths = [2, 1, 1, 1, 1.5, 0.8]  # 默认列宽比例
                
                # 添加统计信息表格到第二列
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=header_values,
                            fill_color='lightgrey',
                            align='center',
                            font=dict(size=12, color='black'),
                            height=25
                        ),
                        cells=dict(
                            values=cell_values,
                            align='center',
                            font=dict(size=11),
                            height=22
                        ),
                        columnwidth=column_widths
                    ),
                    row=1, col=2
                )
                
                # Set layout
                fig.update_layout(
                    title=title,
                    height=height,
                    width=width,
                    hovermode='closest',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(255, 255, 255, 0.8)'
                    ) if show else dict(visible=False),
                    margin=dict(l=50, r=30, t=50, b=50),
                    autosize=True,
                    uirevision='constant'
                )
                
                # 更新第一列的绘图区域
                fig.update_xaxes(
                    title=xlabel,
                    showgrid=grid, 
                    gridwidth=1, 
                    gridcolor='lightgray',
                    showspikes=True,
                    spikemode='across',
                    spikesnap='cursor',
                    showline=True,
                    row=1, col=1
                )
                
                fig.update_yaxes(
                    title=ylabel if ylabel is not None else 'Value',
                    showgrid=grid, 
                    gridwidth=1, 
                    gridcolor='lightgray',
                    showspikes=True,
                    spikemode='across',
                    spikesnap='cursor',
                    showline=True,
                    row=1, col=1
                )
                
                # Set range if xlim is provided
                if xlim is not None:
                    fig.update_xaxes(range=xlim, row=1, col=1)
                
                # Set range if ylim is provided
                if ylim is not None:
                    fig.update_yaxes(range=ylim, row=1, col=1)
                
                # 准备统计数据以便JavaScript使用
                stats_data = {}
                for name in valid_channels:
                    # 存储完整数据以便JavaScript计算统计值
                    x_data = time.tolist()
                    y_data = pydas_obj.data[sseg][name].values[::downsample_factor].tolist()
                    unit = pydas_obj.chInfo.loc[pydas_obj.chInfo['Name'] == name, 'Unit'].values[0]
                    stats_data[name] = {"x": x_data, "y": y_data, "unit": unit}
                
                # 标记Plotly图形创建成功
                plot_created = True
                fig_obj = fig  # 存储图形对象供后续使用
                
                # 处理保存和显示，然后立即返回，完全跳过Matplotlib部分
                # Handle save_path for HTML files
                if save_path and save_path.lower().endswith(('.html', '.htm')):
                    try:
                        # 将统计数据和图表数据嵌入HTML
                        html_content = html_template.replace('{stats_data}', json.dumps(stats_data))
                        html_content = html_content.replace('{plot_data}', fig_obj.to_json())
                        
                        # 保存HTML文件
                        with open(save_path, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        logger.debug(f"Interactive plot saved to {save_path}")
                    except Exception as e:
                        logger.error(f"Failed to save interactive plot to {save_path}: {e}")
                
                # 检查save_html参数格式
                if save_html and not save_html.lower().endswith(('.html', '.htm')):
                    logger.warning(f"save_html parameter should end with .html or .htm. Got: {save_html}")
                    save_html = save_html + '.html'
                    logger.debug(f"Appended .html extension: {save_html}")
                
                # Handle save_html
                if save_html and save_html != save_path:  # 避免重复保存
                    try:
                        # 将统计数据和图表数据嵌入HTML
                        html_content = html_template.replace('{stats_data}', json.dumps(stats_data))
                        html_content = html_content.replace('{plot_data}', fig_obj.to_json())
                        
                        # 保存HTML文件
                        with open(save_html, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        logger.debug(f"Interactive plot saved to {save_html}")
                    except Exception as e:
                        logger.error(f"Failed to save interactive plot to {save_html}: {e}")
                
                # Handle non-HTML save_path
                if save_path and not save_path.lower().endswith(('.html', '.htm')):
                    try:
                        fig_obj.write_image(save_path)
                        logger.debug(f"Plot saved to {save_path}")
                    except Exception as e:
                        logger.error(f"Failed to save plot to {save_path}: {e}")
                
                # Show the plot if requested
                if show:
                    logger.debug("Displaying plot using Plotly")
                    try:
                        # 尝试使用离线HTML方式渲染Plotly
                        import plotly.offline as pyo
                        import tempfile
                        import webbrowser
                        import os
                        
                        # 获取临时文件路径
                        temp_path = os.path.join(tempfile.gettempdir(), 'plotly_temp.html')
                        logger.debug(f"Creating temporary HTML at {temp_path}")
                        
                        # 将图表转换为HTML并保存
                        html_content = html_template.replace('{stats_data}', json.dumps(stats_data))
                        html_content = html_content.replace('{plot_data}', fig_obj.to_json())
                        
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        
                        # 在默认浏览器中打开（只会打开一个窗口）
                        webbrowser.open('file://' + os.path.abspath(temp_path))
                        logger.debug(f"Plot opened in browser at {temp_path}")
                    except Exception as e:
                        logger.error(f"Error with offline display: {e}")
                        # 如果离线方法失败，使用默认方法
                        logger.info("Falling back to standard Plotly show()")
                        fig_obj.show()
                
                # 尝试清理任何可能存在的matplotlib图形，防止显示额外窗口
                try:
                    if 'plt' in locals() or 'plt' in globals():
                        import matplotlib.pyplot as plt
                        plt.close('all')
                        logger.debug("Closed all matplotlib figures to prevent extra windows")
                except Exception as e:
                    logger.debug(f"No need to close matplotlib figures: {e}")
                
                logger.debug("Plotly flow complete - returning figure")
                
                # Return the figure object and exit function
                return fig_obj
            
            except ImportError:
                logger.warning("Plotly not installed. Falling back to matplotlib.")
                plot_created = False
            except Exception as e:
                logger.warning(f"Plotly initialization failed, falling back to matplotlib: {e}")
                plot_created = False
        
        # 只有在Plotly未成功创建图形时才使用Matplotlib
        if not plot_created:
            # 使用Matplotlib创建图形
            import matplotlib.pyplot as plt
            logger.debug("Importing matplotlib for plotting as Plotly was not used")
            # Set figure size
            width_inches = width / 100 if width else 12  # Convert pixels to inches
            height_inches = height / 100 if height else 4  # Convert pixels to inches
            plt.figure(figsize=(width_inches, height_inches), dpi=100)

            # Filter out non-existent channels
            valid_channels = []
            for name in channel_list:
                if name in pydas_obj.chInfo['Name'].values:
                    valid_channels.append(name)
                else:
                    logger.warning(f"Channel '{name}' not found.")
            
            if not valid_channels:
                logger.warning("No valid channels to plot.")
                return None
            
            # Calculate downsampling factor
            n_sample = pydas_obj.segInfo.iloc[sseg]['N sample']
            if downsampling and n_sample > max_points:
                downsample_factor = int(np.ceil(n_sample / max_points))
                logger.debug(f"Data points ({n_sample}) exceed threshold ({max_points}), applying 1:{downsample_factor} downsampling")
            else:
                downsample_factor = 1
            
            # Create downsampled time vector
            time = np.arange(0, n_sample, downsample_factor) / pydas_obj.__fs__
            
            # Plot each channel
            for name in valid_channels:
                # Get unit for label
                unit = pydas_obj.chInfo.loc[pydas_obj.chInfo['Name'] == name, 'Unit'].values[0]
                
                # Format unit label
                if unit and unit.lower() != 'none':
                    unit_label = f" ({unit})"
                else:
                    unit_label = ""
                
                # Get data and downsample
                data = pydas_obj.data[sseg][name].values[::downsample_factor]
                
                # Plot data
                plt.plot(time, data, label=name + unit_label)
            
            # Set labels and title
            plt.xlabel(xlabel)
            plt.ylabel(ylabel if ylabel else 'Value')
            
            # Set title if not provided
            if title is None:
                if len(valid_channels) == 1:
                    title = f'Channel: {valid_channels[0]} - Segment {sseg}'
                else:
                    title = f'Channel Plot - Segment {sseg}'
            
            plt.title(title)
            
            # Set grid
            plt.grid(grid)
            
            # Set limits if provided
            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)
            
            # Show legend if multiple channels
            if len(valid_channels) > 1 and show:
                plt.legend()
            
            # Add statistics as text annotations
            if stats:
                # Create empty strings for stats
                stats_text = "Channel Statistics:\n"
                for name in valid_channels:
                    # Get data
                    data = pydas_obj.data[sseg][name].values[::downsample_factor]
                    
                    # Calculate statistics
                    mean = np.mean(data)
                    max_val = np.max(data)
                    min_val = np.min(data)
                    std = np.std(data)
                    
                    # Get unit
                    unit = pydas_obj.chInfo.loc[pydas_obj.chInfo['Name'] == name, 'Unit'].values[0]
                    
                    # Add to stats text
                    stats_text += f"{name}: Mean={mean:.4g}, Max={max_val:.4g}, Min={min_val:.4g}, Std={std:.4g} {unit}\n"
                
                # Add text to plot
                plt.figtext(0.02, 0.02, stats_text, wrap=True, fontsize=8, 
                            bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()

        # Handle save_path - this only works for matplotlib since Plotly path is handled earlier
        if not plot_created and save_path:
            # For matplotlib plots
            if save_path.lower().endswith(('.html', '.htm')):
                # If we're using matplotlib, we can't save as HTML
                logger.error(f"Cannot save as HTML when using matplotlib. Requested path: {save_path}")
            else:
                # For non-HTML files, proceed with save
                try:
                    plt.savefig(save_path, dpi=dpi)
                    logger.debug(f"Plot saved to {save_path}")
                except Exception as e:
                    logger.error(f"Failed to save plot to {save_path}: {e}")
                    
        # Show plot if requested - Only for matplotlib since Plotly display is handled earlier
        if show and not plot_created and 'plt' in locals():
            # 使用Matplotlib显示
            logger.debug("Displaying plot using Matplotlib")
            plt.show()
                
        # Return the figure object - Plotly return is handled earlier
        return plt.gcf() if not plot_created and 'plt' in locals() else None
            
    except Exception as e:
        logger.error(f"Failed to create plot: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None 

def plot_histogram(pydas_obj, ch_name, sseg=0, title=None, xlabel=None, ylabel='Count', 
                bins=50, xlim=None, ylim=None, grid=True, show=True, save_path=None, 
                use_plotly=True, save_html=None, dpi=300, width=None, height=None, 
                color=None, alpha=0.6, figsize=(12, 6), fit_gaussian=True, fit_color='red'):
    """
    Plot a histogram of a channel from a PyDAS object.
    
    Parameters:
        pydas_obj (PyDAS): The PyDAS object containing channel data
        ch_name (str or list): Channel name or list of channel names to plot histograms for
        sseg (int): Segment index to plot (default: 0)
        title (str): Plot title (default: None, auto-generated)
        xlabel (str): X-axis label (default: None, auto-generated from channel name and unit)
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
        fit_gaussian (bool): Whether to fit a Gaussian distribution to the data (default: True)
        fit_color (str or list): Color of the Gaussian fit curve (default: 'red')
    
    Returns:
        Figure object (matplotlib.figure.Figure or plotly.graph_objects.Figure)
    """
    try:
        logger = logging.getLogger('pydas')
        
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
        else:
            channel_list = ch_name
            
        # Validate all channels exist
        for ch in channel_list:
            if ch not in pydas_obj.chInfo['Name'].values:
                logger.error(f"Channel '{ch}' not found.")
                return None
        
        # Set up color list if needed
        if color is None:
            # Default colors for multiple channels
            default_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
            colors = [default_colors[i % len(default_colors)] for i in range(len(channel_list))]
        elif isinstance(color, str):
            # Single color provided, replicate for all channels
            colors = [color] * len(channel_list)
        else:
            # List of colors provided
            colors = color if len(color) >= len(channel_list) else color + [default_colors[i % len(default_colors)] for i in range(len(channel_list) - len(color))]
            
        # Set up fit colors
        if isinstance(fit_color, str):
            fit_colors = [fit_color] * len(channel_list)
        else:
            # List of fit colors provided
            fit_colors = fit_color if len(fit_color) >= len(channel_list) else fit_color + ['red', 'darkgreen', 'darkblue', 'purple'][:(len(channel_list) - len(fit_color))]
            
        # Set default title if not provided
        if title is None:
            if len(channel_list) == 1:
                title = f"Histogram of {channel_list[0]} - Segment {sseg}"
            else:
                title = f"Histogram Comparison - Segment {sseg}"
                
        # Flag to track if we've successfully created a plot
        plot_created = False
        fig = None
        
        # Prepare data and fit parameters for each channel
        channel_data = []
        gaussian_fits = []
        
        for i, ch in enumerate(channel_list):
            # Get channel data
            data = pydas_obj.data[sseg][ch].values
            
            # Get channel unit
            unit = pydas_obj.chInfo.loc[pydas_obj.chInfo['Name'] == ch, 'Unit'].values[0]
            
            channel_data.append({
                'name': ch,
                'data': data,
                'unit': unit,
                'color': colors[i]
            })
            
            # If fit_gaussian is True, compute Gaussian fit parameters
            if fit_gaussian:
                try:
                    from scipy import stats
                    
                    # Calculate mean and standard deviation for Gaussian fit
                    mu = np.mean(data)
                    sigma = np.std(data)
                    
                    # Create x values for the fit curve
                    if xlim is not None:
                        x_min, x_max = xlim
                    else:
                        x_min = min(data)
                        x_max = max(data)
                        # Add some padding
                        padding = 0.1 * (x_max - x_min)
                        x_min -= padding
                        x_max += padding
                    
                    x_fit = np.linspace(x_min, x_max, 1000)
                    y_fit = stats.norm.pdf(x_fit, mu, sigma)
                    
                    # Calculate y-scale factor to match histogram height
                    hist_data = np.histogram(data, bins=bins)
                    bin_heights = hist_data[0]
                    max_bin_height = np.max(bin_heights)
                    # Scale the PDF to match the histogram height
                    y_fit = y_fit * (max_bin_height / np.max(y_fit))
                    
                    gaussian_fits.append({
                        'x_fit': x_fit,
                        'y_fit': y_fit,
                        'mu': mu,
                        'sigma': sigma,
                        'fit_color': fit_colors[i]
                    })
                except ImportError:
                    logger.warning("scipy.stats not found. Gaussian fit disabled.")
                    gaussian_fits.append(None)
                except Exception as e:
                    logger.warning(f"Gaussian fit failed for {ch}: {str(e)}. Continuing without fit.")
                    gaussian_fits.append(None)
            else:
                gaussian_fits.append(None)
                
        # Set default xlabel if not provided
        if xlabel is None:
            if len(channel_list) == 1:
                ch = channel_list[0]
                unit = pydas_obj.chInfo.loc[pydas_obj.chInfo['Name'] == ch, 'Unit'].values[0]
                if unit and unit.lower() != 'none':
                    xlabel = f"{ch} ({unit})"
                else:
                    xlabel = ch
            else:
                xlabel = "Value"
        
        # If use_plotly is True, try to use Plotly for interactive web-based plotting
        if use_plotly:
            try:
                import plotly.graph_objects as go
                import json
                
                # HTML template for interactive histogram
                hist_html_template = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyDAS Histogram</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }
        #plotDiv {
            width: 100%;
            height: 100vh;
        }
        .loading {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .loading-text {
            font-size: 24px;
            font-family: Arial, sans-serif;
        }
    </style>
</head>
<body>
    <div id="loadingDiv" class="loading">
        <div class="loading-text">Loading, please wait...</div>
    </div>
    <div id="plotDiv"></div>
    <script>
        var plotData = {plot_data};
        
        // Use WebGL rendering for better performance
        var plot = Plotly.newPlot('plotDiv', plotData.data, plotData.layout, {
            responsive: true,
            displayModeBar: true,
            scrollZoom: true,
            showTips: false
        }).then(function() {
            // Hide loading indicator
            document.getElementById('loadingDiv').style.display = 'none';
        });
        
        // Adjust plot size to maintain aspect ratio
        function resizePlot() {
            var width = document.getElementById('plotDiv').offsetWidth;
            var height = width / 2;
            Plotly.relayout('plotDiv', {
                width: width,
                height: height
            });
        }
        
        // Resize on page load
        window.addEventListener('load', resizePlot);
        
        // Resize on window resize
        window.addEventListener('resize', resizePlot);
    </script>
</body>
</html>
'''
                
                # Calculate dimensions if not provided
                if width is None:
                    width = 1200  # Default width is 1200 pixels
                
                if height is None:
                    height = width // 2  # Maintain 2:1 aspect ratio for histogram
                
                # Create figure
                fig = go.Figure()
                
                # Add histogram traces and Gaussian fits for each channel
                annotations = []
                
                for i, ch_data in enumerate(channel_data):
                    # Calculate histogram
                    hist_data = np.histogram(ch_data['data'], bins=bins)
                    bin_values = hist_data[0]
                    bin_edges = hist_data[1]
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Add histogram trace
                    fig.add_trace(
                        go.Bar(
                            x=bin_centers,
                            y=bin_values,
                            width=(bin_edges[1] - bin_edges[0]),  # Width of bars
                            name=ch_data['name'],
                            marker_color=ch_data['color'],
                            opacity=alpha,
                            hovertemplate=f"{ch_data['name']}<br>Value: %{{x:.4g}}<br>Count: %{{y}}<extra></extra>"
                        )
                    )
                    
                    # Add Gaussian fit curve if available
                    if gaussian_fits[i] is not None:
                        fig.add_trace(
                            go.Scatter(
                                x=gaussian_fits[i]['x_fit'],
                                y=gaussian_fits[i]['y_fit'],
                                mode='lines',
                                name=f"{ch_data['name']} Gaussian Fit",
                                line=dict(color=gaussian_fits[i]['fit_color'], width=2),
                                hovertemplate=f"{ch_data['name']} Fit<br>Value: %{{x:.4g}}<br>Density: %{{y:.4g}}<extra></extra>"
                            )
                        )
                    
                    # Add statistics annotations for each channel
                    mean_val = np.mean(ch_data['data'])
                    median_val = np.median(ch_data['data'])
                    min_val = np.min(ch_data['data'])
                    max_val = np.max(ch_data['data'])
                    std_val = np.std(ch_data['data'])
                    
                    stats_text = (
                        f"<b>{ch_data['name']}</b><br>"
                        f"Mean: {mean_val:.4g}<br>"
                        f"Median: {median_val:.4g}<br>"
                        f"Min: {min_val:.4g}<br>"
                        f"Max: {max_val:.4g}<br>"
                        f"Std: {std_val:.4g}<br>"
                        f"Count: {len(ch_data['data'])}"
                    )
                    
                    if gaussian_fits[i] is not None:
                        stats_text += f"<br><br>Gaussian Fit:<br>μ: {gaussian_fits[i]['mu']:.4g}<br>σ: {gaussian_fits[i]['sigma']:.4g}"
                    
                    # Add annotation for this channel
                    annotations.append(
                        dict(
                            x=0.98,
                            y=0.98 - (i * 0.25),  # Stack annotations vertically
                            xref="paper",
                            yref="paper",
                            text=stats_text,
                            showarrow=False,
                            font=dict(size=12),
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            bordercolor="gray",
                            borderwidth=1,
                            borderpad=4,
                            align="left"
                        )
                    )
                
                # Set layout
                fig.update_layout(
                    title=title,
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    height=height,
                    width=width,
                    hovermode='closest',
                    barmode='overlay',  # Overlay histograms
                    bargap=0,  # Gap between bars
                    annotations=annotations,
                    margin=dict(l=50, r=30, t=50, b=50),
                    autosize=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Update axes
                fig.update_xaxes(
                    showgrid=grid, 
                    gridwidth=1, 
                    gridcolor='lightgray',
                    showline=True
                )
                
                fig.update_yaxes(
                    showgrid=grid, 
                    gridwidth=1, 
                    gridcolor='lightgray',
                    showline=True
                )
                
                # Set range if xlim is provided
                if xlim is not None:
                    fig.update_xaxes(range=xlim)
                
                # Set range if ylim is provided
                if ylim is not None:
                    fig.update_yaxes(range=ylim)
                
                # Mark Plotly plot as created
                plot_created = True
                fig_obj = fig
                
                # Handle save_path for HTML files
                if save_path and save_path.lower().endswith(('.html', '.htm')):
                    try:
                        # Embed plot data in HTML
                        html_content = hist_html_template.replace('{plot_data}', fig_obj.to_json())
                        
                        # Save HTML file
                        with open(save_path, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        logger.debug(f"Interactive histogram saved to {save_path}")
                    except Exception as e:
                        logger.error(f"Failed to save interactive histogram to {save_path}: {e}")
                
                # Handle save_html parameter
                if save_html and not save_html.lower().endswith(('.html', '.htm')):
                    logger.warning(f"save_html parameter should end with .html or .htm. Got: {save_html}")
                    save_html = save_html + '.html'
                    logger.debug(f"Appended .html extension: {save_html}")
                
                # Handle save_html
                if save_html and save_html != save_path:  # Avoid duplicate saving
                    try:
                        # Embed plot data in HTML
                        html_content = hist_html_template.replace('{plot_data}', fig_obj.to_json())
                        
                        # Save HTML file
                        with open(save_html, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        logger.debug(f"Interactive histogram saved to {save_html}")
                    except Exception as e:
                        logger.error(f"Failed to save interactive histogram to {save_html}: {e}")
                
                # Handle non-HTML save_path
                if save_path and not save_path.lower().endswith(('.html', '.htm')):
                    try:
                        fig_obj.write_image(save_path)
                        logger.debug(f"Histogram saved to {save_path}")
                    except Exception as e:
                        logger.error(f"Failed to save histogram to {save_path}: {e}")
                
                # Show the plot if requested
                if show:
                    logger.debug("Displaying histogram using Plotly")
                    try:
                        # Try offline HTML rendering for Plotly
                        import plotly.offline as pyo
                        import tempfile
                        import webbrowser
                        import os
                        
                        # Get temporary file path
                        temp_path = os.path.join(tempfile.gettempdir(), 'plotly_histogram_temp.html')
                        logger.debug(f"Creating temporary HTML at {temp_path}")
                        
                        # Convert plot to HTML and save
                        html_content = hist_html_template.replace('{plot_data}', fig_obj.to_json())
                        
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        
                        # Open in default browser (will only open one window)
                        webbrowser.open('file://' + os.path.abspath(temp_path))
                        logger.debug(f"Histogram opened in browser at {temp_path}")
                    except Exception as e:
                        logger.error(f"Error with offline display: {e}")
                        # If offline method fails, use default method
                        logger.info("Falling back to standard Plotly show()")
                        fig_obj.show()
                
                # Try to clean up any matplotlib figures to prevent extra windows
                try:
                    if 'plt' in locals() or 'plt' in globals():
                        import matplotlib.pyplot as plt
                        plt.close('all')
                        logger.debug("Closed all matplotlib figures to prevent extra windows")
                except Exception as e:
                    logger.debug(f"No need to close matplotlib figures: {e}")
                
                logger.debug("Plotly histogram flow complete - returning figure")
                
                # Return the figure object
                return fig_obj
                
            except ImportError:
                logger.warning("Plotly not installed. Falling back to matplotlib.")
                plot_created = False
            except Exception as e:
                logger.warning(f"Plotly initialization failed, falling back to matplotlib: {e}")
                plot_created = False
        
        # If Plotly plot wasn't created, use Matplotlib
        if not plot_created:
            # Create matplotlib figure
            import matplotlib.pyplot as plt
            from matplotlib import patches
            logger.debug("Importing matplotlib for plotting histogram as Plotly was not used")
            
            # Set figure size
            width_inches = width / 100 if width else figsize[0]
            height_inches = height / 100 if height else figsize[1]
            fig = plt.figure(figsize=(width_inches, height_inches))
            
            # Add histograms for each channel
            for i, ch_data in enumerate(channel_data):
                # Plot histogram
                n, bins, patches = plt.hist(ch_data['data'], bins=bins, alpha=alpha, 
                                           color=ch_data['color'], 
                                           label=ch_data['name'],
                                           histtype='bar')
                
                # Add Gaussian fit curve if available
                if gaussian_fits[i] is not None:
                    plt.plot(gaussian_fits[i]['x_fit'], gaussian_fits[i]['y_fit'], 
                            color=gaussian_fits[i]['fit_color'], linewidth=2, 
                            label=f"{ch_data['name']} Fit")
            
            # Add statistics as text annotations for each channel
            stats_texts = []
            for i, ch_data in enumerate(channel_data):
                mean_val = np.mean(ch_data['data'])
                median_val = np.median(ch_data['data'])
                min_val = np.min(ch_data['data'])
                max_val = np.max(ch_data['data'])
                std_val = np.std(ch_data['data'])
                
                stats_text = (
                    f"{ch_data['name']}:\n"
                    f"Mean: {mean_val:.4g}\n"
                    f"Median: {median_val:.4g}\n"
                    f"Min: {min_val:.4g}\n"
                    f"Max: {max_val:.4g}\n"
                    f"Std: {std_val:.4g}\n"
                    f"Count: {len(ch_data['data'])}"
                )
                
                if gaussian_fits[i] is not None:
                    stats_text += f"\n\nGaussian Fit:\nμ: {gaussian_fits[i]['mu']:.4g}\nσ: {gaussian_fits[i]['sigma']:.4g}"
                
                stats_texts.append(stats_text)
            
            # Add text to plot - for multiple channels, stack annotations
            for i, stats_text in enumerate(stats_texts):
                bbox_props = dict(boxstyle='round', facecolor='white', alpha=0.8)
                plt.annotate(stats_text, xy=(0.98, 0.98 - (i * 0.25)), xycoords='axes fraction',
                            horizontalalignment='right', verticalalignment='top',
                            bbox=bbox_props)
            
            # Set labels and title
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            
            # Set grid
            plt.grid(grid)
            
            # Set limits if provided
            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)
            
            plt.tight_layout()
            
            # Handle save_path
            if save_path and not save_path.lower().endswith(('.html', '.htm')):
                try:
                    plt.savefig(save_path, dpi=dpi)
                    logger.debug(f"Histogram saved to {save_path}")
                except Exception as e:
                    logger.error(f"Failed to save histogram to {save_path}: {e}")
            elif save_path:
                logger.error(f"Cannot save as HTML when using matplotlib. Requested path: {save_path}")
            
            # Show plot if requested
            if show:
                logger.debug("Displaying histogram using Matplotlib")
                plt.show()
            
            # Return figure
            return fig
            
    except Exception as e:
        logger.error(f"Failed to create histogram: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
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
    Plot one channel against another channel (XY plot) from a PyDAS object.
    
    Parameters:
        pydas_obj (PyDAS): The PyDAS object containing channel data
        x_ch_name (str): Channel name to use for X-axis
        y_ch_name (str): Channel name to use for Y-axis
        sseg (int): Segment index to plot (default: 0)
        title (str): Plot title (default: None, auto-generated)
        xlabel (str): X-axis label (default: None, auto-generated from channel name and unit)
        ylabel (str): Y-axis label (default: None, auto-generated from channel name and unit)
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
        color (str): Color for the scatter points (default: 'blue')
        alpha (float): Opacity for the scatter points (default: 0.8)
        marker_size (float): Size of the scatter points (default: 5)
        figsize (tuple): Figure size for matplotlib (default: (8, 8))
        line (bool): Connect points with lines (default: False)
        fit_line (bool): Show linear regression fit line (default: False)
        fit_color (str): Color for fit line (default: 'red')
        fit_line_width (float): Width of fit line (default: 2)
        fit_alpha (float): Opacity of fit line (default: 0.8)
        show_stats (bool): Show statistics on the plot (default: False)
        downsampling (bool): Apply downsampling for large datasets (default: True)
        max_points (int): Maximum number of points to show before downsampling (default: 10000)
        density_plot (bool): Show density contour plot for large datasets (default: False)
        density_colorscale (str): Colorscale for density plot (default: 'Viridis')
        density_opacity (float): Opacity for density contours (default: 0.7)
        use_webgl (bool): Use WebGL rendering for better performance (default: True)
        adaptive_sampling (bool): Use adaptive sampling to preserve features (default: False)
        datashade (bool): Use datashading for very large datasets (default: False)
        contour_levels (int): Number of contour levels for density plot (default: 20)
        sampling_algorithm (str): Algorithm for downsampling: 'lttb', 'uniform', or 'peak' (default: 'lttb')
        memory_efficient (bool): Use memory-efficient methods for very large datasets (default: True)
        bin_size (tuple): Bin size for 2D histogram (x_bins, y_bins) (default: None)
        
    Returns:
        tuple: (pandas.DataFrame with x and y data, figure object)
    """
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import logging
        
        # Get logger
        logger = logging.getLogger('pydas')
        
        # Get data for the specified channels
        try:
            x_data = pydas_obj.data[sseg][x_ch_name].values
            y_data = pydas_obj.data[sseg][y_ch_name].values
        except KeyError as e:
            logger.error(f"Channel not found: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error accessing data: {str(e)}")
            return None
        
        # Ensure data is numpy array
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        
        # Check data length
        if len(x_data) != len(y_data):
            logger.error(f"Channel lengths do not match: {x_ch_name} ({len(x_data)}) vs {y_ch_name} ({len(y_data)})")
            return None
        
        if len(x_data) == 0:
            logger.error(f"No data in channel {x_ch_name}")
            return None
            
        # Set default plot title if not provided
        if title is None:
            title = f"{y_ch_name} vs {x_ch_name}"
            
        # Get channel units if available
        x_unit = ""
        y_unit = ""
        try:
            x_unit = pydas_obj.chInfo.loc[pydas_obj.chInfo['Name'] == x_ch_name, 'Unit'].values[0]
            y_unit = pydas_obj.chInfo.loc[pydas_obj.chInfo['Name'] == y_ch_name, 'Unit'].values[0]
        except (IndexError, KeyError, AttributeError):
            pass
        
        # Set default axis labels if not provided
        if xlabel is None:
            xlabel = f"{x_ch_name} ({x_unit})" if x_unit else x_ch_name
        if ylabel is None:
            ylabel = f"{y_ch_name} ({y_unit})" if y_unit else y_ch_name
            
        # Track original number of points for stats
        original_points = len(x_data)
        
        # Initialize statistics data
        stats_data = {}
        
        # Calculate basic statistics if needed
        if show_stats or density_plot:
            stats_data = {
                'x_mean': np.mean(x_data),
                'x_min': np.min(x_data),
                'x_max': np.max(x_data),
                'x_std': np.std(x_data),
                'y_mean': np.mean(y_data),
                'y_min': np.min(y_data),
                'y_max': np.max(y_data),
                'y_std': np.std(y_data),
                'count': len(x_data),
                'original_count': original_points
            }
            
            # Calculate correlation coefficient
            corr_coef = np.corrcoef(x_data, y_data)[0, 1]
            stats_data['corr_coef'] = corr_coef
        
        # Apply downsampling for large datasets
        if downsampling and len(x_data) > max_points:
            # 记录原始数据点数
            original_points = len(x_data)
            
            if sampling_algorithm == 'lttb':
                # 实现Largest Triangle Three Buckets算法 (保留数据形状的更好算法)
                try:
                    # 初始化结果数组
                    sampled_x = []
                    sampled_y = []
                except Exception as e:
                    logger.error(f"Error initializing LTTB downsampling: {str(e)}")
                    # 如果失败则返回原始数据
                    return x_data, y_data
                    # LTTB (Largest Triangle Three Buckets) implementation
                    def _lttb_downsample(data_x, data_y, n_out):
                        """
                        Downsample data using the LTTB algorithm.
                        This algorithm preserves the visual shape of the data much better than uniform sampling.
                        """
                        if len(data_x) <= n_out:
                            return data_x, data_y
                        
                        # Make sure we are working with numpy arrays
                        data_x = np.asarray(data_x)
                        data_y = np.asarray(data_y)
                        
                        # Create points array
                        points = np.column_stack([data_x, data_y])
                        
                        # Bucket size
                        bucket_size = (len(points) - 2) / (n_out - 2)
                        
                        # Always include first and last points
                        result = np.zeros((n_out, 2))
                        result[0] = points[0]
        # Calculate linear regression if fit_line is True
        if fit_line:
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
                
                if show_stats:
                    stats_data['slope'] = slope
                    stats_data['intercept'] = intercept
                    stats_data['r_squared'] = r_value**2
                    stats_data['p_value'] = p_value
                    stats_data['std_err'] = std_err
                
                # Create line points
                x_min, x_max = np.min(x_data), np.max(x_data)
                fit_x = np.array([x_min, x_max])
                fit_y = slope * fit_x + intercept
            except ImportError:
                logger.warning("scipy.stats not found. Linear regression disabled.")
                fit_line = False
            except Exception as e:
                logger.warning(f"Linear regression failed: {str(e)}. Continuing without fit.")
                fit_line = False
        
        # Flag to track if we've successfully created a plot
        plot_created = False
        fig = None
        
        # If use_plotly is True, try to use Plotly for interactive web-based plotting
        if use_plotly:
            try:
                import plotly.graph_objects as go
                import json
                
                # HTML template for interactive scatter plot
                xy_html_template = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyDAS XY Plot</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }
        #plotDiv {
            width: 100%;
            height: 100vh;
        }
        .loading {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .loading-text {
            font-size: 24px;
            font-family: Arial, sans-serif;
        }
    </style>
</head>
<body>
    <div id="loadingDiv" class="loading">
        <div class="loading-text">Loading, please wait...</div>
    </div>
    <div id="plotDiv"></div>
    <script>
        var plotData = {plot_data};
        
        // Use WebGL rendering for better performance
        var plot = Plotly.newPlot('plotDiv', plotData.data, plotData.layout, {
            responsive: true,
            displayModeBar: true,
            scrollZoom: true,
            showTips: false
        }).then(function() {
            // Hide loading indicator
            document.getElementById('loadingDiv').style.display = 'none';
        });
        
        // Adjust plot size to maintain 1:1 aspect ratio
        function resizePlot() {
            var container = document.getElementById('plotDiv');
            var width = container.offsetWidth;
            var height = container.offsetHeight;
            var size = Math.min(width, height);
            
            Plotly.relayout('plotDiv', {
                width: size,
                height: size
            });
        }
        
        // Resize on page load
        window.addEventListener('load', resizePlot);
        
        // Resize on window resize
        window.addEventListener('resize', resizePlot);
    </script>
</body>
</html>
'''
                
                # Calculate dimensions if not provided
                if width is None:
                    width = 800  # Default width is 800 pixels
                
                if height is None:
                    height = width  # Maintain 1:1 aspect ratio
                
                # Create figure
                fig = go.Figure()
                
                # Add density plot if requested
                if density_plot and len(x_data) > 100:  # Only add density for reasonably large datasets
                    fig.add_trace(
                        go.Histogram2dContour(
                            x=x_data,
                            y=y_data,
                            colorscale=density_colorscale,
                            reversescale=False,
                            showscale=True,
                            hoverinfo='skip',
                            opacity=density_opacity,
                            name='Density',
                            contours=dict(
                                showlabels=True,
                                labelfont=dict(
                                    family='Arial',
                                    color='white'
                                )
                            )
                        )
                    )
                
                # Add scatter trace
                scatter_mode = 'markers' if not line else 'markers+lines'
                
                # Add scatter trace with reduced marker size if density plot is used
                adjusted_marker_size = marker_size * 0.7 if density_plot else marker_size
                
                fig.add_trace(
                    go.Scattergl(
                        x=x_data,
                        y=y_data,
                        mode=scatter_mode,
                        name='Data Points',
                        marker=dict(
                            color=color,
                            size=adjusted_marker_size,
                            opacity=alpha
                        ),
                        hovertemplate=f"{x_ch_name}: %{{x:.4g}}<br>{y_ch_name}: %{{y:.4g}}<extra></extra>"
                    )
                )
                
                # Add linear regression line if requested
                if fit_line:
                    fig.add_trace(
                        go.Scatter(
                            x=fit_x,
                            y=fit_y,
                            mode='lines',
                            name='Linear Fit',
                            line=dict(color=fit_color, width=fit_line_width),
                            opacity=fit_alpha,
                            hovertemplate=f"y = {slope:.4g}x + {intercept:.4g}<br>R² = {r_value**2:.4g}<extra></extra>"
                        )
                    )
                
                # Add downsampling info to the title if downsampling was applied
                plot_title = title
                if downsampling and original_points > len(x_data):
                    plot_title += f" (Showing {len(x_data):,} of {original_points:,} points)"
                
                # Construct statistics text if requested
                annotations = []
                if show_stats:
                    stats_text = (
                        f"<b>Statistics:</b><br>"
                        f"<b>{x_ch_name}:</b><br>"
                        f"Mean: {stats_data['x_mean']:.4g}<br>"
                        f"Median: {stats_data['x_median']:.4g}<br>"
                        f"Min: {stats_data['x_min']:.4g}<br>"
                        f"Max: {stats_data['x_max']:.4g}<br>"
                        f"Std: {stats_data['x_std']:.4g}<br><br>"
                        f"<b>{y_ch_name}:</b><br>"
                        f"Mean: {stats_data['y_mean']:.4g}<br>"
                        f"Median: {stats_data['y_median']:.4g}<br>"
                        f"Min: {stats_data['y_min']:.4g}<br>"
                        f"Max: {stats_data['y_max']:.4g}<br>"
                        f"Std: {stats_data['y_std']:.4g}<br><br>"
                        f"<b>Correlation:</b><br>"
                        f"Coefficient: {corr_coef:.4g}<br>"
                        f"Count: {len(x_data)}"
                    )
                    
                    if fit_line:
                        stats_text += (
                            f"<br><br><b>Linear Fit:</b><br>"
                            f"y = {slope:.4g}x + {intercept:.4g}<br>"
                            f"R²: {r_value**2:.4g}<br>"
                            f"p-value: {p_value:.4g}<br>"
                            f"Std Error: {std_err:.4g}"
                        )
                    
                    # Add annotation for statistics
                    annotations.append(
                        dict(
                            x=0.02,
                            y=0.98,
                            xref="paper",
                            yref="paper",
                            text=stats_text,
                            showarrow=False,
                            font=dict(size=12),
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            bordercolor="gray",
                            borderwidth=1,
                            borderpad=4,
                            align="left"
                        )
                    )
                
                # Set layout with 1:1 aspect ratio
                fig.update_layout(
                    title=plot_title,
                    xaxis_title=xlabel,
                    yaxis_title=ylabel,
                    height=height,
                    width=width,
                    hovermode='closest',
                    annotations=annotations,
                    margin=dict(l=50, r=50, t=50, b=50),
                    autosize=False,
                    # Ensure plot maintains 1:1 aspect ratio
                    yaxis=dict(
                        scaleanchor="x",
                        scaleratio=1,
                    )
                )
                
                # Update axes
                fig.update_xaxes(
                    showgrid=grid, 
                    gridwidth=1, 
                    gridcolor='lightgray',
                    showline=True
                )
                
                fig.update_yaxes(
                    showgrid=grid, 
                    gridwidth=1, 
                    gridcolor='lightgray',
                    showline=True
                )
                
                # Set range if xlim is provided
                if xlim is not None:
                    fig.update_xaxes(range=xlim)
                
                # Set range if ylim is provided
                if ylim is not None:
                    fig.update_yaxes(range=ylim)
                
                # Mark Plotly plot as created
                plot_created = True
                fig_obj = fig
                
                # Handle save_path for HTML files
                if save_path and save_path.lower().endswith(('.html', '.htm')):
                    try:
                        # Embed plot data in HTML
                        html_content = xy_html_template.replace('{plot_data}', fig_obj.to_json())
                        
                        # Save HTML file
                        with open(save_path, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        logger.debug(f"Interactive XY plot saved to {save_path}")
                    except Exception as e:
                        logger.error(f"Failed to save interactive XY plot to {save_path}: {e}")
                
                # Handle save_html parameter
                if save_html and not save_html.lower().endswith(('.html', '.htm')):
                    logger.warning(f"save_html parameter should end with .html or .htm. Got: {save_html}")
                    save_html = save_html + '.html'
                    logger.debug(f"Appended .html extension: {save_html}")
                
                # Handle save_html
                if save_html and save_html != save_path:  # Avoid duplicate saving
                    try:
                        # Embed plot data in HTML
                        html_content = xy_html_template.replace('{plot_data}', fig_obj.to_json())
                        
                        # Save HTML file
                        with open(save_html, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        logger.debug(f"Interactive XY plot saved to {save_html}")
                    except Exception as e:
                        logger.error(f"Failed to save interactive XY plot to {save_html}: {e}")
                
                # Handle non-HTML save_path
                if save_path and not save_path.lower().endswith(('.html', '.htm')):
                    try:
                        fig_obj.write_image(save_path)
                        logger.debug(f"XY plot saved to {save_path}")
                    except Exception as e:
                        logger.error(f"Failed to save XY plot to {save_path}: {e}")
                
                # Show the plot if requested
                if show:
                    logger.debug("Displaying XY plot using Plotly")
                    try:
                        # Try offline HTML rendering for Plotly
                        import plotly.offline as pyo
                        import tempfile
                        import webbrowser
                        import os
                        
                        # Get temporary file path
                        temp_path = os.path.join(tempfile.gettempdir(), 'plotly_xy_temp.html')
                        logger.debug(f"Creating temporary HTML at {temp_path}")
                        
                        # Convert plot to HTML and save
                        html_content = xy_html_template.replace('{plot_data}', fig_obj.to_json())
                        
                        with open(temp_path, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        
                        # Open in default browser (will only open one window)
                        webbrowser.open('file://' + os.path.abspath(temp_path))
                        logger.debug(f"XY plot opened in browser at {temp_path}")
                    except Exception as e:
                        logger.error(f"Error with offline display: {e}")
                        # If offline method fails, use default method
                        logger.info("Falling back to standard Plotly show()")
                        fig_obj.show()
                
                # Try to clean up any matplotlib figures to prevent extra windows
                try:
                    if 'plt' in locals() or 'plt' in globals():
                        import matplotlib.pyplot as plt
                        plt.close('all')
                        logger.debug("Closed all matplotlib figures to prevent extra windows")
                except Exception as e:
                    logger.debug(f"No need to close matplotlib figures: {e}")
                
                logger.debug("Plotly XY plot flow complete - returning figure")
                
                # Return the figure object
                return fig_obj
                
            except ImportError:
                logger.warning("Plotly not installed. Falling back to matplotlib.")
                plot_created = False
            except Exception as e:
                logger.warning(f"Plotly initialization failed, falling back to matplotlib: {e}")
                plot_created = False
        
        # If Plotly plot wasn't created, use Matplotlib
        if not plot_created:
            # Create matplotlib figure
            import matplotlib.pyplot as plt
            logger.debug("Importing matplotlib for plotting XY plot as Plotly was not used")
            
            # Set figure size - ensure square aspect ratio (1:1)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, aspect='equal')
            
            # Add downsampling info to the title if downsampling was applied
            plot_title = title
            if downsampling and original_points > len(x_data):
                plot_title += f" (Showing {len(x_data):,} of {original_points:,} points)"
            
            # Add density plot (hexbin) if requested
            if density_plot and len(x_data) > 100:
                hexbin = ax.hexbin(x_data, y_data, gridsize=50, cmap=density_colorscale.lower(), 
                                  alpha=density_opacity, mincnt=1)
                plt.colorbar(hexbin, ax=ax, label='Count')
            
            # Add scatter plot
            scatter = ax.scatter(x_data, y_data, color=color, 
                               alpha=alpha if not density_plot else alpha*0.7, 
                               s=marker_size*2 if not density_plot else marker_size)
            
            # Add connecting line if requested
            if line:
                ax.plot(x_data, y_data, color=color, alpha=alpha*0.7, linewidth=1)
            
            # Add fit line if requested
            if fit_line:
                ax.plot(fit_x, fit_y, color=fit_color, linewidth=fit_line_width, alpha=fit_alpha, label='Linear Fit')
                ax.legend()
            
            # Add statistics if requested
            if show_stats:
                # Construct statistics text
                stats_text = (
                    f"Statistics:\n"
                    f"{x_ch_name}:\n"
                    f"Mean: {stats_data['x_mean']:.4g}\n"
                    f"Median: {stats_data['x_median']:.4g}\n"
                    f"Min: {stats_data['x_min']:.4g}\n"
                    f"Max: {stats_data['x_max']:.4g}\n"
                    f"Std: {stats_data['x_std']:.4g}\n\n"
                    f"{y_ch_name}:\n"
                    f"Mean: {stats_data['y_mean']:.4g}\n"
                    f"Median: {stats_data['y_median']:.4g}\n"
                    f"Min: {stats_data['y_min']:.4g}\n"
                    f"Max: {stats_data['y_max']:.4g}\n"
                    f"Std: {stats_data['y_std']:.4g}\n\n"
                    f"Correlation: {corr_coef:.4g}\n"
                    f"Count: {len(x_data)}"
                )
                
                if fit_line:
                    stats_text += (
                        f"\n\nLinear Fit:\n"
                        f"y = {slope:.4g}x + {intercept:.4g}\n"
                        f"R²: {r_value**2:.4g}\n"
                        f"p-value: {p_value:.4g}\n"
                        f"Std Error: {std_err:.4g}"
                    )
                
                # Add text to plot
                plt.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=8)
            
            # Set labels and title
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(plot_title)
            
            # Set grid
            ax.grid(grid)
            
            # Set limits if provided
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            
            plt.tight_layout()
            
            # Handle save_path
            if save_path and not save_path.lower().endswith(('.html', '.htm')):
                try:
                    plt.savefig(save_path, dpi=dpi)
                    logger.debug(f"XY plot saved to {save_path}")
                except Exception as e:
                    logger.error(f"Failed to save XY plot to {save_path}: {e}")
            elif save_path:
                logger.error(f"Cannot save as HTML when using matplotlib. Requested path: {save_path}")
            
            # Show plot if requested
            if show:
                logger.debug("Displaying XY plot using Matplotlib")
                plt.show()
            
            # Return figure
            return fig
            
    except Exception as e:
        logger.error(f"Failed to create XY plot: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None