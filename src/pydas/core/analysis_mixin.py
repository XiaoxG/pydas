"""PyDAS Core - Analysis Mixin"""
import logging

logger = logging.getLogger(__name__)

from ..analysis import spectral_analysis, statistic_analysis, extreme_analysis

class AnalysisMixin:
    def spectral_analysis(self, channel_name, method='cov', L=1024, plot=False, title=None, 
                         save_path=None, plotbackend=None, save_html=None,
                         fullscale=False, lam=None, rho=1.025, g=9.807, freq_range=(0, 2)):
        """
        Perform spectral analysis on a single channel and return a spectral data object.
        This method calls the spectral_analysis function from the analysis module.
        
        See analysis.spectral_analysis for full documentation.
        """
        return spectral_analysis(self, channel_name, method, L, plot, title, save_path, 
                               plotbackend, save_html, fullscale, lam, rho, g, freq_range)

    def statistic_analysis(self, ch_name, sseg=0, advanced=False, visualization=False, bins=50, 
                          save_fig=False, save_path=None, plotbackend=None, fullscale=False, lam=None, 
                          rho=1.025, g=9.807):
        """
        对通道进行时域统计分析。此方法调用analysis模块中的statistic_analysis函数。
        
        Parameters:
        -----------
        ch_name : str
            要分析的通道名称
        sseg : int, optional
            要分析的数据段索引，默认为0
        advanced : bool, optional
            是否计算高级统计量（偏度、峰度、分位数等），默认为False
        visualization : bool, optional
            是否显示统计量可视化，默认为False
        bins : int, optional
            直方图的箱数，默认为50
        save_fig : bool, optional
            是否保存图形，默认为False
        save_path : str, optional
            图形保存路径，默认为None（当前目录）
        plotbackend : str, optional
            绘图后端 ('plotly', 'matplotlib', 'seaborn' 或 None 自动选择)，默认为None
        fullscale : bool, optional
            是否转换为原型尺度，默认为False
        lam : float, optional
            尺度系数，仅在fullscale=True时使用，默认为None（使用对象的__lam__属性）
        rho : float, optional
            水密度(kg/m³)，仅在fullscale=True时使用，默认为1.025
        g : float, optional
            重力加速度(m/s²)，仅在fullscale=True时使用，默认为9.807
            
        完整文档请参见analysis.statistic_analysis。
        """
        # 如果fullscale=True但未提供lam参数，使用对象的__lam__属性
        if fullscale and lam is None:
            lam = self.__lam__
            
        return statistic_analysis(self, ch_name, sseg, advanced, visualization, bins, 
                                save_fig, save_path, plotbackend, fullscale, lam, rho, g)

    def extreme_analysis(self, ch_name, sseg=0, visualization=True, bins=50, 
                      peak_prominence=1.0, peak_distance=None, 
                      visualization_backend='matplotlib', save_path=None, save_html=None,
                      fullscale=True, lam=None, return_period_multipliers=[1, 5, 10], 
                      peak_height=None, threshold=None, width=None, wlen=None, rel_height=0.5):
        """
        Perform extreme value analysis on a channel.
        This is a wrapper for the extreme_analysis function in the analysis module.
        
        Parameters:
        -----------
        See documentation for analysis.extreme_analysis for details.
        
        Returns:
        --------
        dict
            Dictionary containing analysis results
        """
        # For full-scale analysis without specified lambda, use object default if available
        if fullscale and lam is None and hasattr(self, '__lam__'):
            lam = self.__lam__
            
        return extreme_analysis(self, ch_name, sseg, visualization, bins, 
                                peak_prominence, peak_distance, 
                                visualization_backend, save_path, save_html,
                                fullscale, lam, return_period_multipliers,
                                peak_height, threshold, width, wlen, rel_height)

