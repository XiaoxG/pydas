"""PyDAS Core - Report Mixin"""
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from ..reporting import channel_report as _channel_report

class ReportMixin:
    def print_info(self, printTxt=False, printExcel=False):
        """
        Print and optionally export general information about the data.
        
        Parameters:
        -----------
        printTxt : bool, optional
            If True, export information to a text file, default is False
        printExcel : bool, optional
            If True, export information to an Excel file, default is False
            
        Returns:
        --------
        DataFrame
            DataFrame containing general information about the data
        """
        # Create information DataFrame
        info = pd.DataFrame(columns=['Value'])
        info.loc['Filename'] = self.__filename__
        info.loc['Date'] = self.__date__
        info.loc['Scale'] = self.__scale__
        info.loc['Lambda'] = self.__lam__
        info.loc['Sampling frequency'] = '{0:5.2f} Hz'.format(self.__fs__)
        info.loc['Number of channels'] = self.__chN__
        info.loc['Number of segments'] = self.__segN__
        
        # Print to console
        logger.info('\nGeneral Information:')
        logger.info(info.to_string())
        
        # Export to text file if requested
        if printTxt:
            txt_filename = os.path.splitext(self.__filename__)[0] + '_info.txt'
            with open(txt_filename, 'w') as f:
                f.write('General Information:\n')
                f.write(info.to_string())
                f.write('\n\nSegment Information:\n')
                f.write(self.segInfo.to_string())
                f.write('\n\nChannel Information:\n')
                f.write(self.chInfo.to_string())
            logger.info(f"Information exported to: {txt_filename}")
            
        # Export to Excel file if requested
        if printExcel:
            excel_filename = os.path.splitext(self.__filename__)[0] + '_info.xlsx'
            with pd.ExcelWriter(excel_filename) as writer:
                info.to_excel(writer, sheet_name='General Info')
                self.segInfo.to_excel(writer, sheet_name='Segment Info')
                self.chInfo.to_excel(writer, sheet_name='Channel Info')
            logger.info(f"Information exported to: {excel_filename}")
            
        return None

    def print_channel_info(self, printTxt=False, printExcel=False):
        """
        Print and optionally export channel information.
        
        Parameters:
        -----------
        printTxt : bool, optional
            If True, export information to a text file, default is False
        printExcel : bool, optional
            If True, export information to an Excel file, default is False
            
        Returns:
        --------
        DataFrame
            DataFrame containing channel information
        """
        # Print to console
        logger.info('Channel Information:')
        logger.info('\n' + self.chInfo.to_string())
        
        # Export to text file if requested
        if printTxt:
            txt_filename = os.path.splitext(self.__filename__)[0] + '_channel_info.txt'
            with open(txt_filename, 'w') as f:
                f.write('Channel Information:\n')
                f.write(self.chInfo.to_string())
            logger.info(f"Channel information exported to: {txt_filename}")
            
        # Export to Excel file if requested
        if printExcel:
            excel_filename = os.path.splitext(self.__filename__)[0] + '_channel_info.xlsx'
            self.chInfo.to_excel(excel_filename)
            logger.info(f"Channel information exported to: {excel_filename}")
            
        return None

    def channel_report(self, output_file='channel_report.xlsx', sseg=0, fullscale=True, 
                      lam=None, rho=1.025, g=9.807, header_text=None, include_charts=True, 
                      significant_percentile=33.0, wave_analysis=True, format_sheet=True, 
                      zerocrossing_analysis=True, amplitude_analysis=True, 
                      cutoffperiod=15.0, peak_distance=10, pot_threshold_factor=1.5,mpm_method='POT'):
        """
        为PyDAS对象的所有通道生成详细的Excel分析报告
        
        Parameters
        ----------
        output_file : str, default='channel_report.xlsx'
            输出Excel文件的路径
        sseg : int, default=0
            要分析的数据段索引
        fullscale : bool, default=True
            是否使用实际尺度（原型尺度）值
        lam : float, optional
            尺度因子，仅在fullscale=True且PyDAS对象未设置__lam__属性时使用
        rho : float, default=1.025
            水密度 (kg/m³)，仅用于fullscale=True时
        g : float, default=9.807
            重力加速度 (m/s²)，仅用于fullscale=True时
        header_text : str, optional
            报告中的标题文本
        include_charts : bool, default=True
            是否在报告中包含图表
        significant_percentile : float, default=33.0
            计算显著值的百分位数
        wave_analysis : bool, default=True
            是否进行波浪分析
        format_sheet : bool, default=True
            是否设置Excel格式
        zerocrossing_analysis : bool, default=True
            是否进行过零分析
        amplitude_analysis : bool, default=True
            是否进行振幅分析
        cutoffperiod : float, default=15.0
            高低频分离的截止周期（秒），用于分离高频和低频成分
        peak_distance : int, default=130
            峰值检测的最小距离参数，用于 Weibull 分析中的峰值检测
            
        Returns
        -------
        tuple of pandas.DataFrame
            包含三个 DataFrame 的元组：(总统计, 低频统计, 高频统计)
            
        Notes
        -----
        - 生成一个包含所有通道统计分析的Excel报告
        - 报告包括基本统计值、过零分析、振幅分析和极值估计
        - 默认使用实际尺度值（原型尺度）
        - 报告格式类似于标准海洋工程数据处理软件的输出
        """
        # 如果未提供lam但存在__lam__属性，使用对象的默认值
        if fullscale and lam is None and hasattr(self, '__lam__'):
            lam = self.__lam__
            logger.info(f"Using object's default scale factor: λ = {lam}")
        
        # 调用reporting模块中的channel_report函数
        return _channel_report(
            pydas_obj=self,
            output_file=output_file,
            sseg=sseg,
            fullscale=fullscale,
            lam=lam,
            rho=rho,
            g=g,
            header_text=header_text,
            include_charts=include_charts,
            significant_percentile=significant_percentile,
            wave_analysis=wave_analysis,
            format_sheet=format_sheet,
            zerocrossing_analysis=zerocrossing_analysis,
            amplitude_analysis=amplitude_analysis,
            cutoffperiod=cutoffperiod,
            peak_distance=peak_distance,
            pot_threshold_factor=pot_threshold_factor,
            mpm_method=mpm_method
        )

    def print_statistics(self, printTxt=False, printExcel=False):
        """
        Print and optionally export statistical information for all channels.
        
        Parameters:
        -----------
        printTxt : bool, optional
            If True, export statistics to a text file, default is False
        printExcel : bool, optional
            If True, export statistics to an Excel file, default is False
            
        Returns:
        --------
        None
            Statistics are printed to the console and optionally exported to files
        """
        # Update statistics for all segments
        self.updateST(sseg=0)
        
        # Print separator line and segment count
        logger.info(f'Segment total: {self.__segN__:02d}')
        
        # Print statistics for each segment
        for idx, segment_stats in enumerate(self.segStatis):
            logger.info(f'Seg{idx:02d}')
            logger.info('\n' + segment_stats.to_string(float_format=lambda x: f"% .3E" % x, justify='center'))
        
        
        # Export to files if requested
        if printTxt or printExcel:
            # Prepare file path
            path = os.getcwd()
            base_filename = os.path.splitext(self.__filename__)[0]
            
            # Export to text file
            if printTxt:
                txt_filename = f"{path}/{base_filename}_statistic.txt"
                
                # Write to file
                with open(txt_filename, 'w') as infoFile:
                    infoFile.write(f'Segment total: {self.__segN__:02d}\n')
                    
                    # Write statistics for each segment
                    for idx, segment_stats in enumerate(self.segStatis):
                        infoFile.write('\n')
                        infoFile.write(f'Seg{idx:02d}\n')
                        infoFile.write(segment_stats.to_string(
                            float_format=lambda x: f"% .3E" % x, justify='center'))
                
                logger.info(f"Statistics exported to: {txt_filename}")
            
            # Export to Excel file
            if printExcel:
                excel_filename = f"{path}/{base_filename}_statistic.xlsx"
                
                # Write each segment to a separate sheet
                with pd.ExcelWriter(excel_filename) as writer:
                    for idx, segment_stats in enumerate(self.segStatis):
                        segment_stats.to_excel(writer, sheet_name=f'SEG{idx:02d}')
                
                logger.info(f"Statistics exported to: {excel_filename}")

        return None

    def wave_report(self, ch_name, sseg=0, save_path=None, title=None, L=1024,
                  Hs=None, Tp=None, gamma=None, bins=50, fullscale=True, lam=None, 
                  rho=1.025, g=9.807):
        """
        生成波浪分析报告，包括时间序列、谱分析和峰值统计
        
        Parameters
        ----------
        ch_name : str
            要分析的通道名称
        sseg : int, optional
            数据段索引，默认为0
        save_path : str, optional
            保存图片的路径，默认为None
        title : str, optional
            图表标题，默认为None
        Hs : float, optional
            JONSWAP谱的有效波高，默认为None
        Tp : float, optional
            JONSWAP谱的峰值周期，默认为None
        gamma : float, optional
            JONSWAP谱的峰值增强因子，默认为None
        bins : int, optional
            直方图的bin数量，默认为50
        fullscale : bool, optional
            是否使用实际尺度数据，默认为False
        lam : float, optional
            尺度因子，默认为None
        rho : float, optional
            水密度 (kg/m3)，默认为1.025
        g : float, optional
            重力加速度 (m/s2)，默认为9.807
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            生成的图表对象
        """
        from ..reporting import wave_report
        return wave_report(self, ch_name, sseg, save_path, title, 
                         L, Hs, Tp, gamma, bins, fullscale, lam, rho, g)
