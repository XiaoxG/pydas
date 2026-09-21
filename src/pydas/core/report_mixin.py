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
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing general information about the data.
        """
        info = pd.DataFrame(columns=['Value'])
        info.loc['Filename'] = self.__filename__
        info.loc['Date'] = getattr(self, '__date__', '')
        info.loc['Scale'] = self.__scale__
        info.loc['Lambda'] = self.__lam__
        info.loc['Sampling frequency'] = '{0:5.2f} Hz'.format(self.__fs__)
        info.loc['Number of channels'] = self.__chN__
        info.loc['Number of segments'] = self.__segN__

        logger.info('\nGeneral Information:')
        logger.info(info.to_string())

        base = self.__filename__ or 'pydas'
        if printTxt:
            txt_filename = os.path.splitext(base)[0] + '_info.txt'
            with open(txt_filename, 'w') as f:
                f.write('General Information:\n')
                f.write(info.to_string())
                f.write('\n\nSegment Information:\n')
                f.write(self.segInfo.to_string())
                f.write('\n\nChannel Information:\n')
                f.write(self.chInfo.to_string())
            logger.info(f"Information exported to: {txt_filename}")

        if printExcel:
            excel_filename = os.path.splitext(base)[0] + '_info.xlsx'
            with pd.ExcelWriter(excel_filename) as writer:
                info.to_excel(writer, sheet_name='General Info')
                self.segInfo.to_excel(writer, sheet_name='Segment Info')
                self.chInfo.to_excel(writer, sheet_name='Channel Info')
            logger.info(f"Information exported to: {excel_filename}")

        return info

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
            txt_filename = os.path.splitext(self.__filename__ or 'pydas')[0] + '_channel_info.txt'
            with open(txt_filename, 'w') as f:
                f.write('Channel Information:\n')
                f.write(self.chInfo.to_string())
            logger.info(f"Channel information exported to: {txt_filename}")
            
        # Export to Excel file if requested
        if printExcel:
            excel_filename = os.path.splitext(self.__filename__ or 'pydas')[0] + '_channel_info.xlsx'
            self.chInfo.to_excel(excel_filename)
            logger.info(f"Channel information exported to: {excel_filename}")
            
        return None

    def channel_report(self, output_file='channel_report.xlsx', sseg=0, fullscale=True,
                      lam=None, rho=1.025, g=9.807, header_text=None, include_charts=True,
                      significant_percentile=33.0, wave_analysis=True, format_sheet=True,
                      zerocrossing_analysis=True, amplitude_analysis=True,
                      cutoffperiod=15.0, peak_distance=10, pot_threshold_factor=1.5,
                      mpm_method='POT', frequency_separation=False,
                      wave_type='irregular', metrics=None):
        """
        Generate a detailed Excel analysis report for all channels in this PyDAS object.

        The report content is layered:

        * ``wave_type`` selects a high-level analysis preset
          (``'irregular'`` for the full ocean-engineering report,
          ``'regular'`` for a lean basic + zero-crossing + STD-amplitude report
          where MPM/EEV are intentionally excluded).
        * ``metrics`` lets you customise the exact list / order of statistical
          columns regardless of ``wave_type``.

        Parameters
        ----------
        output_file : str, default='channel_report.xlsx'
            Path to the output Excel file.
        sseg : int, default=0
            Index of the data segment to analyse.
        fullscale : bool, default=True
            Whether to convert data to full (prototype) scale before analysis.
        lam : float, optional
            Scale factor. Used only when ``fullscale=True`` and this object has no
            ``__lam__`` attribute.
        rho : float, default=1.025
            Water density in kg/m^3. Used only when ``fullscale=True``.
        g : float, default=9.807
            Gravitational acceleration in m/s^2. Used only when ``fullscale=True``.
        header_text : str, optional
            Title text for the report. Auto-generated from the filename if *None*.
        include_charts : bool, default=True
            Whether to embed charts in the Excel report.
        significant_percentile : float, default=33.0
            Percentile used to compute significant values (e.g. 33 -> top 1/3).
        wave_analysis : bool, default=True
            Whether to perform wave-by-wave analysis.
        format_sheet : bool, default=True
            Whether to apply Excel formatting.
        zerocrossing_analysis : bool, default=True
            Whether to perform zero-crossing analysis.
        amplitude_analysis : bool, default=True
            Whether to perform amplitude analysis.
        cutoffperiod : float, default=15.0
            Cut-off period in seconds for separating low- and high-frequency components.
        peak_distance : int, default=10
            Minimum sample distance between peaks used in Weibull peak detection.
        pot_threshold_factor : float, default=1.5
            Threshold coefficient for the POT method.
        mpm_method : {'POT', 'STD'}, default='POT'
            MPM/EEV calculation method (only used when ``wave_type='irregular'``
            or when MPM-related metrics are explicitly requested).
        frequency_separation : bool, default=False
            If *True*, additionally analyse low-frequency (T > cutoffperiod) and
            high-frequency (T < cutoffperiod) components in separate sheets.
        wave_type : {'irregular', 'regular'}, default='irregular'
            High-level analysis preset.

            - ``'irregular'``: full report with peak-based amplitudes and
              MPM/EEV extreme-value estimates (default).
            - ``'regular'``: basic statistics + zero-crossing + amplitudes
              derived from ``sqrt(2)*STD`` (single) and ``2*sqrt(2)*STD``
              (double). MPM/EEV are excluded by default and the heavy
              extreme-value pipeline is skipped for performance.
        metrics : list of str, optional
            Explicit list of metric IDs that defines the exact set / order of
            report columns. When *None*, the default set of ``wave_type`` is
            used. See ``pydas.reporting.METRIC_CATALOG`` for valid IDs.

        Returns
        -------
        pandas.DataFrame or tuple of pandas.DataFrame
            Statistical results. See :func:`~pydas.reporting.channel_report` for details.

        Examples
        --------
        Default irregular wave report (full content):

        >>> obj.channel_report('irregular.xlsx')

        Regular wave report (basic stats + zero-crossing + STD amplitudes):

        >>> obj.channel_report('regular.xlsx', wave_type='regular')

        Custom irregular report with only the columns you care about:

        >>> obj.channel_report(
        ...     'custom.xlsx',
        ...     wave_type='irregular',
        ...     metrics=['maximum', 'minimum', 'mean', 'STD',
        ...              'mpm_pos', 'mpm_neg', 'mean_zerocross_period'],
        ... )

        Notes
        -----
        - Default ``wave_type='irregular'`` reproduces the original 19-column
          report.
        - Output format is compatible with standard ocean engineering data
          processing tools.
        """
        # Use the object's default scale factor if lam is not provided but __lam__ exists
        if fullscale and lam is None and hasattr(self, '__lam__'):
            lam = self.__lam__
            logger.info(f"Using object's default scale factor: \u03bb = {lam}")

        # Delegate to the standalone channel_report function in the reporting module
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
            mpm_method=mpm_method,
            frequency_separation=frequency_separation,
            wave_type=wave_type,
            metrics=metrics,
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
        Generate a wave analysis report including time series, spectral analysis, and peak
        statistics.

        Parameters
        ----------
        ch_name : str
            Name of the channel to analyse.
        sseg : int, optional
            Segment index, default is 0.
        save_path : str, optional
            Path to save the figure. If *None*, the figure is not saved.
        title : str, optional
            Figure title. If *None*, no title is added.
        Hs : float, optional
            Significant wave height for a JONSWAP reference spectrum.
        Tp : float, optional
            Peak period for the JONSWAP reference spectrum.
        gamma : float, optional
            Peak enhancement factor for the JONSWAP spectrum. Defaults to 3.3.
        bins : int, optional
            Number of histogram bins, default is 50.
        fullscale : bool, optional
            Whether to convert to full-scale data, default is *True*.
        lam : float, optional
            Scale factor. Inferred from ``self.__lam__`` when *None*.
        rho : float, optional
            Water density in kg/m³, default is 1.025.
        g : float, optional
            Gravitational acceleration in m/s², default is 9.807.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure object.
        """
        from ..reporting import wave_report
        return wave_report(self, ch_name, sseg, save_path, title, 
                         L, Hs, Tp, gamma, bins, fullscale, lam, rho, g)
