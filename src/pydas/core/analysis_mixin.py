"""PyDAS Core - Analysis Mixin"""
import logging

from ..analysis import extreme_analysis, spectral_analysis, statistic_analysis

logger = logging.getLogger(__name__)


class AnalysisMixin:
    def spectral_analysis(self, channel_name, method='cov', L=1024, plot=False, title=None,
                         save_path=None, plotbackend=None, save_html=None,
                         fullscale=False, lam=None, rho=1.025, g=9.807, freq_range=(0, 2)):
        """Perform spectral analysis on a single channel.

        This method calls :func:`pydas.analysis.spectral_analysis`.
        ``method='cov'`` uses the autocovariance path; ``method='psd'`` uses Welch.
        """
        return spectral_analysis(self, channel_name, method, L, plot, title, save_path,
                               plotbackend, save_html, fullscale, lam, rho, g, freq_range)

    def statistic_analysis(self, ch_name, sseg=0, advanced=False, visualization=False, bins=50,
                          save_fig=False, save_path=None, plotbackend=None, fullscale=False, lam=None,
                          rho=1.025, g=9.807):
        """Perform time-domain statistical analysis on a channel.

        Parameters
        ----------
        ch_name : str
            Channel name to analyse.
        sseg : int, optional
            Segment index, default is 0.
        advanced : bool, optional
            Include skewness, kurtosis, and related higher-order stats.
        visualization : bool, optional
            Show statistical plots.
        bins : int, optional
            Histogram bin count.
        save_fig : bool, optional
            Save figures when visualization is enabled.
        save_path : str, optional
            Directory for saved figures.
        plotbackend : str, optional
            ``'plotly'``, ``'matplotlib'``, ``'seaborn'``, or *None* (auto).
        fullscale : bool, optional
            Convert to prototype scale before analysis.
        lam : float, optional
            Scale factor when ``fullscale=True``. Defaults to ``self.__lam__``.
        rho : float, optional
            Water density [kg/m³].
        g : float, optional
            Gravitational acceleration [m/s²].

        See also
        --------
        pydas.analysis.statistic_analysis
        """
        if fullscale and lam is None:
            lam = self.__lam__

        return statistic_analysis(self, ch_name, sseg, advanced, visualization, bins,
                                save_fig, save_path, plotbackend, fullscale, lam, rho, g)

    def extreme_analysis(self, ch_name, sseg=0, visualization=True, bins=50,
                      peak_prominence=1.0, peak_distance=None,
                      visualization_backend='matplotlib', save_path=None, save_html=None,
                      fullscale=True, lam=None, return_period_multipliers=[1, 5, 10],
                      peak_height=None, threshold=None, width=None, wlen=None, rel_height=0.5):
        """Perform extreme value analysis on a channel.

        Wrapper for :func:`pydas.analysis.extreme_analysis`.
        """
        if fullscale and lam is None and hasattr(self, '__lam__'):
            lam = self.__lam__

        return extreme_analysis(self, ch_name, sseg, visualization, bins,
                                peak_prominence, peak_distance,
                                visualization_backend, save_path, save_html,
                                fullscale, lam, return_period_multipliers,
                                peak_height, threshold, width, wlen, rel_height)
