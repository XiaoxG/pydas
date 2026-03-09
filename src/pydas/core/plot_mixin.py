"""PyDAS Core - Plot Mixin

Thin proxy layer that validates channel indices and delegates to the
``pydas.plot`` subpackage.  All plotting keyword arguments are forwarded
via ``**kwargs`` to avoid duplicating long parameter lists.
"""
import logging

from ..plot import (
    plot_channel as _plot_channel,
    plot_histogram as _plot_histogram,
    boxplot_channel as _boxplot_channel,
    plot_xy as _plot_xy,
    validate_channel,
)

logger = logging.getLogger(__name__)


class PlotMixin:
    """Mixin providing plotting capabilities to PyDAS."""

    # -- single-channel plots ------------------------------------------------

    def plot_channel(self, ch_idx, sseg=0, **kwargs):
        """Plot time-series of one or more channels.

        Parameters
        ----------
        ch_idx : int, str, or list
            Channel index, name, or list of channel names to plot.
        sseg : int, optional
            Segment index, default is 0.
        **kwargs
            All additional keyword arguments are forwarded to
            ``pydas.plot.plot_channel`` (e.g. *figsize*, *title*, *xlim*,
            *plotbackend*, *save_path*, *dpi*, etc.).

        Returns
        -------
        Figure or None
        """
        ch_name = validate_channel(self, ch_idx)
        if ch_name is None:
            return None
        return _plot_channel(pydas_obj=self, ch_name=ch_name, sseg=sseg, **kwargs)

    def plot_histogram(self, ch_idx, sseg=0, **kwargs):
        """Plot histogram of one or more channels.

        Parameters
        ----------
        ch_idx : int, str, or list
            Channel index, name, or list of channel indices/names.
        sseg : int, optional
            Segment index, default is 0.
        **kwargs
            Forwarded to ``pydas.plot.plot_histogram`` (e.g. *bins*,
            *fit_gaussian*, *plotbackend*, *save_path*, etc.).

        Returns
        -------
        Figure or None
        """
        ch_name = validate_channel(self, ch_idx)
        if ch_name is None:
            return None
        return _plot_histogram(pydas_obj=self, ch_name=ch_name, sseg=sseg, **kwargs)

    def boxplot_channel(self, ch_idx, sseg=0, **kwargs):
        """Create box-plot of one or more channels.

        Parameters
        ----------
        ch_idx : int, str, or list
            Channel index, name, or list.
        sseg : int, optional
            Segment index, default is 0.
        **kwargs
            Forwarded to ``pydas.plot.boxplot_channel`` (e.g. *notch*,
            *use_peaks*, *plotbackend*, *save_path*, etc.).

        Returns
        -------
        Figure or None
        """
        ch_name = validate_channel(self, ch_idx)
        if ch_name is None:
            return None
        return _boxplot_channel(pydas_obj=self, ch_name=ch_name, sseg=sseg, **kwargs)

    # -- two-channel plots ---------------------------------------------------

    def plot_xy(self, x_ch_idx, y_ch_idx, sseg=0, **kwargs):
        """Create an XY scatter plot of two channels.

        Parameters
        ----------
        x_ch_idx : int or str
            Channel index or name for the X-axis.
        y_ch_idx : int or str
            Channel index or name for the Y-axis.
        sseg : int, optional
            Segment index, default is 0.
        **kwargs
            Forwarded to ``pydas.plot.plot_xy`` (e.g. *fit_line*,
            *density_plot*, *plotbackend*, *save_path*, etc.).

        Returns
        -------
        tuple or None
            ``(DataFrame, Figure)`` on success, ``None`` on failure.
        """
        x_ch_name = validate_channel(self, x_ch_idx)
        y_ch_name = validate_channel(self, y_ch_idx)
        if x_ch_name is None or y_ch_name is None:
            return None
        return _plot_xy(
            pydas_obj=self,
            x_ch_name=x_ch_name,
            y_ch_name=y_ch_name,
            sseg=sseg,
            **kwargs,
        )
