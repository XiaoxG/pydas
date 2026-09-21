"""PyDAS Core - Processing Mixin

Thin proxy that forwards processing operations to ``pydas.process`` and
channel arithmetic to ``pydas.core.channels``. Public method names stay
unchanged.
"""
import logging

from ..process import (
    apply_lowpass_filter,
    apply_highpass_filter,
    remove_mean,
    add_value,
    multiply_value,
    move_data,
    data_wash,
    detrend,
    add_diff1,
    add_diff2,
    fix_unit,
    to_fullscale,
    move_ccor,
    find_move_ccor,
    updateST,
    cut_series,
    channel2fullscale,
)
from ..utils import get_default_transDict, findtrans
from . import channels as _channels

logger = logging.getLogger(__name__)


class ProcessingMixin:
    """Mixin providing processing and scaling methods on a PyDAS object."""

    def fix_unit(self, chName, newunit, pInfo=False):
        """Fix channel unit.

        Parameters
        ----------
        chName : str
            Channel name.
        newunit : str
            New unit to set.
        pInfo : bool, optional
            Whether to print channel information, default is False.
        """
        return fix_unit(self, chName, newunit, pInfo=pInfo)

    def to_fullscale(self, rho=1.025, g=9.807, pInfo=False):
        """Convert model-scale data to prototype scale.

        Parameters
        ----------
        rho : float, optional
            Water density in kg/m³, default is 1.025.
        g : float, optional
            Gravitational acceleration in m/s², default is 9.807.
        pInfo : bool, optional
            Whether to print information, default is False.

        Notes
        -----
        Uses ``self.__lam__`` as the length scale factor. There is no ``lam``
        argument; set ``obj.__lam__`` before calling this method.
        """
        return to_fullscale(self, rho=rho, g=g, pInfo=pInfo)

    def move_ccor(self, to_move_chName, base_chName, reference_ch, sseg=0):
        """Move channel data using cross-correlation.

        Parameters
        ----------
        to_move_chName : str
            Name of channel to move.
        base_chName : str
            Name of base channel.
        reference_ch : str
            Name of reference channel.
        sseg : int, optional
            Segment index to process, default is 0.
        """
        return move_ccor(self, to_move_chName, base_chName, reference_ch, sseg=sseg)

    def find_move_ccor(self, base_chName, reference_ch, sseg=0):
        """Return the lag (in samples) between two channels.

        Parameters
        ----------
        base_chName : str
            Name of base channel.
        reference_ch : str
            Name of reference channel.
        sseg : int, optional
            Segment index to process, default is 0.

        Returns
        -------
        int
            Number of points to pass to :meth:`move_data`.
        """
        return find_move_ccor(self, base_chName, reference_ch, sseg=sseg)

    def apply_lowpass_filter(
        self, chName, cutoffull=2, replace=True, returnValue=False,
        sseg=0, order=6, plot=False,
    ):
        """Apply a lowpass filter to a channel.

        Parameters
        ----------
        chName : str or list
            Channel name, or list of channel names.
        cutoffull : float, optional
            Full-scale cutoff in rad/s, default is 2. In model scale this is
            converted as ``cutoffull / (2*pi) * sqrt(lam)``. This is **not** Hertz.
        replace : bool, optional
            Whether to replace original data, default is True.
        returnValue : bool, optional
            Whether to return filtered data, default is False.
        sseg : int, optional
            Segment index, default is 0.
        order : int, optional
            Filter order, default is 6.
        plot : bool, optional
            Whether to plot before/after comparison, default is False.
        """
        return apply_lowpass_filter(
            self, chName, cutoffull, replace, returnValue, sseg, order, plot
        )

    def apply_highpass_filter(
        self, chName, cutoffull=2, replace=True, returnValue=False,
        sseg=0, order=6, plot=False,
    ):
        """Apply a highpass filter to a channel.

        Parameters
        ----------
        chName : str or list
            Channel name, or list of channel names.
        cutoffull : float, optional
            Full-scale cutoff in rad/s, default is 2. In model scale this is
            converted as ``cutoffull / (2*pi) * sqrt(lam)``. This is **not** Hertz.
        replace : bool, optional
            Whether to replace original data, default is True.
        returnValue : bool, optional
            Whether to return filtered data, default is False.
        sseg : int, optional
            Segment index, default is 0.
        order : int, optional
            Filter order, default is 6.
        plot : bool, optional
            Whether to plot before/after comparison, default is False.
        """
        return apply_highpass_filter(
            self, chName, cutoffull, replace, returnValue, sseg, order, plot
        )

    def remove_mean(self, chName, sseg=0):
        """Remove the mean value from one or more channels."""
        return remove_mean(self, chName, sseg)

    def detrend(self, chName, kind="linear", sseg=0):
        """Remove a linear trend or a constant. Independent of apply_repair."""
        return detrend(self, chName, kind=kind, sseg=sseg)

    def add_value(self, chName, value2add, sseg=0):
        """Add a constant value to one or more channels."""
        return add_value(self, chName, value2add, sseg)

    def multiply_value(self, chName, value2mul, sseg=0):
        """Multiply one or more channels by a constant value."""
        return multiply_value(self, chName, value2mul, sseg)

    def move_data(self, chName, point_of_move, sseg=0):
        """Move data in a channel by a specified number of points."""
        return move_data(self, chName, point_of_move, sseg)

    def data_wash(self, ChName, method="linear", order=5, threshold=3, sseg=0):
        """Global 3σ wash. Not suitable for irregular-wave crests.

        Prefer :meth:`detect_bad_events` / :meth:`apply_repair` for bursts.
        """
        return data_wash(self, ChName, method, order, threshold, sseg)

    def add_diff1(self, name, sseg=0, filter=False, filter_cutoff=2):
        """Calculate and add the first derivative of a channel.

        ``filter_cutoff`` is ``cutoffull`` (full-scale rad/s), not Hertz.
        """
        return add_diff1(self, name, sseg, filter, filter_cutoff)

    def add_diff2(self, name, sseg=0, filter=False, filter_cutoff=2):
        """Calculate and add the second derivative of a channel.

        ``filter_cutoff`` is ``cutoffull`` (full-scale rad/s), not Hertz.
        """
        return add_diff2(self, name, sseg, filter, filter_cutoff)

    def updateST(self, chName="all", sseg=0, engine="pandas"):
        """Update statistical information for channels.

        Parameters
        ----------
        chName : str, optional
            Channel name to update, ``'all'`` for all channels.
        sseg : int, optional
            Segment index to process, default is 0.
        engine : {'pandas', 'dask'}, optional
            Statistics backend. ``'pandas'`` uses ``DataFrame.agg`` (default).
            ``'dask'`` is optional for very large tables.
        """
        return updateST(self, chName=chName, sseg=sseg, engine=engine)

    def update_statistics(self, chName="all", sseg=0, engine="pandas"):
        """Snake-case alias of :meth:`updateST`."""
        return self.updateST(chName=chName, sseg=sseg, engine=engine)

    def cut_series(self, start, stop, sseg=0):
        """Cut a time series to a specified time range in seconds.

        Parameters
        ----------
        start : float
            Start time in seconds (inclusive).
        stop : float
            End time in seconds (exclusive of the sample at ``stop``).
        sseg : int, optional
            Segment index, default is 0.
        """
        return cut_series(self, start, stop, sseg=sseg)

    def channel2fullscale(self, channel_name, lam, rho=1.025, g=9.807):
        """Convert one channel to prototype scale and return a TimeSeries.

        Parameters
        ----------
        channel_name : str
            Name of the channel to convert.
        lam : float
            Scale factor.
        rho : float, optional
            Water density in kg/m³, default is 1.025.
        g : float, optional
            Gravitational acceleration in m/s², default is 9.807.

        Returns
        -------
        waveModel.TimeSeries or None
        """
        return channel2fullscale(self, channel_name, lam, rho=rho, g=g)

    def channel_calculate(self, ch1, ch2, operation, new_chName, sseg=0):
        """Create a new channel from an arithmetic operation on two channels.

        Parameters
        ----------
        ch1, ch2 : str
            Operand channel names.
        operation : str
            ``'add'``/``'+'``, ``'subtract'``/``'-'``, ``'multiply'``/``'*'``,
            or ``'divide'``/``'/'``.
        new_chName : str
            Name of the result channel.
        sseg : int, list, or ``'all'``, optional
            Segment(s) to process, default is 0.
        """
        return _channels.channel_calculate(self, ch1, ch2, operation, new_chName, sseg)

    def channel_apply_function(self, ch, func, new_chName, unit=None, sseg=0):
        """Apply a function to one channel and store the result as a new channel.

        Parameters
        ----------
        ch : str
            Source channel name.
        func : callable or str
            Vectorized callable or a restricted expression string where ``x``
            is the channel array.
        new_chName : str
            Name of the new channel.
        unit : str, optional
            Unit of the result. Defaults to the source channel unit.
        sseg : int, list, or ``'all'``, optional
            Segment(s) to process, default is 0.
        """
        return _channels.channel_apply_function(self, ch, func, new_chName, unit, sseg)

    def _findtrans(self, unit, transDict):
        """Compatibility wrapper for :func:`pydas.utils.findtrans`.

        The historical camelCase name is kept; prefer :meth:`_find_trans`
        in new internal code.
        """
        return findtrans(unit, transDict)

    def _find_trans(self, unit, trans_dict):
        """Snake-case alias of :meth:`_findtrans`."""
        return self._findtrans(unit, trans_dict)

    def _get_default_transDict(self, g=9.807):
        """Compatibility wrapper for :func:`pydas.utils.get_default_transDict`.

        Prefer :meth:`_get_default_trans_dict` in new internal code.
        """
        return get_default_transDict(g)

    def _get_default_trans_dict(self, g=9.807):
        """Snake-case alias of :meth:`_get_default_transDict`."""
        return self._get_default_transDict(g=g)
