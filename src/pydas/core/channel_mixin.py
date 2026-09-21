"""PyDAS Core - Channel Mixin

Thin proxy that forwards channel operations to ``pydas.core.channels``.
Public method names and parameter aliases stay unchanged.
"""
import logging

from . import channels as _channels

logger = logging.getLogger(__name__)


class ChannelMixin:
    """Mixin providing channel management on a PyDAS object."""

    def add_channel(self, name, unit, series, fs, coef=1, point_of_move=0, sseg=0):
        """Add a new channel to the data.

        Parameters
        ----------
        name : str
            Name of the new channel.
        unit : str
            Unit of measurement.
        series : numpy.ndarray
            Channel samples.
        fs : float
            Sampling frequency in Hz.
        coef : float, optional
            Coefficient for data scaling, default is 1.
        point_of_move : int, optional
            Number of points to shift the data, default is 0.
        sseg : int, optional
            Segment index to add the channel to, default is 0.
        """
        return _channels.add_channel(
            self, name, unit, series, fs, coef, point_of_move, sseg
        )

    def delete_channel(self, name):
        """Delete a specified channel from every segment.

        Parameters
        ----------
        name : str
            Name of the channel to delete.
        """
        return _channels.delete_channel(self, name)

    def select_channels(self, chnames):
        """Keep only the named channels.

        Parameters
        ----------
        chnames : str or list of str
            Channel name(s) to keep.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        return _channels.select_channels(self, chnames)

    def change_channel_order(self, newOrder, sseg=0):
        """Change channel order in one segment.

        Parameters
        ----------
        newOrder : list of str
            New order of channel names.
        sseg : int, optional
            Segment index to process, default is 0.
        """
        return _channels.change_channel_order(self, newOrder, sseg=sseg)

    def updateChN(self, sseg=0):
        """Refresh ``__chN__`` after a structural channel change.

        Parameters
        ----------
        sseg : int, optional
            Segment index to process, default is 0.

        Notes
        -----
        ``update_channel_count`` is a snake_case alias of this method.
        The historical name is kept so existing notebooks keep working.
        """
        return _channels.update_channel_count(self, sseg=sseg)

    def update_channel_count(self, sseg=0):
        """Snake-case alias of :meth:`updateChN`."""
        return self.updateChN(sseg=sseg)

    def rename_channel(self, chOld, chNew, sseg=0):
        """Rename a channel in one segment and in ``chInfo``.

        Parameters
        ----------
        chOld : str
            Original channel name.
        chNew : str
            New channel name.
        sseg : int, optional
            Segment index to process, default is 0.
        """
        return _channels.rename_channel(self, chOld, chNew, sseg=sseg)

    def copy_channel(self, chName, new_chName=None, sseg="all"):
        """Copy an existing channel onto a new name.

        Parameters
        ----------
        chName : str
            Name of the channel to copy.
        new_chName : str, optional
            Name for the copy. Defaults to ``original + "_copy"``.
        sseg : int, list, or ``'all'``, optional
            Segment(s) to copy, default is ``'all'``.

        Returns
        -------
        bool
            True if the copy succeeded, False otherwise.
        """
        return _channels.copy_channel(self, chName, new_chName, sseg)
