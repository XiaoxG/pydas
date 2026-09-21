"""PyDAS Core - Main Object"""
import logging
import os

import numpy as np
import pandas as pd

from .analysis_mixin import AnalysisMixin
from .channel_mixin import ChannelMixin
from .io_mixin import IOMixin
from .plot_mixin import PlotMixin
from .processing_mixin import ProcessingMixin
from .quality_mixin import QualityMixin
from .report_mixin import ReportMixin
from .state import empty_seg_statis
from ..logger import setup_logger
from ..quality.repair import empty_repair_log

logger = logging.getLogger(__name__)


class PyDAS(
    IOMixin, ChannelMixin, ProcessingMixin, QualityMixin,
    PlotMixin, AnalysisMixin, ReportMixin,
):
    """Python Data Analysis System for processing and analyzing time series data."""

    def __init__(self, filename=None, lam=1, sseg="all", log_level="info"):
        """Initialize a PyDAS object and optionally read a ``.out`` data file.

        Parameters
        ----------
        filename : str or None, optional
            Path to a binary ``.out`` file. ``None`` creates an empty object
            that can be filled with :meth:`add_channel`.
        lam : float, optional
            Scale factor for model-to-prototype conversion, default is 1.
        sseg : int or str, optional
            Segment index to read, or ``'all'`` for all segments.
        log_level : str, optional
            Logging level (``'debug'``, ``'info'``, ``'warning'``, ``'error'``,
            ``'critical'``), default is ``'info'``.
        """
        setup_logger(log_level)

        self.__lam__ = lam
        self.__fs__ = 1
        self.__chN__ = 0
        self.__segN__ = 0
        self.__scale__ = "model"
        self.__date__ = ""
        self.__desc__ = ""
        self.repair_log = empty_repair_log()

        if not (isinstance(sseg, int) or sseg == "all"):
            logger.error("Input 'sseg' is illegal (should be int or 'all').")
            raise ValueError("sseg must be an integer or 'all'")

        if filename is not None:
            if os.path.exists(filename):
                self.__filename__ = filename
            else:
                raise FileNotFoundError(f"File {filename} does not exist.")
            self.__read__(sseg)
        else:
            self.__filename__ = None
            self.__segN__ = 1
            self.chInfo = pd.DataFrame(columns=["Name", "Unit", "Coef"])
            self.data = [pd.DataFrame()]
            self.segStatis = [empty_seg_statis()]
            self.segInfo = pd.DataFrame(
                [{
                    "Type": 0,
                    "Start": "00:00:00.0",
                    "Stop": "00:00:00.0",
                    "Duration": "     0.0s",
                    "N sample": 0,
                    "Note": "",
                }],
                index=["Seg 0"],
                columns=["Type", "Start", "Stop", "Duration", "N sample", "Note"],
            )
            logger.info(
                "Created empty PyDAS object. Use add_channel() to set data "
                "or pass a .out filename."
            )

    @classmethod
    def from_dataframe(cls, df, fs, lam=1, units=None, desc="", date="01-01"):
        """Build a PyDAS object from a DataFrame of channel columns.

        Parameters
        ----------
        df : pandas.DataFrame
            Each column becomes a channel. All columns must be numeric.
        fs : float
            Sampling frequency in Hz.
        lam : float, optional
            Scale factor, default is 1.
        units : dict, optional
            Mapping of column name to unit string. Missing names use ``'-'``.
        desc : str, optional
            File description stored on the object.
        date : str, optional
            ``MM-DD`` date stamp used when writing ``.out`` files.

        Returns
        -------
        PyDAS
        """
        if df is None or getattr(df, "empty", True):
            raise ValueError("DataFrame is empty")
        obj = cls(filename=None, lam=lam)
        obj.__fs__ = float(fs)
        obj.__desc__ = desc or ""
        obj.__date__ = date or "01-01"
        units = units or {}
        for name in df.columns:
            series = np.asarray(df[name].values, dtype=np.float64)
            obj.add_channel(str(name), units.get(name, "-"), series, obj.__fs__)
        return obj

    @classmethod
    def read_csv(cls, filename, fs, lam=1, units=None, **kwargs):
        """Read a CSV/TSV file into a PyDAS object via :meth:`from_dataframe`.

        Parameters
        ----------
        filename : str
            Path to a delimited text file.
        fs : float
            Sampling frequency in Hz.
        lam : float, optional
            Scale factor, default is 1.
        units : dict, optional
            Mapping of column name to unit string.
        **kwargs
            Forwarded to ``pandas.read_csv``.

        Returns
        -------
        PyDAS
        """
        df = pd.read_csv(filename, **kwargs)
        obj = cls.from_dataframe(df, fs=fs, lam=lam, units=units)
        obj.__filename__ = filename
        return obj
