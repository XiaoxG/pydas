"""PyDAS Core - Main Object"""
import logging
import os

import pandas as pd

from .analysis_mixin import AnalysisMixin
from .channel_mixin import ChannelMixin
from .io_mixin import IOMixin
from .plot_mixin import PlotMixin
from .processing_mixin import ProcessingMixin
from .report_mixin import ReportMixin
from ..logger import setup_logger

logger = logging.getLogger(__name__)

STATS_COLUMNS = ["Mean", "STD", "Max", "Min", "Unit"]


class PyDAS(IOMixin, ChannelMixin, ProcessingMixin, PlotMixin, AnalysisMixin, ReportMixin):
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
            self.segStatis = [pd.DataFrame(columns=STATS_COLUMNS)]
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
