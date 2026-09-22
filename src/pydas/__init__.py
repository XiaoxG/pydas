"""
PyDAS - Python Data Analysis System
====================================
Ocean-engineering time-series toolkit: binary ``.out`` I/O, channel
management, filtering, Froude scaling, spectral/extreme analysis, and
Excel reporting.

Public exports are ``PyDAS``, ``diff1d``, and ``data_change_fs``.  All
other operations live on a ``PyDAS`` instance.

Usage::

    from pydas import PyDAS
    data = PyDAS(filename="data.out", lam=36)
    data.print_statistics()

See the repository README and ``docs/user-guide.md`` for a full tutorial.
"""

from .core.pydas_obj import PyDAS
from .utils import diff1d, data_change_fs

__version__ = "1.4.1"
__all__ = ["PyDAS", "diff1d", "data_change_fs"]
