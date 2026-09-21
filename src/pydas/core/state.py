"""Shared PyDAS object-state helpers.

Keep segment indexing, channel lookup, and statistics columns in one place
so mixins and I/O modules do not each reimplement the same bookkeeping.
"""
import logging

import numpy as np
import pandas as pd

from ..utils import findtrans, get_default_transDict

logger = logging.getLogger(__name__)

STATS_COLUMNS = ("Mean", "STD", "Max", "Min", "Unit")


def empty_seg_statis():
    """Return an empty statistics table with the canonical column order."""
    return pd.DataFrame(columns=list(STATS_COLUMNS))


def normalize_sseg(obj, sseg, on_invalid="reject"):
    """Normalise a segment selector to a list of integer indices.

    Parameters
    ----------
    obj : PyDAS
        Object that exposes ``__segN__``.
    sseg : int, list, tuple, or ``'all'``
        Segment selector.
    on_invalid : {'reject', 'all'}, optional
        When the selector is unusable, return an empty list (``'reject'``)
        or every segment (``'all'``, matching ``write_data``).

    Returns
    -------
    list of int
    """
    nseg = int(getattr(obj, "__segN__", 0) or 0)
    if sseg == "all":
        return list(range(nseg))
    if isinstance(sseg, int):
        if 0 <= sseg < nseg:
            return [sseg]
        logger.warning(
            "Segment %s exceeds the maximum segment number (%s).",
            sseg, max(nseg - 1, 0),
        )
        return list(range(nseg)) if on_invalid == "all" else []
    if isinstance(sseg, (list, tuple)):
        valid = [s for s in sseg if isinstance(s, int) and 0 <= s < nseg]
        if len(valid) != len(sseg):
            logger.warning("Some segment indices were invalid and will be skipped.")
        return valid
    logger.warning("Invalid segment selection. Use an integer, list, or 'all'.")
    if on_invalid == "all":
        logger.warning("Unsupported segment number, using 'all'.")
        return list(range(nseg))
    return []


def require_channel(obj, name, sseg=None):
    """Return True if *name* exists on the object (and optionally a segment)."""
    if name not in obj.chInfo["Name"].values:
        logger.error("Channel '%s' does not exist.", name)
        return False
    if sseg is not None and name not in obj.data[sseg].columns:
        logger.error("Channel '%s' not found in segment %s", name, sseg)
        return False
    return True


def write_channel_stats(obj, name, sseg):
    """Write Mean/STD/Max/Min/Unit for one channel into ``segStatis[sseg]``."""
    series = obj.data[sseg][name]
    unit = obj.chInfo.loc[obj.chInfo["Name"] == name, "Unit"].values[0]
    obj.segStatis[sseg].loc[name] = [
        np.mean(series),
        np.std(series),
        np.amax(series),
        np.amin(series),
        unit,
    ]


def froude_scale_factors(unit, lam, rho=1.025, g=9.807):
    """Return Froude scaling components for one unit.

    Returns
    -------
    tuple
        ``(new_unit, coeff, coeff_unit, coeff_rho, coeff_lam)`` where
        ``coeff = coeff_unit * rho**coeff_rho * lam**coeff_lam``.
    """
    trans_dict = get_default_transDict(g)
    trans = findtrans(unit, trans_dict)
    new_unit = trans[0] if trans and trans[0] is not None else unit
    try:
        coeff_unit = float(trans[1][0])
        coeff_rho = float(trans[1][1])
        coeff_lam = float(trans[1][2])
        coeff = coeff_unit * float(rho ** coeff_rho) * float(lam ** coeff_lam)
    except Exception:
        logger.warning("Could not compute scale factor for unit '%s'; using 1.0.", unit)
        coeff_unit, coeff_rho, coeff_lam, coeff = 1.0, 0.0, 0.0, 1.0
    return new_unit, coeff, coeff_unit, coeff_rho, coeff_lam
