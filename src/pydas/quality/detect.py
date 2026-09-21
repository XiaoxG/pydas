"""Detect bad-data *events* (runs of samples), not isolated indices.

Order of labelling: clip plateaus, then dropouts, then spike bursts.
True irregular-wave crests must not be labelled as spikes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..core.io_format import INT16_SCALE

logger = logging.getLogger(__name__)

SHORT_FRAC = 0.10
MEDIUM_FRAC = 0.30
DEFAULT_T_STAR = 1.0
MAX_N_DEFAULT_TSTAR = 5
MIN_PLATEAU_N = 3
MAD_K = 6.0

EVENT_COLUMNS = [
    "channel",
    "sseg",
    "start_i",
    "stop_i",
    "n",
    "duration_s",
    "kind",
    "at_edge",
    "coincident",
    "coincident_dropout_or_clip",
    "action",
    "refuse_reason",
    "interpolator",
    "t_star",
    "t_star_source",
]


def empty_events():
    """Return an empty event table with a stable schema."""
    return pd.DataFrame(columns=EVENT_COLUMNS)


def estimate_t_star(x, fs, tz=None):
    """Return ``(T*, source)`` used to grade gap length.

    Parameters
    ----------
    x : array-like
        Channel samples.
    fs : float
        Sampling frequency in Hz.
    tz : float, optional
        Caller-supplied characteristic period in seconds.

    Returns
    -------
    t_star : float
        Period in seconds.
    source : {'user', 'zerocross', 'default'}
    """
    if tz is not None and np.isfinite(tz) and tz > 0:
        return float(tz), "user"
    y = np.asarray(x, dtype=float)
    if y.size < 8 or fs <= 0:
        return DEFAULT_T_STAR, "default"
    y = y - np.nanmean(y)
    if not np.isfinite(y).any():
        return DEFAULT_T_STAR, "default"
    up = np.where((y[:-1] <= 0.0) & (y[1:] > 0.0))[0]
    if up.size < 3:
        return DEFAULT_T_STAR, "default"
    duration = (len(y) - 1) / float(fs)
    t_star = duration / float(up.size)
    if not np.isfinite(t_star) or t_star <= 0:
        return DEFAULT_T_STAR, "default"
    return float(t_star), "zerocross"


def channel_quant_step(pydas_obj, name):
    """Amplitude of one int16 count after coefficient scaling, or NaN."""
    info = pydas_obj.chInfo
    if info is None or "Name" not in info.columns:
        return np.nan
    row = info.loc[info["Name"] == name]
    if row.empty or "Coef" not in info.columns:
        return np.nan
    coef = float(row["Coef"].iloc[0])
    return abs(coef) / float(INT16_SCALE)


def _channel_names(pydas_obj, chName):
    names = list(pydas_obj.chInfo["Name"].astype(str))
    if chName == "all":
        return names
    if isinstance(chName, str):
        wanted = [chName]
    else:
        wanted = [str(n) for n in chName]
    missing = [n for n in wanted if n not in names]
    if missing:
        raise KeyError(f"Channel(s) not found: {missing}")
    return wanted


def _runs_from_mask(mask, merge_gap=1):
    """Inclusive-start exclusive-stop runs, merging holes of ``merge_gap``."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return []
    filled = mask.copy()
    if merge_gap > 0:
        for i in range(1, len(mask) - 1):
            if (not mask[i]) and mask[i - 1] and mask[i + 1]:
                filled[i] = True
    delta = np.diff(filled.astype(np.int8), prepend=0, append=0)
    starts = np.where(delta == 1)[0]
    stops = np.where(delta == -1)[0]
    return list(zip(starts.tolist(), stops.tolist()))


def _atol(x, quant_step):
    """Equality tolerance: one int16 count, never coarser than the series range.

    ``from_dataframe`` stores ``Coef=1``, so ``|Coef|/32767`` can be much
    larger than the series amplitude. Using that as a dropout tolerance
    labels slow inflections of a small wave as stuck sensors.
    """
    finite = x[np.isfinite(x)]
    scale = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    q_coef = quant_step if np.isfinite(quant_step) and quant_step > 0 else np.inf
    q_span = (scale / float(INT16_SCALE)) if scale > 0 else np.inf
    q = min(q_coef, q_span)
    if not np.isfinite(q):
        q = 0.0
    return max(q, 1e-12, 1e-9 * max(scale, 1.0))


def _hampel_window(fs, t_star, n):
    max_short_n = max(3, int(round(SHORT_FRAC * t_star * fs)))
    w = 2 * max_short_n + 3
    if w % 2 == 0:
        w += 1
    w = max(11, w)
    cap = max(11, (n // 4) * 2 + 1)
    if cap % 2 == 0:
        cap += 1
    w = min(w, cap)
    if w % 2 == 0:
        w += 1
    return int(w), int(max_short_n)


def _residual_floor(x):
    """Amplitude floor so MAD=0 (noiseless synthetics) can still flag spikes."""
    finite = np.asarray(x, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 1e-12
    dx = np.abs(np.diff(finite))
    dx_med = float(np.nanmedian(dx)) if dx.size else 0.0
    amp = float(np.nanstd(finite))
    return max(3.0 * dx_med, 0.05 * amp, 1e-12)


def _spike_mask(x, fs, t_star, k_mad=MAD_K):
    """Local Hampel residual mask.

    The window is larger than a short burst and smaller than ``T*``.
    Incomplete edge windows stay NaN and are not flagged. When the local
    MAD is zero, ``k_mad`` still multiplies a first-difference floor so a
    noiseless sine is not labelled while a real spike is.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 11:
        return np.zeros(n, dtype=bool)
    w, _max_short_n = _hampel_window(fs, t_star, n)
    s = pd.Series(x)
    med = s.rolling(w, center=True, min_periods=w).median()
    resid = (s - med).abs()
    mad = resid.rolling(w, center=True, min_periods=w).median()
    scale = 1.4826 * mad.to_numpy(dtype=float, na_value=np.nan)
    floor = _residual_floor(x)
    scale = np.where(np.isfinite(scale), scale, np.nan)
    scale = np.where(np.isfinite(scale), np.maximum(scale, floor), np.nan)
    r = resid.to_numpy(dtype=float, na_value=np.nan)
    out = np.zeros(n, dtype=bool)
    ok = np.isfinite(r) & np.isfinite(scale)
    out[ok] = r[ok] > (float(k_mad) * scale[ok])
    return out


def _clip_mask(x, atol):
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros(len(x), dtype=bool)
    span = float(np.nanmax(finite) - np.nanmin(finite))
    if span <= max(2.0 * atol, 1e-15):
        return np.zeros(len(x), dtype=bool)
    xmax = float(np.nanmax(finite))
    xmin = float(np.nanmin(finite))
    at_hi = np.abs(x - xmax) <= atol
    at_lo = np.abs(x - xmin) <= atol
    return at_hi | at_lo


def _dropout_mask(x, atol):
    n = len(x)
    if n < 2:
        return np.zeros(n, dtype=bool)
    stuck = np.zeros(n, dtype=bool)
    dx = np.abs(np.diff(x))
    equal = dx <= atol
    # a sample is stuck if it equals a neighbour
    stuck[:-1] |= equal
    stuck[1:] |= equal
    return stuck


def _events_for_series(name, sseg, x, fs, tz, quant_step, k_mad):
    x = np.asarray(x, dtype=float)
    n = len(x)
    t_star, t_star_source = estimate_t_star(x, fs, tz=tz)
    atol = _atol(x, quant_step)

    clip_m = _clip_mask(x, atol)
    drop_m = _dropout_mask(x, atol)
    # Constant channel: one dropout covering the record, not clip.
    finite = x[np.isfinite(x)]
    span = float(np.nanmax(finite) - np.nanmin(finite)) if finite.size else 0.0
    if finite.size and span <= max(2.0 * atol, 1e-15):
        clip_m[:] = False
        drop_m[:] = True

    drop_min = max(5, int(round(0.06 * fs))) if fs else MIN_PLATEAU_N
    clip_min = max(4, int(round(0.03 * fs))) if fs else MIN_PLATEAU_N

    claimed = np.zeros(n, dtype=bool)
    labelled_runs = []

    for kind, mask, min_n in (
        ("clip", clip_m, clip_min),
        ("dropout", drop_m, drop_min),
    ):
        for start, stop in _runs_from_mask(mask & ~claimed):
            length = stop - start
            if length < min_n:
                continue
            claimed[start:stop] = True
            labelled_runs.append((kind, start, stop, length))

    # Ignore already-claimed plateaus so they do not contaminate the Hampel median.
    work = np.array(x, dtype=float, copy=True)
    work[claimed] = np.nan
    spike_m = _spike_mask(work, fs, t_star, k_mad=k_mad)
    for start, stop in _runs_from_mask(spike_m & ~claimed):
        length = stop - start
        if length < 1:
            continue
        claimed[start:stop] = True
        labelled_runs.append(("spike_burst", start, stop, length))

    rows = []
    for kind, start, stop, length in labelled_runs:
        at_edge = start == 0 or stop == n
        rows.append({
            "channel": name,
            "sseg": int(sseg),
            "start_i": int(start),
            "stop_i": int(stop),
            "n": int(length),
            "duration_s": float(length) / float(fs) if fs else np.nan,
            "kind": kind,
            "at_edge": bool(at_edge),
            "coincident": False,
            "coincident_dropout_or_clip": False,
            "action": "",
            "refuse_reason": "",
            "interpolator": "",
            "t_star": float(t_star),
            "t_star_source": t_star_source,
        })
    return rows, t_star, t_star_source


def mark_coincident(events):
    """Flag overlapping events on different channels in the same segment."""
    events = events.copy()
    if events.empty:
        return events
    events["coincident"] = False
    events["coincident_dropout_or_clip"] = False
    for _, grp in events.groupby("sseg", sort=False):
        recs = grp.to_dict("records")
        indices = list(grp.index)
        for i, a in enumerate(recs):
            for j, b in enumerate(recs):
                if i >= j or a["channel"] == b["channel"]:
                    continue
                overlap = a["start_i"] < b["stop_i"] and b["start_i"] < a["stop_i"]
                if not overlap:
                    continue
                events.loc[indices[i], "coincident"] = True
                events.loc[indices[j], "coincident"] = True
                dc = {"dropout", "clip"}
                if a["kind"] in dc and b["kind"] in dc:
                    events.loc[indices[i], "coincident_dropout_or_clip"] = True
                    events.loc[indices[j], "coincident_dropout_or_clip"] = True
    return events


def detect_bad_events(pydas_obj, chName="all", sseg=0, tz=None, k_mad=MAD_K):
    """Return a table of bad-data events. Does not modify ``data``.

    Parameters
    ----------
    pydas_obj : PyDAS
        Object holding channel series.
    chName : str or list or ``'all'``, optional
        Channel name(s).
    sseg : int, optional
        Segment index, default is 0.
    tz : float, optional
        Characteristic period in seconds used as ``T*``. Estimated from
        zero-upcrossings when omitted.
    k_mad : float, optional
        MAD threshold for residual spike detection, default is 6.

    Returns
    -------
    pandas.DataFrame
        One row per event. See module ``EVENT_COLUMNS``.
    """
    names = _channel_names(pydas_obj, chName)
    fs = float(pydas_obj.__fs__)
    rows = []
    for name in names:
        series = np.asarray(pydas_obj.data[sseg][name].values, dtype=float)
        quant = channel_quant_step(pydas_obj, name)
        ch_rows, t_star, source = _events_for_series(
            name, sseg, series, fs, tz, quant, k_mad
        )
        rows.extend(ch_rows)
        logger.info(
            "detect_bad_events: channel=%s sseg=%s T*=%.4fs (%s) events=%s",
            name, sseg, t_star, source, len(ch_rows),
        )
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS) if rows else empty_events()
    events = mark_coincident(events)
    return events.reset_index(drop=True)
