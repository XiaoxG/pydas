"""Apply short-gap repair from an event table. Default policy is short_only."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .detect import (
    DEFAULT_T_STAR,
    EVENT_COLUMNS,
    MAX_N_DEFAULT_TSTAR,
    MEDIUM_FRAC,
    SHORT_FRAC,
    detect_bad_events,
    empty_events,
)

logger = logging.getLogger(__name__)

REPAIR_LOG_COLUMNS = EVENT_COLUMNS + ["policy"]


def empty_repair_log():
    """Empty audit table stored on a PyDAS object."""
    return pd.DataFrame(columns=REPAIR_LOG_COLUMNS)


class RepairPreview:
    """In-memory preview of a repair; does not write the PyDAS object."""

    def __init__(self, events, series, sseg, policy):
        self.events = events
        self.series = series
        self.sseg = int(sseg)
        self.policy = policy

    @property
    def applied(self):
        ev = self.events
        if ev.empty:
            return ev
        return ev.loc[ev["action"] == "repair"].copy()


def decide_action(kind, n, duration_s, at_edge, t_star, t_star_source, policy="short_only"):
    """Return ``(action, refuse_reason, interpolator)`` for one event."""
    if at_edge:
        return "refuse", "edge", ""
    if kind == "clip":
        return "refuse", "clip", ""
    if kind not in ("spike_burst", "dropout"):
        return "refuse", kind, ""

    short_limit = SHORT_FRAC * float(t_star)
    medium_limit = MEDIUM_FRAC * float(t_star)

    if t_star_source == "default" and n > MAX_N_DEFAULT_TSTAR:
        return "refuse", "t_star_unknown_n_cap", ""

    if policy == "short_only" and duration_s > short_limit:
        if duration_s > medium_limit:
            return "refuse", "long_gap", ""
        return "refuse", "medium_gap", ""

    interpolator = "linear" if n <= 3 else "pchip"
    return "repair", "", interpolator


def _annotate_actions(events, policy):
    events = events.copy()
    if events.empty:
        return events
    actions, reasons, interps = [], [], []
    for rec in events.itertuples(index=False):
        action, reason, interp = decide_action(
            rec.kind,
            int(rec.n),
            float(rec.duration_s),
            bool(rec.at_edge),
            float(rec.t_star) if np.isfinite(rec.t_star) else DEFAULT_T_STAR,
            rec.t_star_source,
            policy=policy,
        )
        actions.append(action)
        reasons.append(reason)
        interps.append(interp)
    events["action"] = actions
    events["refuse_reason"] = reasons
    events["interpolator"] = interps
    return events


def _interpolate_segment(x, start, stop, method):
    """Fill ``x[start:stop]`` from surrounding samples. Returns a copy."""
    y = np.array(x, dtype=float, copy=True)
    n = y.size
    if start <= 0 or stop >= n:
        raise ValueError("Cannot interpolate an event that touches the series edge")
    idx = np.arange(n)
    pad = max(8, stop - start + 4)
    lo = max(0, start - pad)
    hi = min(n, stop + pad)
    good = np.ones(hi - lo, dtype=bool)
    good[(start - lo):(stop - lo)] = False
    xi = idx[lo:hi][good]
    yi = y[lo:hi][good]
    finite = np.isfinite(yi)
    xi = xi[finite]
    yi = yi[finite]
    if xi.size < 2:
        raise ValueError("Not enough surrounding samples to interpolate")
    query = idx[start:stop]
    if method == "linear":
        y[start:stop] = np.interp(query, xi, yi)
    elif method == "pchip":
        spl = PchipInterpolator(xi, yi, extrapolate=False)
        filled = spl(query)
        if np.any(~np.isfinite(filled)):
            raise ValueError("PCHIP produced non-finite samples")
        y[start:stop] = filled
    else:
        raise ValueError(f"Unknown interpolator: {method}")
    return y


def preview_repair(
    pydas_obj, chName="all", sseg=0, policy="short_only", events=None, tz=None, k_mad=6.0,
):
    """Return a :class:`RepairPreview` without writing ``data``.

    Default ``policy='short_only'`` repairs short spike/dropout bursts
    only. Clip, edge, and medium/long gaps are reported and left unchanged.
    Interpolator: linear for ``n<=3``, PCHIP for longer short bursts.
    """
    if events is None:
        events = detect_bad_events(
            pydas_obj, chName=chName, sseg=sseg, tz=tz, k_mad=k_mad
        )
    else:
        events = events.copy()
        if chName != "all":
            names = [chName] if isinstance(chName, str) else list(chName)
            events = events.loc[events["channel"].isin(names)]

    events = events.loc[events["sseg"] == sseg].copy() if not events.empty else events
    events = _annotate_actions(events, policy)

    names = (
        sorted(events["channel"].unique())
        if not events.empty
        else (
            list(pydas_obj.chInfo["Name"].astype(str))
            if chName == "all"
            else ([chName] if isinstance(chName, str) else list(chName))
        )
    )
    series = {}
    for name in names:
        y = np.asarray(pydas_obj.data[sseg][name].values, dtype=float).copy()
        ch_ev = events.loc[events["channel"] == name] if not events.empty else empty_events()
        for rec in ch_ev.itertuples(index=False):
            if rec.action != "repair":
                continue
            try:
                y = _interpolate_segment(y, int(rec.start_i), int(rec.stop_i), rec.interpolator)
                logger.info(
                    "preview_repair: %s [%s:%s] n=%s %s",
                    name, rec.start_i, rec.stop_i, rec.n, rec.interpolator,
                )
            except Exception as exc:
                logger.warning(
                    "preview_repair skipped %s [%s:%s]: %s",
                    name, rec.start_i, rec.stop_i, exc,
                )
                events.loc[
                    (events["channel"] == name)
                    & (events["start_i"] == rec.start_i)
                    & (events["stop_i"] == rec.stop_i),
                    ["action", "refuse_reason", "interpolator"],
                ] = ["refuse", "interpolate_failed", ""]
        series[name] = y
    return RepairPreview(events.reset_index(drop=True), series, sseg, policy)


def apply_repair(
    pydas_obj,
    chName="all",
    sseg=0,
    policy="short_only",
    events=None,
    preview=None,
    tz=None,
    k_mad=6.0,
):
    """Write a short-only repair into ``data`` and append ``repair_log``.

    Parameters
    ----------
    pydas_obj : PyDAS
        Target object.
    chName : str or list or ``'all'``, optional
        Channel name(s).
    sseg : int, optional
        Segment index.
    policy : {'short_only'}, optional
        Only ``short_only`` is implemented; medium/long gaps are not filled.
    events : pandas.DataFrame, optional
        Precomputed event table.
    preview : RepairPreview, optional
        If given, its series are written back.
    tz, k_mad
        Forwarded to detection when ``events`` / ``preview`` are omitted.

    Returns
    -------
    pandas.DataFrame
        The event table that was applied (including refused rows).
    """
    if preview is None:
        preview = preview_repair(
            pydas_obj, chName=chName, sseg=sseg, policy=policy, events=events, tz=tz, k_mad=k_mad
        )
    sseg = preview.sseg
    for name, y in preview.series.items():
        pydas_obj.data[sseg][name] = y
        pydas_obj.updateST(chName=name, sseg=sseg)
        n_rep = int((preview.events["channel"] == name).sum()) if not preview.events.empty else 0
        n_ok = int(len(preview.applied.loc[preview.applied["channel"] == name])) if not preview.applied.empty else 0
        logger.info(
            "apply_repair: wrote channel=%s sseg=%s repaired_events=%s / %s",
            name, sseg, n_ok, n_rep,
        )
    log_rows = preview.events.copy()
    if not log_rows.empty:
        log_rows["policy"] = preview.policy
        if not hasattr(pydas_obj, "repair_log") or pydas_obj.repair_log is None:
            pydas_obj.repair_log = empty_repair_log()
        pydas_obj.repair_log = pd.concat(
            [pydas_obj.repair_log, log_rows[REPAIR_LOG_COLUMNS]],
            ignore_index=True,
        )
    return preview.events
