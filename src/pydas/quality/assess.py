"""Grade each channel-segment for how far downstream analysis may go."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .detect import MEDIUM_FRAC, SHORT_FRAC, channel_quant_step

logger = logging.getLogger(__name__)

GRADE_GOOD = "good"
GRADE_REPAIRED = "repaired"
GRADE_LIMITED = "limited"
GRADE_BAD = "bad"

GRADE_NOTE = {
    GRADE_GOOD: "Usable for spectrum, statistics, and extremes.",
    GRADE_REPAIRED: "Short bursts repaired; note the repair in extreme-value reports.",
    GRADE_LIMITED: "Do not use for MPM/EEV; spectrum/statistics only with caution.",
    GRADE_BAD: "Unusable for formal analysis; recapture or drop the channel.",
}


def _longest(events):
    if events is None or events.empty:
        return 0, 0.0
    i = events["n"].idxmax()
    return int(events.loc[i, "n"]), float(events.loc[i, "duration_s"])


def grade_channel(events, n_samples, t_star, repaired, coincident_dc_segment):
    """Return ``(grade, suggested_action)`` for one channel in one segment."""
    t_star = float(t_star) if t_star and np.isfinite(t_star) else 1.0
    n_samples = max(int(n_samples), 1)
    if events is None or events.empty:
        if coincident_dc_segment:
            return GRADE_LIMITED, "do_not_use_for_extremes"
        return GRADE_GOOD, "none"

    frac = float(events["n"].sum()) / float(n_samples)
    has_clip = bool((events["kind"] == "clip").any())
    has_edge = bool(events["at_edge"].any())
    long_mask = events["duration_s"] > (MEDIUM_FRAC * t_star)
    medium_mask = (
        (events["refuse_reason"].isin(["medium_gap", "long_gap", "t_star_unknown_n_cap"]))
        | ((events["duration_s"] > (SHORT_FRAC * t_star)) & (events["duration_s"] <= (MEDIUM_FRAC * t_star)))
    )
    long_frac = float(events.loc[long_mask, "n"].sum()) / float(n_samples) if long_mask.any() else 0.0

    if coincident_dc_segment:
        if long_mask.any() or frac > 0.05:
            return GRADE_BAD, "unusable"
        return GRADE_LIMITED, "do_not_use_for_extremes"

    if has_clip and frac > 0.02:
        return GRADE_BAD, "unusable"
    if long_mask.any() or long_frac > 0.02 or frac > 0.05:
        action = "cut_series" if has_edge else "unusable"
        return GRADE_BAD, action
    if has_clip or medium_mask.any():
        action = "cut_series" if has_edge else "do_not_use_for_extremes"
        return GRADE_LIMITED, action
    if repaired:
        return GRADE_REPAIRED, "none"
    if has_edge:
        return GRADE_LIMITED, "cut_series"
    return GRADE_GOOD, "none"


def assess_segment(pydas_obj, events, sseg=0, preview=None):
    """Build one quality row per channel in ``sseg``.

    Parameters
    ----------
    pydas_obj : PyDAS
    events : pandas.DataFrame
        Event table for this segment (with action columns if repair was planned).
    sseg : int
    preview : RepairPreview, optional
        Used to compute before/after std and max when a repair was previewed
        or applied.
    """
    names = list(pydas_obj.chInfo["Name"].astype(str))
    fs = float(pydas_obj.__fs__)
    coincident_dc = False
    if events is not None and not events.empty:
        coincident_dc = bool(events["coincident_dropout_or_clip"].any())

    rows = []
    for name in names:
        series = np.asarray(pydas_obj.data[sseg][name].values, dtype=float)
        n = len(series)
        ch_ev = (
            events.loc[events["channel"] == name].copy()
            if events is not None and not events.empty
            else events
        )
        t_star = float(ch_ev["t_star"].iloc[0]) if ch_ev is not None and not ch_ev.empty else np.nan
        t_src = ch_ev["t_star_source"].iloc[0] if ch_ev is not None and not ch_ev.empty else ""
        repaired = False
        if ch_ev is not None and not ch_ev.empty:
            repaired = bool((ch_ev["action"] == "repair").any())
        if preview is not None and name in preview.series:
            repaired = repaired or bool(len(preview.applied.loc[preview.applied["channel"] == name]))

        std_before = float(np.nanstd(series))
        max_before = float(np.nanmax(np.abs(series))) if n else np.nan
        std_after = std_before
        max_after = max_before
        if preview is not None and name in preview.series:
            y = preview.series[name]
            std_after = float(np.nanstd(y))
            max_after = float(np.nanmax(np.abs(y))) if len(y) else np.nan

        grade, suggested = grade_channel(
            ch_ev if ch_ev is not None else pd.DataFrame(),
            n,
            t_star if np.isfinite(t_star) else 1.0,
            repaired,
            coincident_dc,
        )
        longest_n, longest_s = _longest(ch_ev)
        n_events = 0 if ch_ev is None or ch_ev.empty else int(len(ch_ev))
        n_samples_bad = 0 if ch_ev is None or ch_ev.empty else int(ch_ev["n"].sum())
        rows.append({
            "channel": name,
            "sseg": int(sseg),
            "grade": grade,
            "suggested_action": suggested,
            "note": GRADE_NOTE[grade],
            "fs": fs,
            "n_samples": n,
            "duration_s": n / fs if fs else np.nan,
            "quant_step": channel_quant_step(pydas_obj, name),
            "t_star": t_star,
            "t_star_source": t_src,
            "n_events": n_events,
            "n_spike_burst": 0 if ch_ev is None or ch_ev.empty else int((ch_ev["kind"] == "spike_burst").sum()),
            "n_dropout": 0 if ch_ev is None or ch_ev.empty else int((ch_ev["kind"] == "dropout").sum()),
            "n_clip": 0 if ch_ev is None or ch_ev.empty else int((ch_ev["kind"] == "clip").sum()),
            "n_samples_flagged": n_samples_bad,
            "flagged_fraction": n_samples_bad / n if n else 0.0,
            "longest_n": longest_n,
            "longest_duration_s": longest_s,
            "at_edge": False if ch_ev is None or ch_ev.empty else bool(ch_ev["at_edge"].any()),
            "coincident": False if ch_ev is None or ch_ev.empty else bool(ch_ev["coincident"].any()),
            "coincident_dropout_or_clip": coincident_dc,
            "repaired": repaired,
            "std_before": std_before,
            "std_after": std_after,
            "max_abs_before": max_before,
            "max_abs_after": max_after,
        })
        logger.info(
            "qc: channel=%s sseg=%s grade=%s events=%s suggested=%s",
            name, sseg, grade, n_events, suggested,
        )
    return pd.DataFrame(rows)
