"""Gate downstream analysis on quality grades.

``limited`` / ``bad`` must not feed MPM or EEV.  ``good`` and ``repaired``
may proceed; repaired results should be annotated in the log.
This module does not change the frozen ``.out`` pack.
"""

from __future__ import annotations

import logging

from .assess import GRADE_BAD, GRADE_LIMITED, GRADE_REPAIRED

logger = logging.getLogger(__name__)

_GRADE_RANK = {
    "good": 0,
    "repaired": 1,
    "limited": 2,
    "bad": 3,
}


def grade_allows_extremes(grade):
    """Return True if MPM/EEV is allowed for this grade."""
    return str(grade) not in (GRADE_LIMITED, GRADE_BAD)


def worse_grade(a, b):
    """Return the worse of two grade ids."""
    if a is None or a == "":
        return b
    if b is None or b == "":
        return a
    ra = _GRADE_RANK.get(str(a), 0)
    rb = _GRADE_RANK.get(str(b), 0)
    return a if ra >= rb else b


def _segment_indices(pydas_obj, sseg):
    nseg = int(getattr(pydas_obj, "__segN__", 1) or 1)
    nseg = max(nseg, 1)
    if isinstance(sseg, tuple) and len(sseg) == 2:
        idx = int(sseg[0])
        return [min(max(idx, 0), nseg - 1)]
    if isinstance(sseg, int):
        if 0 <= sseg < nseg and nseg == 1:
            return [sseg]
        if 0 <= sseg < nseg:
            # Historical extreme_analysis concatenates every segment when
            # ``sseg`` is not a (start, stop) tuple. Use the worst grade.
            return list(range(nseg))
        return [0]
    return list(range(nseg))


def qc_rows_for_channel(
    pydas_obj, ch_name, sseg=0, tz=None, qc=None, k_mad=6.0,
):
    """Return qc rows covering the segments used for ``ch_name``.

    Parameters
    ----------
    pydas_obj : PyDAS
    ch_name : str
    sseg : int or tuple or None
        Same meaning as :func:`pydas.analysis.extreme_analysis`.
    tz, k_mad
        Forwarded to :func:`pydas.quality.report.qc_report` when ``qc`` is omitted.
    qc : pandas.DataFrame, optional
        Precomputed ``qc_report`` table.
    """
    from .report import qc_report

    segs = _segment_indices(pydas_obj, sseg)
    if qc is not None:
        table = qc
        if table is not None and not table.empty:
            table = table.loc[table["channel"].astype(str) == str(ch_name)]
            if "sseg" in table.columns:
                table = table.loc[table["sseg"].isin(segs)]
        return table

    frames = []
    for seg in segs:
        frames.append(
            qc_report(
                pydas_obj, sseg=int(seg), chName=ch_name, tz=tz, k_mad=k_mad
            )
        )
    if not frames:
        return None
    import pandas as pd

    return pd.concat(frames, ignore_index=True)


def decide_extreme_gate(
    pydas_obj, ch_name, sseg=0, tz=None, qc=None, k_mad=6.0, respect_quality=True,
):
    """Return ``(allowed, grade, row)`` for MPM/EEV on one channel.

    When ``respect_quality`` is False, always ``(True, None, None)``.
    """
    if not respect_quality:
        return True, None, None
    try:
        rows = qc_rows_for_channel(
            pydas_obj, ch_name, sseg=sseg, tz=tz, qc=qc, k_mad=k_mad
        )
    except Exception as exc:
        logger.warning(
            "quality gate skipped for %s: qc_report failed (%s)", ch_name, exc
        )
        return True, None, None
    if rows is None or rows.empty:
        return True, "good", None
    grade = "good"
    chosen = None
    for rec in rows.itertuples(index=False):
        g = getattr(rec, "grade", "good")
        grade = worse_grade(grade, g)
        if str(g) == str(grade):
            chosen = rec
    allowed = grade_allows_extremes(grade)
    if not allowed:
        logger.warning(
            "QC grade=%s forbids MPM/EEV on channel=%s", grade, ch_name
        )
    elif grade == GRADE_REPAIRED:
        logger.warning(
            "QC grade=repaired for channel=%s; MPM/EEV includes short-gap repairs",
            ch_name,
        )
    return allowed, grade, chosen
