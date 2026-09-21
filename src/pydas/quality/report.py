"""Build a per-channel quality table. Independent of channel_report Excel."""

from __future__ import annotations

import logging

from .assess import assess_segment
from .detect import detect_bad_events
from .repair import _annotate_actions

logger = logging.getLogger(__name__)


def qc_report(
    pydas_obj,
    sseg=0,
    output_file=None,
    chName="all",
    tz=None,
    policy="short_only",
    events=None,
    preview=None,
    k_mad=6.0,
):
    """Return a quality DataFrame (optionally write Excel).

    Does not modify channel samples. Grades are English ids
    ``good`` / ``repaired`` / ``limited`` / ``bad``.
    The binary ``.out`` pack cannot store this table.

    Parameters
    ----------
    pydas_obj : PyDAS
    sseg : int, optional
        Segment index.
    output_file : str, optional
        If given, write the table to this ``.xlsx`` path.
    chName : str or list or ``'all'``, optional
    tz : float, optional
        Characteristic period in seconds.
    policy : str, optional
        Used only to annotate what *would* be repaired; default ``short_only``.
    events, preview
        Optional precomputed detection / preview.
    k_mad : float, optional

    Returns
    -------
    pandas.DataFrame
    """
    if events is None:
        events = detect_bad_events(
            pydas_obj, chName=chName, sseg=sseg, tz=tz, k_mad=k_mad
        )
    events = _annotate_actions(events, policy)
    table = assess_segment(pydas_obj, events, sseg=sseg, preview=preview)
    if chName != "all" and table is not None and not table.empty:
        names = [chName] if isinstance(chName, str) else [str(n) for n in chName]
        table = table.loc[table["channel"].isin(names)].reset_index(drop=True)
    if output_file:
        table.to_excel(output_file, index=False)
        logger.info("qc_report: wrote %s", output_file)
    return table
