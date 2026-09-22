"""Data-quality helpers: event detection, short-gap repair, and grading.

These functions do not change the frozen ``.out`` pack layout.  Repair
audit lives on the PyDAS object (``repair_log``), not in the binary file.
"""

from .assess import GRADE_BAD, GRADE_GOOD, GRADE_LIMITED, GRADE_REPAIRED
from .detect import detect_bad_events, estimate_t_star
from .gates import grade_allows_extremes
from .repair import RepairPreview, apply_repair, empty_repair_log, preview_repair
from .report import qc_report

__all__ = [
    "detect_bad_events",
    "estimate_t_star",
    "preview_repair",
    "apply_repair",
    "empty_repair_log",
    "RepairPreview",
    "qc_report",
    "grade_allows_extremes",
    "GRADE_GOOD",
    "GRADE_REPAIRED",
    "GRADE_LIMITED",
    "GRADE_BAD",
]
