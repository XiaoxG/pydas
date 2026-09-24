"""PyDAS Core - Quality Mixin

Thin proxy for bad-event detection, short-gap repair, and qc grading.
"""
import logging

from ..quality.detect import detect_bad_events
from ..quality.repair import apply_repair, preview_repair
from ..quality.report import qc_report

logger = logging.getLogger(__name__)


class QualityMixin:
    """Mixin: diagnose / preview / repair / qc_report."""

    def detect_bad_events(self, chName="all", sseg=0, tz=None, k_mad=6.0):
        """Return a table of clip / dropout / spike-burst events (read-only).

        Consecutive bad samples are one event. Default detection does not
        write ``data``. See :func:`pydas.quality.detect.detect_bad_events`.
        """
        return detect_bad_events(self, chName=chName, sseg=sseg, tz=tz, k_mad=k_mad)

    def preview_repair(
        self, chName="all", sseg=0, policy="short_only", events=None, tz=None, k_mad=6.0,
    ):
        """Preview a ``short_only`` repair without writing ``data``.

        Linear interpolation for ``n<=3``; PCHIP for longer short bursts.
        Clip, edge, and medium/long gaps are refused.
        """
        return preview_repair(
            self, chName=chName, sseg=sseg, policy=policy, events=events, tz=tz, k_mad=k_mad
        )

    def apply_repair(
        self,
        chName="all",
        sseg=0,
        policy="short_only",
        events=None,
        preview=None,
        tz=None,
        k_mad=6.0,
    ):
        """Apply a short-only repair, update stats, and append ``repair_log``.

        The ``.out`` pack cannot store the audit table; keep ``repair_log``
        (or ``qc_report`` Excel) beside the binary file.
        """
        return apply_repair(
            self,
            chName=chName,
            sseg=sseg,
            policy=policy,
            events=events,
            preview=preview,
            tz=tz,
            k_mad=k_mad,
        )

    def qc_report(
        self,
        sseg=0,
        output_file=None,
        chName="all",
        tz=None,
        policy="short_only",
        events=None,
        preview=None,
        k_mad=6.0,
    ):
        """Per-channel quality grades: good / repaired / limited / bad.

        ``repaired`` requires a written ``repair_log`` entry. Unapplied
        short bursts grade ``limited`` so MPM/EEV cannot use the fake peak.
        """
        return qc_report(
            self,
            sseg=sseg,
            output_file=output_file,
            chName=chName,
            tz=tz,
            policy=policy,
            events=events,
            preview=preview,
            k_mad=k_mad,
        )
