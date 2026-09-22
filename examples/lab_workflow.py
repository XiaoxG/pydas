#!/usr/bin/env python
"""Laboratory processing spine with planted defects.

This is the path colleagues should copy for basin tests:

    load .out -> inspect -> cut_series -> detect -> preview -> apply_repair
    -> qc_report -> remove_mean / detrend -> filter -> spectrum -> report
    -> write .out plus sidecar qc / repair_log files

Do not filter before detect/repair. ``qc_report`` only reflects a completed
short repair after ``apply_repair``. ``cutoffull`` is full-scale rad/s.

The constructor reads binary ``.out`` only. Run from the repository root::

    python examples/lab_workflow.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pydas import PyDAS


FS = 50.0
DURATION_S = 40.0
LAM = 36.0
WAVE_HZ = 0.8
TZ_MODEL = 1.0 / WAVE_HZ  # model-scale wave period used as T*


def build_model_record(fs=FS, duration=DURATION_S, lam=LAM, seed=1):
    """Synthetic wave elevation and a force channel at model scale."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / fs)
    eta = 0.04 * np.sin(2 * np.pi * WAVE_HZ * t) + 0.005 * rng.standard_normal(len(t))
    fx = 8.0 * np.sin(2 * np.pi * WAVE_HZ * t + 0.25) + 0.2 * rng.standard_normal(len(t))
    df = pd.DataFrame({"eta": eta, "fx": fx})
    return PyDAS.from_dataframe(
        df, fs=fs, lam=lam, units={"eta": "m", "fx": "N"}, desc="lab-workflow-demo"
    )


def plant_defects(obj, fs=FS):
    """Insert startup, a short spike, a short dropout, and a clip plateau."""
    eta = obj.data[0]["eta"].to_numpy(copy=True)
    fx = obj.data[0]["fx"].to_numpy(copy=True)

    n_edge = int(round(1.2 * fs))
    eta[:n_edge] = 0.35
    fx[:n_edge] = 40.0

    i_spike = int(round(10.0 * fs))
    eta[i_spike] = 2.0
    fx[i_spike] = 80.0

    i_drop = int(round(14.0 * fs))
    eta[i_drop : i_drop + 5] = eta[i_drop]

    i_clip = int(round(22.0 * fs))
    eta[i_clip : i_clip + 16] = 3.0

    obj.data[0]["eta"] = eta
    obj.data[0]["fx"] = fx
    obj.update_statistics()
    print(
        "Planted defects: startup 0-1.2 s (cut, do not interpolate); "
        "coincident short spikes at t=10 s (repair); "
        "5-sample dropout on eta at t=14 s; "
        "clip plateau on eta at t=22 s (refuse)."
    )


def _print_events(events, title):
    print(title)
    cols = [
        "channel", "kind", "start_i", "stop_i", "n", "duration_s",
        "at_edge", "action", "refuse_reason", "interpolator",
    ]
    show = [c for c in cols if c in events.columns]
    if events.empty:
        print("  (no events)")
        return
    print(events[show].to_string(index=False))


def main(out_dir="."):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("PyDAS laboratory workflow (processing spine)")
    print("--------------------------------------------")
    print(f"T* tz={TZ_MODEL:.4f} s (model wave period, not the 1 s default)")
    print(
        "Order: cut -> detect -> preview -> apply_repair -> qc -> "
        "mean/detrend -> filter. Do not reverse detect and filter."
    )

    data = build_model_record()
    plant_defects(data)
    raw_out = out_dir / "lab_demo_raw.out"
    data.write(str(raw_out))
    print(f"Wrote raw binary pack: {raw_out}")

    loaded = PyDAS(filename=str(raw_out), lam=LAM)
    print(loaded.print_info())
    print("Channels:", list(loaded.chInfo["Name"]))

    loaded.plot_channel(
        "eta",
        plotbackend="matplotlib",
        show=False,
        save_path=str(out_dir / "lab_eta_raw.png"),
    )
    print(f"Wrote inspect plot: {out_dir / 'lab_eta_raw.png'}")

    # Startup / edge burst: cut the window; do not interpolate file edges.
    loaded.cut_series(start=2.0, stop=38.0, sseg=0)
    print("cut_series(2.0, 38.0): dropped startup; edge events should be gone.")

    events = loaded.detect_bad_events("all", tz=TZ_MODEL)
    _print_events(events, "detect_bad_events (read-only):")

    preview = loaded.preview_repair("all", tz=TZ_MODEL, events=events)
    _print_events(preview.events, "preview_repair (data not written yet):")
    refused = preview.events.loc[preview.events["action"] == "refuse"]
    if not refused.empty:
        print(
            "Refused events stay as recorded. Clip is never interpolated; "
            "do not use that channel-segment for MPM/EEV."
        )
        print(refused[["channel", "kind", "refuse_reason", "n"]].to_string(index=False))

    loaded.apply_repair("all", tz=TZ_MODEL, preview=preview)
    print(f"apply_repair wrote short bursts; repair_log rows={len(loaded.repair_log)}")

    qc_path = out_dir / "lab_qc.xlsx"
    qc = loaded.qc_report(tz=TZ_MODEL, output_file=str(qc_path))
    print("qc_report after apply_repair:")
    print(
        qc[["channel", "grade", "n_events", "suggested_action", "note"]].to_string(
            index=False
        )
    )
    print(f"Wrote qc table: {qc_path}")

    repair_path = out_dir / "lab_repair_log.csv"
    loaded.repair_log.to_csv(repair_path, index=False)
    print(f"Wrote repair audit (not stored in .out): {repair_path}")

    loaded.remove_mean(["eta", "fx"])
    loaded.detrend(["eta", "fx"], kind="linear")
    # cutoffull is full-scale rad/s; model Hz = cutoffull / 2pi * sqrt(lam)
    loaded.apply_lowpass_filter("eta", cutoffull=2.0, replace=True)
    loaded.update_statistics()

    spec = loaded.spectral_analysis("eta", method="cov", L=256, plot=False)
    print(f"eta spectrum length: {len(spec.data)}")

    loaded.plot_channel(
        "eta",
        plotbackend="matplotlib",
        show=False,
        save_path=str(out_dir / "lab_eta.png"),
    )
    print(f"Wrote processed plot: {out_dir / 'lab_eta.png'}")

    report_path = out_dir / "lab_channel_report.xlsx"
    # Reuse the post-repair qc table. A second detect after the lowpass would
    # smear the clip plateau and might grade eta as repaired.
    report = loaded.channel_report(
        str(report_path),
        wave_type="irregular",
        fullscale=True,
        include_charts=False,
        tz=TZ_MODEL,
        qc=qc,
    )
    print(f"Wrote Excel report: {report_path}")
    mpm_cols = [c for c in ("Name", "MPM_pos", "MPM_neg", "EEV_pos", "EEV_neg") if c in report.columns]
    if mpm_cols:
        print("MPM/EEV (NaN means check qc_report for limited/bad, not the formula):")
        print(report[mpm_cols].to_string(index=False))

    processed_out = out_dir / "lab_demo_processed.out"
    loaded.write(str(processed_out))
    print(f"Wrote processed pack: {processed_out}")
    print(
        "Deliver four files: processed .out, qc xlsx, repair_log csv, "
        "channel_report xlsx. The .out pack layout stays frozen."
    )


if __name__ == "__main__":
    import tempfile

    main(tempfile.mkdtemp(prefix="pydas-lab-"))
