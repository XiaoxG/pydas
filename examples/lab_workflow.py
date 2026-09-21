#!/usr/bin/env python
"""Laboratory workflow: DataFrame -> .out -> filter -> spectrum -> report.

This is the path colleagues should copy for basin tests. The constructor
reads binary ``.out`` only; CSV/TSV must go through ``from_dataframe`` or
``read_csv``. ``cutoffull`` is full-scale rad/s, not Hertz.

Run from the repository root after ``pip install -e .``::

    python examples/lab_workflow.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pydas import PyDAS


def build_model_record(fs=50.0, duration=40.0, lam=36.0, seed=1):
    """Synthetic wave elevation and a force channel at model scale."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / fs)
    eta = 0.04 * np.sin(2 * np.pi * 0.8 * t) + 0.005 * rng.standard_normal(len(t))
    fx = 8.0 * np.sin(2 * np.pi * 0.8 * t + 0.25) + 0.2 * rng.standard_normal(len(t))
    df = pd.DataFrame({"eta": eta, "fx": fx})
    obj = PyDAS.from_dataframe(
        df, fs=fs, lam=lam, units={"eta": "m", "fx": "N"}, desc="lab-workflow-demo"
    )
    return obj


def main(out_dir="."):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("PyDAS laboratory workflow")
    print("-------------------------")

    data = build_model_record()
    out_path = out_dir / "lab_demo.out"
    data.write(str(out_path))
    print(f"Wrote binary pack: {out_path}")

    loaded = PyDAS(filename=str(out_path), lam=36.0)
    print(loaded.print_info())
    print("Channels:", list(loaded.chInfo["Name"]))

    tz_model = 1.0 / 0.8
    qc = loaded.qc_report(tz=tz_model)
    print(qc[["channel", "grade", "n_events", "suggested_action"]].to_string(index=False))

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
    print(f"Wrote plot: {out_dir / 'lab_eta.png'}")

    report_path = out_dir / "lab_channel_report.xlsx"
    loaded.channel_report(
        str(report_path),
        wave_type="irregular",
        fullscale=True,
        include_charts=False,
    )
    print(f"Wrote Excel report: {report_path}")
    print("Done. Keep the .out pack layout unchanged if other software reads it.")


if __name__ == "__main__":
    import tempfile

    main(tempfile.mkdtemp(prefix="pydas-lab-"))
