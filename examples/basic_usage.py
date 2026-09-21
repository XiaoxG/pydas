#!/usr/bin/env python
"""Basic PyDAS usage with the current public API.

Run from the repository root after ``pip install -e .``::

    python examples/basic_usage.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import pydas.waveModel as wm
from pydas import PyDAS


def make_signal(fs=10.0, duration=20.0, seed=0):
    """Return sampling rate, time axis, and a three-tone series."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / fs)
    signal = (
        1.0 * np.sin(2 * np.pi * 0.5 * t)
        + 0.5 * np.sin(2 * np.pi * 1.0 * t)
        + 0.25 * np.sin(2 * np.pi * 2.0 * t)
        + 0.1 * rng.standard_normal(len(t))
    )
    return fs, t, signal


def main(out_dir="."):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("PyDAS basic usage example")
    print("-------------------------")
    print(f"Artifact directory: {out_dir}")

    fs, _t, signal = make_signal()
    df = pd.DataFrame({"signal": signal})
    data = PyDAS.from_dataframe(df, fs=fs, lam=1.0, units={"signal": "m"})
    print(f"Loaded {data.__chN__} channel(s): {list(data.chInfo['Name'])}")

    plot_path = out_dir / "signal_plot.png"
    data.plot_channel(
        "signal",
        plotbackend="matplotlib",
        show=False,
        save_path=str(plot_path),
    )
    print(f"Wrote matplotlib figure: {plot_path}")

    spec = data.spectral_analysis(
        channel_name="signal",
        method="cov",
        L=128,
        plot=False,
    )
    print(f"Spectrum length: {len(spec.data)}")

    freq = np.linspace(0.05, 3, 100)
    S = wm.jonswap(freq, Hs=4.0, Tp=10.0, gamma=3.3)
    print(f"JONSWAP peak density: {np.max(S):.4f}")


if __name__ == "__main__":
    import tempfile

    main(tempfile.mkdtemp(prefix="pydas-basic-"))
