#!/usr/bin/env python
"""Basic PyDAS usage with the current public API."""

from pydas import PyDAS
import pydas.waveModel as wm
import numpy as np
import pandas as pd


def create_synthetic_data(filename="synthetic_data.csv"):
    """Write a small CSV that can be loaded via PyDAS.from a DataFrame workflow.

    The binary ``.out`` reader is still the primary I/O path. This helper only
    builds arrays so the example can construct an empty PyDAS object.
    """
    fs = 10.0
    t = np.arange(0, 20, 1 / fs)
    f1, f2, f3 = 0.5, 1.0, 2.0
    a1, a2, a3 = 1.0, 0.5, 0.25
    signal = (
        a1 * np.sin(2 * np.pi * f1 * t)
        + a2 * np.sin(2 * np.pi * f2 * t)
        + a3 * np.sin(2 * np.pi * f3 * t)
        + 0.1 * np.random.randn(len(t))
    )
    df = pd.DataFrame({"Time": t, "signal": signal})
    df.to_csv(filename, index=False)
    return filename, fs, t, signal


def main():
    print("PyDAS basic usage example")
    print("-------------------------")

    _csv, fs, t, signal = create_synthetic_data()
    data = PyDAS(filename=None, lam=1.0)
    data.add_channel("signal", "m", signal, fs)
    print(f"Loaded {data.__chN__} channel(s): {list(data.chInfo['Name'])}")

    data.plot_channel("signal", plotbackend="plotly", show=False, save_html="signal_plot.html")
    print("Wrote interactive figure: signal_plot.html")

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
    main()
