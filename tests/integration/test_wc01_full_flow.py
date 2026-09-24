# tests/integration/test_wc01_full_flow.py
import pytest
import os
import numpy as np
from pathlib import Path
from pydas import PyDAS
from pydas.analysis import spectral_analysis, statistic_analysis, extreme_analysis

WC01_PATH = Path(__file__).resolve().parents[1] / "legacy" / "WC01.out"

def test_wc01_full_workflow(tmp_path):
    """
    End-to-end spine on the legacy WC01.out file.

    detect -> apply_repair -> qc -> remove_mean -> filter -> spectrum /
    extremes / export. ``data_wash`` is not the laboratory path.
    """
    if not WC01_PATH.exists():
        pytest.skip(f"Legacy test file {WC01_PATH} not found")

    file_path = str(WC01_PATH)
    data = PyDAS(filename=file_path, lam=25)

    assert data.__segN__ >= 1
    assert data.__chN__ >= 1

    wave_ch = None
    for col in data.data[0].columns:
        if 'Wave' in col or col.lower() in ['wave1', 'wave']:
            wave_ch = col
            break
    if wave_ch is None:
        wave_ch = data.data[0].columns[0]

    tz = 1.0
    events = data.detect_bad_events(wave_ch, tz=tz)
    preview = data.preview_repair(wave_ch, tz=tz, events=events)
    data.apply_repair(wave_ch, tz=tz, preview=preview)
    qc = data.qc_report(tz=tz)
    assert not qc.empty

    data.remove_mean(chName=wave_ch)
    data.apply_lowpass_filter(chName=wave_ch, cutoffull=3.0)

    stats_df = statistic_analysis(data, ch_name=wave_ch, advanced=True)
    assert not stats_df.empty

    spec = spectral_analysis(data, channel_name=wave_ch, fullscale=True)
    assert spec is not None
    assert len(spec.args) > 0

    ext_res = extreme_analysis(
        data, ch_name=wave_ch, visualization=False, tz=tz, qc=qc
    )
    assert ext_res is not None
    assert 'exceedance_table' in ext_res or ext_res.get('qc_blocked') is True

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        data.__filename__ = "test_wc01_export.out"
        from pydas.output import export_to_dat
        export_to_dat(data)
        assert (tmp_path / "test_wc01_export_seg00-model.dat").exists()

        from pydas.plot import plot_channel
        plot_channel(data, ch_name=wave_ch, plotbackend='matplotlib', show=False)
    finally:
        os.chdir(cwd)
