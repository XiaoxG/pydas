# tests/unit/test_quality.py
import numpy as np
import pandas as pd
import pytest

from pydas import PyDAS
from pydas.quality.repair import decide_action


FS = 100.0
DURATION = 20.0
FREQ = 1.0
TZ = 1.0


def _sine_frame(extra=None):
    t = np.arange(0.0, DURATION, 1.0 / FS)
    eta = np.sin(2.0 * np.pi * FREQ * t)
    data = {"eta": eta}
    if extra:
        for name, series in extra.items():
            data[name] = series
    return pd.DataFrame(data)


def _obj(extra=None):
    return PyDAS.from_dataframe(
        _sine_frame(extra), fs=FS, lam=1.0, units={"eta": "m"}
    )


def test_clean_sine_has_no_events():
    obj = _obj()
    events = obj.detect_bad_events("eta", tz=TZ)
    assert events.empty


def test_single_spike_is_one_event_and_linear_repair():
    obj = _obj()
    y = obj.data[0]["eta"].to_numpy(copy=True)
    y[500] = 25.0
    obj.data[0]["eta"] = y
    events = obj.detect_bad_events("eta", tz=TZ)
    spikes = events.loc[events["kind"] == "spike_burst"]
    assert len(spikes) == 1
    assert int(spikes.iloc[0]["n"]) == 1
    preview = obj.preview_repair("eta", tz=TZ, events=events)
    row = preview.events.iloc[0]
    assert row["action"] == "repair"
    assert row["interpolator"] == "linear"
    repaired = preview.series["eta"]
    assert repaired[500] == pytest.approx(np.sin(2.0 * np.pi * FREQ * 5.0), abs=0.05)


def test_eight_point_burst_is_one_event_pchip():
    obj = _obj()
    y = obj.data[0]["eta"].to_numpy(copy=True)
    rng = np.random.default_rng(0)
    y[400:408] = 20.0 + rng.normal(0.0, 0.15, 8)
    obj.data[0]["eta"] = y
    events = obj.detect_bad_events("eta", tz=TZ)
    spikes = events.loc[events["kind"] == "spike_burst"]
    assert len(spikes) == 1
    assert int(spikes.iloc[0]["n"]) == 8
    preview = obj.preview_repair("eta", tz=TZ, events=events)
    assert preview.events.iloc[0]["interpolator"] == "pchip"
    assert preview.events.iloc[0]["action"] == "repair"
    repaired = preview.series["eta"]
    t = np.arange(400, 408) / FS
    np.testing.assert_allclose(repaired[400:408], np.sin(2.0 * np.pi * FREQ * t), atol=0.15)
    obj.apply_repair("eta", tz=TZ, preview=preview)
    assert not obj.repair_log.empty
    std_clean = np.std(np.sin(2.0 * np.pi * FREQ * np.arange(0.0, DURATION, 1.0 / FS)))
    assert abs(np.std(obj.data[0]["eta"]) - std_clean) / std_clean < 0.05


def test_long_gap_is_not_filled():
    obj = _obj()
    y = obj.data[0]["eta"].to_numpy(copy=True)
    original = y[200:250].copy()
    y[200:250] = 0.0
    obj.data[0]["eta"] = y
    events = obj.detect_bad_events("eta", tz=TZ)
    drops = events.loc[events["kind"] == "dropout"]
    assert not drops.empty
    assert int(drops.iloc[0]["n"]) >= 40
    preview = obj.preview_repair("eta", tz=TZ, events=events)
    assert (preview.events["action"] == "refuse").all()
    assert preview.events["refuse_reason"].iloc[0] in {"medium_gap", "long_gap"}
    np.testing.assert_array_equal(preview.series["eta"][200:250], original * 0.0)


def test_clip_is_refused():
    obj = _obj()
    y = obj.data[0]["eta"].to_numpy(copy=True)
    y[300:315] = 6.0
    obj.data[0]["eta"] = y
    events = obj.detect_bad_events("eta", tz=TZ)
    clips = events.loc[events["kind"] == "clip"]
    assert not clips.empty
    preview = obj.preview_repair("eta", tz=TZ, events=events)
    assert (preview.events.loc[preview.events["kind"] == "clip", "action"] == "refuse").all()
    assert (preview.events.loc[preview.events["kind"] == "clip", "refuse_reason"] == "clip").all()
    np.testing.assert_array_equal(preview.series["eta"][300:315], 6.0)


def test_edge_burst_is_refused():
    obj = _obj()
    y = obj.data[0]["eta"].to_numpy(copy=True)
    y[:6] = 15.0
    obj.data[0]["eta"] = y
    events = obj.detect_bad_events("eta", tz=TZ)
    assert events["at_edge"].any()
    preview = obj.preview_repair("eta", tz=TZ, events=events)
    assert (preview.events["action"] == "refuse").all()
    assert (preview.events["refuse_reason"] == "edge").all()


def test_short_coincident_spikes_still_repair_each_channel():
    t = np.arange(0.0, DURATION, 1.0 / FS)
    eta = np.sin(2.0 * np.pi * FREQ * t)
    fx = 0.5 * np.sin(2.0 * np.pi * FREQ * t + 0.3)
    eta[500] = 22.0
    fx[500] = 18.0
    obj = PyDAS.from_dataframe(
        pd.DataFrame({"eta": eta, "fx": fx}), fs=FS, lam=1.0, units={"eta": "m", "fx": "N"}
    )
    events = obj.detect_bad_events(["eta", "fx"], tz=TZ)
    assert not events.empty
    assert events["coincident"].all()
    assert not events["coincident_dropout_or_clip"].any()
    preview = obj.preview_repair(["eta", "fx"], tz=TZ, events=events)
    assert (preview.events["action"] == "repair").all()
    qc = obj.qc_report(tz=TZ, events=events, preview=preview)
    assert set(qc["grade"]) <= {"repaired", "good"}


def test_coincident_dropout_marks_segment_limited_or_bad():
    t = np.arange(0.0, DURATION, 1.0 / FS)
    eta = np.sin(2.0 * np.pi * FREQ * t)
    fx = 0.5 * np.sin(2.0 * np.pi * FREQ * t)
    eta[300:320] = 0.0
    fx[300:320] = 0.0
    obj = PyDAS.from_dataframe(
        pd.DataFrame({"eta": eta, "fx": fx}), fs=FS, lam=1.0, units={"eta": "m", "fx": "N"}
    )
    events = obj.detect_bad_events(["eta", "fx"], tz=TZ)
    assert events["coincident_dropout_or_clip"].all()
    preview = obj.preview_repair(["eta", "fx"], tz=TZ, events=events)
    assert (preview.events["action"] == "refuse").all()
    qc = obj.qc_report(tz=TZ, events=events, preview=preview)
    assert set(qc["grade"]) <= {"limited", "bad"}
    assert "good" not in set(qc["grade"])


def test_qc_clean_is_good():
    obj = _obj()
    qc = obj.qc_report(tz=TZ)
    assert (qc["grade"] == "good").all()
    assert (qc["suggested_action"] == "none").all()


def test_decide_action_n_le_3_linear():
    action, reason, interp = decide_action(
        "spike_burst", 3, 0.03, False, 1.0, "user", policy="short_only"
    )
    assert action == "repair"
    assert interp == "linear"
    action, reason, interp = decide_action(
        "spike_burst", 8, 0.08, False, 1.0, "user", policy="short_only"
    )
    assert action == "repair"
    assert interp == "pchip"


def test_detrend_linear_removes_slope():
    t = np.arange(0.0, 5.0, 1.0 / FS)
    y = 0.2 * t + np.sin(2.0 * np.pi * FREQ * t)
    obj = PyDAS.from_dataframe(pd.DataFrame({"eta": y}), fs=FS, lam=1.0)
    obj.detrend("eta", kind="linear")
    x = np.arange(len(obj.data[0]["eta"]))
    slope = np.polyfit(x, obj.data[0]["eta"].to_numpy(), 1)[0]
    assert abs(slope) < 1e-4


def test_irregular_wave_crests_are_not_events():
    import pydas.waveModel as wm

    w = np.linspace(0.3, 4.0, 256)
    S = wm.jonswap(w, Hs=0.08, Tp=1.8, gamma=3.3)
    _t, eta = wm.spectrum_to_timeseries(w, S, duration=80.0, dt=0.05, seed=2)
    obj = PyDAS.from_dataframe(pd.DataFrame({"eta": eta}), fs=20.0, lam=1.0)
    events = obj.detect_bad_events("eta", tz=1.8)
    assert events.empty
    orig_max = float(np.max(np.abs(eta)))
    preview = obj.preview_repair("eta", tz=1.8, events=events)
    repaired = preview.series["eta"]
    assert float(np.max(np.abs(repaired))) == pytest.approx(orig_max)


def test_repair_log_is_not_stored_in_out(tmp_path):
    obj = _obj()
    y = obj.data[0]["eta"].to_numpy(copy=True)
    y[500] = 25.0
    obj.data[0]["eta"] = y
    obj.apply_repair("eta", tz=TZ)
    assert not obj.repair_log.empty
    path = tmp_path / "quality.out"
    obj.write(str(path))
    loaded = PyDAS(filename=str(path), lam=1.0)
    assert loaded.repair_log.empty


def test_updateST_skips_nan():
    obj = _obj()
    y = obj.data[0]["eta"].to_numpy(copy=True)
    y[100] = np.nan
    obj.data[0]["eta"] = y
    obj.updateST("eta")
    assert np.isfinite(obj.segStatis[0].loc["eta", "Mean"])
    assert np.isfinite(obj.segStatis[0].loc["eta", "STD"])
