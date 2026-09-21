# tests/unit/test_channels.py
import numpy as np


def test_add_and_delete_channel(pydas_instance):
    """Adding a channel grows chInfo; deleting it restores the original set."""
    n_orig = pydas_instance.__chN__
    series = np.ones(len(pydas_instance.data[0])) * 3.0
    pydas_instance.add_channel("Const3", "m", series, pydas_instance.__fs__)
    assert "Const3" in pydas_instance.chInfo["Name"].values
    assert pydas_instance.__chN__ == n_orig + 1

    pydas_instance.delete_channel("Const3")
    assert "Const3" not in pydas_instance.chInfo["Name"].values
    assert pydas_instance.__chN__ == n_orig


def test_copy_rename_reorder_select(pydas_instance):
    """Channel copy / rename / reorder / select stay on the object state."""
    ch1 = "Wave1"

    pydas_instance.copy_channel(ch1, "Wave1_Copy")
    assert "Wave1_Copy" in pydas_instance.chInfo["Name"].values
    np.testing.assert_array_equal(
        pydas_instance.data[0][ch1], pydas_instance.data[0]["Wave1_Copy"]
    )

    pydas_instance.rename_channel("Wave1_Copy", "Wave1_Renamed")
    assert "Wave1_Renamed" in pydas_instance.chInfo["Name"].values
    assert "Wave1_Copy" not in pydas_instance.chInfo["Name"].values

    names = pydas_instance.chInfo["Name"].tolist()
    new_order = names[::-1]
    pydas_instance.change_channel_order(new_order)
    assert pydas_instance.chInfo["Name"].tolist() == new_order

    pydas_instance.select_channels(["Wave1"])
    assert pydas_instance.__chN__ == 1
    assert "Wave1_Renamed" not in pydas_instance.chInfo["Name"].values


def test_from_dataframe_and_read_csv(tmp_path):
    """CSV/DataFrame construction must not go through binary __read__."""
    import pandas as pd
    from pydas import PyDAS

    n = 64
    df = pd.DataFrame({"eta": np.sin(np.linspace(0, 2 * np.pi, n))})
    obj = PyDAS.from_dataframe(df, fs=10.0, lam=4.0, units={"eta": "m"})
    assert obj.__fs__ == 10.0
    assert obj.__lam__ == 4.0
    assert "eta" in obj.chInfo["Name"].values
    assert obj.chInfo.loc[obj.chInfo["Name"] == "eta", "Unit"].values[0] == "m"
    assert len(obj.data[0]["eta"]) == n

    csv_path = tmp_path / "eta.csv"
    df.to_csv(csv_path, index=False)
    loaded = PyDAS.read_csv(str(csv_path), fs=10.0, lam=1.0, units={"eta": "m"})
    assert loaded.__filename__ == str(csv_path)
    np.testing.assert_allclose(loaded.data[0]["eta"], df["eta"].values)


def test_snake_case_aliases(pydas_instance):
    """Historical camelCase methods keep snake_case aliases."""
    pydas_instance.update_channel_count()
    pydas_instance.update_statistics(chName="all")
    assert pydas_instance.__chN__ == len(pydas_instance.chInfo)
