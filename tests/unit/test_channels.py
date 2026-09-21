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
