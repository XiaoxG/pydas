# tests/integration/test_pydas_io.py
import pytest
import os
import numpy as np
from pathlib import Path
from pydas import PyDAS
from pydas.output import write_data

WC01_PATH = Path(__file__).resolve().parents[1] / "legacy" / "WC01.out"

@pytest.fixture
def legacy_out_file():
    """Path to the real WC01.out file."""
    if not WC01_PATH.exists():
        pytest.skip(f"Legacy test file {WC01_PATH} not found")
    return str(WC01_PATH)

def test_pydas_read_write_loop(legacy_out_file, tmp_path):
    """Test reading an .out file, writing it back, and verifying consistency."""
    # 1. Read
    pydas = PyDAS(legacy_out_file, lam=60.0)
    assert pydas.__chN__ > 0
    assert pydas.__fs__ > 0
    
    original_data = pydas.data[0].copy()
    original_channels = pydas.chInfo['Name'].tolist()
    
    # 2. Write
    save_path = str(tmp_path / "WC01_recompute.out")
    pydas.write(save_path)
    
    # 3. Read back
    pydas_new = PyDAS(save_path, lam=60.0)
    
    # 4. Verify
    assert pydas_new.__chN__ == pydas.__chN__
    assert pydas_new.__fs__ == pydas.__fs__
    
    # Check data integrity (using allclose for float precision)
    for col in original_channels:
        np.testing.assert_allclose(
            pydas_new.data[0][col].values, 
            original_data[col].values, 
            rtol=1e-5
        )

def test_export_formats(pydas_instance, tmp_path):
    """Test exporting to various supported formats."""
    # We change directory to tmp_path so exports land there
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Mat file
        mat_path = tmp_path / "test.mat"
        pydas_instance.to_mat(str(mat_path))
        assert os.path.exists(mat_path)
        
        # Feather and Parquet use hardcoded logic based on __filename__
        # synthetic.csv -> synthetic_seg00.feather
        pydas_instance.to_feather()
        assert os.path.exists("synthetic_seg00.feather")
        
        pydas_instance.to_parquet()
        assert os.path.exists("synthetic_seg00.parquet")
    finally:
        os.chdir(old_cwd)


def test_generated_out_roundtrip(pydas_instance, tmp_path):
    """Generate a mini .out in tmp and read it back without WC01.out."""
    out_file = tmp_path / "mini.out"
    write_data(pydas_instance, str(out_file))
    assert out_file.exists()

    loaded = PyDAS(filename=str(out_file), lam=1.0)
    assert loaded.__chN__ == pydas_instance.__chN__
    assert loaded.__fs__ == pydas_instance.__fs__
    assert "Wave1" in loaded.chInfo["Name"].values
    assert len(loaded.data[0]["Wave1"]) == len(pydas_instance.data[0]["Wave1"])
    corr = np.corrcoef(
        loaded.data[0]["Wave1"].values, pydas_instance.data[0]["Wave1"].values
    )[0, 1]
    assert corr > 0.99
