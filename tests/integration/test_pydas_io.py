# tests/integration/test_pydas_io.py
import pytest
import os
import numpy as np
from pydas import PyDAS

@pytest.fixture
def legacy_out_file():
    """Path to the real WC01.out file."""
    path = os.path.join(os.path.dirname(__file__), '../legacy/WC01.out')
    if not os.path.exists(path):
        pytest.skip(f"Legacy test file {path} not found")
    return path

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
        # Note: to_mat now supports filename after my recent fix!
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
