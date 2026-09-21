# tests/unit/test_output.py
import pytest
import os
import pandas as pd
import numpy as np
from pydas.output import (
    write_data, export_to_dat, export_to_mat, 
    export_to_parquet, export_to_feather, export_to_hdf5
)

def test_write_data(pydas_instance, tmp_path):
    """Test exporting data to custom binary .out format."""
    out_file = tmp_path / "test_write.out"
    write_data(pydas_instance, str(out_file))
    
    assert out_file.exists()
    assert os.path.getsize(out_file) > 256  # Header is at least 256 bytes

def test_export_to_dat(pydas_instance, tmp_path, monkeypatch):
    """Test exporting data to text .dat format."""
    # Temporarily change working directory to tmp_path so files are saved there
    monkeypatch.chdir(tmp_path)
    
    pydas_instance.__scale__ = 'model'
    pydas_instance.__filename__ = "test_data.csv"
    export_to_dat(pydas_instance)
    
    # Check if file was created
    expected_file = tmp_path / "test_data_seg00-model.dat"
    assert expected_file.exists()
    
    # Basic content check
    with open(expected_file, 'r') as f:
        content = f.read()
        assert "OUTFILE NAME" in content

def test_export_to_mat(pydas_instance, tmp_path):
    """Test exporting data to MATLAB .mat format."""
    mat_file = tmp_path / "test_data.mat"
    
    res = export_to_mat(pydas_instance, filename=str(mat_file))
    assert res is True
    assert mat_file.exists()

def test_export_to_parquet(pydas_instance, tmp_path, monkeypatch):
    """Test exporting data to Parquet format."""
    monkeypatch.chdir(tmp_path)
    pydas_instance.__filename__ = "test_data.csv"
    
    res = export_to_parquet(pydas_instance)
    assert res is True
    
    parquet_file = tmp_path / "test_data_seg00.parquet"
    json_file = tmp_path / "test_data_seg00_metadata.json"
    assert parquet_file.exists()
    assert json_file.exists()
    
    # Verify data can be read
    df = pd.read_parquet(parquet_file)
    assert 'Wave1' in df.columns

def test_export_to_feather(pydas_instance, tmp_path, monkeypatch):
    """Test exporting data to Feather format."""
    monkeypatch.chdir(tmp_path)
    pydas_instance.__filename__ = "test_data.csv"
    
    res = export_to_feather(pydas_instance)
    assert res is True
    
    feather_file = tmp_path / "test_data_seg00.feather"
    assert feather_file.exists()
    
    # Verify data can be read
    df = pd.read_feather(feather_file)
    assert 'Wave1' in df.columns
    assert '__metadata_fs__' in df.columns  # Feather test appends metadata to cols

def test_export_to_hdf5(pydas_instance, tmp_path):
    """Test exporting data to HDF5 format."""
    h5_file = tmp_path / "test_data.h5"
    
    res = export_to_hdf5(pydas_instance, filename=str(h5_file))
    assert res is True
    assert h5_file.exists()
