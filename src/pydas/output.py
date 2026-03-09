#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
PyDAS Output Module
Provides data output and file export functionality for the PyDAS system.
"""

import os
import datetime
import re
import math
import struct
import numpy as np
import pandas as pd
import scipy.io as sio
from .logger import logger


def write_data(pydas_obj, filename, sseg='all', ch='all'):
    """Write data to a new *.out file.

    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data.
    filename : str
        Path to the output *.out file.
    sseg : int, list, or 'all', optional
        Segment(s) to write to the file, default is 'all'.
    ch : list or 'all', optional
        Channels to write to the file, default is 'all'.

    Notes
    -----
    This method will automatically append '.out' extension if not provided.
    """
    # Ensure filename has .out extension
    if not filename.endswith('.out'):
        filename += '.out'

    # Determine which segments to write
    if sseg == 'all':
        sseg = list(range(pydas_obj.__segN__))
    elif isinstance(sseg, int):
        sseg = [sseg]
    else:
        logger.warning("Unsupported segment number, using 'all'.")
        sseg = list(range(pydas_obj.__segN__))

    logger.info(f'Saving segment(s) No. {sseg} to file {filename}')

    with open(filename, 'wb') as fOut:
        # Write file header (256 bytes)
        datemmdd = pydas_obj.__date__.split('-')

        # Pack header information
        buf = struct.pack('=hhlhh',
                          -2,                    # File format version
                          pydas_obj.__chN__,     # Number of channels
                          0x0d,                  # Reserved
                          pydas_obj.__fs__,      # Sampling frequency
                          len(sseg))             # Number of segments

        # Pack date and description
        buf += struct.pack('2s2s240s',
                           datemmdd[0].encode('utf-8'),
                           datemmdd[1].encode('utf-8'),
                           pydas_obj.__desc__.encode('utf-8')).replace(b'\x00', b' ')

        # Write header
        if fOut.write(buf) != 256:
            logger.error("Error when saving out file!")
            raise IOError("Failed to write file header")

        # Write channel names (16 bytes per channel)
        fOut.write(struct.pack(pydas_obj.__chN__ * '16s',
                               *[pydas_obj.chInfo['Name'].iloc[i].encode('utf-8')
                                 for i in range(pydas_obj.__chN__)]).replace(b'\x00', b' '))

        # Write channel units (4 bytes per channel)
        fOut.write(struct.pack(pydas_obj.__chN__ * '4s',
                               *[pydas_obj.chInfo['Unit'].iloc[i].encode('utf-8')
                                 for i in range(pydas_obj.__chN__)]).replace(b'\x00', b' '))

        # Calculate new coefficients for optimal data range
        # Find maximum absolute value for each channel across selected segments
        chMagMax = np.amax(np.array(
            [np.amax(abs(pydas_obj.data[i].values), axis=0) for i in sseg]),
            axis=0)

        # Calculate coefficients to scale data to 16-bit range (-32767 to 32767)
        chCoef_ = (chMagMax / 32767).astype(np.float32)

        # Write channel coefficients (4 bytes per channel)
        fOut.write(struct.pack('=' + pydas_obj.__chN__ * 'f', *chCoef_))

        # Write channel indices (2 bytes per channel)
        fOut.write(struct.pack('=' + pydas_obj.__chN__ * 'h', *pydas_obj.chInfo.index))

        # Write each segment
        for iseg in sseg:
            # Align to 128-byte boundary
            p_cur = fOut.tell()
            fOut.seek(128 * math.ceil(p_cur / 128))

            # Write segment information (256 bytes)
            fOut.write(struct.pack('=h', pydas_obj.segInfo['Type'].iloc[iseg]))
            fOut.write(struct.pack('=h', pydas_obj.__chN__))
            fOut.write(struct.pack(
                '=l', pydas_obj.segInfo['N sample'].iloc[iseg] + 5))

            # Write start and stop times (8 bytes)
            # Convert time strings to bytes: HH:MM:SS.s -> [s, SS, MM, HH]
            start_time_parts = re.split(r':|\\.', pydas_obj.segInfo.Start.iloc[iseg])[::-1]
            stop_time_parts = re.split(r':|\\.', pydas_obj.segInfo.Stop.iloc[iseg])[::-1]
            time_parts = start_time_parts + stop_time_parts
            time_parts_int = list(map(int, time_parts))
            fOut.write(struct.pack(8 * 'B', *time_parts_int))

            # Write segment note (240 bytes)
            fOut.write(struct.pack('240s', pydas_obj.segInfo.Note.iloc[iseg].encode(
                'utf-8')).replace(b'\x00', b' '))

            # Calculate statistical information for each channel
            mean_ = np.mean(pydas_obj.data[iseg].values, axis=0) / chCoef_
            std_ = np.std(pydas_obj.data[iseg].values, axis=0) / chCoef_
            max_ = np.amax(pydas_obj.data[iseg].values, axis=0) / chCoef_
            min_ = np.amin(pydas_obj.data[iseg].values, axis=0) / chCoef_

            # Write statistical information
            fOut.write(struct.pack('=' + pydas_obj.__chN__ * 'h',
                                   *np.round(mean_).astype(np.int16)))
            fOut.write(struct.pack('=' + pydas_obj.__chN__ * 'f', *std_))
            fOut.write(struct.pack('=' + pydas_obj.__chN__ * 'h',
                                   *np.round(max_).astype(np.int16)))
            fOut.write(struct.pack('=' + pydas_obj.__chN__ * 'h',
                                   *np.round(min_).astype(np.int16)))

            # Convert data to 16-bit integers and write
            raw_ = np.round(pydas_obj.data[iseg].values / np.repeat(chCoef_.reshape(
                1, -1), pydas_obj.segInfo['N sample'].iloc[iseg], axis=0)).astype(np.int16)
            fOut.write(raw_.tobytes())


def export_to_dat(pydas_obj, Time=True, sseg='all'):
    """Export data to DAT file format.

    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data.
    Time : bool, optional
        If True, include time column in the output, default is True.
    sseg : int or 'all', optional
        Segment(s) to export, default is 'all'.

    Notes
    -----
    The output file will be named based on the original filename with
    segment number and scale (model or full) appended.
    """
    def _write_dat_file(pydas_obj, idx):
        """Helper function to write a single segment to a DAT file."""
        path = os.getcwd()

        # Create filename based on scale (model or prototype)
        if pydas_obj.__scale__ == 'model':
            filename = f"{path}/{os.path.splitext(pydas_obj.__filename__)[0]}_seg{idx:02d}-model.dat"
        else:
            filename = f"{path}/{os.path.splitext(pydas_obj.__filename__)[0]}_seg{idx:02d}-full.dat"

        # Prepare header information
        header = [
            f"OUTFILE NAME: {pydas_obj.__filename__}",
            f"CHANNEL NO.: {pydas_obj.__chN__}",
            f"SAMPLING FREQUENCY: {pydas_obj.__fs__:.1f}"
        ]

        # Add channel names and units to header
        if Time:
            header.append("Time " + " ".join(pydas_obj.chInfo['Name']))
            header.append("S " + " ".join(pydas_obj.chInfo['Unit']))
        else:
            header.append(" ".join(pydas_obj.chInfo['Name']))
            header.append(" ".join(pydas_obj.chInfo['Unit']))

        header_str = "\n".join(header)
        n_sample = pydas_obj.segInfo.iloc[idx]['N sample']

        # Write data with or without time column
        if Time:
            time_vector = np.arange(0, n_sample / pydas_obj.__fs__, 1 / pydas_obj.__fs__)
            datawrite = np.zeros((n_sample, pydas_obj.__chN__ + 1))
            datawrite[:, 0] = time_vector
            datawrite[:, 1:] = pydas_obj.data[idx].values
            np.savetxt(filename, datawrite, fmt='% .5E', delimiter=' ', header=header_str)
        else:
            with open(filename, 'w') as f:
                f.write(header_str + "\n")
                f.write(
                    pydas_obj.data[idx].to_string(
                        header=False, index=False, justify='left',
                        float_format=lambda x: "% .5E" % x
                    )
                )

        logger.info(f"Data exported to: {filename}")

    # Determine which segments to export
    if sseg == 'all':
        for idx in range(pydas_obj.__segN__):
            _write_dat_file(pydas_obj, idx)
    elif isinstance(sseg, int):
        if sseg < pydas_obj.__segN__:
            _write_dat_file(pydas_obj, sseg)
        else:
            logger.warning(f"Segment {sseg} exceeds the maximum segment number ({pydas_obj.__segN__ - 1}).")
    else:
        logger.warning("Invalid segment selection. Use an integer or 'all'.")


def export_to_mat(pydas_obj, filename=None, sseg=0):
    """Export data to MATLAB MAT file format.

    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data.
    filename : str, optional
        Path to the output MAT file. If None, uses original filename.
    sseg : int, optional
        Segment index to export, default is 0.

    Returns
    -------
    bool
        True if export was successful, False otherwise.
    """
    # Validate segment index
    if not isinstance(sseg, int):
        logger.warning("Selected segment id must be an integer.")
        return False

    if sseg >= pydas_obj.__segN__:
        logger.warning(f"Segment {sseg} exceeds the maximum segment number ({pydas_obj.__segN__ - 1}).")
        return False

    # Create dictionary with data to export
    data_dic = {
        'Data': pydas_obj.data[sseg].values,
        'chName': pydas_obj.chInfo['Name'].values,
        'chUnit': pydas_obj.chInfo['Unit'].values,
        'Date': pydas_obj.__date__,
        'fs': pydas_obj.__fs__,
        'chN': pydas_obj.__chN__,
        'Readme': 'Generated by PyDAS from python, SKLOE/SJTU'
    }

    # Prepare file path and name
    if filename is None:
        path = os.getcwd()
        if pydas_obj.__filename__:
            mat_filename = f"{path}/{os.path.splitext(pydas_obj.__filename__)[0]}.mat"
        else:
            mat_filename = f"{path}/pydas_export.mat"
    else:
        mat_filename = filename

    # Save to MAT file
    try:
        sio.savemat(mat_filename, data_dic)
        logger.info(f"Data exported to: {mat_filename}")
        return True
    except Exception as e:
        logger.error(f"Error exporting to MAT file: {str(e)}")
        return False


def export_to_parquet(pydas_obj, sseg='all', compression='zstd', compression_level=9):
    """Export data to Apache Parquet file format.

    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data.
    sseg : int, list, or 'all', optional
        Segment(s) to export, default is 'all'.
    compression : str, optional
        Compression type ('snappy', 'gzip', 'brotli', 'zstd', 'lz4', 'none').
        Default is 'zstd'.
    compression_level : int, optional
        Compression level for 'gzip', 'brotli', and 'zstd'. Default is 9.

    Returns
    -------
    bool
        True if export was successful, False otherwise.
    """
    # Determine which segments to export
    if sseg == 'all':
        segments = list(range(pydas_obj.__segN__))
    elif isinstance(sseg, int):
        if sseg < pydas_obj.__segN__:
            segments = [sseg]
        else:
            logger.warning(f"Segment {sseg} exceeds the maximum segment number ({pydas_obj.__segN__ - 1}).")
            return False
    elif isinstance(sseg, list):
        segments = [s for s in sseg if s < pydas_obj.__segN__]
        if len(segments) != len(sseg):
            logger.warning("Some segment indices were invalid and will be skipped.")
    else:
        logger.warning("Invalid segment selection. Use an integer, list, or 'all'.")
        return False

    # Verify compression options
    valid_compressions = ['snappy', 'gzip', 'brotli', 'zstd', 'lz4', 'none']
    if compression not in valid_compressions:
        logger.warning(f"Invalid compression type. Using 'zstd' instead. Valid options are: {valid_compressions}")
        compression = 'zstd'

    # Get base filename without extension
    path = os.getcwd()
    base_filename = os.path.splitext(pydas_obj.__filename__)[0]

    success = True
    for idx in segments:
        parquet_filename = f"{path}/{base_filename}_seg{idx:02d}.parquet"
        try:
            metadata = {
                'date': pydas_obj.__date__,
                'fs': pydas_obj.__fs__,
                'chN': pydas_obj.__chN__,
                'scale': pydas_obj.__scale__,
                'desc': pydas_obj.__desc__,
                'segment_type': pydas_obj.segInfo['Type'].iloc[idx],
                'segment_start': pydas_obj.segInfo['Start'].iloc[idx],
                'segment_stop': pydas_obj.segInfo['Stop'].iloc[idx],
                'segment_note': pydas_obj.segInfo['Note'].iloc[idx],
                'n_sample': pydas_obj.segInfo['N sample'].iloc[idx],
                'source': 'PyDAS from SKLOE/SJTU',
                'channel_units': dict(zip(pydas_obj.chInfo['Name'], pydas_obj.chInfo['Unit']))
            }
            df = pydas_obj.data[idx].copy()

            compression_args = {'compression': compression}
            if compression in ['gzip', 'brotli', 'zstd']:
                compression_args['compression_level'] = compression_level

            df.to_parquet(
                parquet_filename,
                engine='pyarrow',
                index=False,
                **compression_args
            )

            # Save metadata separately as JSON
            metadata_filename = f"{path}/{base_filename}_seg{idx:02d}_metadata.json"
            pd.Series(metadata).to_json(metadata_filename)
            logger.info(f"Data exported to: {parquet_filename}")
            logger.info(f"Metadata exported to: {metadata_filename}")
        except Exception as e:
            logger.error(f"Error exporting to Parquet file: {str(e)}")
            success = False

    return success


def export_to_feather(pydas_obj, sseg='all', compression='zstd'):
    """Export data to Feather file format.

    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data.
    sseg : int, list, or 'all', optional
        Segment(s) to export, default is 'all'.
    compression : str or None, optional
        Compression type ('zstd', 'lz4', 'uncompressed'). Default is 'zstd'.

    Returns
    -------
    bool
        True if export was successful, False otherwise.
    """
    # Determine which segments to export
    if sseg == 'all':
        segments = list(range(pydas_obj.__segN__))
    elif isinstance(sseg, int):
        if sseg < pydas_obj.__segN__:
            segments = [sseg]
        else:
            logger.warning(f"Segment {sseg} exceeds the maximum segment number ({pydas_obj.__segN__ - 1}).")
            return False
    elif isinstance(sseg, list):
        segments = [s for s in sseg if s < pydas_obj.__segN__]
        if len(segments) != len(sseg):
            logger.warning("Some segment indices were invalid and will be skipped.")
    else:
        logger.warning("Invalid segment selection. Use an integer, list, or 'all'.")
        return False

    # Verify compression options
    valid_compressions = ['zstd', 'lz4', 'uncompressed', None]
    if compression not in valid_compressions:
        logger.warning(f"Invalid compression type. Using 'zstd' instead. Valid options are: {valid_compressions}")
        compression = 'zstd'

    # Get base filename without extension
    path = os.getcwd()
    base_filename = os.path.splitext(pydas_obj.__filename__)[0]

    success = True
    for idx in segments:
        feather_filename = f"{path}/{base_filename}_seg{idx:02d}.feather"
        try:
            df = pydas_obj.data[idx].copy()
            metadata = {
                'date': pydas_obj.__date__,
                'fs': pydas_obj.__fs__,
                'chN': pydas_obj.__chN__,
                'scale': pydas_obj.__scale__,
                'desc': pydas_obj.__desc__,
                'segment_type': pydas_obj.segInfo['Type'].iloc[idx],
                'segment_start': pydas_obj.segInfo['Start'].iloc[idx],
                'segment_stop': pydas_obj.segInfo['Stop'].iloc[idx],
                'segment_note': pydas_obj.segInfo['Note'].iloc[idx],
                'n_sample': pydas_obj.segInfo['N sample'].iloc[idx],
                'source': 'PyDAS from SKLOE/SJTU',
                'channel_units': dict(zip(pydas_obj.chInfo['Name'], pydas_obj.chInfo['Unit']))
            }

            # Add metadata as additional columns with prefix 'metadata_'
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    df[f"__metadata_{key}__"] = value

            # Save to feather file with specified compression
            if compression == 'uncompressed':
                compression = None
            df.to_feather(feather_filename, compression=compression)
            logger.info(f"Data exported to: {feather_filename}")

            # Save complex metadata as separate JSON
            metadata_filename = f"{path}/{base_filename}_seg{idx:02d}_metadata.json"
            pd.Series(metadata).to_json(metadata_filename)
            logger.info(f"Full metadata exported to: {metadata_filename}")
        except Exception as e:
            logger.error(f"Error exporting to Feather file: {str(e)}")
            success = False

    return success


def export_to_hdf5(pydas_obj, filename=None, sseg='all', compression='gzip',
                   compression_opts=9, include_metadata=True,
                   chunks=True, fletcher32=True):
    """Export PyDAS data to HDF5 format.

    Parameters
    ----------
    pydas_obj : PyDAS
        PyDAS object containing the data.
    filename : str, optional
        Output filename, if None, an auto-generated name will be used.
    sseg : int, list, or 'all', optional
        Segment(s) to export, default is 'all'.
    compression : str, optional
        Compression algorithm ('gzip', 'lzf', 'szip' or None). Default is 'gzip'.
    compression_opts : int, optional
        Compression options, for gzip 0-9 (9 highest compression).
    include_metadata : bool, optional
        Whether to include metadata, default is True.
    chunks : bool or tuple, optional
        Chunking strategy, True for auto-chunking.
    fletcher32 : bool, optional
        Whether to apply Fletcher32 checksum.

    Returns
    -------
    bool
        True if export was successful, False otherwise.
    """
    try:
        import h5py
    except ImportError as e:
        logger.error(f"Export to HDF5 failed: Missing required library - {str(e)}")
        logger.error("Please install h5py: pip install h5py")
        return False

    # Check if there is data to export
    if not hasattr(pydas_obj, 'data') or len(pydas_obj.data) == 0:
        logger.error("No data to export")
        return False

    # Auto-generate filename if not specified
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pydas_export_{timestamp}.h5"

    # Ensure filename has correct extension
    if not filename.endswith('.h5') and not filename.endswith('.hdf5'):
        filename += '.h5'

    # Determine which segments to export
    if sseg == 'all':
        segments = list(range(pydas_obj.__segN__))
    elif isinstance(sseg, int):
        if sseg < pydas_obj.__segN__:
            segments = [sseg]
        else:
            logger.warning(f"Segment {sseg} exceeds the maximum segment number ({pydas_obj.__segN__ - 1}).")
            return False
    elif isinstance(sseg, list):
        segments = [s for s in sseg if s < pydas_obj.__segN__]
        if len(segments) != len(sseg):
            logger.warning("Some segment indices were invalid and will be skipped.")
    else:
        logger.warning("Invalid segment selection. Use an integer, list, or 'all'.")
        return False

    try:
        with h5py.File(filename, 'w') as f:
            grp_data = f.create_group('data')

            # Export each data segment
            for i in segments:
                if i >= len(pydas_obj.data):
                    logger.warning(f"Segment index {i} out of range, skipping")
                    continue

                seg_group = grp_data.create_group(f'segment_{i}')
                df = pydas_obj.data[i]

                # Save each channel's data
                for col in df.columns:
                    data = df[col].values
                    seg_group.create_dataset(
                        col, data=data,
                        compression=compression,
                        compression_opts=compression_opts,
                        chunks=chunks,
                        fletcher32=fletcher32
                    )

                # Add time index if available
                if hasattr(df, 'index') and not df.index.empty:
                    seg_group.create_dataset(
                        'time_index', data=df.index.values,
                        compression=compression,
                        compression_opts=compression_opts
                    )

            # Save metadata if requested
            if include_metadata:
                meta_group = f.create_group('metadata')

                # Save channel information
                if hasattr(pydas_obj, 'chInfo'):
                    ch_group = meta_group.create_group('channel_info')
                    for col in pydas_obj.chInfo.columns:
                        if pydas_obj.chInfo[col].dtype.kind in ['U', 'O']:
                            data = np.array(pydas_obj.chInfo[col].values, dtype='S100')
                        else:
                            data = pydas_obj.chInfo[col].values
                        ch_group.create_dataset(col, data=data)

                # Save segment information
                if hasattr(pydas_obj, 'segInfo'):
                    seg_info_group = meta_group.create_group('segment_info')
                    for col in pydas_obj.segInfo.columns:
                        if pydas_obj.segInfo[col].dtype.kind in ['U', 'O']:
                            data = np.array(pydas_obj.segInfo[col].values, dtype='S100')
                        else:
                            data = pydas_obj.segInfo[col].values
                        seg_info_group.create_dataset(col, data=data)

                if hasattr(pydas_obj, '__fs__'):
                    meta_group.attrs['sampling_rate'] = pydas_obj.__fs__
                if hasattr(pydas_obj, '__lam__'):
                    meta_group.attrs['scale_factor'] = pydas_obj.__lam__

                meta_group.attrs['export_time'] = datetime.datetime.now().isoformat()
                meta_group.attrs['pydas_version'] = getattr(pydas_obj, '__version__', '1.0.0')

                # Save channel units as attributes
                if hasattr(pydas_obj, 'chInfo'):
                    for i, ch_name in enumerate(pydas_obj.chInfo['Name']):
                        ch_unit = pydas_obj.chInfo['Unit'].iloc[i]
                        meta_group.attrs[f'channel_unit_{ch_name}'] = ch_unit

            # Add compression info
            f.attrs['compression'] = str(compression)
            f.attrs['compression_level'] = compression_opts

        file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        logger.info(f"Successfully exported data to HDF5 file: {filename} (size: {file_size:.2f} MB)")
        return True
    except Exception as e:
        logger.error(f"Export to HDF5 failed: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return False