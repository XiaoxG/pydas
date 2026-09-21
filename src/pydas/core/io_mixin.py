"""PyDAS Core - IO Mixin"""
import os
import struct
import numpy as np
import pandas as pd
import logging

from scipy.signal import correlate
from scipy.spatial.transform import Rotation as R

from ..output import (
    export_to_dat, export_to_mat, write_data, export_to_feather,
    export_to_parquet, export_to_hdf5
)
from ..utils import diff1d, data_change_fs
from .io_format import (
    CH_NAME_WIDTH,
    CH_UNIT_WIDTH,
    FILE_HEADER_SIZE,
    align_offset,
    unpack_channel_names,
    unpack_channel_units,
    unpack_file_header,
    unpack_seg_header,
)

logger = logging.getLogger(__name__)

class IOMixin:
    def __read__(self, sseg):
        """
        Read data from the *.out file.
        
        This method reads the file header, channel information, and data segments.
        It populates the object's properties with the read data.
        
        Parameters:
        -----------
        sseg : int or 'all'
            Selected segment number to load, or 'all' to load all segments
        """
        with open(self.__filename__, 'rb') as fIn:
            # Read file header (256 bytes)
            buf = fIn.read(FILE_HEADER_SIZE)
            if not buf:
                logger.warning(f"Reading data file {self.__filename__} failed, exiting...")
                return None
                
            header = unpack_file_header(buf)
            index = header["index"]
            self.__chN__ = header["chN"]
            self.__fs__ = header["fs"]
            self.__segN__ = header["segN"]
            self.__date__ = f'{header["date_mm"]}-{header["date_dd"]}'
            self.__desc__ = header["desc"]

            # Read channel names (16 bytes per channel)
            chName = unpack_channel_names(
                fIn.read(self.__chN__ * CH_NAME_WIDTH), self.__chN__
            )
            
            # Read channel units (4 bytes per channel)
            chUnit = unpack_channel_units(
                fIn.read(self.__chN__ * CH_UNIT_WIDTH), self.__chN__
            )
            
            # Read channel coefficients (4 bytes per channel)
            chCoef = struct.unpack('=' + self.__chN__ * 'f',
                                   fIn.read(self.__chN__ * 4))

            # Read channel IDs if available
            if (index < -1):
                chIdx = struct.unpack(
                    '=' + self.__chN__ * 'h', fIn.read(self.__chN__ * 2))
            else:
                chIdx = list(range(1, self.__chN__ + 1))

            # Create channel information dictionary and DataFrame
            chInfoDict = {'Index': chIdx, 'Name': chName, 'Unit': chUnit,
                          'Coef': chCoef}
            column = ['Name', 'Unit', 'Coef']
            self.chInfo = pd.DataFrame(chInfoDict, columns=column)
            self.chInfo.index = range(1, self.__chN__ + 1)

            # Initialize arrays for segment data
            sampNum = [0] * self.__segN__  # Number of samples in each segment
            segInfo = [[] for _ in range(self.__segN__)]  # Segment information
            segStatis = [[] for _ in range(self.__segN__)]  # Statistical values
            dataRaw = [[] for _ in range(self.__segN__)]  # Raw data
            note = [[] for _ in range(self.__segN__)]  # Notes for each segment
            
            # Record on-disk offsets for later memmap reads
            segment_positions = []
            segment_sizes = []

            # Read segment headers first; defer sample payloads
            for iseg in range(self.__segN__):
                # Align to 128-byte boundary
                p_cur = fIn.tell()
                aligned_pos = align_offset(p_cur)
                fIn.seek(aligned_pos)
                segment_positions.append(aligned_pos)

                # Read segment information (256 bytes)
                buf = fIn.read(FILE_HEADER_SIZE)
                segInfo[iseg] = unpack_seg_header(buf)

                # Extract segment details
                segChN = segInfo[iseg][1]  # Number of channels in this segment
                sampNum[iseg] = segInfo[iseg][2] - 5  # Number of samples (minus 5)
                note[iseg] = segInfo[iseg][11].decode('utf-8').rstrip()  # Segment note

                # Read statistical values for each channel
                fmtstr = '=' + segChN * 'h' + segChN * 'f' + segChN * 2 * 'h'
                buf = fIn.read(segChN * (2 * 3 + 4))
                segStatis[iseg] = struct.unpack(fmtstr, buf)
                
                # Skip the int16 payload for now
                data_pos = fIn.tell()
                data_size = sampNum[iseg] * segChN * 2
                segment_sizes.append(data_size)
                
                # Advance past this segment's samples
                fIn.seek(data_pos + data_size)

        # Second pass: load samples, memmap when large
        with open(self.__filename__, 'rb') as fIn:
            for iseg in range(self.__segN__):
                segChN = segInfo[iseg][1]
                
                # Seek to the sample payload
                fIn.seek(segment_positions[iseg] + 256 + segChN * (2 * 3 + 4))
                
                # Memmap large segments
                if sampNum[iseg] * segChN > 1000000:  # memmap threshold
                    # numpy.memmap of int16 samples
                    mm = np.memmap(self.__filename__, dtype=np.int16, mode='r',
                                  offset=fIn.tell(),
                                  shape=(sampNum[iseg], segChN))
                    # Copy into RAM so the file can be closed
                    dataRaw[iseg] = np.array(mm, dtype=np.int16)
                    # Drop the memmap handle
                    del mm
                else:
                    # Small segments: read into an array
                    dataRaw[iseg] = np.frombuffer(
                        fIn.read(sampNum[iseg] * segChN * 2),
                        dtype=np.int16
                    ).reshape((sampNum[iseg], segChN))

        # Process segment information
        segType = []
        startTime = []
        stopTime = []
        index = []
        duration = []
        
        for n in range(self.__segN__):
            # Segment type: 0-sampling, 1-pre-calibration, 2-post-calibration
            segType.append(segInfo[n][0])
            
            # Format start and stop times
            startTime.append('{0:02d}:{1:02d}:{2:02d}.{3:1d}'.format(
                segInfo[n][6], segInfo[n][5], segInfo[n][4], segInfo[n][3]))
            stopTime.append('{0:02d}:{1:02d}:{2:02d}.{3:1d}'.format(
                segInfo[n][10], segInfo[n][9], segInfo[n][8], segInfo[n][7]))
            
            # Segment index and duration
            index.append(f'Seg{n:2d}')
            duration.append(f'{(sampNum[n] - 1) / self.__fs__:8.1f}s')
        
        # Create segment information DataFrame
        segInfoDict = {
            'Type': segType,
            'Start': startTime,
            'Stop': stopTime,
            'Duration': duration,
            'N sample': sampNum,
            'Note': note
        }
        column = ['Type', 'Start', 'Stop', 'Duration', 'N sample', 'Note']
        segInfo = pd.DataFrame(segInfoDict, index=index, columns=column)

        # Convert header stats and payloads (threaded)
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing
        
        # Worker count
        num_cores = min(multiprocessing.cpu_count(), self.__segN__)
        
        # Per-segment statistics table
        def process_statistics(iseg):
            segStatis_temp = np.reshape(
                np.array(segStatis[iseg], dtype='float64'),
                (4, self.__chN__)).transpose()
            
            for m in range(self.__chN__):
                segStatis_temp[m] *= chCoef[m]
            
            # Create DataFrame with statistics
            column = ['Mean', 'STD', 'Max', 'Min']
            stats_df = pd.DataFrame(
                segStatis_temp, index=chName, columns=column)
            stats_df['Unit'] = chUnit
            return stats_df
        
        # Per-segment sample DataFrame
        def process_raw_data(iseg):
            # Cast to float64
            data_temp = dataRaw[iseg].astype('float64')
            
            # Apply channel coefficients by broadcasting
            coef_array = np.array(chCoef, dtype='float64')
            data_temp = data_temp * coef_array
            
            # Column names are channel names
            return pd.DataFrame(data_temp, columns=chName, dtype='float64')
        
        # Statistics tables
        self.segStatis = [None] * self.__segN__
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            for iseg, result in enumerate(executor.map(process_statistics, range(self.__segN__))):
                self.segStatis[iseg] = result
        
        # Sample tables
        self.data = [None] * self.__segN__
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            for iseg, result in enumerate(executor.map(process_raw_data, range(self.__segN__))):
                self.data[iseg] = result

        # Handle segment selection
        if sseg == 'all':
            self.segInfo = segInfo
        else:
            # If a specific segment is selected, keep only that segment
            self.__segN__ = 1
            self.segInfo = segInfo[sseg:sseg + 1]
            self.segInfo = segInfo[sseg:sseg + 1].rename(index={'Seg{0:2d}'.format(sseg): 'Seg 0'})
            self.segStatis = [self.segStatis[sseg]]
            self.data = [self.data[sseg]]

        return None

    def write(self, filename, sseg='all', ch='all'):
        """
        Write data to a new *.out file.
        
        Parameters:
        -----------
        filename : str
            Path to the output *.out file
        sseg : int, list, or 'all', optional
            Segment(s) to write to the file, default is 'all'
        ch : list or 'all', optional
            Channels to write to the file, default is 'all'
        
        Notes:
        ------
        This method will automatically append '.out' extension if not provided.
        """
        return write_data(self, filename, sseg, ch)

    def to_dat(self, Time=True, sseg='all'):
        """
        Export data to DAT file format.
        
        Parameters:
        -----------
        Time : bool, optional
            If True, include time column in the output, default is True
        sseg : int or 'all', optional
            Segment(s) to export, default is 'all'
            
        Notes:
        ------
        The output file will be named based on the original filename with
        segment number and scale (model or full) appended.
        """
        return export_to_dat(self, Time, sseg)

    def to_mat(self, filename=None, sseg=0):
        """Export data to MATLAB MAT file format.

        Parameters
        ----------
        filename : str, optional
            Output path. When *None*, a name is derived from the original file.
        sseg : int, optional
            Segment index to export, default is 0.

        Returns
        -------
        bool
            True if export was successful, False otherwise.
        """
        return export_to_mat(self, filename=filename, sseg=sseg)

    def to_feather(self, sseg='all', compression='zstd'):
        """
        Export data to feather file format

        Parameters:
        -----------
        sseg : int, list, or 'all', optional
            Segment(s) to export
        compression : str, optional
            Compression to use, default is 'zstd', other options include 'lz4' and 'uncompressed'

        Returns:
        --------
        bool
            True if export was successful
        """
        return export_to_feather(self, sseg, compression)

    def to_parquet(self, sseg='all', compression='zstd', compression_level=9):
        """
        Export data to Apache Parquet file format.
        
        Parameters:
        -----------
        sseg : int or 'all', optional
            Segment index to export, default is 'all'
        compression : str, optional
            Compression type to use. Options include: 'snappy', 'gzip', 'brotli', 'zstd', 'lz4', 'none'
            Default is 'zstd'
        compression_level : int, optional
            Compression level for 'gzip', 'brotli', and 'zstd' compressors
            Default is 9
            
        Returns:
        --------
        bool
            True if export was successful
        """
        return export_to_parquet(self, sseg, compression, compression_level)

    def to_hdf5(self, filename=None, sseg='all', compression='gzip', 
               compression_opts=9, include_metadata=True):
        """
        Export data to HDF5 file format.
        
        Parameters:
        -----------
        filename : str, optional
            Output filename, if None, an auto-generated name will be used
        sseg : int or 'all', optional
            Segment index to export, default is 'all'
        compression : str, optional
            Compression algorithm, options include 'gzip', 'lzf', 'szip' or None
            Default is 'gzip'
        compression_opts : int, optional
            Compression options, for gzip 0-9 (9 highest compression)
        include_metadata : bool, optional
            Whether to include metadata, default is True
            
        Returns:
        --------
        bool
            True if export was successful
        """
        return export_to_hdf5(self, filename, sseg, compression, 
                             compression_opts, include_metadata)

    def read_waveCal(self, wavefname, sseg=0, YBname='YBS', YBcalname='YBS', alignFlag=True):
        """
        Read wave calibration data.
        
        Parameters:
        -----------
        wavefname : str
            Path to wave calibration file
        sseg : int, optional
            Segment index to process, default is 0
        YBname : str, optional
            Name of wave gauge channel, default is 'YBS'
        YBcalname : str, optional
            Name of calibration wave gauge channel, default is 'YBS'
        alignFlag : bool, optional
            Whether to align data, default is True
            
        Notes:
        ------
        - Reads wave calibration data from file
        - Supports data alignment
        - Handles multiple wave gauges
        - Updates channel information
        """
        from .pydas_obj import PyDAS as _PyDAS
        wavecase_cal = _PyDAS(wavefname, lam = self.__lam__)
        fs_cal = wavecase_cal.__fs__
        nch = wavecase_cal.data[0].shape[1]
        for i in range(nch):
            iname = wavecase_cal.chInfo['Name'].loc[i+1]
            iunit = wavecase_cal.chInfo['Unit'].loc[i+1]
            icoef = wavecase_cal.chInfo['Coef'].loc[i+1]
            iseries = wavecase_cal.data[0][iname].values
            self.add_channel('Cal.'+iname, iunit, iseries, fs_cal, coef=icoef, point_of_move=0, sseg=sseg)
        if alignFlag:
            for ich in wavecase_cal.chInfo['Name'].values:
                self.move_ccor('Cal.'+ich, 'Cal.'+YBcalname, YBname, sseg=sseg)

        return None

    def read_motion(self, motionfname, alignAccName=None, alignMethod='acc', zerofilename='', lowpassfilter=-1, rotation=True, NameList=['Platform']):
        """
        Read motion data and add as channels.
        
        Parameters:
        -----------
        motionfname : str
            Path to motion data file
        alignAccName : str, optional
            Acceleration channel name for alignment, default is None
        alignMethod : str, optional
            Alignment method (``'acc'``, ``'time'``, or ``'none'``),
            default is ``'acc'``.
        zerofilename : str, optional
            Path to zero reference file, default is ''
        lowpassfilter : float, optional
            Lowpass filter cutoff frequency, default is -1 (no filter)
        rotation : bool, optional
            Whether to apply rotation, default is True
        NameList : list of str, optional
            List of object names to process, default is ['Platform']
            
        Notes:
        ------
        - Reads motion data from file
        - Supports data alignment and filtering
        - Handles coordinate transformations
        - Updates channel information
        """
        try:
            # Try to open and read the motion file
            with open(motionfname, 'r') as f:
                try:
                    f.seek(0)
                    lines = f.readlines()
                    # Parse header information
                    try:
                        n_body = int(lines[2].replace('\n', '').split('\t')[1])
                        # n_frames = int(lines[0].replace('\n', '').split('\t')[1])
                        motion_fs = float(lines[3].replace('\n', '').split('\t')[1])
                        Time_start = pd.Timestamp(lines[7].replace('\n', '').split('\t')[1])
                        rotationname = lines[10].replace('\n', '').split('\t')[3:6]
                    except (IndexError, ValueError) as e:
                        logger.error(f"Invalid motion file format: {str(e)}")
                        raise ValueError(f"Motion file format is invalid: {str(e)}")
                except Exception as e:
                    logger.error(f"Error reading motion file: {str(e)}")
                    raise
        except FileNotFoundError:
            logger.error(f"Motion file not found: {motionfname}")
            raise

        motionName = ['Surge','Sway','Heave'] + rotationname
        motionDataRawList = []
        
        # Process zero reference file if provided
        if zerofilename:
            try:
                for ibody in range(n_body):
                    try:
                        # Read zero reference data
                        ZeromotionDataRaw = np.genfromtxt(
                            zerofilename, 
                            skip_header=12, 
                            delimiter='\t', 
                            usecols=(0+ibody*17, 1+ibody*17, 2+ibody*17, 3+ibody*17, 4+ibody*17, 5+ibody*17)
                        )
                        Zeromean = ZeromotionDataRaw.mean(axis=0)
                        Zeromean[3] = 0  # Don't apply zero correction to yaw
                        logger.info(f"Zero reference data: {Zeromean}")
                        # Read motion data and apply zero correction
                        motionDataRawList.append(
                            np.genfromtxt(
                                motionfname, 
                                skip_header=12, 
                                delimiter='\t', 
                                usecols=(0+ibody*17, 1+ibody*17, 2+ibody*17, 3+ibody*17, 4+ibody*17, 5+ibody*17)
                            ) - Zeromean
                        )
                    except Exception as e:
                        logger.error(f"Error processing data for body {ibody}: {str(e)}")
                        raise ValueError(f"Failed to process motion data for body {ibody}")
            except FileNotFoundError:
                logger.error(f"Zero reference file not found: {zerofilename}")
                raise
        else:
            # Read motion data without zero correction
            try:
                for ibody in range(n_body):
                    motionDataRawList.append(
                        np.genfromtxt(
                            motionfname, 
                            skip_header=12, 
                            delimiter='\t', 
                            usecols=(0+ibody*17, 1+ibody*17, 2+ibody*17, 3+ibody*17, 4+ibody*17, 5+ibody*17)
                        )
                    )
            except Exception as e:
                logger.error(f"Error reading motion data: {str(e)}")
                raise ValueError(f"Failed to read motion data: {str(e)}")

        # Process each body's motion data
        for ibody, motionDataRaw in enumerate(motionDataRawList):
            try:
                # Convert units from mm to cm for position data
                motionDataRaw[:, 0:3] /= 10
                
                # Apply rotation if requested
                if rotation:
                    try:
                        Yaw = np.mean(motionDataRaw[:, motionName.index('Yaw')])
                        # Yaw = 180
                        r = R.from_euler('z', Yaw, degrees=True)
                        motionDataRaw[:, 0:3] = r.apply(motionDataRaw[:, 0:3])
                        logger.info(f'motion rotated: {Yaw: 0.2f} DEG')
                    except Exception as e:
                        logger.warning(f"Failed to apply rotation: {str(e)}")
                        # Continue without rotation

                # Add channels for each motion component
                unit = ['cm']*3 + ['deg']*3
                for i, iName in enumerate(motionName):
                    try:
                        self.add_channel(
                            NameList[ibody] + '.' + iName, 
                            unit[i], 
                            motionDataRaw[:, i], 
                            fs=motion_fs
                        )
                    except Exception as e:
                        logger.error(f"Failed to add channel {NameList[ibody]}.{iName}: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing motion data for body {ibody}: {str(e)}")
                # Continue with next body
        
        # Align motion data with existing data
        try:
            if alignMethod == 'acc':
                try:
                    # Calculate acceleration from heave motion
                    heave = self.apply_lowpass_filter(NameList[0]+'.Heave', replace=False, returnValue=True)
                    vz = diff1d(heave/100, 1 / self.__fs__)
                    az = diff1d(vz, 1 / self.__fs__) * -1
                    n_sample = self.segInfo.iloc[0]['N sample']
                    
                    # Find correlation with acceleration channel
                    base = self.apply_lowpass_filter(alignAccName, replace=False, returnValue=True)
                    lag = np.argmax(correlate(base, az, method='fft')) - n_sample + 1
                    
                    # Add calculated acceleration channel
                    self.add_channel('azfromHeave', unit='m/s2', series=az, point_of_move=lag, fs=self.__fs__)
                    
                    # Move all motion channels by the calculated lag
                    for ibody in NameList:
                        for iname in motionName:
                            try:
                                self.move_data(ibody+'.'+iname, point_of_move=lag)
                            except Exception as e:
                                logger.warning(f"Failed to move channel {ibody}.{iname}: {str(e)}")
                except Exception as e:
                    logger.error(f"Failed to align using acceleration method: {str(e)}")
                    logger.warning("Continuing without alignment")
            elif alignMethod == 'time':
                try:
                    # Calculate time difference and convert to sample points
                    das_start = self.segInfo['Start'].iloc[0]
                    das_date = self.__date__ or Time_start.strftime('%m-%d')
                    das_ts = pd.Timestamp(
                        f"{Time_start.year}-{das_date} {das_start}"
                    )
                    timedelta = Time_start - das_ts
                    lag = round(timedelta.total_seconds() * self.__fs__)
                    
                    # Move all motion channels by the calculated lag
                    for ibody in NameList:
                        for iname in motionName:
                            try:
                                self.move_data(ibody+'.'+iname, point_of_move=lag)
                            except Exception as e:
                                logger.warning(f"Failed to move channel {ibody}.{iname}: {str(e)}")
                except Exception as e:
                    logger.error(f"Failed to align using time method: {str(e)}")
                    logger.warning("Continuing without alignment")
            elif alignMethod != 'none':
                logger.warning(f"Unknown alignment method: {alignMethod}. No alignment applied.")
        except Exception as e:
            logger.error(f"Error during alignment: {str(e)}")
            logger.warning("Continuing without alignment")

        # Apply lowpass filter if requested
        if lowpassfilter > 0:
            try:
                for ibody in NameList:
                    for iname in motionName:
                        try:
                            self.apply_lowpass_filter(ibody+'.'+iname, cutoffull=lowpassfilter)
                        except Exception as e:
                            logger.warning(f"Failed to apply filter to {ibody}.{iname}: {str(e)}")
            except Exception as e:
                logger.error(f"Error applying lowpass filter: {str(e)}")
                logger.warning("Continuing without filtering")

        return None

