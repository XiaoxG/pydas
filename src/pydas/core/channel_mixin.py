"""PyDAS Core - Channel Mixin"""
import numpy as np
import pandas as pd
import logging

from ..utils import data_change_fs

logger = logging.getLogger(__name__)

class ChannelMixin:
    def add_channel(self, name, unit, series, fs, coef=1, point_of_move=0, sseg=0):
        """
        Add a new channel to the data.
        
        Parameters:
        -----------
        name : str
            Name of the new channel
        unit : str
            Unit of measurement
        series : numpy.ndarray
            Channel data
        fs : float
            Sampling frequency in Hz
        coef : float, optional
            Coefficient for data scaling, default is 1
        point_of_move : int, optional
            Number of points to shift the data, default is 0
        sseg : int, optional
            Segment index to add channel to, default is 0
            
        Notes:
        ------
        - Validates input data and parameters
        - Handles data alignment and scaling
        - Updates channel information and statistics
        """
        if name not in self.chInfo['Name'].values:
            n_sample = int(self.segInfo.iloc[sseg]['N sample'])
            # First channel on an empty object defines length and sampling rate.
            if n_sample == 0:
                n_sample = len(series)
                self.segInfo.iloc[sseg, self.segInfo.columns.get_loc('N sample')] = n_sample
                self.__fs__ = fs
            elif fs != self.__fs__:
                series = data_change_fs(series, fs, self.__fs__)

            if len(series) > n_sample:
                series = series[:n_sample]
            elif len(series) < n_sample:
                series = np.pad(series, (0, n_sample - len(series)), 'constant', constant_values=0)
            
            # Add to data
            self.data[sseg][name] = series
            
            # Update channel info
            new_idx = self.__chN__ + 1
            # Build row data by column name to support dynamic chInfo schemas.
            # After to_fullscale(), chInfo may contain additional coefficient columns.
            new_row = {col: np.nan for col in self.chInfo.columns}
            if 'Name' in new_row:
                new_row['Name'] = name
            if 'Unit' in new_row:
                new_row['Unit'] = unit
            if 'Coef' in new_row:
                new_row['Coef'] = coef
            if 'CoeffUnit' in new_row:
                new_row['CoeffUnit'] = 1.0
            if 'CoeffRho' in new_row:
                new_row['CoeffRho'] = 0.0
            if 'CoeffLam' in new_row:
                new_row['CoeffLam'] = 0.0
            self.chInfo.loc[new_idx] = new_row
            
            # Update statistics
            self.segStatis[sseg].loc[name] = [
                np.mean(series), np.std(series), np.amax(series), np.amin(series), unit]
                
            # Update channel count
            self.__chN__ += 1
            
            # Move data if requested
            if point_of_move != 0:
                self.move_data(name, point_of_move, sseg=sseg)
                
            logger.info(f"Channel '{name}' has been added")
        else:
            logger.warning(f"Channel '{name}' already exists.")

    def delete_channel(self, name):
        """
        Delete a specified channel from the data.
        
        Parameters:
        -----------
        name : str
            Name of the channel to delete
            
        Notes:
        ------
        - Removes channel from all data segments
        - Updates channel information
        - Recalculates statistics
        """
        if name in self.chInfo['Name'].values:
            # Find the index of the channel
            idx = self.chInfo.index[self.chInfo['Name'] == name].tolist()[0]
            
            # Remove from channel info
            self.chInfo = self.chInfo.drop(idx)
            
            # Remove from data in all segments
            for sseg in range(self.__segN__):
                self.data[sseg] = self.data[sseg].drop(name, axis=1)
                self.segStatis[sseg] = self.segStatis[sseg].drop(name)
            
            # Update channel count
            self.__chN__ -= 1
            
            # Reset index
            self.chInfo.index = range(1, self.__chN__ + 1)
            
            logger.info(f"Channel '{name}' has been removed")
        else:
            logger.warning(f"Channel '{name}' does not exist.")

    def select_channels(self, chnames):
        """
        Select and keep only specified channels, removing others.
        
        Parameters:
        -----------
        chnames : str or list of str
            Channel name(s) to keep. Can be a single string for one channel
            or a list of strings for multiple channels.
            
        Returns:
        --------
        bool
            True if successful, False otherwise
            
        Notes:
        ------
        - Validates channel names
        - Removes all channels not in the specified list
        - Updates channel information and statistics
        """
        # Check if input is a string (single channel) and convert to list
        if isinstance(chnames, str):
            chnames = [chnames]
            
        # Validate channel names
        valid_chnames = []
        for name in chnames:
            if name in self.chInfo['Name'].values:
                valid_chnames.append(name)
            else:
                logger.warning(f"Channel '{name}' does not exist and will be ignored.")
        
        if not valid_chnames:
            logger.warning("No valid channels specified.")
            return False
            
        # Get indices of channels to keep
        keep_indices = []
        for name in valid_chnames:
            idx = self.chInfo.index[self.chInfo['Name'] == name].tolist()[0]
            keep_indices.append(idx)
            
        # Keep only selected channels in channel info
        self.chInfo = self.chInfo.loc[keep_indices]
        
        # Keep only selected channels in data and statistics
        for sseg in range(self.__segN__):
            self.data[sseg] = self.data[sseg][valid_chnames]
            self.segStatis[sseg] = self.segStatis[sseg].loc[valid_chnames]
            
        # Update channel count and reset indices
        self.__chN__ = len(valid_chnames)
        self.chInfo.index = range(1, self.__chN__ + 1)
        
        logger.info(f"Selected {len(valid_chnames)} channels: {', '.join(valid_chnames)}")
        return True

    def change_channel_order(self,
                     newOrder,
                     sseg=0):
        """
        Change channel order in the data.
        
        Parameters:
        -----------
        newOrder : list of str
            New order of channel names
        sseg : int, optional
            Segment index to process, default is 0
            
        Notes:
        ------
        - Reorders channels in data structure
        - Updates channel information
        - Maintains data integrity
        - Validates channel names
        """
        if len(newOrder) == self.__chN__:
            indexNew = []
            for inewOrder in newOrder:
                indexNew.append(
                    list(
                        self.data[sseg].columns).index(inewOrder) +
                    1)
            self.chInfo = self.chInfo.reindex(indexNew)
            self.segStatis[sseg] = self.segStatis[sseg].reindex(newOrder)
            self.chInfo.index = np.arange(1, len(self.chInfo) + 1)
            self.data[sseg] = self.data[sseg][newOrder]
            self.updateChN()
            logger.info('Changed the Channel order.')
        else:
            raise ValueError("Number of channels does not match!")

    def updateChN(self, sseg=0):
        """
        Update channel count information.
        
        Parameters:
        -----------
        sseg : int, optional
            Segment index to process, default is 0
            
        Notes:
        ------
        - Updates channel count in segment information
        - Validates channel consistency
        - Maintains data integrity
        """
        if self.data[sseg].shape[1] == self.chInfo.shape[0] == self.segStatis[0].shape[0]:
            self.__chN__ = self.chInfo.shape[0]
        else:
            raise ValueError("Number of channels does not match!")

        return None

    def rename_channel(self,
                     chOld,
                     chNew,
                     sseg=0):
        """
        Rename a channel in the dataset.
        
        Parameters:
        -----------
        chOld : str
            Original channel name
        chNew : str
            New channel name
        sseg : int, optional
            Segment index to process, default is 0
            
        Notes:
        ------
        - Updates channel name in data, chInfo and statistics
        - Maintains all data and properties
        """
        try:
            # Check if the old channel exists
            if chOld not in self.data[sseg].columns:
                logger.error(f"Channel '{chOld}' not found in segment {sseg}")
                raise KeyError(f"Channel '{chOld}' not found")
                
            # Check if the new channel name already exists
            if chNew in self.data[sseg].columns:
                logger.error(f"Channel '{chNew}' already exists in segment {sseg}")
                raise ValueError(f"Channel '{chNew}' already exists")
                
            self.data[sseg].rename(columns={chOld: chNew}, inplace=True)
            ch_mask = self.chInfo['Name'] == chOld
            self.chInfo.loc[ch_mask, 'Name'] = chNew
            if chOld in self.segStatis[sseg].index:
                self.segStatis[sseg].rename(index={chOld: chNew}, inplace=True)
            
            logger.info(f"Renamed channel '{chOld}' to '{chNew}' in segment {sseg}")
            return True
            
        except Exception as e:
            logger.error(f"Channel renaming failed: {str(e)}")
            return False

    def copy_channel(self, chName, new_chName=None, sseg='all'):
        """
        Copy an existing channel to create a new channel with the same data.
        
        Parameters:
        -----------
        chName : str
            Name of the channel to copy
        new_chName : str, optional
            Name for the new channel. If None, will use original name + "_copy"
        sseg : int, list, or 'all', optional
            Segment(s) to apply the copy operation, default is 'all'
            
        Returns:
        --------
        bool
            True if copy was successful, False otherwise
            
        Notes:
        ------
        - The copy will have the same unit and coefficient as the original channel
        - If a channel with the new name already exists, it will be overwritten
        """
        # Check if source channel exists
        if chName not in self.chInfo['Name'].values:
            logger.warning(f"Channel '{chName}' does not exist.")
            return False
            
        # Create new channel name if not provided
        if new_chName is None:
            new_chName = f"{chName}_copy"
            
        # Get unit and coefficient of original channel
        idx = self.chInfo.index[self.chInfo['Name'] == chName].tolist()[0]
        unit = self.chInfo.loc[idx, 'Unit']
        coef = self.chInfo.loc[idx, 'Coef']
            
        # Determine which segments to process
        if sseg == 'all':
            segments = list(range(self.__segN__))
        elif isinstance(sseg, int):
            if sseg < self.__segN__:
                segments = [sseg]
            else:
                logger.warning(f"Segment {sseg} exceeds the maximum segment number ({self.__segN__ - 1}).")
                return False
        elif isinstance(sseg, list):
            segments = [s for s in sseg if s < self.__segN__]
            if len(segments) != len(sseg):
                logger.warning("Some segment indices were invalid and will be skipped.")
        else:
            logger.warning("Invalid segment selection. Use an integer, list, or 'all'.")
            return False
            
        # Delete the channel first if it already exists
        if new_chName in self.chInfo['Name'].values:
            logger.warning(f"Channel '{new_chName}' already exists. Operation canceled.")
            return False
            
        # Copy channel data for first segment
        first_seg = segments[0]
        series = self.data[first_seg][chName].copy()
        self.add_channel(new_chName, unit, series, self.__fs__, coef, 0, first_seg)
        
        # For additional segments (if any), manually copy the data
        for seg in segments[1:]:
            if chName in self.data[seg].columns:
                # Copy the data for this segment
                self.data[seg][new_chName] = self.data[seg][chName].copy()
                
                # Update statistics for this segment
                self.segStatis[seg].loc[new_chName] = [
                    np.mean(self.data[seg][new_chName]), 
                    np.std(self.data[seg][new_chName]),
                    np.amax(self.data[seg][new_chName]), 
                    np.amin(self.data[seg][new_chName]), 
                    unit
                ]
                
        logger.info(f"Channel '{chName}' copied to '{new_chName}'")
        return True

