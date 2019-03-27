#!/usr/bin/python3
# -*- coding: utf-8 -*-
import re
import sys
import os
import struct
import math
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import correlate, butter, lfilter, freqz

class CaseData:
    """
    CaseData class for SKLOE
    Please read the readme file.
    """

    def __init__(self, filename, lam=1, sseg='all'):
        """
        @param: filename - input *.out file name
        @param: lam      - scale ratio
        @param: sseg     - selected segment number
        """

        if os.path.exists(filename):
            self.filename = filename
        else:
            print(
                "Error: File {:s} does not exist. Breaking!".format(filename))
            sys.exit()

        self.lam = lam
        self.fs = 1
        self.chN = 0
        self.segN = 0
        self.scale = 'model'
        if not (isinstance(sseg, int) or sseg == 'all'):
            print("Error: Input 'sseg' is illegal (should be int or 'all').")
            raise
        self.read(sseg)

    def read(self, sseg):
        """read the *.out file"""

        print('Reading file {}...'.format(self.filename), end='')

        with open(self.filename, 'rb') as fIn:
            # file head
            fmtstr = '=hhlhh2s2s240s'
            buf = fIn.read(256)
            if not buf:
                warnings.warn("Reading data file {0} failed, exiting...".format(
                    self.filename))
            tmp = struct.unpack(fmtstr, buf)
            index, self.chN, self.fs, self.segN =\
                tmp[0], tmp[1], tmp[3], tmp[4]
            datemm, datedd = tmp[5].decode('utf-8'), tmp[6].decode('utf-8')
            # date mm/dd
            self.date = '{:2s}/{:2s}'.format(datemm, datedd)
            # global description
            self.desc = tmp[7].decode('utf-8').rstrip()

            # read the name of each channel
            chName = [namei.decode('utf-8').rstrip() for namei in
                      struct.unpack(self.chN * '16s', fIn.read(self.chN * 16))]
            # read the unit of each channel
            chUnit = [uniti.decode('utf-8').rstrip() for uniti in
                      struct.unpack(self.chN * '4s', fIn.read(self.chN * 4))]
            # read the coefficient of each channel
            chCoef = struct.unpack('=' + self.chN * 'f',
                                   fIn.read(self.chN * 4))

            # read the id of each channel, if there are
            if (index < -1):
                chIdx = struct.unpack(
                    '=' + self.chN * 'h', fIn.read(self.chN * 2))
            else:
                chIdx = list(range(1, self.chN + 1))

            chInfoDict = {'Index': chIdx, 'Name': chName, 'Unit': chUnit,
                          'Coef': chCoef}

            column = ['Name', 'Unit', 'Coef']
            self.chInfo = pd.DataFrame(chInfoDict, index=chIdx, columns=column)

            # sampNum[i] is the number of samples in segment i
            sampNum = [0 for i in range(self.segN)]
            # segInfo[i] is the information of the segment i
            # [segType, segChN, sampNum, ds, s, min, h, desc]
            segInfo = [[] for i in range(self.segN)]
            # seg_satistic[i] is the satistical values of the segment i
            # [mean[segChN], std[segChN], max[segChN], min[segChN]]
            segStatis = [[] for i in range(self.segN)]
            # dataRaw[i] are the data of the segment i
            dataRaw = [[] for i in range(self.segN)]
            # note for each segment
            note = [[] for i in range(self.segN)]

            # read data of each segment
            for iseg in range(self.segN):
                # jump over the blank section
                p_cur = fIn.tell()
                fIn.seek(128 * math.ceil(p_cur / 128))

                # read segment informantion
                fmtstr = '=hhlBBBBBBBB240s'
                buf = fIn.read(256)
                segInfo[iseg] = struct.unpack(fmtstr, buf)

                segChN = segInfo[iseg][1]
                sampNum[iseg] = segInfo[iseg][2] - 5  # NOTE -5 here
                note[iseg] = segInfo[iseg][11].decode('utf-8').rstrip()

                # read the statiscal values of each channel
                fmtstr = '=' + segChN * 'h' + segChN * 'f' + segChN * 2 * 'h'
                buf = fIn.read(segChN * (2 * 3 + 4))
                segStatis[iseg] = struct.unpack(fmtstr, buf)

                # read the data in each channel
                dataRaw[iseg] = np.frombuffer(fIn.read(sampNum[iseg] *
                                                       segChN * 2),
                                              dtype=np.int16).reshape(
                    (sampNum[iseg], segChN))

        # segment information
        segType = []
        startTime = []
        stopTime = []
        index = []
        duration = []
        for n in range(self.segN):
            # type: 0 - 采样段，1 - 前标定段，2 - 后标定段
            segType.append(segInfo[n][0])
            startTime.append('{0:02d}:{1:02d}:{2:02d}.{3:1d}'.format(
                segInfo[n][6], segInfo[n][5], segInfo[n][4], segInfo[n][3]))
            stopTime.append('{0:02d}:{1:02d}:{2:02d}.{3:1d}'.format(
                segInfo[n][10], segInfo[n][9], segInfo[n][8], segInfo[n][7]))
            index.append('Seg{0:2d}'.format(n))
            duration.append('{0:8.1f}s'.format((sampNum[n] - 1) / self.fs))
        segInfoDict = {'Type': segType,
                       'Start': startTime,
                       'Stop': stopTime,
                       'Duration': duration,
                       'N sample': sampNum,
                       'Note': note}
        column = ['Type', 'Start', 'Stop', 'Duration', 'N sample', 'Note']
        segInfo = pd.DataFrame(segInfoDict, index=index, columns=column)

        # convert the statistics into data matrix
        self.segStatis = []
        for iseg in range(self.segN):
            segStatis_temp = np.reshape(
                np.array(segStatis[iseg], dtype='float64'),
                (4, self.chN)).transpose()
            for m in range(self.chN):
                segStatis_temp[m] *= chCoef[m]
            column = ['Mean', 'STD', 'Max', 'Min']
            self.segStatis.append(pd.DataFrame(
                segStatis_temp, index=chName, columns=column))
            self.segStatis[iseg]['Unit'] = chUnit

        # convert the dataRaw into data matrix
        self.data = [[] for i in range(self.segN)]
        for iseg in range(self.segN):
            data_temp = dataRaw[iseg].astype('float64')
            for m in range(self.chN):
                data_temp[:, m] *= chCoef[m]
            # index = np.arange(segInfo['N sample'].iloc[iseg]) / self.fs
            # self.data[iseg] = pd.DataFrame(
            #     data_temp, index=index, columns=chName, dtype='float64')
            self.data[iseg] = pd.DataFrame(
                 data_temp, columns=chName, dtype='float64')

        if sseg == 'all':
            self.segInfo = segInfo
        else:
            self.segN = 1
            self.segInfo = segInfo[sseg:sseg + 1]
            self.segStatis = [self.segStatis[sseg]]
            self.data = [self.data[sseg]]

        print(" Done!")

    def write(self, filename, sseg='all', ch='all'):
        """write the *.out file"""

        if not filename.endswith('.out'):
            filename += '.out'

        # write data of selected segment(s)
        if sseg == 'all':
            sseg = list(range(self.segN))
        elif type(sseg) is int:
            sseg = [sseg]
        else:
            warnings.warn("Unsupported segment number, using 'all'.")
            sseg = list(range(self.segN))

        print('Saving segment(s) No. {} to file {}'.format(sseg, filename))

        with open(filename, 'wb') as fOut:
            # file head
            datemmdd = self.date.split('/')
            buf = struct.pack('=hhlhh', -2, self.chN, 0x0d, self.fs, len(sseg)) \
                + struct.pack('2s2s240s', datemmdd[0].encode('utf-8'),
                              datemmdd[1].encode('utf-8'),
                              self.desc.encode('utf-8')).replace(b'\x00', b' ')
            if fOut.write(buf) != 256:
                print("Error when saving out file!")
                raise

            # write the name of each channel
            fOut.write(struct.pack(self.chN * '16s',
                                   *[self.chInfo['Name'].iloc[i].encode('utf-8')
                                     for i in range(self.chN)]).replace(b'\x00', b' '))

            # write the unit of each channel
            fOut.write(struct.pack(self.chN * '4s',
                                   *[self.chInfo['Unit'].iloc[i].encode('utf-8')
                                     for i in range(self.chN)]).replace(b'\x00', b' '))

            # write the coefficient of each channel
            # calculate new coefficient for each channel
            # max short: 32767
            chMagMax = np.amax(np.array(
                [np.amax(abs(self.data[i].values), axis=0) for i in sseg]),
                axis=0)
            chCoef_ = (chMagMax / 32767).astype(np.float32)
            fOut.write(struct.pack('=' + self.chN * 'f', *chCoef_))

            # write the index of each channel (write case -2)
            fOut.write(struct.pack('=' + self.chN * 'h', *self.chInfo.index))

            for iseg in sseg:
                # jump over the blank section
                p_cur = fOut.tell()
                fOut.seek(128 * math.ceil(p_cur / 128))

                # write segment informantion
                fOut.write(struct.pack('=h', self.segInfo['Type'][iseg]))
                fOut.write(struct.pack('=h', self.chN))
                fOut.write(struct.pack(
                    '=l', self.segInfo['N sample'][iseg] + 5))
                fOut.write(struct.pack(8 * 'B', *list(
                    map(int, re.split(':|\.', self.segInfo.Start[iseg])[::-1] +
                        re.split(':|\.', self.segInfo.Stop[iseg])[::-1]))))
                fOut.write(struct.pack(
                    '240s', self.segInfo.Note[iseg].encode('utf-8')).replace(b'\x00', b' '))

                # calculate the statistical information of each channel
                # as short
                mean_ = np.mean(self.data[iseg].values, axis=0) / chCoef_
                # as float
                std_ = np.std(self.data[iseg].values, axis=0) / chCoef_
                # as short
                max_ = np.amax(self.data[iseg].values, axis=0) / chCoef_
                min_ = np.amin(self.data[iseg].values, axis=0) / chCoef_
                # write the statistical information of each channel
                fOut.write(struct.pack('=' + self.chN * 'h',
                                       *np.round(mean_).astype(np.int16)))
                fOut.write(struct.pack('=' + self.chN * 'f', *std_))
                fOut.write(struct.pack('=' + self.chN * 'h',
                                       *np.round(max_).astype(np.int16)))
                fOut.write(struct.pack('=' + self.chN * 'h',
                                       *np.round(min_).astype(np.int16)))

                # write the data in each channel
                raw_ = np.round(self.data[iseg].values / np.repeat(chCoef_.reshape(
                    1, -1), self.segInfo['N sample'][iseg], axis=0)).astype(np.int16)
                fOut.write(raw_.tobytes())

    def addCh(self, name, unit, series):
        """Add a channel to the case data
        (currently only one-segment data is supported)."""
        if self.segN > 1:
            warnings.warn(
                "Only support one-segment data file. The data remain unchanged.")
            return

        # check if the data size equals the number of samples
        if self.segInfo.iloc[0]['N sample'] != len(series):
            warnings.warn("DIFFERENT data size! The data remain unchanged.")
            return

        if self.scale == 'model':
            self.chInfo.loc[max(self.chInfo.index) + 1] = [name, unit, 1]
        else:
            self.chInfo.loc[max(self.chInfo.index) +
                            1] = [name, unit, 1, 1, 0, 0]
        self.segStatis[0].loc[name] = [np.mean(series), np.std(series),
                                       np.amax(series), np.amin(series), unit]
        self.data[0].insert(self.chN, name, series)
        chInfo_dic = {'Name': name,
                        'Unit': unit,
                        'Coef': 1}
        Column = ['Name', 'Unit', 'Coef']
        chIdx = '{0:02d}'.format(self.chN + 1)
        addchInfo = pd.DataFrame(chInfo_dic, index=chIdx, columns=Column)
        self.chInfo = pd.concat([self.chInfo, addchInfo])
        self.chN += 1

    def delCh(self, name):
        """Drop channel(s) of the case data"""

        if name not in self.chInfo.Name.values:
            print('Channel name does not exist, returning.')
            return

        self.chInfo = self.chInfo.drop(
            self.chInfo.index[self.chInfo.Name == name])
        for iseg in range(self.segN):
            self.segStatis[iseg] = self.segStatis[iseg].drop(name)
            del self.data[iseg][name]

        # print(self.chInfo)
        # print(self.segStatis[0])
        # print(self.data[0])
        # update the number of channels
        if self.data[0].shape[1] == self.chInfo.shape[0] == self.segStatis[0].shape[0]:
            self.chN = self.chInfo.shape[0]
        else:
            raise ValueError("Number of channels does not match!")

    def pickCh(self, chnames):
        "pick out the channels in chnames and drop the rest"

        for chn in chnames:
            if chn not in self.data[0].columns:
                warnings.warn("Ch-{:s} does not exist. Ignor it".format(chn))
                chnames.remove(chn)

        self.chInfo.set_index('Name', inplace=True)
        self.chInfo = self.chInfo.loc[chnames]
        self.chInfo.reset_index(level='Name', inplace=True)
        self.chN = self.chInfo.shape[0]

        for segi in range(self.segN):
            self.data[segi] = self.data[segi][chnames]
            self.segStatis[segi] = self.segStatis[segi].loc[chnames]

    def pInfo(self, printTxt=False, printExcel=False):
        print('-' * 50)
        print('Segment: {0:2d}; Channel: {1:3d}; Sampling frequency: {2:4d}Hz.'.format(
            self.segN, self.chN, self.fs))
        print(self.segInfo.to_string(justify='center'))
        print('-' * 50)
        path = os.getcwd()
        path += '/' + os.path.splitext(self.filename)[0]
        if printTxt:
            fname = path + '_Info.txt'
            self.segInfo.to_csv(path_or_buf=fname, sep='\t')
        if printExcel:
            fname = path + '_Info.xlsx'
            self.segInfo.to_excel(fname, sheet_name='Sheet01')

    def pChInfo(self, printTxt=False, printExcel=False):
        print('-' * 50)
        print(self.chInfo.to_string(justify='center'))
        print('-' * 50)
        if printTxt:
            path = os.getcwd()
            fname = path + '/' + \
                os.path.splitext(self.filename)[0] + 'ChInfo.txt'
            infoFile = open(fname, 'w')
            infoFile.write('Channel total: {0:3d} \n'.format(self.chN))
            formatters = {'Name': "{:16s}".format,
                          "Unit": "{:4s}".format,
                          "Coef": "{: .7f}".format}
            infoFile.write(self.chInfo.to_string(
                formatters=formatters, justify='center'))
            infoFile.close()
        if printExcel:
            file_name = path + '/' + \
                os.path.splitext(self.filename)[0] + '_ChInfo.xlsx'
            self.chInfo.to_excel(file_name, sheet_name='Sheet01')

    def to_dat(self, sseg='all'):
        def writefile(self, idx):
            path = os.getcwd()
            if self.scale == 'model':
                filename = path + '/' + \
                    os.path.splitext(self.filename)[
                        0] + '_seg{0:02d}-model.txt'.format(idx)
                hearder_fmt_str = 'File: {0:s}, Seg{1:02d}, fs:{2:4d}Hz\nDate: {3:5s} from: {4:8s} to {5:8s}\nNote:{6:s}\n'
                header2write = hearder_fmt_str.format(
                    self.filename, idx, self.fs, self.date,
                    self.segInfo['Start'].iloc[idx], self.segInfo['Stop'].iloc[idx],
                    self.segInfo['Note'].iloc[idx])
            else:
                filename = path + '/' + \
                    os.path.splitext(self.filename)[
                        0] + '_seg{0:02d}-full.txt'.format(idx)
                hearder_fmt_str = 'File: {0:s}, Seg{1:02d}, fs:{2:4d}Hz(model scale)\nDate: {3:5s} from: {4:8s} to {5:8s}\nNote:{6:s}\nFull scale results, scale ratio = {7:02d}\n'
                header2write = hearder_fmt_str.format(
                    self.filename, idx, self.fs, self.date,
                    self.segInfo['Start'].iloc[idx], self.segInfo['Stop'].iloc[idx],
                    self.segInfo['Note'].iloc[idx], self.lam)
            comments = '\t'.join(
                self.chInfo['Name']) + '\n' + '\t'.join(self.chInfo['Unit']) + '\n'
            header2write += comments
            infoFile = open(filename, 'w')
            infoFile.write(header2write)
            data2write = self.data[idx].to_string(header=False,
                                                  index=False, justify='left', float_format='% .5E')
            infoFile.write(data2write)
            infoFile.close()
            print('Export: {0:s}'.format(filename))

        if sseg == 'all':
            for idx in range(self.segN):
                writefile(self, idx)
        elif isinstance(sseg, int):
            if sseg <= self.segN:
                writefile(self, sseg)
            else:
                warnings.warn('seg exceeds the max.')
        else:
            warnings.warn('Input sseg is illegal. (int or defalt)')

    def pst(self, printTxt=False, printExcel=False):
        print('-' * 50)
        print('Segment total: {0:02d}'.format(self.segN))
        for idx, istatictis in enumerate(self.segStatis):
            print('')
            print('Seg{0:02d}'.format(idx))
            print(istatictis.to_string(float_format='% .3E', justify='center'))
            print('')
        print('-' * 50)
        path = os.getcwd()
        if printTxt:
            file_name = path + '/' + \
                os.path.splitext(self.filename)[0] + '_statistic.txt'
            infoFile = open(file_name, 'w')
            infoFile.write('Segment total: {0:02d}\n'.format(self.segN))
            for idx, istatictis in enumerate(self.segStatis):
                infoFile.write('\n')
                infoFile.write('Seg{0:02d}\n'.format(idx))
                infoFile.write(istatictis.to_string(
                    float_format='% .3E', justify='center'))
            infoFile.close()
            print('Export: {0:s}'.format(file_name))
        if printExcel:
            file_name = path + '/' + \
                os.path.splitext(self.filename)[0] + '_statistic.xlsx'
            for idx, istatictis in enumerate(self.segStatis):
                istatictis.to_excel(
                    file_name, sheet_name='SEG{:02d}'.format(idx))
            print('Export: {0:s}'.format(file_name))

    def to_mat(self, sseg=0):
        if isinstance(sseg, int):
            if sseg <= self.segN:
                data_dic = {'Data': self.data[sseg].values,
                            'chInfo': self.chInfo,
                            'Date': self.date,
                            'Nseg': 1,
                            'fs': self.fs,
                            'chN': self.chN,
                            'Seg_sta': self.segStatis[sseg],
                            'SegInfo': self.segInfo[sseg:sseg + 1],
                            'Readme': 'Generated by CaseData from python'
                            }
                path = os.getcwd()
                fname = path + '/' + \
                    os.path.splitext(self.filename)[0] +\
                    '_seg{:02d}.mat'.format(sseg)
                sio.savemat(fname, data_dic)
                print('Export: {0:s}'.format(fname))
            else:
                warnings.warn('seg exceeds the max.')
        else:
            warnings.warn('Selected segment id is illegal (should be int).')

    def fix_unit(self, chIdx, newunit, pInfo=False):
        """Fix the incorrect unit due to the limitation of the 'out' format"""
        self.chInfo.loc[chIdx, 'Unit'] = newunit
        if pInfo:
            print('-' * 50)
            print(self.chInfo.to_string(justify='center'))
            print('-' * 50)

    def to_fullscale(self, rho=1.025, g=9.807, lam=1, pInfo=False):
        if self.scale == 'prototype':
            print('The data is already upscaled.')
            return
        else:
            print('Please make sure the channel units are all checked!')
            if pInfo:
                print(self.chInfo.to_string(
                    justify='center', columns=['Name', 'Unit']))
            self.rho = rho
            self.lam = lam
            self.scale = 'prototype'
            transDict = {'kg': ['kN', np.array([g * 0.001, 1.0, 3.0])],
                         'cm': ['m', np.array([0.01, 0.0, 1.0])],
                         'mm': ['m', np.array([0.001, 0.0, 1.0])],
                         'm': ['m', np.array([1, 0.0, 1.0])],
                         's': ['s', np.array([1, 0.0, 0.5])],
                         'deg': ['deg', np.array([1, 0.0, 0.0])],
                         'rad': ['rad', np.array([1, 0.0, 0.0])],
                         'N': ['kN', np.array([0.001, 1.0, 3.0])]
                         }

            def findtrans(transDict, unit):
                unit = unit.lower()
                if unit in transDict:
                    trans = transDict[unit]
                    return trans
                elif '/' in unit:
                    unitUpper, unitLower = unit.split('/')
                    transUpper = findtrans(transDict, unitUpper)
                    transLower = findtrans(transDict, unitLower)
                    trans = [transUpper[0] + '/' +
                             transLower[0], np.array([0.0, 0.0, 0.0])]
                    trans[1][0] = transUpper[1][0] / transLower[1][0]
                    trans[1][1] = transUpper[1][1] - transLower[1][1]
                    trans[1][2] = transUpper[1][2] - transLower[1][2]
                    return trans
                elif '.' in unit:
                    unitWithDot = unit.split('.')
                    transU = []
                    transN1 = np.array([])
                    transN2 = np.array([])
                    transN3 = np.array([])
                    for idx, uWithDot in enumerate(unitWithDot):
                        transWithDot = findtrans(transDict, uWithDot)
                        transU.append(transWithDot[0])
                        transN1 = np.append(transN1, transWithDot[1][0])
                        transN2 = np.append(transN2, transWithDot[1][1])
                        transN3 = np.append(transN3, transWithDot[1][2])
                    trans = ['.'.join(transU), np.array([1.0, 0.0, 0.0])]
                    for x in np.nditer(transN1):
                        trans[1][0] *= x
                    trans[1][1] = transN2.sum()
                    trans[1][2] = transN3.sum()
                    return trans
                elif unit[-1].isdigit():
                    n = int(unit[-1])
                    unit = unit[0:-1]
                    if unit in transDict:
                        trans_temp = transDict[unit]
                        trans = [trans_temp[0] +
                                 str(n), np.array([1.0, 0.0, 0.0])]
                        trans[1][0] = trans_temp[1][0]**n
                        trans[1][1] = trans_temp[1][1] * n
                        trans[1][2] = trans_temp[1][2] * n
                        return trans
                    else:
                        warnings.warn(
                            "input unit cannot identified, please check the unit.")
                else:
                    warnings.warn(
                        "input unit cannot identified, please check the unit.")

            transUnit = []
            transCoeffUnit = np.zeros(self.chN)
            transCoeffRho = np.zeros(self.chN)
            transCoeffLam = np.zeros(self.chN)
            for idx, unit in enumerate(self.chInfo['Unit']):
                trans_temp = findtrans(transDict, unit)
                transUnit.append(trans_temp[0])
                transCoeffUnit[idx] = trans_temp[1][0]
                transCoeffRho[idx] = trans_temp[1][1]
                transCoeffLam[idx] = trans_temp[1][2]
            self.chInfo['Unit'] = transUnit
            self.chInfo['CoeffUnit'] = transCoeffUnit
            self.chInfo['CoeffRho'] = transCoeffRho
            self.chInfo['CoeffLam'] = transCoeffLam

            if pInfo:
                print(self.chInfo.to_string(justify='center'))

            for idx1 in range(self.segN):
                for idx2, name in enumerate(self.chInfo['Name']):
                    C1 = self.chInfo['CoeffUnit'].iloc[idx2]
                    C2 = rho ** self.chInfo['CoeffRho'].iloc[idx2]
                    C3 = self.lam ** self.chInfo['CoeffLam'].iloc[idx2]
                    C = C1 * C2 * C3
                    self.data[idx1][name] *= C

            print('The data is upscaled.')

    def read_waveCal(self,
                    wavefname, 
                    waveChName,
                    waveUnit,
                    sseg = 0,
                    alignFlag = True):
        if 'YB.cal' in self.data[sseg]:
            print('Calibrated wave data already exists.')
            return
        else:
            waveCalDataRaw = np.genfromtxt(wavefname)
            t = waveCalDataRaw[:,0]
            tInterp = np.arange(1 / self.fs, t[-1], 1 / self.fs)
            nCh = waveCalDataRaw.shape[1] - 1
            waveCalDataInterp = np.zeros((tInterp.shape[0], nCh))
            for i in range(nCh):
                waveCalDataInterp[:, i] = np.interp(tInterp, t, waveCalDataRaw[:,i+1])
            if alignFlag:
                YBcal = waveCalDataInterp[:, waveChName.index('YB.cal')]
                YBme = self.data[sseg]['YB'].values
                YBlag = np.argmax(
                    correlate(YBme, YBcal, method='fft')) - tInterp.shape[0]+1
                index = np.arange(tInterp.shape[0]) + YBlag
                waveCalDF = pd.DataFrame(
                    waveCalDataInterp, index=index, columns=waveChName)
                self.data[sseg] = pd.concat(
                    [self.data[sseg], waveCalDF], axis=1, join_axes=[self.data[sseg].index]).fillna(0.0)
            else:
                waveCalDF = pd.DataFrame(
                    waveCalDataInterp, columns=waveChName)
                self.data[sseg] = pd.concat(
                    [self.data[sseg], waveCalDF], axis=1, join_axes=[self.data[sseg].index]).fillna(0.0)
            
            coef = [1.0 for i in range(nCh)]
            chInfo_dic = {'Name': waveChName,
                          'Unit': waveUnit,
                          'Coef': coef}
            Column = ['Name', 'Unit', 'Coef']
            chIdx = list(range(self.chN + 1,self.chN + nCh + 1))
            # for idx in range(nCh):
            #     chIdx.append('{0:02d}'.format(idx + self.chN + 1))
            waveCalchInfo = pd.DataFrame(chInfo_dic, index=chIdx, columns=Column)
            self.chInfo = pd.concat([self.chInfo, waveCalchInfo])
            self.chN += nCh

    def move_ccor(self,
                 to_move_chName,
                 base_chName,
                 sseg = 0,
                 pLag = False):

        toMove = self.data[sseg][to_move_chName]
        base = self.data[sseg][base_chName]        
        n_sample = self.segInfo['N sample'].iloc[sseg]
        lag = np.argmax(correlate(base.values, toMove.values, method='fft')) - n_sample + 1
        if pLag:
            print('Move points: {: d}',format(lag))
        new_index = range(n_sample) + lag
        toMove_dic = {'to move': toMove.values}
        toMoveDF = pd.DataFrame(toMove_dic, index=new_index)
        temp = pd.concat([self.data[sseg], toMoveDF], axis=1, join_axes=[self.data[sseg].index]).fillna(0.0)

        self.data[sseg][to_move_chName] = temp['to move'].values
        
    def changeChOder(self,
                    newOrder,
                    sseg = 0):
        if newOrder.shape[0] == self.chN:
            self.data[sseg] = self.data[sseg][newOrder]
            indexNew = []
            for inewOrder in newOrder:
                indexNew.append(list(self.data[sseg].columns).index(inewOrder))
            self.chInfo = self.chInfo.reindex(indexNew)
            
            #self.chInfo.index = range(1,self.chN+1)
        else:
            print('The size of new orders are not equal to chN.')

    def lowpassFilter(self,
                    chName,
                    cutoffull=2,
                    sseg=0,
                    order = 6):
        def butter_lowpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        def butter_lowpass_filter(data, cutoff, fs, order=5):
            b, a = butter_lowpass(cutoff, fs, order=order)
            y = lfilter(b, a, data)
            return y

        cutoff = cutoffull / 2 / np.pi * np.sqrt(self.lam)
        print('Attention, Lambda = {0:02d}'.format(self.lam))
        data = self.data[sseg][chName].values
        self.data[sseg][chName] = butter_lowpass_filter(data, cutoff, self.fs, order)

    def rmMean(self,
                chName,
                sseg = 0):
        data = self.data[sseg][chName].values
        self.data[sseg][chName] = data-data.mean()

    def cutSeries(self,
                    start,
                    stop,
                    sseg=0):
        startIndx = int(start * self.fs)
        stopIndx = int(stop * self.fs)
        lngth = self.data[sseg].index[-1]
        self.data[sseg] = self.data[sseg].drop(range(startIndx+1))
        self.data[sseg] = self.data[sseg].drop(range(stopIndx,lngth+1))
        self.data[sseg] = self.data[sseg].reset_index(drop=True)
        
        sampNum = self.data[sseg].shape[0]
        # self.segInfo['Start']['Seg{0:2d}'.format(sseg)] = ''
        # self.segInfo['Stop']['Seg{0:2d}'.format(sseg)] = ''
        self.segInfo['Duration']['Seg{0:2d}'.format(sseg)] = '{0:8.1f}s'.format((sampNum- 1) / self.fs)
        self.segInfo['N sample']['Seg{0:2d}'.format(sseg)] = sampNum


    # def read_motion(self,
    #                 motionfname,
    #                 fs_motion,
    #                 sseg=0,
    #                 alignFlag=False,
    #                 zRotate=False,
    #                 rpCorrect=False):

    #     if 'Surge' in self.data[sseg]:
    #         print('Motion data already exists.')
    #         return
    #     else:
    #         motionDataRaw = np.genfromtxt(wavefname,skip_heaeder=11,delimiter='\t')
    #         f = open(wavefname, 'r')
    #         f.seek(0)
    #         fs_str = f.readlines()[3].replace('\n', '')
    #         fs_motion = float(fs_str.split('\t')[1])
    #         tmax = motionDataRaw.shape[1]*fs_motion
    #         tInterp = np.arange(1 / self.fs, t[-1], 1 / self.fs)
    #         tRaw = np.arange(1, motionDataRaw.shape[1] + 1) / self.fs
    #         waveCalDataInterp = np.zeros(
    #             (tInterp.shape[0], waveCalDataRaw.shape[1] - 1))
    #         for i in range(waveCalDataInterp.shape[1]):
    #             waveCalDataInterp[:, i] = np.interp(
    #                 tInterp, t, waveCalDataRaw[:, i + 1])
