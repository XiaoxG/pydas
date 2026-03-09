# %%
from CaseData import CaseData
import pandas as pd
import numpy as np
from nptdms import TdmsFile

def diff1d(y, dx):
    """Calculate the first-order derivative of the signal y.
    @param: y - the signal
    @param: dx - interval"""

    y = np.asarray(y, dtype='float')
    dy = np.zeros_like(y)
    if y.shape[0] <= 5:
        warnings.warn("Array size is too small!")
        return dy
    else:
        dy[0] = (-y[2] + 4 * y[1] - 3 * y[0]) / (2 * dx)
        dy[1] = (-y[3] + 6 * y[2] - 3 * y[1] - 2 * y[0]) / (6 * dx)
        dy[2] = (8 * (y[3] - y[1]) - (y[4] - y[0])) / (12 * dx)
        dy[3:-3] = (45 * (y[4:-2] - y[2:-4]) - 9 *
                    (y[5:-1] - y[1:-5]) + (y[6:] - y[:-6])) / (60 * dx)
        dy[-3] = (8 * (y[-2] - y[-4]) - (y[-1] - y[-5])) / (12 * dx)
        dy[-2] = (2 * y[-1] + 3 * y[-2] - 6 * y[-3] + y[-4]) / (6 * dx)
        dy[-1] = (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * dx)
        return dy
# %%
casename = 'W04'
wavename = '45'
wavecase = CaseData('../Data/{0:s}/{1:s}R3.out'.format(casename,casename))
wavecalibration = CaseData('../WAVE/IRR{0:s}.out'.format(wavename))

# wavecase.cutSeries(0, 106055/50)
filenames = ['20231115154206']
wavecase.__lam__ = 40
# %%
wavecase.addCh('Wave.C1', unit='cm', series=wavecalibration.data[0]['waveC'].values, fs=wavecalibration.__fs__)
wavecase.addCh('Wave.C2', unit='cm', series=wavecalibration.data[0]['waveW'].values, fs=wavecalibration.__fs__)
wavecase.addCh('Wave.C3', unit='cm', series=wavecalibration.data[0]['waveN3'].values, fs=wavecalibration.__fs__)
# %%
column_names = ['F.line1', 'F.line2', 'F.line3', 'F.line4', 'Acc1.x', 'Acc1.y', 'Acc1.z', 'Acc2.x', 'Acc2.y', 'Acc2.z', 'Airgap1', 'Airgap2', 'Airgap3', 'Airgap4', 'Wave.basin']
unit = ['kg', 'kg', 'kg', 'kg', 'm/s2', 'm/s2', 'm/s2', 'm/s2', 'm/s2', 'm/s2', 'cm', 'cm', 'cm', 'cm', 'cm']
data_pd_list = []

for inames in filenames:
    tdms_file = TdmsFile.read('../Data/{0:s}/{1:s}.tdms'.format(casename, inames))
    tdms_pd = tdms_file.as_dataframe(time_index=True, absolute_time=True)
    # set index time shift 8 hours
    tdms_pd.index = tdms_pd.index + pd.Timedelta(hours=8)
    # drop the last column
    # tdms_pd = tdms_pd.iloc[:, :-1]
    # rename the columns
    tdms_pd.columns = column_names
    data_pd_list.append(tdms_pd)

data_contactpd = pd.concat(data_pd_list)
data_contactpd.index.name = 'Timestamp'
data_contactpd.sort_values(by='Timestamp',inplace=True )
data_contactpd.to_pickle('{0:s}.pkl'.format(casename))

for i, icol in enumerate(column_names):
    wavecase.addCh(icol, unit=unit[i], series=data_contactpd[icol].values, fs=50)
# %%
heave = wavecase.lowpassFilter('Heave',replace=False,returnValue=True, cutoffull=2)
vz = diff1d(heave/100, 1 / wavecase.__fs__)
az_motion = diff1d(vz, 1 / wavecase.__fs__)

wavecase.addCh('Acc_motion', unit='m/s2', series=az_motion, fs=50)
wavecase.addvalue('Acc2.z', - wavecase.data[0]['Acc2.z'].values[0])
wavecase.lowpassFilter('Acc2.z',cutoffull=2)

lag = wavecase.find_move_ccor('Acc_motion', 'Acc2.z')
for inames in column_names:
    wavecase.moveData(inames, lag)

wavecase.data[0].plot(y = ['Acc_motion', 'Acc2.z'], backend = 'plotly')
# %%
wavecase.delCh('Acc_motion')
wavecase.delCh('X')
wavecase.delCh('Y')
wavecase.delCh('Z')

# %%
for i in range(4):
    Fname = 'F.line{0:d}'.format(i+1)
    wavecase.data[0][Fname] = wavecase.data[0][Fname].values - np.mean(wavecase.data[0][Fname].values[0:1000] + 1.89)

# %%
for i in ['Airgap1', 'Airgap2', 'Airgap3', 'Airgap4', 'Wave.basin']:
    wavecase.data[0][i] = wavecase.data[0][i].values - np.mean(wavecase.data[0][i].values[0:1000])
for i in range(4):
    wavecase.data[0]['Airgap{0:d}'.format(i+1)] = 33.25 - wavecase.data[0]['Airgap{0:d}'.format(i+1)].values
wavecase.lowpassFilter('Wave.basin',cutoffull=2)
wavecase.lowpassFilter('Wave.C1',cutoffull=2.5)
wavecase.lowpassFilter('Wave.C2',cutoffull=2.5)
wavecase.lowpassFilter('Wave.C3',cutoffull=2.5)

# %%
for iacc in ['Acc1.x', 'Acc1.y', 'Acc1.z', 'Acc2.x', 'Acc2.y', 'Acc2.z']:
    wavecase.lowpassFilter(iacc, cutoffull=6)
    wavecase.data[0][iacc] = wavecase.data[0][iacc].values - np.mean(wavecase.data[0][iacc].values[0:1000])

# %%
for imotion in ['Surge', 'Sway', 'Heave', 'Pitch', 'Roll', 'Yaw']:
    wavecase.lowpassFilter(imotion, cutoffull=2)

# %%
wavecase.data[0].plot(y = ['F.line1', 'F.line2', 'F.line3', 'F.line4'], backend = 'plotly')

# %%
wavecase.data[0].plot(y = ['Acc1.x', 'Acc1.y', 'Acc1.z', 'Acc2.x', 'Acc2.y', 'Acc2.z'], backend = 'plotly')

# %%
wavecase.data[0].plot(y = ['Airgap1', 'Airgap2', 'Airgap3', 'Airgap4'], backend = 'plotly')
# %%
wavecase.data[0].plot(y = ['Surge', 'Sway', 'Heave', 'Pitch', 'Roll', 'Yaw'], backend = 'plotly')
# %%
wavecase.data[0].plot(y = [ 'Wave.C1', 'Wave.C2', 'Wave.C3'], backend = 'plotly')

# %%
wavecase.write('{0:s}_V1.out'.format(casename))
# %%
start = 60
end = 3*60*60/np.sqrt(40)
wavecase.cutSeries(start, end)
# %%
wavecase.to_fullscale(lam=40)
wavecase.to_dat()
wavecase.pst(printExcel=True)