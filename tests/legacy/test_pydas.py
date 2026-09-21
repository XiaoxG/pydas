# %%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydas import PyDAS

wavedata = PyDAS('WC01.out',lam=60)
wavedata.print_channel_info()
# %% 测试多通道绘图-默认设置

wavedata.plot_xy('Vessel.Surge', 'Vessel.Sway', datashade=True,use_webgl=True,adaptive_sampling=True)

# %%
channels = ['Vessel.Surge']
wavedata.spectral_analysis(channels)
# %%
wavedata.print_channel_info()
# %%
# %%
