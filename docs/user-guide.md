# PyDAS 使用指南

面向实验室同事：如何用当前代码库处理水池 / 海洋工程时序数据。仓库根目录的 [README.md](../README.md) 是英文总览；本页是中文操作说明。报告指标含义见 [channel_report_metrics.md](channel_report_metrics.md) 与 [report_appendix_metrics.md](report_appendix_metrics.md)。

---

## 1. 这个库做什么

PyDAS 把一次试验（或一段合成信号）放进 **一个对象** 里：读实验室二进制 `.out` → 管通道 → 滤波 / 对齐 / Froude 换算 → 统计 / 谱 / 极值 → 画图或写 Excel → 再导出。

你平时只需要：

```python
from pydas import PyDAS
```

包公开导出只有三项：`PyDAS`、`diff1d`、`data_change_fs`。其余能力都是实例方法，或 `pydas.waveModel` 子包。

---

## 2. 安装

需要 Python 3.11+。

```bash
git clone https://github.com/XiaoxG/pydas.git
cd pydas
pip install -e .
```

跑测试再装开发依赖：

```bash
pip install -e ".[dev]"
pytest
```

---

## 3. 十分钟上手

### 3.1 读实验室 `.out`（主路径）

```python
from pydas import PyDAS

data = PyDAS(filename="case.out", lam=36)  # lam 是模型比尺，默认 1
data.print_info()
data.print_channel_info()
data.print_statistics()
print(data.chInfo)
print(data.__fs__, data.__chN__, data.__segN__, data.__lam__, data.__scale__)
```

构造函数 **只按二进制 `.out` 解包**。不要把 CSV / TSV / MAT 路径丢给 `PyDAS(filename=...)`。

### 3.2 没有 `.out` 时：从 DataFrame 或 CSV 建对象

```python
import numpy as np
import pandas as pd
from pydas import PyDAS

fs = 50.0
t = np.arange(0, 30, 1 / fs)
df = pd.DataFrame({
    "eta": 0.05 * np.sin(2 * np.pi * 0.8 * t),
    "fx": 10.0 * np.sin(2 * np.pi * 0.8 * t + 0.3),
})
data = PyDAS.from_dataframe(df, fs=fs, lam=40.0, units={"eta": "m", "fx": "N"})

# 或显式读文本表（分隔符等参数转给 pandas.read_csv）
data = PyDAS.read_csv("signals.csv", fs=50.0, lam=40.0, units={"eta": "m"})
```

空对象也可以手工填通道：

```python
data = PyDAS(filename=None, lam=1.0)
data.add_channel("eta", "m", df["eta"].to_numpy(), fs)
```

### 3.3 一条典型试验处理链

```python
data.remove_mean("eta")
data.apply_lowpass_filter("eta", cutoffull=2.0)  # 足尺 rad/s，不是 Hz
data.plot_channel("eta", plotbackend="matplotlib", show=False, save_path="eta.png")
spec = data.spectral_analysis("eta", method="cov", L=512, plot=False)
data.channel_report("eta_report.xlsx", wave_type="irregular")
data.write("eta_processed.out")
```

完整可运行脚本：`examples/basic_usage.py`、`examples/lab_workflow.py`。

---

## 4. 对象里有什么

| 属性 | 含义 |
|------|------|
| `__filename__` / `__date__` / `__desc__` | 文件名、日期戳、描述 |
| `__fs__` | 采样频率，Hz |
| `__chN__` / `__segN__` | 通道数、段数 |
| `__lam__` / `__scale__` | 比尺；`'model'` 或 `'full'` |
| `chInfo` | 通道表，列含 `Name`、`Unit`、`Coef` |
| `data` | **list**，每段一个 DataFrame，列为通道 |
| `segInfo` | 段起止时间、`N sample` |
| `segStatis` | 每段统计，列名固定为 `Mean` / `STD` / `Max` / `Min` / `Unit` |

取一段某个通道：

```python
eta = data.data[0]["eta"]          # pandas Series
eta_np = data.data[0]["eta"].values
```

段索引 `sseg`：整数表示某一段，`'all'` 表示全部（部分方法支持 list）。默认常常是 `0`。

---

## 5. 通道管理

公开方法名不要改，参数也沿用历史驼峰（`chName`、`chOld`、`newOrder`）。

```python
data.add_channel("const", "-", np.ones(len(data.data[0])), data.__fs__)
data.copy_channel("eta", "eta_copy")
data.rename_channel("eta_copy", "wave")          # 参数名是 chOld, chNew
data.change_channel_order(["wave", "eta"])       # 参数名是 newOrder
data.channel_calculate("fx", "fy", "add", "fxy")
data.channel_apply_function("eta", np.abs, "eta_abs", unit="m")
data.select_channels(["eta", "fx"])              # 只保留这些通道
data.delete_channel("const")
data.updateChN()                                 # 或 data.update_channel_count()
```

`channel_apply_function` 的 `func` 可以是向量化可调用对象，或只含 `x` 的受限表达式字符串。不要自己 `eval` 整段代码。

---

## 6. 处理：滤波、清洗、差分、对齐、切段

### 6.1 `cutoffull` 单位（必读）

`apply_lowpass_filter` / `apply_highpass_filter` 的 `cutoffull` 是 **足尺角频率，单位 rad/s**，不是 Hz。

模型尺度下，实现里的截止频率（Hz）为：

```text
cutoff_hz = cutoffull / (2π) * sqrt(λ)
```

例如 `lam=36`、`cutoffull=2` 时，模型截止约为 `2 / (2π) * 6 ≈ 1.91 Hz`。按 README 旧写法把 `2.0` 当成 Hz 会滤错。

```python
data.apply_lowpass_filter("eta", cutoffull=2.0, order=6, replace=True)
data.apply_highpass_filter("eta", cutoffull=0.5)
```

### 6.2 其它常用处理

```python
data.remove_mean("eta")
data.add_value("eta", value2add=0.01)
data.multiply_value("fx", value2mul=1.02)
data.data_wash("eta", method="linear", threshold=3)
data.add_diff1("eta", filter=True, filter_cutoff=2)   # 新通道 eta_d1
data.add_diff2("eta")                                 # 新通道 eta_d2
data.cut_series(start=5.0, stop=25.0, sseg=0)         # 按秒切，iloc 语义
data.move_data("eta", point_of_move=3)                # 平移采样点
lag = data.find_move_ccor("eta", "Cal.eta")
data.move_ccor("Cal.eta", "Cal.eta", "eta")
data.updateST()                                       # 或 data.update_statistics()
```

`updateST(..., engine='pandas')` 默认用 `DataFrame.agg`。只有表极大时才考虑 `engine='dask'`。

### 6.3 尺度换算

```python
data.fix_unit("eta", "m")
data.to_fullscale(rho=1.025, g=9.807)   # 就地改所有通道，使用 data.__lam__
ts = data.channel2fullscale("eta", lam=data.__lam__)  # 返回 waveModel.TimeSeries
```

`to_fullscale` **没有** `lam=` 参数，比尺只看 `data.__lam__`。换算前先设好 `__lam__`。

---

## 7. 绘图

没有 `use_plotly=` 这个参数。后端用 `plotbackend`：

```python
data.plot_channel("eta", plotbackend="matplotlib", show=False, save_path="eta.png")
data.plot_channel("eta", plotbackend="plotly", show=False, save_html="eta.html")
data.plot_histogram("eta", bins=50, show=False)
data.plot_xy("eta", "fx", show=False)
data.boxplot_channel("eta", show=False)
```

`show=False` 适合无界面环境。大数据会走 LTTB / WebGL，这是内部实现，调用方不必自己下采样。

---

## 8. 谱、统计、极值

```python
spec = data.spectral_analysis(
    "eta",
    method="cov",      # 'cov' = 自协方差；'psd' = Welch
    L=1024,
    plot=False,
    freq_range=(0, 2), # 足尺 rad/s 窗口
)
# spec 是 waveModel.SpecData1D，密度在 spec.data，频率在 spec.args

stats = data.statistic_analysis("eta", advanced=True, visualization=False)
ext = data.extreme_analysis("eta", visualization=False, fullscale=True)
```

`method='cov'` 与 `method='psd'` 都会真正换估计器，不要以为只能 Welch。

---

## 9. 报告

```python
data.channel_report(
    "channel_report.xlsx",
    wave_type="irregular",   # 或不规则波默认；规则波用 'regular'
    fullscale=True,
    frequency_separation=False,
)

data.wave_report("eta", save_path="wave_report")
```

`wave_type='irregular'` 走完整海洋工程指标（含 MPM / EEV）；`'regular'` 只保留基本统计、零穿越和基于 STD 的幅值，默认不算 MPM。列名与算法见 `docs/channel_report_metrics.md`。可以用 `metrics=[...]` 精确挑列。

`print_info()` / `print_channel_info()` / `print_statistics()` 会打日志；`print_info` 同时 **返回** 一张总览 DataFrame。需要文件时再开 `printTxt=True` 或 `printExcel=True`。

---

## 10. 导出

```python
data.write("case_out.out")          # 实验室二进制，布局冻结
data.to_dat(Time=True)
data.to_mat(filename="case.mat", sseg=0)
data.to_parquet()
data.to_feather()
data.to_hdf5()
```

`.out` 的 pack 格式 **不能改**：其它软件也在读同一套头、通道名宽度、int16 量化、128 字节对齐。只调用 `write`，不要手改 `core/io_format.py` 里的常量。

标定类读入（仍是实例方法，参数名历史驼峰）：

```python
data.read_waveCal("wave_cal.out", YBname="YBS", YBcalname="YBS", alignFlag=True)
data.read_motion("motion.out", alignAccName="AccX", alignMethod="acc")
```

---

## 11. waveModel

```python
import numpy as np
import pydas.waveModel as wm

w = np.linspace(0.05, 3.0, 256)
S = wm.jonswap(w, Hs=4.0, Tp=10.0, gamma=3.3)          # 别名 jonswap_spectrum
S_pm = wm.PM(w, Hs=4.0, Tp=10.0)
t, eta = wm.spectrum_to_timeseries(w, S, duration=600.0, dt=0.05, seed=1)
```

理论谱按 DNV-RP-C205 方向实现。`TimeSeries.tospecdata(method='psd'|'cov')` 与对象上的 `spectral_analysis` 对应。

---

## 12. 公开 API 速查

**通道：** `add_channel` / `delete_channel` / `select_channels` / `rename_channel` / `change_channel_order` / `copy_channel` / `channel_calculate` / `channel_apply_function` / `updateChN`（别名 `update_channel_count`）

**处理：** `apply_lowpass_filter` / `apply_highpass_filter` / `remove_mean` / `detrend` / `add_value` / `multiply_value` / `move_data` / `data_wash` / `add_diff1` / `add_diff2` / `cut_series` / `move_ccor` / `find_move_ccor` / `fix_unit` / `to_fullscale` / `channel2fullscale` / `updateST`（别名 `update_statistics`）

**质量：** `detect_bad_events` / `preview_repair` / `apply_repair` / `qc_report`

**I/O：** `write` / `to_dat` / `to_mat` / `to_feather` / `to_parquet` / `to_hdf5` / `read_waveCal` / `read_motion` / `from_dataframe` / `read_csv`

**图 / 分析 / 报告：** `plot_channel` / `plot_histogram` / `boxplot_channel` / `plot_xy` / `spectral_analysis` / `statistic_analysis` / `extreme_analysis` / `print_info` / `print_channel_info` / `print_statistics` / `channel_report` / `wave_report`

方法名是用户资产，**不要重命名**。新内部函数用 snake_case。

---

## 13. 坏段检测、替换与质量分级

默认只修**短段**（`policy='short_only'`）。中长洞、削波、贴文件头尾的段只进报告，不插值编造波浪。`n≤3` 用线性插值，更长的短 burst 用 PCHIP。多通道同一时刻的短尖刺仍可各修；同一时刻的 dropout/clip 会把**整段**标成 `limited` 或 `bad`。

```python
events = data.detect_bad_events("eta", tz=1.0)    # 只读，一行一个连续事件
preview = data.preview_repair("eta", tz=1.0)
data.apply_repair("eta", tz=1.0, preview=preview)  # 写回 data，追加 repair_log
qc = data.qc_report(tz=1.0)                       # good / repaired / limited / bad

data.detrend("fx", kind="linear")                 # 独立去趋势，不是 repair 的一部分
```

旧的 `data_wash` 仍是全局 mean±kσ，**不适合**不规则波 η / 一阶力。`.out` 存不下 mask（pack 冻结），审计在 `repair_log` 或 `qc_report` 的 Excel。

`tz` 是特征周期（秒），用来判断段长相对 `T*`。不传则用零上穿估计，估不出时默认 1 s，并且最多只允许修 5 个点。

相对 `T*` 的段长：短段 `≤0.10 T*` 可替换；中段 `0.10–0.30 T*` 与长段 `>0.30 T*` 只进报告。贴边事件建议 `cut_series`，不要插值。

---

## 14. 常见坑

1. **不要** `PyDAS("file.csv")` 或 `PyDAS("file.mat")`。文本表走 `read_csv` / `from_dataframe`。
2. **不要** 传 `use_plotly=True`，用 `plotbackend=`。
3. **不要** 把 `cutoffull=2` 理解成 2 Hz。
4. **不要** 把 `examples/proc.py` 当教程。那是重构前的 `CaseData` / `addCh` 笔记本。
5. **不要** 改 `.out` pack。往返测试在 `tests/unit/test_out_pack_compat.py`。
6. 空对象用 `filename=None`（或省略），`lam` 默认为 1；没有 `load()` 方法。
7. 统计列是 `STD` 不是 `Std`。
8. `to_fullscale` 不接收 `lam=`，只读 `__lam__`。
9. `to_mat(filename=None, sseg=0)` 必须用关键字传 `sseg`，避免把段号当成文件名。
10. 旧 `data_wash`（全局 3σ）会误伤不规则波峰；坏段请走 `detect_bad_events` / `apply_repair`。

---

## 15. 仓库地图（给要改代码的人）

```
src/pydas/core/     PyDAS 门面、薄 mixin、state / channels / io_format
src/pydas/process.py    滤波、换算、互相关、updateST、detrend
src/pydas/quality/      坏段检测、短段替换、qc_report
src/pydas/analysis.py   谱 / 统计 / 极值
src/pydas/output.py     写出（消费 io_format）
src/pydas/reporting.py  Excel 报告
src/pydas/plot/         绘图
src/pydas/waveModel/    波浪模型
```

贡献约定见 [coding-standards.md](coding-standards.md)。改行为前先跑 `pytest`。
