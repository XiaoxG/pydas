# PyDAS 使用指南

面向实验室同事：一次水池 / 海洋工程试验记录，从读进对象到交付，该按什么顺序处理。仓库根目录的 [README.md](../README.md) 是中文总览（含十分钟路径）；**本页是中文教学正文**。报告每一列的公式见 [channel_report_metrics.md](channel_report_metrics.md) 与 [report_appendix_metrics.md](report_appendix_metrics.md)。

可运行的脊骨示例：[examples/lab_workflow.py](../examples/lab_workflow.py)（合成记录里种了坏段）。最小 API 示例：[examples/basic_usage.py](../examples/basic_usage.py)。不要运行 [examples/historical/proc.py](../examples/historical/proc.py)。

你平时只需要：

```python
from pydas import PyDAS
```

包公开导出只有三项：`PyDAS`、`diff1d`、`data_change_fs`。其余能力都是实例方法，或 `pydas.waveModel` 子包。

---

## 1. 这个库做什么：一条处理流水线

PyDAS 把一次试验放进 **一个对象**。功能不是一堆平行工具，而是挂在下面这条链上的站点。每天处理 `.out`，按这个顺序走；跳站会得到看起来干净、其实不可信的谱和极值。

```
读 .out / from_dataframe / read_csv
  → 看对象（print_info, plot_channel）
  → 切稳态窗（cut_series：启动、停车、贴边坏段；不要插值头尾）
  → 检测成段事件（detect_bad_events）
  → 预览短修（preview_repair）
  → 短修或拒绝（apply_repair；clip / 中长洞 / 贴边不填）
  → 质量分级（qc_report：这条通道还能干什么）
  → 去均值 / 线性 detrend
  → 滤波（cutoffull = 足尺 rad/s）
  → 尺度（to_fullscale，先设好 __lam__）
  → 谱 / 统计 / 极值（极值尊重 grade）
  → 交付 Excel + 再写 .out
    + 把 repair_log / qc 表放在 .out 旁边
```

| 站 | 方法 | 这一站在干什么 |
|----|------|----------------|
| 读入 | 构造函数 / `from_dataframe` / `read_csv` | 构造函数 **只** 解包二进制 `.out` |
| 查看 | `print_*` / `plot_channel` | 先看，再改 |
| 切窗 | `cut_series` | 头尾启动停车用切，不用插值 |
| 检测 | `detect_bad_events` | 只读；一行一个 **连续事件**，不是孤立点 |
| 预览 | `preview_repair` | 不写 `data`；看哪些会修、哪些拒绝 |
| 短修 | `apply_repair` | 默认只修短段；写回 `data`，追加 `repair_log` |
| 质量 | `qc_report` | `good` / `repaired` / `limited` / `bad` |
| 去均值/趋势 | `remove_mean` / `detrend` | `detrend` 不是 repair 的一部分 |
| 滤波 | `apply_lowpass_filter` 等 | **先修再滤**；`cutoffull` 不是 Hz |
| 尺度 | `to_fullscale` | 只读 `__lam__`，没有 `lam=` 参数 |
| 谱/极值 | `spectral_analysis` / `extreme_analysis` | `limited`/`bad` 不进 MPM/EEV |
| 报告 | `channel_report` | 19 列主表 **不加** grade 列 |
| 写出 | `write` | pack 冻结；审计 **不** 进 `.out` |
| 旁路 | `data_wash` | 全局 3σ，**不要** 用在不规则波 η / 一阶力 |
| 对照 | `waveModel` | 理论谱，不是处理链 |

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

## 3. 顺序纪律（必读）

库 **不会** 拦住错误顺序。下面四条是实验室约定，不是可选项。

### 3.1 先检测 / 短修，再滤波

Butterworth 会把尖刺涂开到两侧。先低通再 `detect_bad_events`，Hampel 往往只看见一圈残差，qc 还可能打 `good`。正确顺序：切窗 → 检测 → 预览 → 短修 → **然后** 去均值 / detrend → 滤波。

### 3.2 `qc_report` 不等于已经修好

- `detect_bad_events`：只读事件表。
- `preview_repair`：内存里预览，**不** 写 `data`。
- `apply_repair`：这才把短段写回，并追加 `repair_log`。
- `qc_report`：分级。`repaired` **只** 在 `repair_log` 里已有写回记录时出现。事件表上 `action='repair'` 只是「建议短修」；若还没 `apply_repair`，grade 是 `limited`，`suggested_action='apply_repair'`，MPM/EEV **拒绝**。未修的短尖刺不能当极值。

因此：**只有 `apply_repair` 之后的 `qc_report` 才可能是 `repaired`。** 只跑检测 / 预览 / `qc_report` 就出极值，库会挡住。完整链见第 4.3 节和 `examples/lab_workflow.py`。

### 3.3 怎么选 `tz`

`tz` 是特征周期（秒），用作段长尺子 `T*`：短段 `≤ 0.10 T*` 默认可替换；中段 `0.10–0.30 T*`、长段 `> 0.30 T*` 只报告不填。

| 通道角色 | 传什么 |
|----------|--------|
| 浪高仪、波频力 / 运动 | 该工况模型尺度的波周期（零上穿或谱峰周期） |
| 系泊力、慢漂 | 慢漂周期，**不要** 用波频 Tz |
| 不知道、也不传 | 库用零上穿估计；估不出则默认 **1 s**，并且最多只修 5 个点 |

默默依赖 1 s 会让慢漂通道的「短段」完全没物理意义。能传就传。

### 3.4 MPM / EEV 为 NaN 时先看 `qc_report`

`extreme_analysis` 和 `channel_report` 在 `respect_quality=True`（默认）时：`limited` / `bad` **拒绝** 极值，单元格为 NaN，19 列主表 **不加** grade 列。这不是 Tz=0 或 σ=0 的同一种空。未修的短尖刺、以及 `qc_report` 自身失败，同样拒绝（fail-closed）。先打开 qc 表，不要先改 `wave_type`。需要旧行为时再传 `respect_quality=False`。

### 3.5 审计不在 `.out` 里

二进制 pack 冻结（其它软件在读同一套头和 int16）。`write` 存不下 mask。每次交付请留下四件套：

`case.out` + `case_qc.xlsx` + `case_repair_log.csv` + `channel_report.xlsx`

### 3.6 看见 `filter_assessed=False` 请忽略

qc 表里这一列是占位，滤波频率建议 **尚未实现**。它不是在提醒你忘了某一步。

---

## 4. 十分钟：按流水线走一遍

### 4.1 读实验室 `.out`（主路径）

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

### 4.2 没有 `.out` 时：从 DataFrame 或 CSV 建对象

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

### 4.3 一条典型试验处理链

下面是缩微版。种了缺陷的完整可运行脚本是 `examples/lab_workflow.py`。

```python
# --- 查看 ---
data.plot_channel("eta", plotbackend="matplotlib", show=False, save_path="eta_raw.png")

# --- 切稳态窗（头尾用切，不用插值）---
data.cut_series(start=5.0, stop=25.0, sseg=0)

# --- 检测 → 预览 → 短修 → 再分级（tz 用模型波周期）---
tz = 1.25
events = data.detect_bad_events("all", tz=tz)
preview = data.preview_repair("all", tz=tz, events=events)
data.apply_repair("all", tz=tz, preview=preview)   # 这才写回 data
qc = data.qc_report(tz=tz, output_file="case_qc.xlsx")
data.repair_log.to_csv("case_repair_log.csv", index=False)

# --- 去均值 / 去线性趋势 / 滤波（修完以后）---
data.remove_mean("eta")
data.detrend("eta", kind="linear")
data.apply_lowpass_filter("eta", cutoffull=2.0)  # 足尺 rad/s，不是 Hz

# --- 谱 / 报告 / 写出 ---
spec = data.spectral_analysis("eta", method="cov", L=512, plot=False)
data.channel_report("eta_report.xlsx", wave_type="irregular", tz=tz, qc=qc)
data.write("eta_processed.out")
```

`examples/basic_usage.py` 只演示最小 API（建对象、画图、谱），**不是** 试验全流程。

---

## 5. 对象里有什么

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
| `repair_log` | 已应用短修的审计表；**不会** 写入 `.out` |

取一段某个通道：

```python
eta = data.data[0]["eta"]          # pandas Series
eta_np = data.data[0]["eta"].values
```

段索引 `sseg`：整数表示某一段，`'all'` 表示全部（部分方法支持 list）。默认常常是 `0`。

---

## 6. 查看与切稳态窗

先看再改：

```python
print(data.print_info())           # 同时返回总览 DataFrame
data.print_channel_info()
data.print_statistics()
data.plot_channel("eta", plotbackend="matplotlib", show=False, save_path="eta.png")
```

文件头几秒启动、尾几秒停车，看起来像坏点，其实该切掉：

```python
data.cut_series(start=5.0, stop=25.0, sseg=0)   # 按秒；iloc 语义
```

贴文件头尾的坏段 **禁止** 插值（两侧没有真锚点）。`preview_repair` 会 `refuse` + `refuse_reason='edge'`。不要用旧习惯 `ffill`/`bfill` 把端点编成波浪。

---

## 7. 坏段检测、短修与质量分级

水池里的「坏」几乎从不是教科书上的单点脉冲，而是 **连续一段**：接线抖动、DAQ 丢包卡住、满量程削波、头尾启停。检测对象是事件 `(channel, sseg, start_i, stop_i, kind, n, duration_s)`。

默认只修 **短段**（`policy='short_only'`）。中长洞、削波、贴边只进报告，不插值编造波浪。`n≤3` 用线性插值，更长的短 burst 用 PCHIP。多通道同一时刻的短尖刺仍可各修；同一时刻的 dropout/clip 会把 **整段** 标成 `limited` 或 `bad`。

### 7.1 检测（只读）

```python
events = data.detect_bad_events("eta", tz=1.25)
print(events[["channel", "kind", "start_i", "stop_i", "n", "duration_s", "at_edge"]])
```

| kind | 形态 | 默认能不能替换 |
|------|------|----------------|
| `spike_burst` | 相对局部轨迹的短脉冲 | 短段可以 |
| `dropout` | 卡住的恒值 / 突然变 0 | 短段可以，中长段不行 |
| `clip` | 贴满量程的平台 | **不替换**（会造假波峰） |
| 贴边（`at_edge`） | 发生在序列开头或末尾 | **不插值**；应 `cut_series` |

不规则波峰、砰击、慢漂包络 **禁止** 当坏点。旧的全局 3σ（`data_wash`）会误伤它们，见第 15 节。

### 7.2 预览与短修

```python
preview = data.preview_repair("eta", tz=1.25, events=events)
print(preview.events[["kind", "n", "action", "refuse_reason", "interpolator"]])
# action == "repair" 的区间此时还没写进 data

data.apply_repair("eta", tz=1.25, preview=preview)
print(data.repair_log)
```

`preview.events` 里 `action='refuse'` 时看 `refuse_reason`：`clip`、`edge`、`medium_gap`、`long_gap`、`t_star_unknown_n_cap`。拒绝的区间保持原样。

### 7.3 质量四档

```python
qc = data.qc_report(tz=1.25, output_file="case_qc.xlsx")
print(qc[["channel", "grade", "suggested_action", "n_events", "note"]])
```

| grade | 含义 | 极值（MPM/EEV） |
|-------|------|-----------------|
| `good` | 可用 | 算 |
| `repaired` | 短段已写回 `data`（`repair_log` 有记录）；交付时注明含短修 | 仍算，log 会写明 |
| `limited` | 不要做极值；谱/统计需谨慎。含「建议短修但未 `apply_repair`」 | **拒绝**（NaN / 空峰值） |
| `bad` | 不宜正式分析；重采或丢掉该通道 | **拒绝** |

`suggested_action` 常见值：`none`、`apply_repair`、`cut_series`、`do_not_use_for_extremes`、`unusable`。

只读警告列（`too_short_for_mpm`、`startup_unsteady`、`constant_channel`、`nyquist_warning`、`n_nan` 等）是提示，**不自动切窗、不自动滤波**。过短记录、恒通道会升级 grade。`filter_assessed` 恒为 `False`，忽略即可。

再强调第 3.2 节：`repaired` 只表示已经 `apply_repair`。未写回的建议短修是 `limited`。

---

## 8. 去均值、detrend、滤波、尺度

这一站在短修 **之后**。

### 8.1 `cutoffull` 单位（必读）

`apply_lowpass_filter` / `apply_highpass_filter` 的 `cutoffull` 是 **足尺角频率，单位 rad/s**，不是 Hz。

模型尺度下，实现里的截止频率（Hz）为：

```text
cutoff_hz = cutoffull / (2π) * sqrt(λ)
```

例如 `lam=36`、`cutoffull=2` 时，模型截止约为 `2 / (2π) * 6 ≈ 1.91 Hz`。把 `2.0` 当成 Hz 会滤错。

```python
data.apply_lowpass_filter("eta", cutoffull=2.0, order=6, replace=True)
data.apply_highpass_filter("eta", cutoffull=0.5)
```

### 8.2 其它常用处理

```python
data.remove_mean("eta")
data.detrend("eta", kind="linear")   # 或 kind="constant"；独立方法，不是 repair
data.add_value("eta", value2add=0.01)
data.multiply_value("fx", value2mul=1.02)
data.add_diff1("eta", filter=True, filter_cutoff=2)   # 新通道 eta_d1；filter_cutoff 同 cutoffull
data.add_diff2("eta")                                 # 新通道 eta_d2
data.move_data("eta", point_of_move=3)                # 平移采样点
lag = data.find_move_ccor("eta", "Cal.eta")
data.move_ccor("Cal.eta", "Cal.eta", "eta")
data.updateST()                                       # 或 data.update_statistics()
```

`updateST(..., engine='pandas')` 默认用 `DataFrame.agg`。只有表极大时才考虑 `engine='dask'`。NaN 样本在统计时会被跳过。

坏段 **不要** 在这里调用 `data_wash`。见第 15 节。

### 8.3 尺度换算

```python
data.fix_unit("eta", "m")
data.to_fullscale(rho=1.025, g=9.807)   # 就地改所有通道，使用 data.__lam__
ts = data.channel2fullscale("eta", lam=data.__lam__)  # 返回 waveModel.TimeSeries
```

`to_fullscale` **没有** `lam=` 参数，比尺只看 `data.__lam__`。换算前先设好 `__lam__`。

---

## 9. 绘图

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

## 10. 谱、统计、极值

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
ext = data.extreme_analysis("eta", visualization=False, fullscale=True, tz=1.25, qc=qc)
# ext["qc_blocked"] 为 True 时没有可用峰值；先看 ext["qc_grade"] / qc_report
```

`method='cov'` 与 `method='psd'` 都会真正换估计器，不要以为只能 Welch。

谱和基本统计 **目前不看** grade（`limited` 的 note 写「谱/统计需谨慎」，代码不硬拒绝）。极值看 grade。MPM 为空时走第 3.4 节，不要先怀疑公式。

---

## 11. 报告与交付

```python
data.channel_report(
    "channel_report.xlsx",
    wave_type="irregular",   # 规则波用 'regular'
    fullscale=True,
    frequency_separation=False,
    tz=1.25,                 # 转给质量门，与检测用同一把尺子
    qc=qc,                   # 用短修之后、滤波之前的那张表
)

data.wave_report("eta", save_path="wave_report")
```

`wave_type='irregular'` 走完整海洋工程指标（含 MPM / EEV）；`'regular'` 只保留基本统计、零穿越和基于 STD 的幅值，默认不算 MPM。列名与算法见 `docs/channel_report_metrics.md`。可以用 `metrics=[...]` 精确挑列。

**19 列主表故意不加 qc 列**（历史 Excel 模板、其它软件）。`limited`/`bad` 时 MPM/EEV 仍在表里，但是 NaN。对照第 3.5 节的四件套里的 `case_qc.xlsx`。

把 **短修之后、滤波之前** 的 `qc` 表传给 `channel_report(..., qc=qc)` 和 `extreme_analysis(..., qc=qc)`。若不传，报告会在滤波后的波形上再跑一遍检测：clip 平台被涂开后，grade 可能从 `limited` 变成 `repaired`，假峰就会进 MPM。

`print_info()` / `print_channel_info()` / `print_statistics()` 会打日志；`print_info` 同时 **返回** 一张总览 DataFrame。需要文件时再开 `printTxt=True` 或 `printExcel=True`。

---

## 12. 导出

```python
data.write("case_out.out")          # 实验室二进制，布局冻结
data.to_dat(Time=True)
data.to_mat(filename="case.mat", sseg=0)
data.to_parquet()
data.to_feather()
data.to_hdf5()
```

`.out` 的 pack 格式 **不能改**：其它软件也在读同一套头、通道名宽度、int16 量化、128 字节对齐。只调用 `write`，不要手改 `core/io_format.py` 里的常量。修过的波形 roundtrip 后仍受 int16 台阶限制；审计在旁路文件，不在 pack 里。

标定类读入（仍是实例方法，参数名历史驼峰）：

```python
data.read_waveCal("wave_cal.out", YBname="YBS", YBcalname="YBS", alignFlag=True)
data.read_motion("motion.out", alignAccName="AccX", alignMethod="acc")
```

---

## 13. 通道管理

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

## 14. waveModel

理论谱是对照用的，不是处理链上的一站。

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

## 15. 遗留：`data_wash`（不要当常用处理）

`data_wash` 仍是公开方法，算法未改：全局 `|x-mean| > k·std`（默认 k=3），然后无限长插值，端点还会填。

它不适合不规则波 η / 一阶力：真波峰会被当成离群点；连续饱和或死值 0 反而可能漏检。坏段请走第 7 节。只有近似高斯的噪声通道、并且你明确知道自己在做什么时，才考虑它。

---

## 16. 公开 API 速查

**通道：** `add_channel` / `delete_channel` / `select_channels` / `rename_channel` / `change_channel_order` / `copy_channel` / `channel_calculate` / `channel_apply_function` / `updateChN`（别名 `update_channel_count`）

**处理：** `apply_lowpass_filter` / `apply_highpass_filter` / `remove_mean` / `detrend` / `add_value` / `multiply_value` / `move_data` / `data_wash` / `add_diff1` / `add_diff2` / `cut_series` / `move_ccor` / `find_move_ccor` / `fix_unit` / `to_fullscale` / `channel2fullscale` / `updateST`（别名 `update_statistics`）

**质量：** `detect_bad_events` / `preview_repair` / `apply_repair` / `qc_report`

**I/O：** `write` / `to_dat` / `to_mat` / `to_feather` / `to_parquet` / `to_hdf5` / `read_waveCal` / `read_motion` / `from_dataframe` / `read_csv`

**图 / 分析 / 报告：** `plot_channel` / `plot_histogram` / `boxplot_channel` / `plot_xy` / `spectral_analysis` / `statistic_analysis` / `extreme_analysis` / `print_info` / `print_channel_info` / `print_statistics` / `channel_report` / `wave_report`

方法名是用户资产，**不要重命名**。新内部函数用 snake_case。

---

## 17. 常见坑

1. **不要** `PyDAS("file.csv")` 或 `PyDAS("file.mat")`。文本表走 `read_csv` / `from_dataframe`。
2. **不要** 传 `use_plotly=True`，用 `plotbackend=`。
3. **不要** 把 `cutoffull=2` 理解成 2 Hz。
4. **不要** 把 `examples/historical/proc.py` 当教程。那是重构前的 `CaseData` / `addCh` 笔记本。
5. **不要** 改 `.out` pack。往返测试在 `tests/unit/test_out_pack_compat.py`。
6. 空对象用 `filename=None`（或省略），`lam` 默认为 1；没有 `load()` 方法。
7. 统计列是 `STD` 不是 `Std`。
8. `to_fullscale` 不接收 `lam=`，只读 `__lam__`。
9. `to_mat(filename=None, sseg=0)` 必须用关键字传 `sseg`，避免把段号当成文件名。
10. 旧 `data_wash`（全局 3σ）会误伤不规则波峰；坏段请走 `detect_bad_events` / `apply_repair`。
11. **不要** 先滤波再检测。
12. **不要** 只调用 `qc_report` 就认为数据已修；必须 `apply_repair`。未修短尖刺现在会打 `limited` 并挡住 MPM。
13. `channel_report` 里 MPM 为 NaN 时先看 `qc_report`，不是先怀疑公式。
14. 不传 `tz` 时慢漂通道不要默默依赖 1 s。

---

## 18. 仓库地图（给要改代码的人）

```
src/pydas/core/         PyDAS 门面、薄 mixin、state / channels / io_format
src/pydas/process.py    滤波、换算、互相关、updateST、detrend
src/pydas/quality/      坏段检测、短段替换、qc_report
src/pydas/analysis.py   谱 / 统计 / 极值
src/pydas/output.py     写出（消费 io_format）
src/pydas/reporting.py  Excel 报告
src/pydas/plot/         绘图
src/pydas/waveModel/    波浪模型
```

贡献约定见 [coding-standards.md](coding-standards.md)。改行为前先跑 `pytest`。
