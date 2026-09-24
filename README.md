# PyDAS

海洋工程时序数据处理系统。

PyDAS 读实验室二进制 `.out`，把通道和段放在一个对象上，然后查看、切窗、检测并短修坏段、质量分级、滤波、尺度变换、分析、绘图、写报告。公开入口是：

```python
from pydas import PyDAS

data = PyDAS(filename="case.out", lam=36)
data.print_info()
```

包公开导出只有 `PyDAS`、`diff1d`、`data_change_fs`。其余能力都是实例方法，或在 `pydas.waveModel` 里。

## 依赖

- Python >= 3.11
- numpy、pandas（>= 2.0）、scipy、matplotlib、plotly、numba，以及 `pyproject.toml` 里的其它包

## 安装

```bash
git clone https://github.com/XiaoxG/pydas.git
cd pydas
pip install -e .
pip install -e ".[dev]"          # pytest、sphinx、格式化工具
pip install -e ".[performance]"  # 可选，大数据集
```

## 十分钟路径

**不要** 把「读入 → 低通 → 谱 → 写出」当成实验室路径。
处理脊骨是：

```
读 .out → 查看 → cut_series → detect_bad_events → preview_repair
  → apply_repair → qc_report → remove_mean / detrend → 滤波
  → 谱 / 极值 / Excel → 写出 .out
  + 旁路 qc.xlsx 和 repair_log.csv（审计不进 pack）
```

检测和短修必须在滤波 **之前**。`qc_report` 里的 `repaired` 只表示已经 `apply_repair` 写回。建议短修但未写回，grade 是 `limited`，挡住 MPM/EEV。若 `channel_report` 里 MPM/EEV 为 NaN，先打开 qc 表（`limited` / `bad`），不要先改公式。`cutoffull` 是足尺 rad/s，不是 Hz。

```python
from pydas import PyDAS

# 主路径：二进制 .out（构造函数不会猜 CSV / MAT）
data = PyDAS(filename="case.out", lam=36)
data.print_info()
data.plot_channel("Wave1", plotbackend="matplotlib", show=False, save_path="wave1.png")
```

完整链——含种进去的缺陷、`tz`、拒绝 clip/贴边、四件套交付——见 [`docs/user-guide.md`](docs/user-guide.md)（中文教学指南）和 [`examples/lab_workflow.py`](examples/lab_workflow.py)。
[`examples/basic_usage.py`](examples/basic_usage.py) 只冒烟测试导入、画图和谱，**不是** 水池全流程。

没有 `.out` 时：

```python
import numpy as np
import pandas as pd
from pydas import PyDAS

fs = 50.0
t = np.arange(0, 20, 1 / fs)
df = pd.DataFrame({"eta": np.sin(2 * np.pi * 0.5 * t)})
data = PyDAS.from_dataframe(df, fs=fs, lam=1.0, units={"eta": "m"})
# 或: data = PyDAS.read_csv("eta.csv", fs=50.0, lam=1.0, units={"eta": "m"})
```

## 对象里有什么

| 属性 | 含义 |
|------|------|
| `__filename__`、`__date__`、`__desc__` | 源文件元数据 |
| `__fs__` | 采样频率，Hz |
| `__chN__`、`__segN__` | 通道数、段数 |
| `__lam__`、`__scale__` | 长度比尺；`'model'` / `'full'` |
| `chInfo` | 通道表，列含 `Name`、`Unit`、`Coef` |
| `data` | 每段一个 DataFrame 的 list（列为通道） |
| `segInfo` | 段起止时间 / 样本数 |
| `segStatis` | 每段统计，列名 `Mean`、`STD`、`Max`、`Min`、`Unit` |

空对象（`PyDAS(filename=None, lam=1)`）一开始有一段空数据，供 `add_channel` 填入。

## 公开 API（方法名冻结）

历史 mixin 参数名保持驼峰（`chName`、`chOld`、`newOrder`、`updateST`、`cutoffull`、`plotbackend`）。调用代码不要改这些名字。

**通道：** `add_channel`、`delete_channel`、`select_channels`、`rename_channel`、`change_channel_order`、`copy_channel`、`channel_calculate`、`channel_apply_function`、`updateChN`（别名 `update_channel_count`）

**处理：** `apply_lowpass_filter`、`apply_highpass_filter`、`remove_mean`、`detrend`、`add_value`、`multiply_value`、`move_data`、`data_wash`、`add_diff1`、`add_diff2`、`cut_series`、`move_ccor`、`find_move_ccor`、`fix_unit`、`to_fullscale`、`channel2fullscale`、`updateST`（别名 `update_statistics`）

**质量（1.4）：** `detect_bad_events`、`preview_repair`、`apply_repair`、`qc_report`（档位 `good` / `repaired` / `limited` / `bad`）

**I/O：** `write`（`.out`）、`to_dat`、`to_mat`、`to_feather`、`to_parquet`、`to_hdf5`、`read_waveCal`、`read_motion`，类方法 `from_dataframe` / `read_csv`

**图 / 分析 / 报告：** `plot_channel`、`plot_histogram`、`boxplot_channel`、`plot_xy`、`spectral_analysis`、`statistic_analysis`、`extreme_analysis`、`print_info`、`print_channel_info`、`print_statistics`、`channel_report`、`wave_report`

**waveModel：** `jonswap_spectrum`（别名 `jonswap`）、`pm_spectrum`（`PM`）、`torsethaugen_spectrum`、`TimeSeries`、`SpecData1D`、`spectrum_to_timeseries`，以及相关 DNV-RP-C205 辅助函数。

报告列含义见 [`docs/channel_report_metrics.md`](docs/channel_report_metrics.md) 与 [`docs/report_appendix_metrics.md`](docs/report_appendix_metrics.md)。

## 旧 README 里的坑

- 构造函数 **只读二进制 `.out`**。CSV/DAT/MAT 不会自动识别。用 `from_dataframe`、`read_csv` 或导出方法。
- 绘图用 `plotbackend='plotly'|'matplotlib'|'seaborn'`。没有 `use_plotly=True`。
- `cutoffull` 是 **足尺 rad/s**。模型尺度下实现里的截止是 `cutoffull / (2π) * sqrt(λ)`（Hz）。它不是 Hz 参数。
- `spectral_analysis(..., method='cov')` 是自协方差估计；`method='psd'` 是 Welch。
- `examples/historical/proc.py` 是历史实验室笔记本（`CaseData`、`addCh` …）。这些名字不在 `PyDAS` 上。不要运行它。
- `.out` 磁盘布局 **冻结**。其它软件读同一套 pack。不要改头宽、保留字节、int16 缩放或 128 字节对齐。
- `data_wash` 是全局 3σ 插值。不要用在不规则波峰上。成段坏点走 `detect_bad_events` / `apply_repair`（`short_only`）。审计在 `repair_log`，不在 `.out` 里。

## 数据质量与短段修复

这些方法挂在上面的脊骨上，位于切窗和去均值之间。跳过它们，25 计数的尖刺就会进 MPM。

```python
events = data.detect_bad_events("eta", tz=1.0)   # 只读事件表
preview = data.preview_repair("eta", tz=1.0)     # 不写 data
data.apply_repair("eta", tz=1.0, preview=preview)
qc = data.qc_report(tz=1.0)                      # good/repaired/limited/bad
data.detrend("eta", kind="linear")               # 独立于 repair
# limited/bad 在 extreme_analysis 和 channel_report 里跳过 MPM/EEV
```

默认策略 `short_only`：只填短尖刺 / 短 dropout（`n<=3` 线性，更长的短 burst 用 PCHIP）。clip、贴边、中长洞只报告、不编造。`repaired` 只表示 `apply_repair` 已经写回；未写回的建议短修是 `limited`，挡住 MPM。详见 `docs/user-guide.md`。

`respect_quality=True`（默认）时，`qc_report` 失败也会拒绝 MPM（fail-closed）。逃生口是 `respect_quality=False`。

## 包结构

```
src/pydas/
  __init__.py          # PyDAS, diff1d, data_change_fs, __version__
  core/                # PyDAS 门面 + 薄 mixin + state / channels / io_format
  process.py           # 滤波、尺度、统计、相关、detrend
  quality/             # 坏段检测、短修、qc 分级
  analysis.py          # 谱 / 统计 / 极值
  output.py            # 写出（消费 core.io_format）
  reporting.py         # Excel 通道 / 波浪报告
  plot/                # matplotlib / plotly 辅助
  waveModel/           # 谱、TimeSeries、DNV-RP-C205 模型
  utils.py, logger.py
examples/              # 当前 API 的可运行脚本
tests/                 # pytest（tests/legacy 默认不收集）
docs/                  # 使用指南和报告指标手册
```

`PyDAS` 上的 mixin 是薄代理。共享内核在 `core/state.py`、`core/channels.py`、`core/io_format.py`、`process.py`、`quality/`。

## 示例

```bash
python examples/lab_workflow.py    # 实验室脊骨（种了缺陷）
python examples/basic_usage.py     # 最小 API 冒烟，不是水池全流程
```

见 [`examples/README.md`](examples/README.md)。不要运行 `examples/historical/proc.py`。

## 测试

```bash
pytest
```

`pytest.ini` 设置 `pythonpath=src` 和 `testpaths=tests`。`tests/legacy` 故意排除。

## 编码约定

注释、docstring、log 用英文。面向同事的 README / 指南 / 示例说明用中文。公开方法名和历史驼峰参数不改。新内部函数用 snake_case。细节见 [`docs/coding-standards.md`](docs/coding-standards.md)。

## License

MIT。见 [`LICENSE`](LICENSE)。
