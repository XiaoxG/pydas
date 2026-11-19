# PyDAS

海洋工程水池模型试验的时序库。一次试验进一个对象：读二进制 `.out`，切窗、检测并短修坏段、质量分级，再滤波、尺度、谱 / 极值、绘图和 Excel。

公开入口只有：

```python
from pydas import PyDAS

data = PyDAS(filename="case.out", lam=36)
data.print_info()
```

包导出三项：`PyDAS`、`diff1d`、`data_change_fs`。其余都是实例方法，或 `pydas.waveModel`。当前版本见 `pydas.__version__`（1.4.3）。

仓库：https://github.com/XiaoxG/pydas

## 处理脊骨（请从这里开始）

不要走「读入 → 低通 → 谱 → 写出」。实验室路径是：

```
读 .out → 查看 → cut_series → detect_bad_events → preview_repair
  → apply_repair → qc_report → remove_mean / detrend → 滤波
  → 谱 / 极值 / Excel → 写出 .out
  + 旁路 qc.xlsx 和 repair_log.csv（审计不进 pack）
```

检测和短修必须在滤波之前。`repaired` 只表示已经 `apply_repair` 写回；建议短修但未写回是 `limited`，挡住 MPM/EEV。`cutoffull` 是足尺 rad/s，不是 Hz。`channel_report` 里 MPM/EEV 为 NaN 时，先看 qc 表，不要先改公式。

```python
from pydas import PyDAS

data = PyDAS(filename="case.out", lam=36)  # 构造函数只读二进制 .out
data.print_info()
data.plot_channel("Wave1", plotbackend="matplotlib", show=False, save_path="wave1.png")

tz = 1.25  # 模型尺度特征周期，秒
events = data.detect_bad_events("all", tz=tz)
preview = data.preview_repair("all", tz=tz, events=events)
data.apply_repair("all", tz=tz, preview=preview)
qc = data.qc_report(tz=tz, output_file="case_qc.xlsx")
data.repair_log.to_csv("case_repair_log.csv", index=False)

data.remove_mean("Wave1")
data.detrend("Wave1", kind="linear")
data.apply_lowpass_filter("Wave1", cutoffull=2.0)  # 足尺 rad/s

data.channel_report("channel_report.xlsx", wave_type="irregular", tz=tz, qc=qc)
data.write("case_processed.out")
```

完整教学（含种进去的缺陷、clip/贴边拒绝、四件套交付）在 [`docs/user-guide.md`](docs/user-guide.md)。可运行脊骨：[`examples/lab_workflow.py`](examples/lab_workflow.py)。[`examples/basic_usage.py`](examples/basic_usage.py) 只冒烟导入 / 画图 / 谱，不是水池全流程。

没有 `.out` 时用类方法，不要把 CSV 丢给构造函数：

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

## 安装

Python >= 3.11。依赖见 `pyproject.toml`（numpy、pandas>=2、scipy、matplotlib、plotly、numba 等）。

```bash
git clone https://github.com/XiaoxG/pydas.git
cd pydas
pip install -e .
pip install -e ".[dev]"          # pytest、sphinx、格式化工具
pip install -e ".[performance]"  # 可选，大数据集
```

## 对象里有什么

| 属性 | 含义 |
|------|------|
| `__filename__`、`__date__`、`__desc__` | 源文件元数据 |
| `__fs__` | 采样频率，Hz |
| `__chN__`、`__segN__` | 通道数、段数 |
| `__lam__`、`__scale__` | 长度比尺；`'model'` / `'full'` |
| `chInfo` | 通道表，列 `Name`、`Unit`、`Coef` |
| `data` | **list**：每段一个 DataFrame，列为通道名 |
| `segInfo` | 段起止时间 / 样本数 |
| `segStatis` | 每段统计，列名固定 `Mean`、`STD`、`Max`、`Min`、`Unit` |
| `repair_log` | 短修审计表；**不**写入 `.out` |

没有 `.channels`。读一段一通道用 `data.data[sseg][name]`。空对象：`PyDAS(filename=None, lam=1)`，再 `add_channel`。

## 给 AI 编程工具

用这个库写脚本或改调用代码时，按下面契约走。不要从旧 README、Gitee 镜像、或 `examples/historical/proc.py` 猜 API。

**入口**

- `from pydas import PyDAS`。不要 `import pydas.pydas`，没有顶层 `pydas.py`，没有 `CaseData` / `addCh` / `load()`。
- 构造函数：`PyDAS(filename=None, lam=1, sseg="all", log_level="info")`。`filename` 只接受实验室二进制 `.out`。CSV / DAT / MAT / 文本表用 `PyDAS.from_dataframe` 或 `PyDAS.read_csv`。
- 比尺 `lam` 要设对；`to_fullscale` **没有** `lam=` 参数，只读对象上的 `__lam__`。

**对象与通道**

- 状态见上一节。通道名在 `chInfo["Name"]`，样本在 `data.data[sseg]` 的列上。
- 公开方法名和历史驼峰参数冻结：`chName`、`chOld`、`newOrder`、`updateST`、`cutoffull`、`plotbackend`。不要重命名。
- 写出二进制：`data.write("out.out")`。没有用户该调用的 `write_data`。`to_mat(filename=None, sseg=0)` 必须用关键字传 `sseg`。

**处理顺序（强制纪律，库不会拦）**

1. 读入 / 查看（`print_info`、`plot_channel`）
2. `cut_series`：启动、停车、贴边坏段用切，不要插值头尾
3. `detect_bad_events` → `preview_repair` → `apply_repair` → `qc_report`
4. `remove_mean` / `detrend`（`detrend` 独立，不塞进 repair）
5. `apply_lowpass_filter` / `apply_highpass_filter`
6. 需要足尺再 `to_fullscale`
7. `spectral_analysis` / `extreme_analysis` / `channel_report`
8. `write`，并把 `qc_report` Excel 与 `repair_log.csv` 放在 `.out` 旁边

先滤波再检测会把尖刺涂开。把短修后的 `qc` 表传给 `channel_report(..., qc=qc)` 和 `extreme_analysis(..., qc=qc)`，避免滤波后重跑 qc 改掉 grade。

**单位**

- `cutoffull` 是 **足尺角频率，rad/s**，不是 Hz。模型尺度下实现里的截止（Hz）为 `cutoffull / (2π) * sqrt(λ)`。
- `__fs__` 是 Hz。`tz` 是特征周期，秒（浪高仪用波周期；慢漂不要用波频 Tz）。
- `spectral_analysis(..., method="cov")` 是自协方差；`method="psd"` 是 Welch。

**质量规则**

- 档位英文 id：`good` / `repaired` / `limited` / `bad`。
- `repaired` **只**在 `repair_log` 记录了写回（`action=="repair"`）时出现。只 `qc_report` / `preview_repair` 不够。
- 建议短修但未 `apply_repair` → `limited`，`suggested_action='apply_repair'`，**挡住** MPM/EEV。
- `limited` / `bad` 在 `extreme_analysis` 和 `channel_report` 里跳过 MPM/EEV（NaN / 空峰值）。19 列交付表 **不加** grade 列。
- `respect_quality=True`（默认）：`qc_report` 失败也拒绝 MPM（fail-closed）。逃生口：`respect_quality=False`。
- 默认 `policy="short_only"`：短尖刺 / 短 dropout 可填（`n<=3` 线性，更长短 burst 用 PCHIP）；clip、贴边、中长洞只报告不填。
- 审计在对象的 `repair_log`，不进冻结的 `.out` pack。

**不要调用 / 不要假设**

- 不要 `use_plotly=True`。绘图：`plotbackend="plotly"|"matplotlib"|"seaborn"`，无头用 `show=False`。
- 不要把 `data_wash` 当正常路径。它是全局 mean±kσ，会误伤不规则波峰。坏段走 `detect_bad_events` / `apply_repair`。
- 不要改 `.out` pack（头、保留字节、16 字节名、int16 `/32767`、128 字节对齐）。
- 不要给 19 列 `channel_report` 主表加列。
- 不要运行 `examples/historical/proc.py`（历史 `CaseData` 笔记本）。
- 不要指向 Gitee；源码在 GitHub `XiaoxG/pydas`。
- 代码注释、docstring、log 用英文；同事文档用中文。不要把 log 改成中文。

**该读的文件**

- 中文教程：[`docs/user-guide.md`](docs/user-guide.md)
- 该抄的脚本：[`examples/lab_workflow.py`](examples/lab_workflow.py)
- 报告列公式：[`docs/channel_report_metrics.md`](docs/channel_report_metrics.md)、[`docs/report_appendix_metrics.md`](docs/report_appendix_metrics.md)
- 改库时的规范：[`docs/coding-standards.md`](docs/coding-standards.md)

## 公开 API（方法名冻结）

**通道：** `add_channel`、`delete_channel`、`select_channels`、`rename_channel`、`change_channel_order`、`copy_channel`、`channel_calculate`、`channel_apply_function`、`updateChN`（别名 `update_channel_count`）

**处理：** `apply_lowpass_filter`、`apply_highpass_filter`、`remove_mean`、`detrend`、`add_value`、`multiply_value`、`move_data`、`data_wash`、`add_diff1`、`add_diff2`、`cut_series`、`move_ccor`、`find_move_ccor`、`fix_unit`、`to_fullscale`、`channel2fullscale`、`updateST`（别名 `update_statistics`）

**质量：** `detect_bad_events`、`preview_repair`、`apply_repair`、`qc_report`

**I/O：** `write`、`to_dat`、`to_mat`、`to_feather`、`to_parquet`、`to_hdf5`、`read_waveCal`、`read_motion`；类方法 `from_dataframe` / `read_csv`

**图 / 分析 / 报告：** `plot_channel`、`plot_histogram`、`boxplot_channel`、`plot_xy`、`spectral_analysis`、`statistic_analysis`、`extreme_analysis`、`print_info`、`print_channel_info`、`print_statistics`、`channel_report`、`wave_report`

**waveModel：** `jonswap_spectrum`（`jonswap`）、`pm_spectrum`（`PM`）、`torsethaugen_spectrum`、`TimeSeries`、`SpecData1D`、`spectrum_to_timeseries` 等。

## 包结构

```
src/pydas/
  __init__.py          # PyDAS, diff1d, data_change_fs, __version__
  core/                # 门面 + 薄 mixin + state / channels / io_format
  process.py           # 滤波、尺度、统计、相关、detrend
  quality/             # 坏段检测、短修、qc 分级
  analysis.py          # 谱 / 统计 / 极值
  output.py            # 写出（消费 core.io_format）
  reporting.py         # Excel 通道 / 波浪报告
  plot/                # matplotlib / plotly
  waveModel/           # 理论谱与 DNV-RP-C205 辅助
examples/              # 当前 API；historical/ 不要跑
tests/                 # pytest（tests/legacy 默认不收集）
docs/                  # 中文指南和报告指标手册
```

## 测试

```bash
pytest
```

`pytest.ini`：`pythonpath=src`，`testpaths=tests`。`tests/legacy` 故意排除。

## 提交问题

用 GitHub 模板，不要发空白单：

- [缺陷报告](https://github.com/XiaoxG/pydas/issues/new?template=bug_report.yml)
- [功能请求](https://github.com/XiaoxG/pydas/issues/new?template=feature_request.yml)
- [使用提问](https://github.com/XiaoxG/pydas/issues/new?template=usage_question.yml)

写明 PyDAS 版本、Python 版本、操作系统，附最小复现代码和 traceback。协作约定见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## License

MIT。见 [`LICENSE`](LICENSE)。
