# 示例

这些脚本使用 **当前** `PyDAS` API。在仓库根目录 `pip install -e .` 之后运行。

实验室处理脊骨写在中文指南 [`docs/user-guide.md`](../docs/user-guide.md)。水池试验请抄 `lab_workflow.py`。不要从「去均值 → 滤波 → 出报告」起步。

| 脚本 | 演示什么 | 要不要跑 |
|------|----------|----------|
| [`lab_workflow.py`](lab_workflow.py) | 合成记录上的完整脊骨（种了启动段、尖刺、dropout、clip）：切窗、检测、预览、短修、qc 分级、旁路审计文件，然后去均值 / detrend / 滤波 / 谱 / Excel。`cutoffull` 是足尺 rad/s。 | **要跑** — 这是该抄的路径。 |
| [`basic_usage.py`](basic_usage.py) | 最小 API：`from_dataframe`、画通道、协方差谱、JONSWAP。没有质量链。 | 可以跑，用来冒烟测试导入。 |
| [`historical/proc.py`](historical/proc.py) | 历史实验室笔记本。`CaseData` / `addCh` 这类名字 **不在** `PyDAS` 上。 | **不要跑。** |

无头绘图用 `plotbackend='matplotlib'` 和 `show=False`。交互 Plotly 用 `plotbackend='plotly'` 和 `save_html=...`。
