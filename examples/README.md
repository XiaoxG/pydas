# Examples

These scripts use the **current** `PyDAS` API. Run them from the repository
root after `pip install -e .`.

The laboratory processing spine is documented in Chinese in
[`docs/user-guide.md`](../docs/user-guide.md). Copy `lab_workflow.py` for
basin tests. Do not start from `remove_mean` → filter → report.

| Script | What it shows | Run it? |
|--------|----------------|---------|
| [`lab_workflow.py`](lab_workflow.py) | Full spine on a synthetic record **with planted defects**: cut startup, detect, preview, apply short repair, qc grades, sidecar audit files, then mean / detrend / filter / spectrum / Excel. `cutoffull` is full-scale rad/s. | **Yes** — this is the path to copy. |
| [`basic_usage.py`](basic_usage.py) | Smallest API: `from_dataframe`, channel plot, covariance spectrum, JONSWAP. No quality chain. | Yes, for a smoke check of imports. |
| [`historical/proc.py`](historical/proc.py) | Historical lab notebook. Names such as `CaseData` / `addCh` are **not** on `PyDAS`. | **Do not run.** |

Headless plotting uses `plotbackend='matplotlib'` and `show=False`. Interactive
Plotly is available with `plotbackend='plotly'` and `save_html=...`.
