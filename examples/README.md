# Examples

These scripts use the **current** `PyDAS` API. Run them from the repository
root after `pip install -e .`.

| Script | What it shows |
|--------|----------------|
| `basic_usage.py` | `from_dataframe`, channel plot, covariance spectrum, JONSWAP |
| `lab_workflow.py` | Write/read `.out`, filter, scale notes, spectrum, Excel report |
| `proc.py` | Historical lab notebook. Names such as `CaseData` / `addCh` are **not** on `PyDAS`. Kept only as a reference of an old workflow. |

Headless plotting uses `plotbackend='matplotlib'` and `show=False`. Interactive
Plotly is available with `plotbackend='plotly'` and `save_html=...`.

Chinese walkthrough: [`docs/user-guide.md`](../docs/user-guide.md).
