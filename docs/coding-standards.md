# Coding standards

This note is for people who will change PyDAS, not only call it. The public
facade is frozen; internal code should look like one library.

## Language

- Comments, docstrings, and log messages are English.
- User-facing teaching docs are Chinese (`README.md`, `docs/user-guide.md`,
  `examples/*.md`, report metric catalogues).

## Docstrings

Use NumPy style:

```python
def apply_lowpass_filter(pydas_obj, chName, cutoffull=2):
    """Apply a lowpass filter to a channel.

    Parameters
    ----------
    pydas_obj : PyDAS
        Object that holds the channel data.
    chName : str or list
        Channel name, or list of names.
    cutoffull : float, optional
        Full-scale cutoff in rad/s, default is 2. This is not Hertz.

    Returns
    -------
    numpy.ndarray, optional
        Filtered samples when ``returnValue`` is True.
    """
```

Do not mix `Parameters:` / `Returns:` Google headers with NumPy underlines.

## Names

| Layer | Rule | Examples |
|-------|------|----------|
| Public methods on `PyDAS` | Frozen. Never rename. | `add_channel`, `updateST`, `updateChN`, `to_fullscale` |
| Public mixin parameters | Keep historical camelCase | `chName`, `chOld`, `newOrder`, `cutoffull`, `plotbackend` |
| New internal functions | snake_case | `normalize_sseg`, `_apply_butterworth`, `update_channel_count` |
| Optional aliases | Add snake_case wrappers; do not delete the old name | `update_channel_count` → `updateChN`, `update_statistics` → `updateST` |

Internal state fields (`__fs__`, `__chN__`, `segStatis`, …) are a historical
contract. Do not name-mangle or rename them.

## Architecture

- Users import `from pydas import PyDAS` and call instance methods.
- Mixins stay thin: validate, then `return module.fn(self, ...)`.
- One implementation per algorithm (`_apply_butterworth`, `_correlation_lag`,
  `plot.lttb_downsample`, `core.io_format` pack helpers, `quality/` detect-repair-assess).
- Statistics columns are always `Mean`, `STD`, `Max`, `Min`, `Unit`.
- `data` and `segStatis` are lists of DataFrames, not dicts.
- Repair audit is `repair_log` on the object, never the frozen `.out` pack.
- Quality grades are the English ids `good` / `repaired` / `limited` / `bad`.

## Binary `.out` pack

The layout in `src/pydas/core/io_format.py` is frozen. Other software reads
the same file. Do not change:

- 256-byte file header (`=hhlhh` + `2s2s240s`)
- version `-2`, reserved `0x0D`
- 16-byte names, 4-byte units, float32 coefficients, int16 indices
- 128-byte segment alignment
- int16 payload scaled by `/32767`

`tests/unit/test_out_pack_compat.py` checks byte-for-byte compatibility.

## Line endings

`src/` and `tests/` use CRLF, matching the existing tree. Keep that when you
edit those files.

## Tests

Add pytest cases under `tests/unit` or `tests/integration` with the existing
markers. Do not put new `test_*.py` files at the `tests/` root. Do not revive
`tests/legacy` as the default suite.
