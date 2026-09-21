"""Channel operations for PyDAS.

Implementation lives here; :class:`ChannelMixin` only forwards so the
public ``data.add_channel(...)`` API stays unchanged.
"""
import logging
import math

import numpy as np

from ..utils import data_change_fs
from .state import normalize_sseg, require_channel, write_channel_stats

logger = logging.getLogger(__name__)


def add_channel(pydas_obj, name, unit, series, fs, coef=1, point_of_move=0, sseg=0):
    """Add a new channel to the object."""
    if name in pydas_obj.chInfo["Name"].values:
        logger.warning("Channel '%s' already exists.", name)
        return None

    n_sample = int(pydas_obj.segInfo.iloc[sseg]["N sample"])
    if n_sample == 0:
        n_sample = len(series)
        pydas_obj.segInfo.iloc[sseg, pydas_obj.segInfo.columns.get_loc("N sample")] = n_sample
        pydas_obj.__fs__ = fs
    elif fs != pydas_obj.__fs__:
        series = data_change_fs(series, fs, pydas_obj.__fs__)

    if len(series) > n_sample:
        series = series[:n_sample]
    elif len(series) < n_sample:
        series = np.pad(series, (0, n_sample - len(series)), "constant", constant_values=0)

    pydas_obj.data[sseg][name] = series

    new_idx = pydas_obj.__chN__ + 1
    new_row = {col: np.nan for col in pydas_obj.chInfo.columns}
    if "Name" in new_row:
        new_row["Name"] = name
    if "Unit" in new_row:
        new_row["Unit"] = unit
    if "Coef" in new_row:
        new_row["Coef"] = coef
    if "CoeffUnit" in new_row:
        new_row["CoeffUnit"] = 1.0
    if "CoeffRho" in new_row:
        new_row["CoeffRho"] = 0.0
    if "CoeffLam" in new_row:
        new_row["CoeffLam"] = 0.0
    pydas_obj.chInfo.loc[new_idx] = new_row
    write_channel_stats(pydas_obj, name, sseg)
    pydas_obj.__chN__ += 1

    if point_of_move != 0:
        pydas_obj.move_data(name, point_of_move, sseg=sseg)

    logger.info("Channel '%s' has been added", name)
    return None


def delete_channel(pydas_obj, name):
    """Delete a channel from every segment."""
    if name not in pydas_obj.chInfo["Name"].values:
        logger.warning("Channel '%s' does not exist.", name)
        return None

    idx = pydas_obj.chInfo.index[pydas_obj.chInfo["Name"] == name].tolist()[0]
    pydas_obj.chInfo = pydas_obj.chInfo.drop(idx)
    for sseg in range(pydas_obj.__segN__):
        pydas_obj.data[sseg] = pydas_obj.data[sseg].drop(name, axis=1)
        pydas_obj.segStatis[sseg] = pydas_obj.segStatis[sseg].drop(name)
    pydas_obj.__chN__ -= 1
    pydas_obj.chInfo.index = range(1, pydas_obj.__chN__ + 1)
    logger.info("Channel '%s' has been removed", name)
    return None


def select_channels(pydas_obj, chnames):
    """Keep only the named channels."""
    if isinstance(chnames, str):
        chnames = [chnames]

    valid_chnames = []
    for name in chnames:
        if name in pydas_obj.chInfo["Name"].values:
            valid_chnames.append(name)
        else:
            logger.warning("Channel '%s' does not exist and will be ignored.", name)

    if not valid_chnames:
        logger.warning("No valid channels specified.")
        return False

    keep_indices = [
        pydas_obj.chInfo.index[pydas_obj.chInfo["Name"] == name].tolist()[0]
        for name in valid_chnames
    ]
    pydas_obj.chInfo = pydas_obj.chInfo.loc[keep_indices]
    for sseg in range(pydas_obj.__segN__):
        pydas_obj.data[sseg] = pydas_obj.data[sseg][valid_chnames]
        pydas_obj.segStatis[sseg] = pydas_obj.segStatis[sseg].loc[valid_chnames]
    pydas_obj.__chN__ = len(valid_chnames)
    pydas_obj.chInfo.index = range(1, pydas_obj.__chN__ + 1)
    logger.info("Selected %s channels: %s", len(valid_chnames), ", ".join(valid_chnames))
    return True


def change_channel_order(pydas_obj, new_order, sseg=0):
    """Reorder channels in one segment."""
    if len(new_order) != pydas_obj.__chN__:
        raise ValueError("Number of channels does not match!")
    index_new = [list(pydas_obj.data[sseg].columns).index(name) + 1 for name in new_order]
    pydas_obj.chInfo = pydas_obj.chInfo.reindex(index_new)
    pydas_obj.segStatis[sseg] = pydas_obj.segStatis[sseg].reindex(new_order)
    pydas_obj.chInfo.index = np.arange(1, len(pydas_obj.chInfo) + 1)
    pydas_obj.data[sseg] = pydas_obj.data[sseg][new_order]
    update_channel_count(pydas_obj, sseg=sseg)
    logger.info("Changed the Channel order.")
    return None


def update_channel_count(pydas_obj, sseg=0):
    """Refresh ``__chN__`` after a structural channel change."""
    if pydas_obj.data[sseg].shape[1] == pydas_obj.chInfo.shape[0] == pydas_obj.segStatis[0].shape[0]:
        pydas_obj.__chN__ = pydas_obj.chInfo.shape[0]
    else:
        raise ValueError("Number of channels does not match!")
    return None


def rename_channel(pydas_obj, ch_old, ch_new, sseg=0):
    """Rename a channel in one segment and in ``chInfo``."""
    try:
        if ch_old not in pydas_obj.data[sseg].columns:
            logger.error("Channel '%s' not found in segment %s", ch_old, sseg)
            raise KeyError(f"Channel '{ch_old}' not found")
        if ch_new in pydas_obj.data[sseg].columns:
            logger.error("Channel '%s' already exists in segment %s", ch_new, sseg)
            raise ValueError(f"Channel '{ch_new}' already exists")
        pydas_obj.data[sseg].rename(columns={ch_old: ch_new}, inplace=True)
        pydas_obj.chInfo.loc[pydas_obj.chInfo["Name"] == ch_old, "Name"] = ch_new
        if ch_old in pydas_obj.segStatis[sseg].index:
            pydas_obj.segStatis[sseg].rename(index={ch_old: ch_new}, inplace=True)
        logger.info("Renamed channel '%s' to '%s' in segment %s", ch_old, ch_new, sseg)
        return True
    except Exception as exc:
        logger.error("Channel renaming failed: %s", exc)
        return False


def copy_channel(pydas_obj, ch_name, new_ch_name=None, sseg="all"):
    """Copy a channel onto a new name."""
    if not require_channel(pydas_obj, ch_name):
        logger.warning("Channel '%s' does not exist.", ch_name)
        return False
    if new_ch_name is None:
        new_ch_name = f"{ch_name}_copy"
    idx = pydas_obj.chInfo.index[pydas_obj.chInfo["Name"] == ch_name].tolist()[0]
    unit = pydas_obj.chInfo.loc[idx, "Unit"]
    coef = pydas_obj.chInfo.loc[idx, "Coef"]
    segments = normalize_sseg(pydas_obj, sseg)
    if not segments:
        return False
    if new_ch_name in pydas_obj.chInfo["Name"].values:
        logger.warning("Channel '%s' already exists. Operation canceled.", new_ch_name)
        return False
    first_seg = segments[0]
    series = pydas_obj.data[first_seg][ch_name].copy()
    add_channel(pydas_obj, new_ch_name, unit, series, pydas_obj.__fs__, coef, 0, first_seg)
    for seg in segments[1:]:
        if ch_name in pydas_obj.data[seg].columns:
            pydas_obj.data[seg][new_ch_name] = pydas_obj.data[seg][ch_name].copy()
            write_channel_stats(pydas_obj, new_ch_name, seg)
    logger.info("Channel '%s' copied to '%s'", ch_name, new_ch_name)
    return True


def channel_calculate(pydas_obj, ch1, ch2, operation, new_ch_name, sseg=0):
    """Create a new channel from an arithmetic operation on two channels."""
    if not require_channel(pydas_obj, ch1) or not require_channel(pydas_obj, ch2):
        return False

    ops_map = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide"}
    if operation in ops_map:
        operation = ops_map[operation]
    elif operation not in ["add", "subtract", "multiply", "divide"]:
        logger.warning(
            "Invalid operation '%s'. Valid operations are: %s",
            operation,
            ["'add' or '+'", "'subtract' or '-'", "'multiply' or '*'", "'divide' or '/'"],
        )
        return False

    if new_ch_name in pydas_obj.chInfo["Name"].values:
        logger.warning("Channel '%s' already exists. Operation canceled.", new_ch_name)
        return False

    ch1_idx = pydas_obj.chInfo.index[pydas_obj.chInfo["Name"] == ch1].tolist()[0]
    ch2_idx = pydas_obj.chInfo.index[pydas_obj.chInfo["Name"] == ch2].tolist()[0]
    ch1_unit = pydas_obj.chInfo.loc[ch1_idx, "Unit"]
    ch2_unit = pydas_obj.chInfo.loc[ch2_idx, "Unit"]

    if operation in ["add", "subtract"] and ch1_unit != ch2_unit:
        logger.warning(
            "Cannot %s channels with different units: '%s' and '%s'",
            operation, ch1_unit, ch2_unit,
        )
        return False

    if operation in ["add", "subtract"]:
        new_unit = ch1_unit
    elif operation == "multiply":
        if ch1_unit == "-" or ch2_unit == "-":
            new_unit = ch1_unit if ch2_unit == "-" else ch2_unit
        elif ch1_unit == "" or ch2_unit == "":
            new_unit = ch1_unit if ch2_unit == "" else ch2_unit
        else:
            new_unit = f"{ch1_unit}·{ch2_unit}"
    else:
        if ch1_unit in ("-", ""):
            new_unit = "-"
        elif ch2_unit in ("-", ""):
            new_unit = ch1_unit
        else:
            new_unit = f"{ch1_unit}/{ch2_unit}"

    segments = normalize_sseg(pydas_obj, sseg)
    if not segments:
        return False

    success = True
    first_seg = segments[0]

    def _operate(seg):
        left = pydas_obj.data[seg][ch1]
        right = pydas_obj.data[seg][ch2]
        if operation == "add":
            return left + right
        if operation == "subtract":
            return left - right
        if operation == "multiply":
            return left * right
        divisor = right.copy().replace(0, np.nan)
        return (left / divisor).fillna(0)

    try:
        result = _operate(first_seg)
        add_channel(pydas_obj, new_ch_name, new_unit, result.values, pydas_obj.__fs__, 1.0, 0, first_seg)
        for seg in segments[1:]:
            if ch1 in pydas_obj.data[seg].columns and ch2 in pydas_obj.data[seg].columns:
                result = _operate(seg)
                pydas_obj.data[seg][new_ch_name] = result
                write_channel_stats(pydas_obj, new_ch_name, seg)
    except Exception as exc:
        logger.error("Error performing %s operation: %s", operation, exc)
        if new_ch_name in pydas_obj.chInfo["Name"].values:
            delete_channel(pydas_obj, new_ch_name)
        success = False

    if success:
        ops_symbol = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}.get(operation, operation)
        logger.info("Created new channel '%s' as %s %s %s", new_ch_name, ch1, ops_symbol, ch2)
    return success


def channel_apply_function(pydas_obj, ch, func, new_ch_name, unit=None, sseg=0):
    """Apply a vectorized function or restricted expression to a channel."""
    if not require_channel(pydas_obj, ch):
        logger.warning("Channel '%s' does not exist.", ch)
        return False
    if new_ch_name in pydas_obj.chInfo["Name"].values:
        logger.warning("Channel '%s' already exists. Operation canceled.", new_ch_name)
        return False

    ch_idx = pydas_obj.chInfo.index[pydas_obj.chInfo["Name"] == ch].tolist()[0]
    ch_unit = pydas_obj.chInfo.loc[ch_idx, "Unit"]
    if unit is None:
        unit = ch_unit
        logger.info(
            "Using source unit '%s' for '%s'. Adjust the unit if the transform changes dimensionality.",
            ch_unit, new_ch_name,
        )

    segments = normalize_sseg(pydas_obj, sseg)
    if not segments:
        return False

    try:
        if isinstance(func, str):
            unsafe_terms = [
                "import", "eval", "exec", "compile", "open", "file",
                "os.", "sys.", "subprocess", "shutil", "__",
            ]
            if any(term in func for term in unsafe_terms):
                logger.error("Unsafe expression detected: %s", func)
                return False

            def apply_func(arr):
                local_vars = {"np": np, "math": math, "x": np.asarray(arr)}
                return eval(func, {"__builtins__": {}}, local_vars)
        else:
            apply_func = func

        for seg_idx, seg in enumerate(segments):
            if ch not in pydas_obj.data[seg].columns:
                logger.warning("Channel '%s' not found in segment %s, skipping.", ch, seg)
                continue
            values = np.asarray(pydas_obj.data[seg][ch].values, dtype=np.float64)
            try:
                result = np.asarray(apply_func(values), dtype=np.float64)
                if result.shape != values.shape:
                    result = np.asarray([apply_func(v) for v in values], dtype=np.float64)
            except TypeError:
                result = np.asarray([apply_func(v) for v in values], dtype=np.float64)

            if seg_idx == 0:
                add_channel(pydas_obj, new_ch_name, unit, result, pydas_obj.__fs__, 1.0, 0, seg)
            else:
                pydas_obj.data[seg][new_ch_name] = result
                write_channel_stats(pydas_obj, new_ch_name, seg)

        logger.info("Created channel '%s' by applying a function to '%s'", new_ch_name, ch)
        return True
    except Exception as exc:
        logger.error("Error applying function to channel: %s", exc)
        if new_ch_name in pydas_obj.chInfo["Name"].values:
            delete_channel(pydas_obj, new_ch_name)
        return False
