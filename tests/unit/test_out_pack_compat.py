# tests/unit/test_out_pack_compat.py
"""Lock the on-disk *.out layout against the pre-Phase-3 writer.

Other software reads this format. The current writer must emit the same
bytes as the historical pack sequence for the same object state.
"""
import math
import re
import struct

import numpy as np

from pydas.output import write_data


def _legacy_write_data(pydas_obj, filename, sseg="all"):
    """Replica of master write_data packing (do not 'improve' this)."""
    if not filename.endswith(".out"):
        filename += ".out"

    if sseg == "all":
        sseg = list(range(pydas_obj.__segN__))
    elif isinstance(sseg, int):
        sseg = [sseg]
    else:
        sseg = list(range(pydas_obj.__segN__))

    with open(filename, "wb") as fOut:
        datemmdd = pydas_obj.__date__.split("-")
        buf = struct.pack(
            "=hhlhh",
            -2,
            pydas_obj.__chN__,
            0x0D,
            int(pydas_obj.__fs__),
            len(sseg),
        )
        buf += struct.pack(
            "2s2s240s",
            datemmdd[0].encode("utf-8"),
            datemmdd[1].encode("utf-8"),
            pydas_obj.__desc__.encode("utf-8"),
        ).replace(b"\x00", b" ")
        if fOut.write(buf) != 256:
            raise IOError("Failed to write file header")

        fOut.write(
            struct.pack(
                pydas_obj.__chN__ * "16s",
                *[
                    pydas_obj.chInfo["Name"].iloc[i].encode("utf-8")
                    for i in range(pydas_obj.__chN__)
                ],
            ).replace(b"\x00", b" ")
        )
        fOut.write(
            struct.pack(
                pydas_obj.__chN__ * "4s",
                *[
                    pydas_obj.chInfo["Unit"].iloc[i].encode("utf-8")
                    for i in range(pydas_obj.__chN__)
                ],
            ).replace(b"\x00", b" ")
        )

        chMagMax = np.amax(
            np.array([np.amax(abs(pydas_obj.data[i].values), axis=0) for i in sseg]),
            axis=0,
        )
        chCoef_ = (chMagMax / 32767).astype(np.float32)
        fOut.write(struct.pack("=" + pydas_obj.__chN__ * "f", *chCoef_))
        fOut.write(struct.pack("=" + pydas_obj.__chN__ * "h", *pydas_obj.chInfo.index))

        for iseg in sseg:
            p_cur = fOut.tell()
            fOut.seek(128 * math.ceil(p_cur / 128))
            fOut.write(struct.pack("=h", pydas_obj.segInfo["Type"].iloc[iseg]))
            fOut.write(struct.pack("=h", pydas_obj.__chN__))
            fOut.write(struct.pack("=l", pydas_obj.segInfo["N sample"].iloc[iseg] + 5))
            start_time_parts = re.split(r":|\.", pydas_obj.segInfo.Start.iloc[iseg])[::-1]
            stop_time_parts = re.split(r":|\.", pydas_obj.segInfo.Stop.iloc[iseg])[::-1]
            time_parts_int = list(map(int, start_time_parts + stop_time_parts))
            fOut.write(struct.pack(8 * "B", *time_parts_int))
            fOut.write(
                struct.pack(
                    "240s", pydas_obj.segInfo.Note.iloc[iseg].encode("utf-8")
                ).replace(b"\x00", b" ")
            )
            mean_ = np.mean(pydas_obj.data[iseg].values, axis=0) / chCoef_
            std_ = np.std(pydas_obj.data[iseg].values, axis=0) / chCoef_
            max_ = np.amax(pydas_obj.data[iseg].values, axis=0) / chCoef_
            min_ = np.amin(pydas_obj.data[iseg].values, axis=0) / chCoef_
            fOut.write(
                struct.pack("=" + pydas_obj.__chN__ * "h", *np.round(mean_).astype(np.int16))
            )
            fOut.write(struct.pack("=" + pydas_obj.__chN__ * "f", *std_))
            fOut.write(
                struct.pack("=" + pydas_obj.__chN__ * "h", *np.round(max_).astype(np.int16))
            )
            fOut.write(
                struct.pack("=" + pydas_obj.__chN__ * "h", *np.round(min_).astype(np.int16))
            )
            raw_ = np.round(
                pydas_obj.data[iseg].values
                / np.repeat(
                    chCoef_.reshape(1, -1),
                    pydas_obj.segInfo["N sample"].iloc[iseg],
                    axis=0,
                )
            ).astype(np.int16)
            fOut.write(raw_.tobytes())


def test_write_data_bytes_match_legacy_pack(pydas_instance, tmp_path):
    """Current writer must be a byte-for-byte match of the frozen layout."""
    new_path = tmp_path / "current.out"
    old_path = tmp_path / "legacy.out"
    write_data(pydas_instance, str(new_path), sseg="all")
    _legacy_write_data(pydas_instance, str(old_path), sseg="all")
    new_bytes = new_path.read_bytes()
    old_bytes = old_path.read_bytes()
    assert len(new_bytes) == len(old_bytes)
    assert new_bytes == old_bytes
    # File header is 256 bytes and starts with version -2 (little endian).
    assert new_bytes[:2] == b"\xfe\xff"
