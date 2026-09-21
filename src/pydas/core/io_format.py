"""Binary ``.out`` pack layout used by both the reader and the writer.

The on-disk layout is frozen. Other software reads this format, so field
widths, endianness, reserved bytes, padding, and alignment must not change.
Keep pack and unpack in this one module so they cannot drift apart.
"""
import math
import struct

FILE_HEADER_SIZE = 256
FILE_HEADER_FMT = "=hhlhh2s2s240s"
SEG_HEADER_FMT = "=hhlBBBBBBBB240s"
CH_NAME_WIDTH = 16
CH_UNIT_WIDTH = 4
ALIGN = 128
INT16_SCALE = 32767
FILE_VERSION = -2
RESERVED = 0x0D


def align_offset(pos):
    """Return the next 128-byte aligned file offset."""
    return ALIGN * math.ceil(pos / ALIGN)


def pack_file_header(ch_n, fs, nseg, date_mm, date_dd, desc):
    """Pack the 256-byte file header."""
    buf = struct.pack("=hhlhh", FILE_VERSION, ch_n, RESERVED, int(fs), nseg)
    buf += struct.pack(
        "2s2s240s",
        str(date_mm).encode("utf-8"),
        str(date_dd).encode("utf-8"),
        str(desc or "").encode("utf-8"),
    ).replace(b"\x00", b" ")
    return buf


def unpack_file_header(buf):
    """Unpack a 256-byte file header into a dict."""
    tmp = struct.unpack(FILE_HEADER_FMT, buf)
    return {
        "index": tmp[0],
        "chN": tmp[1],
        "fs": tmp[3],
        "segN": tmp[4],
        "date_mm": tmp[5].decode("utf-8"),
        "date_dd": tmp[6].decode("utf-8"),
        "desc": tmp[7].decode("utf-8").rstrip(),
    }


def pack_channel_names(names):
    """Pack channel names as 16-byte space-padded records."""
    return struct.pack(
        len(names) * f"{CH_NAME_WIDTH}s",
        *[str(name).encode("utf-8") for name in names],
    ).replace(b"\x00", b" ")


def pack_channel_units(units):
    """Pack channel units as 4-byte space-padded records."""
    return struct.pack(
        len(units) * f"{CH_UNIT_WIDTH}s",
        *[str(unit).encode("utf-8") for unit in units],
    ).replace(b"\x00", b" ")


def unpack_channel_names(buf, ch_n):
    """Unpack ``ch_n`` 16-byte channel name records."""
    return [
        name.decode("utf-8").rstrip()
        for name in struct.unpack(ch_n * f"{CH_NAME_WIDTH}s", buf)
    ]


def unpack_channel_units(buf, ch_n):
    """Unpack ``ch_n`` 4-byte channel unit records."""
    return [
        unit.decode("utf-8").rstrip()
        for unit in struct.unpack(ch_n * f"{CH_UNIT_WIDTH}s", buf)
    ]


def unpack_seg_header(buf):
    """Unpack a 256-byte segment header."""
    return struct.unpack(SEG_HEADER_FMT, buf)
