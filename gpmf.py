"""GoPro GPMF telemetry: stream loading, GPS decoding, HiLight tags."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

STREAMS = ["ACCL", "GYRO", "GRAV", "GPS9", "GPS5", "MWET"]
GPS9_SCALES = np.array([1e7, 1e7, 1e3, 1e3, 1e2, 1, 1e3, 1e2, 1])


def _times(s) -> np.ndarray:
    """Video-timeline seconds if available (pts_data), else the GPMF clock."""
    if s.pts_data:
        return np.array([t for t, _ in s.pts_data], dtype=float)
    return np.array([t for t, _ in s.data], dtype=float) / 1000.0


def _values(s) -> list:
    return [v for _, v in (s.pts_data or s.data)]


def _extract(path: Path) -> dict:
    from telemetrik import extract_all_telemetry
    try:
        return extract_all_telemetry(str(path), streams=STREAMS)
    except ValueError:
        return {}  # no GPMF track
    except Exception:
        out = {}
        for k in STREAMS:  # a single odd stream can break bulk parsing
            try:
                out.update(extract_all_telemetry(str(path), streams=[k]))
            except Exception:
                pass
        return out


def load_telemetry(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {name: (t_seconds, values)} with values as float arrays.

    Names: accel (N,3) m/s², gyro (N,3) rad/s, grav (N,3) unit, speed (N,) m/s, wet (N,)
    """
    streams = _extract(path)
    out = {}
    for key, name in (("ACCL", "accel"), ("GYRO", "gyro"), ("GRAV", "grav")):
        if key in streams:
            s = streams[key]
            try:
                out[name] = (_times(s), np.array(_values(s), dtype=float).reshape(len(s.data), -1))
            except (ValueError, TypeError):
                pass

    t_gps, v_gps = [], []
    if "GPS9" in streams:
        s = streams["GPS9"]
        for t, v in zip(_times(s), _values(s)):
            if isinstance(v, (bytes, bytearray)):
                for off in range(0, len(v) - 31, 32):
                    rec = np.array(struct.unpack(">7i2H", bytes(v[off:off + 32])), float) / GPS9_SCALES
                    if rec[8] >= 3:
                        t_gps.append(t)
                        v_gps.append(rec[3])
    elif "GPS5" in streams:
        s = streams["GPS5"]
        for t, v in zip(_times(s), _values(s)):
            if len(v) >= 7 and v[5] >= 3:
                t_gps.append(t)
                v_gps.append(v[3])
    if len(t_gps) >= 2:
        tg, vg = np.array(t_gps), np.array(v_gps)
        ok = vg < 70  # drop GPS glitches
        out["speed"] = (tg[ok], vg[ok])

    if "MWET" in streams:
        s = streams["MWET"]
        vals = [v[0] if isinstance(v, (list, tuple)) else v for v in _values(s)]
        try:
            out["wet"] = (_times(s), (np.array(vals, dtype=float) > 0).astype(float))
        except (ValueError, TypeError):
            pass
    return out


# ------------------------------------------------------------------ HiLights

def _iter_boxes(f, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        size, typ = struct.unpack(">I4s", f.read(8))
        hdr = 8
        if size == 1:
            size, hdr = struct.unpack(">Q", f.read(8))[0], 16
        elif size == 0:
            size = end - pos
        if size < hdr:
            break
        yield typ, pos, hdr, size
        pos += size


def _walk_klv(buf: bytes, off: int, end: int, depth: int = 0):
    while off + 8 <= end:
        key = buf[off:off + 4]
        if key == b"\0\0\0\0":
            break
        typ = chr(buf[off + 4]) if buf[off + 4] else "\0"
        length = buf[off + 5] * int.from_bytes(buf[off + 6:off + 8], "big")
        data = buf[off + 8:off + 8 + length]
        yield depth, key.decode("latin-1"), typ, data
        if typ == "\0":
            yield from _walk_klv(buf, off + 8, off + 8 + length, depth + 1)
        off += 8 + ((length + 3) & ~3)


def read_hilights(path: Path) -> tuple[list[float], list[str]]:
    """In-camera HiLight tag times (s) + notes. Legacy HMMT box and HERO13 GPMF device."""
    tags, notes = [], []
    size = path.stat().st_size
    with open(path, "rb") as f:
        moov = next(((p, h, s) for t, p, h, s in _iter_boxes(f, 0, size) if t == b"moov"), None)
        if not moov:
            return tags, notes
        udta = next(((p, h, s) for t, p, h, s in _iter_boxes(f, moov[0] + moov[1], moov[0] + moov[2])
                     if t == b"udta"), None)
        if not udta:
            return tags, notes
        for typ, p, h, s in _iter_boxes(f, udta[0] + udta[1], udta[0] + udta[2]):
            f.seek(p + h)
            payload = f.read(s - h)
            if typ == b"HMMT" and len(payload) >= 4:
                count = struct.unpack(">I", payload[:4])[0]
                tags += [ms / 1000 for ms in struct.unpack(f">{count}I", payload[4:4 + 4 * count])]
            elif typ == b"GPMF":
                in_hl = False
                for depth, key, t, data in _walk_klv(payload, 0, len(payload)):
                    if depth == 0 and key == "DEVC":
                        in_hl = False
                    elif key in ("DVID", "DVNM") and data[:4] in (b"HLMT", b"High"):
                        in_hl = True
                    elif in_hl and t in ("L", "J") and key != "DVID":
                        sz, code = (4, "I") if t == "L" else (8, "Q")
                        vals = struct.unpack(f">{len(data) // sz}{code}", data[:len(data) // sz * sz])
                        tags += [v / 1000 for v in vals]
                        notes.append(f"HiLights from GPMF key {key} (unverified layout)")
    return sorted(set(round(t, 3) for t in tags if t > 0)), notes
