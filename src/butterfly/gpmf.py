"""GoPro GPMF telemetry: stream loading and GPS decoding."""

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

