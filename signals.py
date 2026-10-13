"""Telemetry and audio features on the common DT grid."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from config import DT, GATES


def bin_mean(t: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    """Average samples into n bins of DT; empty bins interpolated."""
    if len(t) == 0:
        return np.zeros(n)
    idx = np.clip((t / DT).astype(int), 0, n - 1)
    sums = np.bincount(idx, weights=v, minlength=n)
    cnts = np.bincount(idx, minlength=n)
    have = cnts > 0
    out = np.zeros(n)
    out[have] = sums[have] / cnts[have]
    if have.sum() >= 2 and not have.all():
        g = np.arange(n)
        out[~have] = np.interp(g[~have], g[have], out[have])
    return out


def _rate(t: np.ndarray) -> float:
    return max((len(t) - 1) / max(t[-1] - t[0], 1e-3), 1.0)


def detect_drops(t: np.ndarray, accel: np.ndarray) -> list[tuple[float, float]]:
    """Free-fall runs: |accel| below threshold for at least drop_min_s."""
    mag = np.linalg.norm(accel, axis=1)
    low = mag < GATES["drop_accel"]
    min_len = max(int(GATES["drop_min_s"] * _rate(t)), 2)
    drops, i = [], 0
    while i < len(low):
        if low[i]:
            j = i
            while j < len(low) and low[j]:
                j += 1
            if j - i >= min_len:
                drops.append((float(t[i]), float(t[j - 1])))
            i = j
        else:
            i += 1
    return drops


def telemetry_features(tel: dict, n: int) -> tuple[dict, dict]:
    """-> (features {name: array[n]}, events {drops: [(s,e)], water_entries: [t]})"""
    f, ev = {}, {"drops": [], "water_entries": []}

    if "accel" in tel:
        t, a = tel["accel"]
        ev["drops"] = detect_drops(t, a)

    if "gyro" in tel:
        t, g = tel["gyro"]
        r = _rate(t)
        lp = uniform_filter1d(g, size=max(int(0.5 * r), 1), axis=0)     # intentional motion
        hp2 = ((g - lp) ** 2).sum(axis=1)                                  # erratic motion
        shake = np.sqrt(np.clip(uniform_filter1d(hp2, size=max(int(0.25 * r), 1)), 0, None))
        f["shake"] = bin_mean(t, shake, n)
        f["turn"] = uniform_filter1d(bin_mean(t, np.linalg.norm(lp, axis=1), n), size=10)

    if "grav" in tel:
        t, gv = tel["grav"]
        norm = np.linalg.norm(gv, axis=1, keepdims=True)
        gv = gv / np.where(norm > 0, norm, 1)
        ref = np.median(gv, axis=0)
        ref /= np.linalg.norm(ref) or 1
        ang = np.degrees(np.arccos(np.clip(gv @ ref, -1, 1)))
        f["tumble"] = bin_mean(t, ang, n)

    if "speed" in tel:
        t, v = tel["speed"]
        if len(t) >= 2:
            f["speed"] = np.interp(np.arange(n) * DT, t, v)

    if "wet" in tel:
        t, w = tel["wet"]
        wet = uniform_filter1d(bin_mean(t, w, n), size=10) > 0.5
        f["wet"] = wet.astype(float)
        ev["water_entries"] = [float(i * DT) for i in np.flatnonzero(np.diff(wet.astype(int)) == 1) + 1]
    return f, ev


def audio_features(path: Path, n: int) -> dict:
    """Loudness (dB) and peaks above a 5 s rolling median, on the DT grid."""
    sr = 8000
    try:
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                              "-ar", str(sr), "-f", "s16le", "-"],
                             capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        return {}
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(sr * DT)
    frames = min(len(x) // hop, n)
    if frames < 2:
        return {}
    rms = np.sqrt((x[:frames * hop].reshape(frames, hop) ** 2).mean(axis=1))
    db = np.full(n, -100.0)
    db[:frames] = 20 * np.log10(rms + 1e-5)
    base = median_filter(db, size=51, mode="nearest")
    return {"loudness": db, "audio_peak": np.clip(db - base, 0, None)}
