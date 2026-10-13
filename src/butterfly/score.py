"""Folder-wide normalisation, quality gates, interest and final score."""

from __future__ import annotations

import numpy as np

from . import config as C
from .analyse import Clip

G = C.GATES


def _ramp(x, lo, hi):
    """0 at/below lo, 1 at/above hi, linear between."""
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def _folder_norm(clips: list[Clip], key: str) -> dict:
    """Per-clip arrays scaled against the whole folder: p50 -> 0, p95 -> 1, capped at 3."""
    have = [c for c in clips if key in c.feats]
    if not have:
        return {}
    vals = np.concatenate([c.feats[key] for c in have])
    p50, p95 = np.percentile(vals, [50, 95])
    return {id(c): np.clip((c.feats[key] - p50) / (p95 - p50 + 1e-9), 0, 3) for c in have}


def _bumps(n: int, centres: list[float], sigma: float) -> np.ndarray:
    t = np.arange(n) * C.DT
    out = np.zeros(n)
    for c in centres:
        out += np.exp(-((t - c) ** 2) / (2 * sigma ** 2))
    return np.clip(out, 0, 1.5)


def compute(clips: list[Clip]) -> None:
    norm = {k: _folder_norm(clips, k)
            for k in ("semantic", "aesthetic", "colour", "speed", "turn", "audio_peak", "shake")}

    sharp_all = np.concatenate([c.feats["sharpness"] for c in clips if "sharpness" in c.feats] or [np.zeros(0)])
    blur_thr = np.percentile(sharp_all, G["blur_pct"]) if sharp_all.size else None
    sharp_mid = np.percentile(sharp_all, 50) if sharp_all.size else None

    W = C.INTEREST_WEIGHTS
    for c in clips:
        n, f = c.n, c.feats
        t = np.arange(n) * C.DT
        z = np.zeros(n)
        get = lambda k: norm[k].get(id(c), z)[:n] if norm[k] else z

        # ---------------------------------------------------------- quality gates
        gates = {}

        drop = np.ones(n)
        for s, e in c.events.get("drops", []):
            drop[(t >= s - G["drop_before"]) & (t <= e + G["drop_after"])] = 0.0
        gates["drop"] = drop

        blurry = (1 - np.clip(f["sharpness"][:n] / (sharp_mid + 1e-9), 0, 1)) if sharp_mid else np.full(n, 0.5)
        tumble = _ramp(f["tumble"][:n], G["tumble_deg"] - 15, G["tumble_deg"] + 15) if "tumble" in f else z
        shaky = np.clip(get("shake"), 0, 1)                      # 0 calm … 1 folder-p95 shake
        riding = (f["speed"][:n] > G["shake_ride_speed"]) if "speed" in f else np.zeros(n, bool)
        handling = shaky * (0.5 + 0.5 * np.maximum(blurry, tumble))
        handling = np.where(riding, handling * G["shake_ride_factor"], handling)
        gates["handling"] = 1 - G["shake_penalty"] * handling

        gates["tumble"] = 1 - (1 - G["tumble_penalty"]) * tumble
        if blur_thr is not None:
            gates["blur"] = G["blur_penalty"] + (1 - G["blur_penalty"]) * np.clip(f["sharpness"][:n] / (blur_thr + 1e-9), 0, 1)
        if "luma" in f:
            gates["dark"] = G["dark_penalty"] + (1 - G["dark_penalty"]) * _ramp(f["luma"][:n], G["dark_luma"] - 10, G["dark_luma"] + 10)
            gates["blown"] = 1 - (1 - G["clipped_penalty"]) * _ramp(f["blown"][:n], G["clipped_frac"], G["clipped_frac"] + 0.2)
        if "clip_bad" in f:
            gates["clip_bad"] = 1 - (1 - G["clip_bad_floor"]) * _ramp(f["clip_bad"][:n], G["clip_bad_start"], G["clip_bad_full"])
        edge = np.ones(n)
        edge[(t < G["edge_s"]) | (t > c.duration - G["edge_s"])] = G["edge_penalty"]
        gates["edge"] = edge

        q = np.ones(n)
        for g in gates.values():
            q *= g

        # ---------------------------------------------------------- interest
        parts = {
            "semantic": W["semantic"] * get("semantic"),
            "aesthetic": W["aesthetic"] * get("aesthetic"),
            "colour": W["colour"] * get("colour"),
            # motion only counts when it's smooth, intentional motion
            "action": W["action"] * (get("speed") + 0.5 * get("turn")) * (1 - np.clip(shaky, 0, 1)),
            "audio": W["audio"] * get("audio_peak"),
            "water": W["water"] * _bumps(n, [w + 0.5 for w in c.events.get("water_entries", [])], 1.0),
        }
        interest = sum(parts.values()) + 0.05

        c.gates, c.parts, c.q, c.interest = gates, parts, q, interest
        c.score = interest * q


def why(c: Clip, lo: int, hi: int) -> str:
    """Short human explanation of a window: top interest terms + any notable gate."""
    contrib = sorted(((k, float(v[lo:hi].mean())) for k, v in c.parts.items()), key=lambda kv: -kv[1])
    bits = [f"{k} {v:.2f}" for k, v in contrib[:3] if v > 0.05]
    worst = min(((k, float(g[lo:hi].mean())) for k, g in c.gates.items()), key=lambda kv: kv[1])
    if worst[1] < 0.85:
        bits.append(f"penalised: {worst[0]} ×{worst[1]:.2f}")
    return ", ".join(bits) or "quiet"


def unusable_fraction(c: Clip) -> float:
    return float((c.q < G["window_min_q"]).mean())
