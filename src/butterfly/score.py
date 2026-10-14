"""Folder-wide normalisation, quality gates, interest and final score."""

from __future__ import annotations

import numpy as np

from . import config as C
from .analyse import Clip

G = C.GATES


def _ramp(x, lo, hi):
    """0 at/below lo, 1 at/above hi, linear between."""
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def _folder_norm(clips: list[Clip], key: str, steady: bool = False) -> dict:
    """Per-clip arrays scaled against the whole folder: p50 -> 0, p95 -> 1, capped at 3.

    steady=True takes the percentiles only from moments without visible camera shake: optical
    flow is unreliable while the image jitters, and shaky footage shouldn't set everyone's baseline.
    """
    have = [c for c in clips if key in c.feats]
    if not have:
        return {}
    vals = np.concatenate([c.feats[key] for c in have])
    if steady:
        calm = np.concatenate([c.feats["camera_shake"] < G["cam_shake_lo"] if "camera_shake" in c.feats
                               else np.ones(len(c.feats[key]), bool) for c in have])
        vals = vals[calm] if calm.any() else vals
    p50, p95 = np.percentile(vals, [50, 95])
    return {id(c): np.clip((c.feats[key] - p50) / (p95 - p50 + 1e-9), 0, 3) for c in have}


def _bumps(n: int, centres: list[float], sigma: float) -> np.ndarray:
    t = np.arange(n) * C.DT
    out = np.zeros(n)
    for c in centres:
        out += np.exp(-((t - c) ** 2) / (2 * sigma ** 2))
    return np.clip(out, 0, 1.5)


def _people_parts(f: dict, n: int) -> dict:
    """Faces and people as absolute, unweighted interest parts (presence means the same in any folder)."""
    if "face_area" not in f:
        return {}
    return {"faces": _ramp(f["face_area"][:n], C.FACE_LO, C.FACE_HI),
            "people": np.clip(f["n_people"][:n] / 3, 0, 1)}


def _weigh(c: Clip, raw: dict, W: dict) -> dict:
    """Keep the unweighted parts (ranker features) and return the weighted ones."""
    c.raw = raw
    return {k: W.get(k, 0.0) * v for k, v in raw.items()}


def compute(clips: list[Clip], weights: dict | None = None) -> None:
    """Gates -> q, parts -> interest, score = interest × q. Videos and photos are normalised separately.

    weights: interest weights overriding config.INTEREST_WEIGHTS (e.g. from a learned ranker).
    """
    W = {**C.INTEREST_WEIGHTS, **(weights or {})}
    _compute_videos([c for c in clips if c.kind == "video"], W)
    _compute_photos([c for c in clips if c.kind == "photo"], W)


def _compute_photos(photos: list[Clip], W: dict) -> None:
    """A photo is one frame: blur (against the folder's photos), exposure and CLIP gates; semantic,
    aesthetic and colour interest, normalised against photos only."""
    if not photos:
        return
    P = C.PHOTO
    norm = {k: _folder_norm(photos, k) for k in ("semantic", "aesthetic", "colour")}
    mid = float(np.median([c.feats["sharpness"][0] for c in photos]))
    for c in photos:
        f, z = c.feats, np.zeros(1)
        get = lambda k: norm[k].get(id(c), z) if norm[k] else z
        gates = {
            "blur": P["blur_floor"] + (1 - P["blur_floor"]) * _ramp(f["sharpness"] / (mid + 1e-9), P["blur_lo"], P["blur_hi"]),
            "dark": G["dark_penalty"] + (1 - G["dark_penalty"]) * _ramp(f["luma"], G["dark_luma"] - 10, G["dark_luma"] + 10),
            "blown": 1 - (1 - G["clipped_penalty"]) * _ramp(f["blown"], G["clipped_frac"], G["clipped_frac"] + 0.2),
        }
        if "clip_bad" in f:
            gates["clip_bad"] = 1 - (1 - G["clip_bad_floor"]) * _ramp(f["clip_bad"], G["clip_bad_start"], G["clip_bad_full"])
        q = np.ones(1)
        for g in gates.values():
            q = q * g
        parts = _weigh(c, {**{k: get(k) for k in ("semantic", "aesthetic", "colour")}, **_people_parts(f, 1)}, W)
        c.gates, c.parts, c.q = gates, parts, q
        c.interest = sum(parts.values()) + 0.05
        c.score = c.interest * q


def _compute_videos(clips: list[Clip], W: dict) -> None:
    if not clips:
        return
    norm = {k: _folder_norm(clips, k) for k in ("semantic", "aesthetic", "colour", "speed", "turn", "audio_peak", "shake")}
    norm["subject_motion"] = _folder_norm(clips, "subject_motion", steady=True)

    sharp_all = np.concatenate([c.feats["sharpness"] for c in clips if "sharpness" in c.feats] or [np.zeros(0)])
    blur_thr = np.percentile(sharp_all, G["blur_pct"]) if sharp_all.size else None
    sharp_mid = np.percentile(sharp_all, 50) if sharp_all.size else None

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
        riding = (f["speed"][:n] > G["shake_ride_speed"]) if "speed" in f else np.zeros(n, bool)
        if "shake" in f:                                         # gyro: 0 calm … 1 folder-p95 shake
            shaky = np.clip(get("shake"), 0, 1)
            # HyperSmooth hides much gyro shake, so it needs blur or tumble to count in full
            handling = shaky * (0.5 + 0.5 * np.maximum(blurry, tumble))
        elif "camera_shake" in f:                                # no gyro: shake seen in the image itself
            shaky = _ramp(f["camera_shake"][:n], G["cam_shake_lo"], G["cam_shake_hi"])
            handling = shaky
        else:
            shaky = handling = z
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
        if "ev_wind" in f:
            gates["wind"] = 1 - G["wind_penalty"] * _ramp(f["ev_wind"][:n], G["wind_lo"], G["wind_hi"])
        edge = np.ones(n)
        edge[(t < G["edge_s"]) | (t > c.duration - G["edge_s"])] = G["edge_penalty"]
        gates["edge"] = edge

        q = np.ones(n)
        for g in gates.values():
            q *= g

        # ---------------------------------------------------------- interest
        parts = _weigh(c, {
            "semantic": get("semantic"),
            "aesthetic": get("aesthetic"),
            "colour": get("colour"),
            # motion only counts when it's smooth, intentional motion
            "action": (get("speed") + 0.5 * get("turn")) * (1 - np.clip(shaky, 0, 1)),
            "subject": get("subject_motion") * (1 - np.clip(shaky, 0, 1)),
            "audio": get("audio_peak"),
            **{e: _ramp(f[f"ev_{e}"][:n], C.EVENT_LO, C.EVENT_HI)
               for e in ("laughter", "cheering", "splash", "speech") if f"ev_{e}" in f},
            **_people_parts(f, n),
            "water": _bumps(n, [w + 0.5 for w in c.events.get("water_entries", [])], 1.0),
        }, W)
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
