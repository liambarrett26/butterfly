"""Candidate windows, diversity-aware (MMR) selection, and the cut-length plan."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as C
from .analyse import Clip
from .score import why


@dataclass
class Pick:
    clip: Clip
    start: float
    dur: float
    score: float
    reasons: str
    emb: np.ndarray | None = None


# ------------------------------------------------------------------ cut plan

def plan_durations(length: float, seg: float, music: Path | None, music_start: float) -> list[float]:
    if music is None:
        n = max(1, round(length / seg))
        return [length / n] * n
    import librosa
    y, sr = librosa.load(str(music), sr=22050, mono=True, offset=music_start, duration=length + 10)
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 4:
        print("  (no clear beat in the music — using even cuts)")
        return plan_durations(length, seg, None, 0)
    period = float(np.median(np.diff(beats)))
    k = min((1, 2, 4, 8), key=lambda b: abs(b * period - seg))  # musical grouping
    bounds = [0.0] + [b for b in beats[k::k] if b <= length + 1e-6]
    if length - bounds[-1] > 0.3:
        bounds.append(length)
    print(f"  music: ~{60 / period:.0f} BPM, cutting every {k} beat(s) ≈ {k * period:.2f} s")
    return list(np.diff(bounds))


# ------------------------------------------------------------------ candidates

def _window_emb(c: Clip, s: float, e: float) -> np.ndarray | None:
    if c.emb is None or len(c.emb) == 0:
        return None
    m = (c.emb_t >= s) & (c.emb_t <= e)
    v = c.emb[m].astype(np.float32).mean(axis=0) if m.any() else \
        c.emb[int(np.argmin(np.abs(c.emb_t - (s + e) / 2)))].astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def candidates(c: Clip, L: float) -> list[Pick]:
    G = C.GATES
    if c.duration <= L:
        starts = [0.0]
        L = c.duration
    else:
        starts = list(np.arange(0, c.duration - L + 1e-6, C.WINDOW_STRIDE))
    win = []
    for s in starts:
        lo, hi = int(s / C.DT), max(int((s + L) / C.DT), int(s / C.DT) + 1)
        qwin = c.q[lo:hi]
        qmin = float(np.percentile(qwin, 5))
        if qmin < G["window_min_q"]:
            continue  # part of this cut is unusable (drop, handling, black…)
        sc = float(c.score[lo:hi].mean()) * (0.7 + 0.3 * qmin)
        win.append((sc, s, lo, hi))
    win.sort(reverse=True)

    kept: list[Pick] = []
    for sc, s, lo, hi in win:  # non-max suppression
        if all(abs(s - k.start) >= L for k in kept):
            kept.append(Pick(c, s, L, sc, why(c, lo, hi), _window_emb(c, s, s + L)))
            if len(kept) >= C.CANDIDATES_PER_CLIP:
                break
    return kept


# ------------------------------------------------------------------ selection

def select(clips: list[Clip], durations: list[float], diversity: float = C.DIVERSITY) -> list[Pick]:
    L = float(np.median(durations))
    clips = [c for c in clips if c.kind == "video"]          # photos join selection in v0.5 step 2
    pool = [p for c in clips for p in candidates(c, L)]
    if not pool:
        return []
    n_seg = len(durations)
    cap = max(2, math.ceil(n_seg / max(len(clips), 1)) + C.PICKS_PER_CLIP_SLACK)
    top = max(p.score for p in pool) or 1.0

    chosen: list[Pick] = []
    per_clip: dict[Path, int] = {}
    remaining = list(pool)
    while remaining and len(chosen) < n_seg:
        best, best_val = None, -np.inf
        for p in remaining:
            if per_clip.get(p.clip.path, 0) >= cap:
                continue
            red = 0.0
            if p.emb is not None:
                sims = [float(p.emb @ q.emb) for q in chosen if q.emb is not None]
                red = max(sims) if sims else 0.0
            val = (1 - diversity) * (p.score / top) - diversity * red
            if val > best_val:
                best, best_val = p, val
        if best is None:
            break
        chosen.append(best)
        per_clip[best.clip.path] = per_clip.get(best.clip.path, 0) + 1
        remaining.remove(best)

    chosen.sort(key=lambda p: (p.clip.created, p.clip.path.name, p.start))
    for p, d in zip(chosen, durations):
        centre = p.start + p.dur / 2
        p.dur = min(d, p.clip.duration)
        p.start = float(np.clip(centre - p.dur / 2, 0, p.clip.duration - p.dur))
    return chosen
