"""Candidate windows, diversity-aware (MMR) selection, and the cut-length plan."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
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

def photo_candidate(c: Clip, L: float) -> Pick | None:
    """A usable photo as one candidate cut (start 0, a slot of length L); None if it is gated."""
    if float(c.q[0]) < C.GATES["window_min_q"]:
        return None
    emb = None
    if c.emb is not None and len(c.emb):
        emb = c.emb[0].astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-9
    return Pick(c, 0.0, L, float(c.score[0]) * C.PHOTO_SCORE_FACTOR, why(c, 0, 1), emb)


def moment(p: Pick) -> datetime:
    """When the picked moment happened (UTC): the clip's start time plus the cut's offset."""
    return p.clip.created + timedelta(seconds=p.start)


def select(clips: list[Clip], durations: list[float], diversity: float = C.DIVERSITY,
           photos: str = "auto") -> list[Pick]:
    """MMR over video windows and photos in one pool -> picks in time order, with planned durations.

    photos: "auto" (photos fill at most MAX_PHOTO_SHARE of the cuts), "off", or "only".
    """
    L = float(np.median(durations))
    n_seg = len(durations)
    videos = [] if photos == "only" else [c for c in clips if c.kind == "video"]
    stills = [] if photos == "off" else [c for c in clips if c.kind == "photo"]
    pool = [p for c in videos for p in candidates(c, L)]
    pool += [p for p in (photo_candidate(c, L) for c in stills) if p is not None]
    if not pool:
        return []
    cap = max(2, math.ceil(n_seg / max(len(videos), 1)) + C.PICKS_PER_CLIP_SLACK)   # per video clip
    max_photos = n_seg if photos == "only" else int(C.MAX_PHOTO_SHARE * n_seg)
    top = max(p.score for p in pool) or 1.0

    chosen: list[Pick] = []
    per_clip: dict[Path, int] = {}
    n_photos = 0
    remaining = list(pool)
    while remaining and len(chosen) < n_seg:
        best, best_val = None, -np.inf
        for p in remaining:
            if p.clip.kind == "photo":
                if n_photos >= max_photos:
                    continue
            elif per_clip.get(p.clip.path, 0) >= cap:
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
        n_photos += best.clip.kind == "photo"
        per_clip[best.clip.path] = per_clip.get(best.clip.path, 0) + 1
        remaining.remove(best)

    chosen.sort(key=lambda p: (moment(p), p.clip.path.name))
    for p, d in zip(chosen, durations):
        if p.clip.kind == "photo":
            p.dur = d
            continue
        centre = p.start + p.dur / 2
        p.dur = min(d, p.clip.duration)
        p.start = float(np.clip(centre - p.dur / 2, 0, p.clip.duration - p.dur))
    return chosen
