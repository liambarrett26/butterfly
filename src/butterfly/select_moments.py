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
    speed: float = 1.0                    # < 1 = slow motion: the slot (dur) is filled from dur × speed of source

    @property
    def src_dur(self) -> float:
        """Seconds of source footage the cut uses (start .. start + src_dur)."""
        return self.dur * self.speed


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
    """A usable photo as one candidate cut (start 0, a slot of length L); None if gated or a burst duplicate."""
    if float(c.q[0]) < C.GATES["window_min_q"] or c.info.get("burst_of"):
        return None
    emb = None
    if c.emb is not None and len(c.emb):
        emb = c.emb[0].astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-9
    return Pick(c, 0.0, L, float(c.score[0]) * C.PHOTO_SCORE_FACTOR, why(c, 0, 1), emb)


def same_moment(photo: Pick, video: Pick) -> bool:
    """Does the photo show what the video window already shows? Within DEDUPE_S of the window
    and CLIP-similar (DEDUPE_SIM). Needs embeddings on both; without CLIP, nothing is a duplicate."""
    if photo.emb is None or video.emb is None:
        return False
    t = photo.clip.created
    lo = video.clip.created + timedelta(seconds=video.start - C.DEDUPE_S)
    hi = video.clip.created + timedelta(seconds=video.start + video.dur + C.DEDUPE_S)
    return lo <= t <= hi and float(photo.emb @ video.emb) >= C.DEDUPE_SIM


def moment(p: Pick) -> datetime:
    """When the picked moment happened (UTC): the clip's start time plus the cut's offset."""
    return p.clip.created + timedelta(seconds=p.start)


def onset_strength(c: Clip) -> np.ndarray:
    """Per DT step: how much activity rises over the next ONSET_WINDOW_S versus the previous one,
    scaled so the clip's 95th percentile is 1. Activity is frame change, else audio loudness."""
    key = "change" if "change" in c.feats else "loudness" if "loudness" in c.feats else None
    if key is None:
        return np.zeros(c.n)
    a = np.asarray(c.feats[key][:c.n], float)
    k = max(int(round(C.ONSET_WINDOW_S / C.DT)), 1)
    cs = np.concatenate([[0.0], np.cumsum(a)])
    i = np.arange(len(a))
    hi, lo = np.minimum(i + k, len(a)), np.maximum(i - k, 0)
    after = (cs[hi] - cs[i]) / np.maximum(hi - i, 1)
    before = (cs[i] - cs[lo]) / np.maximum(i - lo, 1)
    rise = np.clip(after - before, 0, None)
    return rise / (np.percentile(rise, 95) + 1e-9)


def refine(picks: list[Pick]) -> None:
    """Smart in/out points: move each video cut by up to ±INOUT_SHIFT_S to raise its minimum q,
    then to start on a motion onset; ties keep the smallest move. Cuts stay inside their clip,
    never overlap another cut of the same clip, and never end up below window_min_q or their old q
    (5th percentile, as in window rejection)."""
    steps = int(round(C.INOUT_SHIFT_S / C.DT))
    onsets: dict[int, np.ndarray] = {}
    for p in picks:
        c = p.clip
        if c.kind != "video" or p.dur >= c.duration - 1e-6:
            continue
        on = onsets.setdefault(id(c), onset_strength(c))
        others = [(o.start, o.start + o.dur) for o in picks if o is not p and o.clip is c]

        def qwin(s: float) -> np.ndarray:
            lo = int(round(s / C.DT))
            return c.q[lo:max(int(round((s + p.dur) / C.DT)), lo + 1)]

        floor = min(float(np.percentile(qwin(p.start), 5)), C.GATES["window_min_q"])
        best_key, best_s = None, p.start
        for k in range(-steps, steps + 1):
            s = round(p.start + k * C.DT, 6)
            if s < 0 or s + p.dur > c.duration + 1e-6 or any(s < e and s + p.dur > b for b, e in others):
                continue
            q = qwin(s)
            if float(np.percentile(q, 5)) < floor:
                continue
            onset = float(on[min(int(round(s / C.DT)), len(on) - 1)])
            key = (round(float(q.min()) / C.INOUT_Q_STEP), int(onset / C.ONSET_STEP), -abs(k))
            if best_key is None or key > best_key:
                best_key, best_s = key, s
        p.start = best_s


def plan_slowmo(picks: list[Pick]) -> None:
    """Slow down the best cuts from high-frame-rate clips: up to SLOWMO_MAX_CUTS picks from clips at
    >= SLOWMO_MIN_FPS play at SLOWMO_SPEED, filling their slot from the best-scoring stretch of
    dur × speed seconds inside their window."""
    fast = [p for p in picks if p.clip.kind == "video" and p.clip.fps >= C.SLOWMO_MIN_FPS]
    for p in sorted(fast, key=lambda p: -p.score)[:C.SLOWMO_MAX_CUTS]:
        c, span = p.clip, p.dur * C.SLOWMO_SPEED
        lo, hi, k = int(round(p.start / C.DT)), int(round((p.start + p.dur) / C.DT)), max(int(round(span / C.DT)), 1)
        sc = c.score[lo:hi]
        if len(sc) > k:
            best = int(np.argmax(np.convolve(sc, np.ones(k), "valid")))
            p.start = round((lo + best) * C.DT, 6)
        p.speed = C.SLOWMO_SPEED
        p.reasons += f", slow ×{C.SLOWMO_SPEED:g}"


def select(clips: list[Clip], durations: list[float], diversity: float = C.DIVERSITY,
           photos: str = "auto", slowmo: bool = True) -> list[Pick]:
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
        remaining.remove(best)
        if best.clip.kind == "photo":
            if any(same_moment(best, q) for q in chosen if q.clip.kind == "video"):
                continue                                     # the video already shows this moment
        else:
            dupes = [q for q in chosen if q.clip.kind == "photo" and same_moment(q, best)]
            for q in dupes:                                  # video leads: it replaces the photo
                chosen.remove(q)
            n_photos -= len(dupes)
        chosen.append(best)
        n_photos += best.clip.kind == "photo"
        per_clip[best.clip.path] = per_clip.get(best.clip.path, 0) + 1

    chosen.sort(key=lambda p: (moment(p), p.clip.path.name))
    for p, d in zip(chosen, durations):
        if p.clip.kind == "photo":
            p.dur = d
            continue
        centre = p.start + p.dur / 2
        p.dur = min(d, p.clip.duration)
        p.start = float(np.clip(centre - p.dur / 2, 0, p.clip.duration - p.dur))
    refine(chosen)
    if slowmo:
        plan_slowmo(chosen)
    return chosen
