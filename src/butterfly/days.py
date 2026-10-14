"""Multi-day folders: local days, a reel per day, or a trip reel with day title cards."""

from __future__ import annotations

import subprocess
import time
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from . import config as C
from .analyse import Clip, location_of
from .select_moments import Pick, moment, select


def local_offsets(clips: list[Clip], tz: str | None = None) -> tuple[dict[int, timedelta], str]:
    """Each item's UTC offset (by id) -> (offsets, a one-line summary).

    --tz applies to everything. Otherwise each item uses its own best evidence: a photo's EXIF
    offset; a video's GPS longitude (≈ lon / 15 h, good to about an hour, enough to split days).
    Items with neither take the most common offset among those that have one, else this
    machine's zone.
    """
    if tz:
        from .photos import _offset
        off = _offset(tz)
        if off is None:
            raise ValueError(f"--tz wants ±HH:MM, got {tz!r}")
        return {id(c): off for c in clips}, f"UTC{off.total_seconds() / 3600:+g} for everything (--tz)"
    own: dict[int, timedelta] = {}
    for c in clips:
        if c.kind == "photo" and c.info.get("source") == "exif+offset":
            own[id(c)] = timedelta(seconds=c.info["offset_s"])
        elif c.kind == "video":
            loc = location_of(c.path)
            if loc:
                own[id(c)] = timedelta(hours=round(loc[1] / 15))
    if own:
        fallback = Counter(own.values()).most_common(1)[0][0]
        how = "per item from EXIF / GPS"
    else:
        fallback = timedelta(seconds=time.localtime().tm_gmtoff)
        how = "this computer's time zone"
    zones = Counter(own.get(id(c), fallback).total_seconds() / 3600 for c in clips)
    return ({id(c): own.get(id(c), fallback) for c in clips},
            how + ": " + ", ".join(f"UTC{h:+g} ×{n}" for h, n in sorted(zones.items())))


def day_of(c: Clip, offsets: dict[int, timedelta]) -> date:
    return (c.created + offsets[id(c)]).date()


def group_by_day(clips: list[Clip], offsets: dict[int, timedelta]) -> dict[date, list[Clip]]:
    days: dict[date, list[Clip]] = {}
    for c in sorted(clips, key=lambda c: c.created + offsets[id(c)]):
        days.setdefault(day_of(c, offsets), []).append(c)
    return dict(sorted(days.items()))


def title_clip(n: int, day: date, offset: timedelta) -> Clip:
    """A title card for day n, timed at that local midnight (offset: the day's zone)."""
    start = datetime(day.year, day.month, day.day) - offset
    return Clip(Path(f"Day {n}"), 0.0, start, False, 0.0, kind="title",
                info={"title": f"Day {n}", "subtitle": f"{day:%a %-d %b %Y}"})


def _usable_seconds(items: list[Clip], slot: float, photos: str = "auto") -> float:
    """How much a day has to offer: usable video seconds, plus one slot per usable photo."""
    total = 0.0
    for c in items:
        if c.q is None:
            continue
        if c.kind == "video":
            total += float((c.q >= C.GATES["window_min_q"]).sum()) * C.DT
        elif c.kind == "photo" and photos != "off" and float(c.q[0]) >= C.GATES["window_min_q"] and not c.info.get("burst_of"):
            total += slot
    return total


def _share(n: int, weights: list[float]) -> list[int]:
    """Split n slots by weight (largest remainder), at least 1 each (needs n >= len(weights))."""
    raw = [n * w / sum(weights) for w in weights]
    out = [max(1, int(r)) for r in raw]
    while sum(out) > n:                                      # the minimum of 1 overshot: trim the largest
        out[out.index(max(out))] -= 1
    for k in sorted(range(len(raw)), key=lambda k: raw[k] - int(raw[k]), reverse=True)[: n - sum(out)]:
        out[k] += 1
    return out


def select_with_titles(clips: list[Clip], durations: list[float], offsets: dict[int, timedelta],
                       diversity: float = C.DIVERSITY, photos: str = "auto", slowmo: bool = True) -> list[Pick]:
    """One trip reel: each day with usable footage opens with a title card, then its own cuts.

    Days share the cut slots in proportion to their usable footage, with at least one each; if
    the reel is too short for every day to get a title and a cut, the days with the most usable
    footage are kept. A title takes one slot, so the reel keeps its planned length (and beat grid).
    """
    slot = float(np.median(durations)) if durations else 0.0
    days = group_by_day(clips, offsets)
    order = list(days)
    weight = {d: _usable_seconds(days[d], slot, photos) for d in order}
    active = [d for d in order if weight[d] > 0]
    while active and len(durations) < 2 * len(active):       # each day needs a title slot and a cut slot
        active.remove(min(active, key=lambda d: weight[d]))
    if not active:
        return []
    counts = _share(len(durations) - len(active), [weight[d] for d in active])
    seq, k = [], 0
    for d, n_cuts in zip(active, counts):
        items = days[d]
        has_video = any(c.kind == "video" for c in items)
        mode = photos if has_video or photos == "off" else "only"     # a photo-only day is all photos
        title = Pick(title_clip((d - order[0]).days + 1, d, offsets[id(items[0])]), 0.0, float(durations[k]), 0.0,
                     "day title")                            # calendar days since the first: gaps still count
        cuts = select(items, list(durations[k + 1:k + 1 + n_cuts]), diversity, mode, slowmo)
        k += 1 + n_cuts
        if cuts:
            title.clip.info["first"] = min(cuts, key=moment)
            seq += [title] + cuts
    return seq


def title_image(p: Pick, size: tuple[int, int]):
    """Day N and the date, centred, over a blurred and darkened frame of the day's first cut."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    w, h = size
    first = p.clip.info.get("first")
    bg = None
    try:
        if first is not None and first.clip.kind == "photo":
            from .photos import open_photo
            bg, _ = open_photo(first.clip.path)
        elif first is not None:
            raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{first.start + 0.5:.2f}", "-i", str(first.clip.path),
                                  "-frames:v", "1", "-vf", "scale=640:-2", "-f", "image2pipe", "-vcodec", "png", "-"],
                                 capture_output=True, check=True).stdout
            import io
            bg = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        bg = None
    if bg is None:
        img = Image.new("RGB", (w, h), (20, 20, 24))
    else:
        scale = max(w / bg.width, h / bg.height)
        bg = bg.resize((max(w, round(bg.width * scale)), max(h, round(bg.height * scale))))
        left, top = (bg.width - w) // 2, (bg.height - h) // 2
        img = bg.crop((left, top, left + w, top + h)).filter(ImageFilter.GaussianBlur(w / 60))
        img = Image.fromarray((np.asarray(img) * C.TITLE_DIM).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    big, small = ImageFont.load_default(size=h // 7), ImageFont.load_default(size=h // 18)
    for text, font, y in ((p.clip.info["title"], big, 0.42), (p.clip.info["subtitle"], small, 0.58)):
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(((w - (box[2] - box[0])) / 2, h * y - (box[3] - box[1]) / 2), text, font=font, fill=(255, 255, 255))
    return img
