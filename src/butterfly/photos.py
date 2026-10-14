"""Photos: discovery, EXIF orientation and capture time, and analysis as one-frame items.

A photo becomes a `Clip` with kind="photo" and duration 0 (so `n == 1`): the same quality
metrics and CLIP scores as a video frame, as length-1 arrays. Scoring (score.py) compares
photos only with other photos.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from . import config as C
from .analyse import Clip, cache_path, probe
from .vision import classic_metrics

# timestamps in file names: IMG_20261010_123526, PXL_20261010_123526123, 2026-10-10 12.35.26
_STAMP = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[ _T-]?(\d{2})[.:_-]?(\d{2})[.:_-]?(\d{2})")


def find_photos(folder: Path) -> list[Path]:
    """Photo files in the folder (not recursive), skipping macOS ._ files."""
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in C.PHOTO_EXTS and not p.name.startswith("._"))


def open_photo(path: Path):
    """-> (RGB PIL image with EXIF orientation applied, EXIF). HEIC needs the `photos` extra."""
    from PIL import Image, ImageOps

    if path.suffix.lower() in (".heic", ".heif"):
        try:
            from pillow_heif import register_heif_opener
        except ImportError as e:
            raise RuntimeError("HEIC photos need the photos extra: python -m pip install -e '.[photos]'") from e
        register_heif_opener()
    im = Image.open(path)
    exif = im.getexif()
    return ImageOps.exif_transpose(im).convert("RGB"), exif


def _offset(s) -> timedelta | None:
    m = re.fullmatch(r"\s*([+-])(\d{2}):?(\d{2})\s*", str(s or ""))
    if not m:
        return None
    d = timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))
    return d if m.group(1) == "+" else -d


def capture_time(path: Path, exif) -> dict:
    """When the photo was taken -> {'local': iso, 'offset_s': int|None, 'source': str}.

    Sources, best first: EXIF time + offset; EXIF time alone (offset unknown); a timestamp
    in the file name; the file's modification time (stored as UTC, unreliable after copying).
    """
    sub = exif.get_ifd(0x8769) if exif else {}
    stamp = sub.get(0x9003) or (exif.get(0x0132) if exif else None)   # DateTimeOriginal, else DateTime
    if stamp:
        try:
            t = datetime.strptime(str(stamp).strip()[:19], "%Y:%m:%d %H:%M:%S")
            off = _offset(sub.get(0x9011) or sub.get(0x9010))            # OffsetTimeOriginal / OffsetTime
            return {"local": t.isoformat(), "offset_s": None if off is None else int(off.total_seconds()),
                    "source": "exif+offset" if off is not None else "exif"}
        except ValueError:
            pass
    m = _STAMP.search(path.stem)
    if m:
        try:
            return {"local": datetime(*map(int, m.groups())).isoformat(), "offset_s": None, "source": "filename"}
        except ValueError:
            pass
    utc = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(tzinfo=None)
    return {"local": utc.isoformat(), "offset_s": 0, "source": "mtime"}


def signature(im) -> np.ndarray:
    """32×24 grey thumbnail, zero-mean and unit-norm: near-identical shots correlate close to 1."""
    a = np.asarray(im.convert("L").resize((32, 24)), dtype=np.float32).ravel()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def device_of(exif) -> str:
    model = str(exif.get(0x0110, "") or "").strip() if exif else ""          # Model
    make = str(exif.get(0x010F, "") or "").strip() if exif else ""           # Make
    return model or make or "unknown"


def analyse_photo(path: Path, scorer, cache_dir: Path, tag: str, refresh: bool = False) -> Clip:
    """Quality metrics + CLIP for one photo, cached like a clip. `created` is set by assign_times."""
    cache = cache_path(path, cache_dir, tag)
    clip = Clip(path, 0.0, datetime.min, False, 0.0, kind="photo")
    if cache.exists() and not refresh:
        z = np.load(cache, allow_pickle=False)
        meta = json.loads(str(z["__meta__"]))
        clip.feats = {k[2:]: z[k] for k in z.files if k.startswith("f_")}
        if "emb" in z.files:
            clip.emb_t, clip.emb = z["emb_t"], z["emb"]
        clip.size, clip.info, clip.notes = tuple(meta["size"]), meta["info"], meta["notes"] + ["cached"]
        clip.sig = z["sig"] if "sig" in z.files else None
        return clip

    im, exif = open_photo(path)
    clip.size, clip.info = im.size, {**capture_time(path, exif), "device": device_of(exif)}
    clip.sig = signature(im)
    w = min(C.PHOTO["width"], im.width)
    frame = np.asarray(im.resize((w, max(2, round(im.height * w / im.width)))))[None]
    m, _ = classic_metrics(frame, None)
    clip.feats = {k: v.astype(float) for k, v in m.items() if k != "change"}   # no "change" for a still
    if scorer is not None:
        s = scorer.score(frame)
        clip.feats.update({k: s[k].astype(float) for k in ("semantic", "clip_bad", "aesthetic")})
        clip.emb_t, clip.emb = np.zeros(1), s["embedding"]

    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {f"f_{k}": v for k, v in clip.feats.items()}
    arrays["sig"] = clip.sig
    if clip.emb is not None:
        arrays["emb_t"], arrays["emb"] = clip.emb_t, clip.emb
    meta = json.dumps({"size": list(clip.size), "info": clip.info, "notes": clip.notes})
    np.savez_compressed(cache, __meta__=np.array(meta), **arrays)
    return clip


def assign_times(photos: list[Clip]) -> None:
    """Set each photo's `created` (naive UTC, like video creation_time).

    Photos whose time zone is unknown (EXIF without offset, or a file-name timestamp) take the
    most common offset among the folder's photos that have one, and say so in their notes.
    """
    known = Counter(p.info["offset_s"] for p in photos if p.info.get("source") == "exif+offset")
    majority = known.most_common(1)[0][0] if known else None
    for p in photos:
        local, off = datetime.fromisoformat(p.info["local"]), p.info.get("offset_s")
        src = p.info.get("source")
        if src in ("exif", "filename"):
            off = majority
            p.notes.append(f"time from {src}, " + ("time zone assumed from other photos" if majority is not None
                                                   else "time zone unknown (treated as UTC)"))
        elif src == "mtime":
            p.notes.append("time from file date (unreliable)")
        p.created = local - timedelta(seconds=off or 0)


def live_photo_motion(videos: list[Path], photos: list[Path]) -> set[Path]:
    """Videos that are a Live Photo's motion: same file stem as a photo, at most LIVE_PHOTO_MAX_S long."""
    stems = {p.stem.lower() for p in photos}
    out = set()
    for v in videos:
        if v.stem.lower() in stems:
            try:
                if probe(v)[0] <= C.LIVE_PHOTO_MAX_S:
                    out.add(v)
            except Exception:
                pass
    return out


def parse_clock_offset(spec: str) -> tuple[str, timedelta]:
    """'iPhone 14 Plus=+01:00' -> ('iPhone 14 Plus', +1 h). Added to that device's times."""
    device, _, off = spec.rpartition("=")
    d = _offset(off)
    if not device.strip() or d is None:
        raise ValueError(f"--clock-offset wants DEVICE=±HH:MM, got {spec!r}")
    return device.strip(), d


def apply_clock_offsets(items: list[Clip], offsets: dict[str, timedelta]) -> None:
    for c in items:
        d = offsets.get(c.info.get("device", "unknown"))
        if d:
            c.created += d
            c.notes.append(f"clock corrected by {d}")


def collapse_bursts(photos: list[Clip]) -> None:
    """Mark all but the best-scoring photo of each burst with info['burst_of'] (call after scoring).

    A burst is a run of photos at most BURST_GAP_S apart whose signatures correlate ≥ BURST_SIM
    with the previous shot in the run.
    """
    runs: list[list[Clip]] = []
    for c in sorted(photos, key=lambda c: c.created):
        c.info.pop("burst_of", None)
        prev = runs[-1][-1] if runs else None
        if (prev is not None and c.sig is not None and prev.sig is not None
                and (c.created - prev.created).total_seconds() <= C.BURST_GAP_S
                and float(c.sig @ prev.sig) >= C.BURST_SIM):
            runs[-1].append(c)
        else:
            runs.append([c])
    for run in runs:
        if len(run) > 1:
            keep = max(run, key=lambda c: float(c.score[0]))
            for c in run:
                if c is not keep:
                    c.info["burst_of"] = keep.path.name
                    c.notes.append(f"burst: kept {keep.path.name}")
