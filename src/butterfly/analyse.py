"""Per-clip analysis (telemetry + audio + vision) with an on-disk cache."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C
from .gpmf import load_telemetry
from .signals import audio_features, telemetry_features
from .vision import analyse_vision


@dataclass
class Clip:
    path: Path
    duration: float
    created: datetime
    has_audio: bool
    fps: float
    feats: dict = field(default_factory=dict)      # name -> array on the DT grid
    emb_t: np.ndarray | None = None                 # frame times for embeddings
    emb: np.ndarray | None = None                   # [N, D] float16 CLIP embeddings
    events: dict = field(default_factory=dict)      # drops, water_entries
    notes: list = field(default_factory=list)
    kind: str = "video"                             # "video" or "photo" (a photo is one frame, duration 0)
    size: tuple | None = None                       # display (width, height) when known (photos)
    info: dict = field(default_factory=dict)        # device; photos: capture-time details, burst_of
    sig: np.ndarray | None = None                   # photos: 32×24 grey signature (zero-mean, unit-norm)
    # filled by score.py
    q: np.ndarray | None = None
    interest: np.ndarray | None = None
    score: np.ndarray | None = None
    parts: dict = field(default_factory=dict)       # interest components (weighted)
    gates: dict = field(default_factory=dict)       # quality multipliers per gate

    @property
    def n(self) -> int:
        return max(int(np.ceil(self.duration / C.DT)), 1)


def device_of(format_tags: dict, streams: list) -> str:
    """Camera or phone a video came from: QuickTime model tags (phones), GoPro handler names, else 'unknown'."""
    t = {k.lower(): v for k, v in format_tags.items()}
    for key in ("com.apple.quicktime.model", "model", "com.android.model"):
        if t.get(key):
            return str(t[key]).strip()
    if any("gopro" in str(s.get("tags", {}).get("handler_name", "")).lower() for s in streams):
        return "GoPro"
    return "unknown"


def probe(path: Path) -> tuple[float, datetime, bool, float, str]:
    """-> (duration s, creation time as naive UTC, has audio, fps, device)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration:format_tags:stream=codec_type,avg_frame_rate:stream_tags=handler_name",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    dur = float(info["format"].get("duration", 0))
    ct = info["format"].get("tags", {}).get("creation_time")
    try:
        created = datetime.fromisoformat(ct.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        created = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(tzinfo=None)
    streams = info.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    fps = 0.0
    for s in streams:
        if s.get("codec_type") == "video":
            num, _, den = s.get("avg_frame_rate", "0/1").partition("/")
            fps = float(num) / float(den or 1) if float(den or 1) else 0.0
            break
    return dur, created, has_audio, fps, device_of(info["format"].get("tags", {}), streams)


def cache_path(path: Path, cache_dir: Path, tag: str) -> Path:
    """Full file name (so IMG_1.HEIC and IMG_1.MOV don't collide) + size + mtime + version + model tag."""
    st = path.stat()
    return cache_dir / f"{path.name}_{st.st_size}_{int(st.st_mtime)}_v{C.CACHE_VERSION}_{tag}.npz"


def analyse_clip(path: Path, scorer, cache_dir: Path, tag: str, refresh: bool = False,
                 vision: bool = True) -> Clip:
    dur, created, has_audio, fps, device = probe(path)
    clip = Clip(path, dur, created, has_audio, fps, info={"device": device})
    cache = cache_path(path, cache_dir, tag)

    if cache.exists() and not refresh:
        z = np.load(cache, allow_pickle=False)
        meta = json.loads(str(z["__meta__"]))
        clip.feats = {k[2:]: z[k] for k in z.files if k.startswith("f_")}
        if "emb" in z.files:
            clip.emb_t, clip.emb = z["emb_t"], z["emb"]
        clip.events, clip.notes = meta["events"], meta["notes"] + ["cached"]
        return clip

    n = clip.n
    tel = load_telemetry(path)
    if not tel:
        clip.notes.append("no GPMF telemetry")
    feats, events = telemetry_features(tel, n)
    if has_audio:
        feats.update(audio_features(path, n))

    vis = analyse_vision(path, scorer) if vision else {}
    if vis:
        grid = np.arange(n) * C.DT
        for k, v in vis.items():
            if k in ("t", "flow_t", "embedding", "source"):
                continue
            t = vis["flow_t"] if k in ("camera_shake", "subject_motion") else vis["t"]
            feats[k] = np.interp(grid, t, v)
        if "embedding" in vis:
            clip.emb_t, clip.emb = vis["t"], vis["embedding"]
        if str(vis["source"]) != path.name:
            clip.notes.append(f"vision from proxy {vis['source']}")
    clip.feats, clip.events = feats, events

    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {f"f_{k}": v for k, v in feats.items()}
    if clip.emb is not None:
        arrays["emb_t"], arrays["emb"] = clip.emb_t, clip.emb
    meta = json.dumps({"events": events, "notes": clip.notes})
    np.savez_compressed(cache, __meta__=np.array(meta), **arrays)
    return clip
