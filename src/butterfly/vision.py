"""Vision analysis: frame sampling, classic CV quality metrics, CLIP scoring."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import laplace

from . import config as C


# ------------------------------------------------------------------ frames

def proxy_for(path: Path) -> Path | None:
    """GoPro low-res proxy (GL01xxxx.LRV) next to GX01xxxx.MP4 / GH01xxxx.MP4, if present."""
    stem = path.stem
    if len(stem) < 3 or stem[:2].upper() not in ("GX", "GH"):
        return None
    for ext in (".LRV", ".lrv"):
        cand = path.with_name("GL" + stem[2:] + ext)
        if cand.exists():
            return cand
    return None


def _dims(path: Path) -> tuple[int, int]:
    """Display (width, height): the stored size, swapped when the video is rotated by ±90°.

    ffmpeg auto-rotates when decoding, so frames come out at the display size.
    """
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:stream_tags=rotate:stream_side_data=rotation",
                          "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    rot = next((sd["rotation"] for sd in s.get("side_data_list", []) if "rotation" in sd),
               s.get("tags", {}).get("rotate", 0))       # side data (ffmpeg ≥ 5), or legacy tag
    return (h, w) if round(float(rot)) % 180 == 90 else (w, h)


def iter_frames(path: Path, fps: float, width: int, batch: int):
    """Yield (times, frames uint8 [B,H,W,3]) batches, decoding with VideoToolbox on macOS."""
    w0, h0 = _dims(path)
    h = max(2, int(round(width * h0 / w0 / 2)) * 2)
    cmd = ["ffmpeg", "-v", "error"]
    if platform.system() == "Darwin":
        cmd += ["-hwaccel", "videotoolbox"]
    cmd += ["-i", str(path), "-map", "0:v:0", "-vf", f"fps={fps},scale={width}:{h}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    fsize = width * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(fsize * batch)
            k = len(buf) // fsize
            if k == 0:
                break
            frames = np.frombuffer(buf[:k * fsize], np.uint8).reshape(k, h, width, 3)
            yield (np.arange(idx, idx + k) / fps), frames
            idx += k
    finally:
        proc.stdout.close()
        proc.wait()


def classic_metrics(frames: np.ndarray, prev_gray: np.ndarray | None):
    """Per-frame sharpness, luma, blown-out fraction, colourfulness, frame change."""
    f = frames.astype(np.float32)
    gray = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    sharp = np.array([laplace(g).var() for g in gray])
    luma = gray.mean(axis=(1, 2))
    blown = (gray > 250).mean(axis=(1, 2))
    rg = f[..., 0] - f[..., 1]
    yb = 0.5 * (f[..., 0] + f[..., 1]) - f[..., 2]
    colour = (np.sqrt(rg.std(axis=(1, 2)) ** 2 + yb.std(axis=(1, 2)) ** 2)
              + 0.3 * np.sqrt(rg.mean(axis=(1, 2)) ** 2 + yb.mean(axis=(1, 2)) ** 2))
    prevs = np.concatenate([gray[:1] if prev_gray is None else prev_gray[None], gray[:-1]])
    change = np.abs(gray - prevs).mean(axis=(1, 2))
    return {"sharpness": sharp, "luma": luma, "blown": blown, "colour": colour,
            "change": change}, gray[-1]


# ------------------------------------------------------------------ CLIP

class ClipScorer:
    def __init__(self, model_name: str = C.CLIP_MODEL, pretrained: str | None = C.CLIP_PRETRAINED):
        import torch
        import open_clip

        self.torch = torch
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model = open_clip.create_model(model_name, pretrained=pretrained).to(self.device).eval()
        cfg = getattr(self.model.visual, "preprocess_cfg", {}) or {}
        size = cfg.get("size", 224)
        self.size = size[0] if isinstance(size, (tuple, list)) else size
        self.mean = torch.tensor(cfg.get("mean", (0.48145466, 0.4578275, 0.40821073)),
                                 device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.get("std", (0.26862954, 0.26130258, 0.27577711)),
                                device=self.device).view(1, 3, 1, 1)
        tok = open_clip.get_tokenizer(model_name)

        with torch.no_grad():
            def enc(texts):
                e = self.model.encode_text(tok(texts).to(self.device)).float()
                return e / e.norm(dim=-1, keepdim=True)
            self.pos = enc(C.POSITIVE_PROMPTS)
            self.neg = enc(C.NEGATIVE_PROMPTS)
            self.aes = enc([C.AESTHETIC_POS, C.AESTHETIC_NEG])
        self.scale = float(self.model.logit_scale.exp().item()) if hasattr(self.model, "logit_scale") else 100.0
        print(f"  CLIP {model_name} ({pretrained or 'random'}) on {self.device}", file=sys.stderr)

    def _prep(self, frames: np.ndarray):
        torch = self.torch
        x = torch.from_numpy(frames).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        h, w = x.shape[-2:]
        s = self.size / min(h, w)
        x = torch.nn.functional.interpolate(x, size=(max(self.size, round(h * s)), max(self.size, round(w * s))),
                                            mode="bilinear", antialias=True, align_corners=False)
        h, w = x.shape[-2:]
        top, left = (h - self.size) // 2, (w - self.size) // 2
        x = x[..., top:top + self.size, left:left + self.size]
        return (x - self.mean) / self.std

    def score(self, frames: np.ndarray) -> dict:
        torch = self.torch
        with torch.no_grad():
            e = self.model.encode_image(self._prep(frames)).float()
            e = e / e.norm(dim=-1, keepdim=True)
            sp, sn, sa = e @ self.pos.T, e @ self.neg.T, e @ self.aes.T
            k = min(3, sp.shape[1])
            semantic = sp.topk(k, dim=1).values.mean(dim=1)
            probs = torch.softmax(self.scale * torch.cat([sp, sn], dim=1), dim=1)
            bad = probs[:, sp.shape[1]:].sum(dim=1)
            aesthetic = sa[:, 0] - sa[:, 1]
        return {"semantic": semantic.cpu().numpy(), "clip_bad": bad.cpu().numpy(),
                "aesthetic": aesthetic.cpu().numpy(), "embedding": e.half().cpu().numpy()}


# ------------------------------------------------------------------ per clip

def analyse_vision(path: Path, scorer: ClipScorer | None) -> dict:
    """Sample frames at VISION_FPS -> {'t': times, metric: array, 'embedding': [N,D]}."""
    src = proxy_for(path) or path
    parts, prev = [], None
    for times, frames in iter_frames(src, C.VISION_FPS, C.VISION_WIDTH, C.CLIP_BATCH):
        m, prev = classic_metrics(frames, prev)
        m["t"] = times
        if scorer is not None:
            m.update(scorer.score(frames))
        parts.append(m)
    if not parts:
        return {}
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["source"] = np.array(src.name)
    return out
