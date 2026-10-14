"""Cut, concatenate and (optionally) mix music with ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config as C
from .vision import display_size


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{r.stderr[-2000:]}")


def _encoder(res: int) -> list[str]:
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "h264_videotoolbox" in enc:  # Apple Silicon hardware encoder
        return ["-c:v", "h264_videotoolbox", "-b:v", "50M" if res >= 2160 else "20M"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]


def fit_graph(w: int, h: int, src_w: int, src_h: int) -> str:
    """filter_complex fitting input 0's video into w×h as [v].

    Matching aspect: plain scale (letterbox only for rounding). Otherwise the item sits,
    whole, on a blurred and dimmed copy of itself that fills the frame, instead of black bars.
    """
    tail = "setsar=1,fps=30000/1001,format=yuv420p[v]"
    if abs((src_w / src_h) / (w / h) - 1) <= C.FILL_ASPECT_TOL:
        return (f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,{tail}")
    bw, bh = w // 8 // 2 * 2, h // 8 // 2 * 2                  # blur at low resolution: cheap
    return (f"[0:v]split=2[bg][fg];"
            f"[bg]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
            f"gblur=sigma={C.FILL_BLUR},lutyuv=y=val*{C.FILL_DIM},scale={w}:{h}[bgb];"
            f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,{tail}")


def render(picks, out: Path, res: int = 1080, music: Path | None = None, music_start: float = 0.0,
           clip_audio: float = 0.25, workers: int = 4) -> None:
    w, h = (3840, 2160) if res >= 2160 else (1920, 1080)
    enc = _encoder(res)

    with tempfile.TemporaryDirectory(prefix="butterfly_") as tmp:
        tmp = Path(tmp)

        def cut(item):
            i, p = item
            seg, d = tmp / f"seg_{i:03d}.mp4", p.dur
            af = f"afade=t=in:d=0.05,afade=t=out:st={max(d - 0.05, 0):.3f}:d=0.05"
            cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{p.start:.3f}", "-t", f"{d:.3f}",
                   "-i", str(p.clip.path)]
            if p.clip.has_audio:
                amap = ["-map", "[v]", "-map", "0:a:0"]
            else:
                cmd += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
                amap = ["-map", "[v]", "-map", "1:a:0"]
            graph = fit_graph(w, h, *display_size(p.clip.path))
            _run(cmd + ["-filter_complex", graph] + amap + ["-af", af] + enc +
                 ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", str(seg)])
            print(f"  cut {i + 1}/{len(picks)}  {p.clip.path.name} {p.start:.1f}–{p.start + d:.1f}s")
            return seg

        with ThreadPoolExecutor(max_workers=workers) as ex:
            segs = list(ex.map(cut, enumerate(picks)))

        listing = tmp / "list.txt"
        listing.write_text("".join(f"file '{s}'\n" for s in segs))
        reel = tmp / "reel.mp4"
        _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
              "-c", "copy", str(reel)])

        if music is None:
            shutil.move(str(reel), out)
            return

        total = sum(p.dur for p in picks)
        fade = min(2.0, total / 4)
        mix = (f"[1:a]atrim=0:{total:.3f},asetpts=PTS-STARTPTS[m];"
               + (f"[0:a]volume={clip_audio}[c];[m][c]amix=inputs=2:duration=first:normalize=0[mx];"
                  if clip_audio > 0 else "[m]anull[mx];")
               + f"[mx]afade=t=out:st={total - fade:.3f}:d={fade:.3f}[a]")
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(reel), "-ss", f"{music_start:.3f}",
              "-i", str(music), "-filter_complex", mix, "-map", "0:v", "-map", "[a]",
              "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest",
              "-movflags", "+faststart", str(out)])
