"""Cut, concatenate and (optionally) mix music with ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{r.stderr[-2000:]}")


def _encoder(res: int) -> list[str]:
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "h264_videotoolbox" in enc:  # Apple Silicon hardware encoder
        return ["-c:v", "h264_videotoolbox", "-b:v", "50M" if res >= 2160 else "20M"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]


def render(picks, out: Path, res: int = 1080, music: Path | None = None, music_start: float = 0.0,
           clip_audio: float = 0.25, workers: int = 4) -> None:
    w, h = (3840, 2160) if res >= 2160 else (1920, 1080)
    enc = _encoder(res)
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30000/1001,format=yuv420p")

    with tempfile.TemporaryDirectory(prefix="gopro_reel_") as tmp:
        tmp = Path(tmp)

        def cut(item):
            i, p = item
            seg, d = tmp / f"seg_{i:03d}.mp4", p.dur
            af = f"afade=t=in:d=0.05,afade=t=out:st={max(d - 0.05, 0):.3f}:d=0.05"
            cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{p.start:.3f}", "-t", f"{d:.3f}",
                   "-i", str(p.clip.path)]
            if p.clip.has_audio:
                amap = ["-map", "0:v:0", "-map", "0:a:0"]
            else:
                cmd += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
                amap = ["-map", "0:v:0", "-map", "1:a:0"]
            _run(cmd + amap + ["-vf", vf, "-af", af] + enc +
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
