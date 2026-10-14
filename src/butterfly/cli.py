"""
butterfly — build a highlight reel from a folder of GoPro clips (gate-then-score + CLIP).

  butterfly ~/Movies/GoPro --dry-run        # analyse, plan, write the HTML report
  butterfly ~/Movies/GoPro                  # render a 30 s reel
  butterfly ~/Movies/GoPro --music song.mp3 # cuts on the beat

See README.md for how it works and how to tune it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from . import config as C
from .analyse import analyse_clip
from .render import render
from .report import write_report
from .score import compute
from .select_moments import plan_durations, select


def find_videos(folder: Path) -> list[Path]:
    """Video files in the folder (not recursive), skipping macOS ._ files and our own reels."""
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in C.VIDEO_EXTS
                  and not p.name.startswith(("._", "highlights")))


def load_scorer(no_vision: bool, no_clip: bool, model: str, pretrained: str):
    """-> (ClipScorer or None, cache tag). Falls back to classic CV if CLIP can't load."""
    if no_vision:
        return None, "novis"
    if not no_clip:
        try:
            from .vision import ClipScorer
            return ClipScorer(model, pretrained), f"{model}-{pretrained}".replace("/", "_")
        except Exception as e:  # missing torch/open_clip or no weights
            print(f"  CLIP unavailable ({type(e).__name__}: {e}) — continuing with classic CV only")
    return None, "cv"


def analyse_folder(folder: Path, files: list[Path], scorer, tag: str, refresh: bool, vision: bool) -> list:
    """Analyse (or load cached) clips, printing one line each; -> clips ≥ 1 s, in shooting order."""
    print(f"Analysing {len(files)} clip(s) in {folder} ...")
    clips = []
    for i, f in enumerate(files, 1):
        t0 = time.time()
        c = analyse_clip(f, scorer, folder / C.OUTPUT_DIR / "cache", tag, refresh, vision=vision)
        if c.duration < 1.0:
            continue
        clips.append(c)
        extra = [f"{len(c.events['drops'])} drop(s)"] if c.events.get("drops") else []
        extra += c.notes
        print(f"  [{i}/{len(files)}] {f.name} {c.duration:6.1f}s  {time.time() - t0:5.1f}s  "
              f"{'; '.join(extra)}")
    clips.sort(key=lambda c: (c.created, c.path.name))
    return clips


def main() -> None:
    ap = argparse.ArgumentParser(prog="butterfly", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, nargs="?", default=Path("~/Movies/GoPro"))
    ap.add_argument("--length", type=float, default=30, help="reel length in seconds (default 30)")
    ap.add_argument("--seg", type=float, default=2.5, help="target seconds per cut (default 2.5)")
    ap.add_argument("--music", type=Path, help="audio file to sync cuts to")
    ap.add_argument("--music-start", type=float, default=0, help="offset into the music (s)")
    ap.add_argument("--clip-audio", type=float, default=0.25, help="original audio under music (0 = mute)")
    ap.add_argument("--res", type=int, choices=[1080, 2160], default=1080)
    ap.add_argument("--diversity", type=float, default=C.DIVERSITY, help="0 = pure score, 1 = pure variety")
    ap.add_argument("--no-vision", action="store_true", help="skip CLIP/frame analysis (telemetry + audio only)")
    ap.add_argument("--no-clip", action="store_true", help="classic CV metrics only, no CLIP model")
    ap.add_argument("--clip-model", default=C.CLIP_MODEL)
    ap.add_argument("--clip-pretrained", default=C.CLIP_PRETRAINED)
    ap.add_argument("--refresh", action="store_true", help="ignore the analysis cache")
    ap.add_argument("--out", type=Path, help="output file (default <folder>/highlights.mp4)")
    ap.add_argument("--dry-run", action="store_true", help="analyse + plan + report, no render")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="parallel encode jobs")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found — install with `brew install ffmpeg`")

    folder = args.folder.expanduser()
    files = find_videos(folder)
    if not files:
        sys.exit(f"No videos ({', '.join(C.VIDEO_EXTS)}) in {folder}")
    out = (args.out or folder / "highlights.mp4").expanduser()
    work = folder / C.OUTPUT_DIR  # cache/ and report/; an old .reel_cache/ is left alone

    scorer, tag = load_scorer(args.no_vision, args.no_clip, args.clip_model, args.clip_pretrained)
    clips = analyse_folder(folder, files, scorer, tag, args.refresh, not args.no_vision)
    if not clips:
        sys.exit("Nothing to work with.")

    # ---------------------------------------------------------------- score + select
    compute(clips)
    durations = plan_durations(args.length, args.seg, args.music, args.music_start)
    picks = select(clips, durations, args.diversity)

    total = sum(p.dur for p in picks)
    print(f"\nReel plan: {len(picks)} cuts, {total:.1f}s")
    print(f"  {'#':>2}  {'clip':<14} {'in':>7} {'out':>7}  {'score':>5}  why")
    for i, p in enumerate(picks, 1):
        print(f"  {i:>2}  {p.clip.path.name:<14} {p.start:7.1f} {p.start + p.dur:7.1f}  {p.score:5.2f}  {p.reasons}")
    if total < args.length - 0.5:
        print("  (shorter than requested: not enough usable footage)")

    if not args.no_report:
        settings = (f"{'no vision' if args.no_vision else ('CLIP ' + args.clip_model if scorer else 'classic CV')}"
                    f", diversity {args.diversity}, {args.seg}s cuts" + (f", music {args.music.name}" if args.music else ""))
        index = write_report(clips, picks, folder, work / "report", settings)
        print(f"\nReport: {index}")

    if args.dry_run or not picks:
        return
    print(f"\nRendering {args.res}p ...")
    render(picks, out, args.res, args.music, args.music_start, args.clip_audio, args.workers)
    print(f"\nDone: {out}")


if __name__ == "__main__":
    main()
