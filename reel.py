#!/usr/bin/env python3
"""
reel.py — build a highlight reel from a folder of GoPro clips (v2: gate-then-score + CLIP).

  python reel.py ~/Movies/GoPro --dry-run        # analyse, plan, write the HTML report
  python reel.py ~/Movies/GoPro                  # render a 30 s reel
  python reel.py ~/Movies/GoPro --music song.mp3 # cuts on the beat

See README.md for how it works and how to tune it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
from analyse import analyse_clip  # noqa: E402
from render import render  # noqa: E402
from report import write_report  # noqa: E402
from score import compute  # noqa: E402
from select_moments import plan_durations, select  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() == ".mp4" and not p.name.startswith(("._", "highlights")))
    if not files:
        sys.exit(f"No MP4 files in {folder}")
    out = (args.out or folder / "highlights.mp4").expanduser()

    # ---------------------------------------------------------------- vision model
    scorer, tag = None, "cv"
    if args.no_vision:
        tag = "novis"
    elif not args.no_clip:
        try:
            from vision import ClipScorer
            scorer = ClipScorer(args.clip_model, args.clip_pretrained)
            tag = f"{args.clip_model}-{args.clip_pretrained}".replace("/", "_")
        except Exception as e:  # missing torch/open_clip or no weights
            print(f"  CLIP unavailable ({type(e).__name__}: {e}) — continuing with classic CV only")

    # ---------------------------------------------------------------- analysis
    print(f"Analysing {len(files)} clip(s) in {folder} ...")
    clips = []
    for i, f in enumerate(files, 1):
        t0 = time.time()
        c = analyse_clip(f, scorer, folder / ".reel_cache", tag, args.refresh, vision=not args.no_vision)
        if c.duration < 1.0:
            continue
        clips.append(c)
        sig = ", ".join(sorted(k for k in c.feats)) or "none"
        extra = []
        if c.events.get("drops"):
            extra.append(f"{len(c.events['drops'])} drop(s)")
        if c.hilights:
            extra.append("HiLights " + ", ".join(f"{h:.1f}s" for h in c.hilights))
        extra += c.notes
        print(f"  [{i}/{len(files)}] {f.name} {c.duration:6.1f}s  {time.time() - t0:5.1f}s  "
              f"{'; '.join(extra)}")
    if not clips:
        sys.exit("Nothing to work with.")
    clips.sort(key=lambda c: (c.created, c.path.name))

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
        index = write_report(clips, picks, folder, out.with_name(out.stem + "_report"), settings)
        print(f"\nReport: {index}")

    if args.dry_run or not picks:
        return
    print(f"\nRendering {args.res}p ...")
    render(picks, out, args.res, args.music, args.music_start, args.clip_audio, args.workers)
    print(f"\nDone: {out}")


if __name__ == "__main__":
    main()
