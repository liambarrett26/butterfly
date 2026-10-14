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
from .audio_events import load_tagger
from .days import group_by_day, local_offsets, select_with_titles
from .people import load_people
from .photos import (analyse_photo, apply_clock_offsets, assign_times, collapse_bursts, find_photos,
                     live_photo_motion, parse_clock_offset)
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


def analyse_folder(folder: Path, files: list[Path], scorer, tag: str, refresh: bool, vision: bool,
                   photos: list[Path] = (), tagger=None, people=None) -> list:
    """Analyse (or load cached) videos and photos, printing one line per video;
    -> videos ≥ 1 s and photos, in shooting order. Photos need vision, so --no-vision skips them."""
    live = live_photo_motion(files, photos) if vision else set()
    files = [f for f in files if f not in live]
    print(f"Analysing {len(files)} clip(s)" + (f" and {len(photos)} photo(s)" if photos and vision else "")
          + f" in {folder} ..." + (f" ({len(live)} Live Photo motion clip(s) kept with their photos)" if live else ""))
    clips = []
    for i, f in enumerate(files, 1):
        t0 = time.time()
        c = analyse_clip(f, scorer, folder / C.OUTPUT_DIR / "cache", tag, refresh, vision=vision, tagger=tagger,
                         people=people)
        if c.duration < 1.0:
            continue
        clips.append(c)
        extra = [f"{len(c.events['drops'])} drop(s)"] if c.events.get("drops") else []
        extra += c.notes
        print(f"  [{i}/{len(files)}] {f.name} {c.duration:6.1f}s  {time.time() - t0:5.1f}s  "
              f"{'; '.join(extra)}")
    stills, skipped = [], []
    for f in (photos if vision else []):
        try:
            stills.append(analyse_photo(f, scorer, folder / C.OUTPUT_DIR / "cache", tag, refresh, people))
        except Exception as e:  # unreadable file, or HEIC without the photos extra
            skipped.append(f"{f.name} ({e})")
    if stills:
        assign_times(stills)
        print(f"  {len(stills)} photo(s)")
    for s in skipped[:5]:
        print(f"  skipped photo {s}")
    clips += stills
    clips.sort(key=lambda c: (c.created, c.path.name))
    return clips


def score_folder(clips: list, clock_offsets: list[str] = ()) -> None:
    """Clock corrections, time order, scoring, then burst collapsing (which needs the scores)."""
    apply_clock_offsets(clips, dict(parse_clock_offset(s) for s in clock_offsets))
    clips.sort(key=lambda c: (c.created, c.path.name))
    compute(clips)
    collapse_bursts([c for c in clips if c.kind == "photo"])


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
    ap.add_argument("--photos", choices=["auto", "off", "only"], default="auto",
                    help="photos in the reel: auto (up to a quarter of the cuts), off, or only photos")
    ap.add_argument("--no-audio-events", action="store_true",
                    help="skip laughter / cheering / splash / wind tagging (needs the audio extra anyway)")
    ap.add_argument("--no-faces", action="store_true", help="skip face and people detection (Apple Vision, macOS)")
    ap.add_argument("--days", choices=["off", "split", "titles"], default="off",
                    help="multi-day folders: split = one reel per day; titles = one trip reel with a title card per day")
    ap.add_argument("--tz", metavar="±HH:MM", help="the trip's UTC offset for --days (default: from photos, GPS, "
                                                  "or this computer)")
    ap.add_argument("--no-slowmo", action="store_true", help="never slow down high-frame-rate clips")
    ap.add_argument("--clock-offset", action="append", default=[], metavar="DEVICE=±HH:MM",
                    help="correct a device's clock, e.g. 'iPhone 14 Plus=+01:00' (repeatable; device names "
                         "are listed in the report)")
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
    photos = [] if args.photos == "off" else find_photos(folder)
    if not files and not photos:
        sys.exit(f"No videos ({', '.join(C.VIDEO_EXTS)}) or photos ({', '.join(C.PHOTO_EXTS)}) in {folder}")
    out = (args.out or folder / "highlights.mp4").expanduser()
    work = folder / C.OUTPUT_DIR  # cache/ and report/; an old .reel_cache/ is left alone

    scorer, tag = load_scorer(args.no_vision, args.no_clip, args.clip_model, args.clip_pretrained)
    tagger, suffix = load_tagger(not (args.no_vision or args.no_audio_events))
    people, psuffix = load_people(not (args.no_vision or args.no_faces))
    clips = analyse_folder(folder, files, scorer, tag + suffix + psuffix, args.refresh, not args.no_vision, photos,
                           tagger, people)
    if not clips:
        sys.exit("Nothing to work with.")

    # ---------------------------------------------------------------- score + select
    try:
        score_folder(clips, args.clock_offset)
    except ValueError as e:
        sys.exit(str(e))
    durations = plan_durations(args.length, args.seg, args.music, args.music_start)
    opts = dict(diversity=args.diversity, photos=args.photos, slowmo=not args.no_slowmo)
    reels: list[tuple[str, Path, list]] = []                  # (heading, output file, picks)
    if args.days == "off":
        reels.append(("Reel plan", out, select(clips, durations, **opts)))
    else:
        try:
            offsets, how = local_offsets(clips, args.tz)
        except ValueError as e:
            sys.exit(str(e))
        days = group_by_day(clips, offsets)
        print(f"\n{len(days)} day(s); local time {how}")
        if args.days == "titles":
            reels.append(("Trip reel plan", out, select_with_titles(clips, durations, offsets, **opts)))
        else:
            for d, items in days.items():
                mode = args.photos if any(c.kind == "video" for c in items) or args.photos == "off" else "only"
                reels.append((f"Reel plan for {d:%a %-d %b %Y}", out.with_name(f"{out.stem}_{d.isoformat()}{out.suffix}"),
                              select(items, durations, args.diversity, mode, not args.no_slowmo)))

    for heading, _, picks in reels:
        print_plan(heading, picks, args.length)

    if not args.no_report:
        settings = (f"{'no vision' if args.no_vision else ('CLIP ' + args.clip_model if scorer else 'classic CV')}"
                    f", diversity {args.diversity}, {args.seg}s cuts, photos {args.photos}, days {args.days}"
                    + (f", music {args.music.name}" if args.music else ""))
        index = write_report(clips, [p for _, _, ps in reels for p in ps], folder, work / "report", settings)
        print(f"\nReport: {index}")

    if args.dry_run:
        return
    for _, path, picks in reels:
        if picks:
            print(f"\nRendering {args.res}p -> {path.name} ...")
            render(picks, path, args.res, args.music, args.music_start, args.clip_audio, args.workers)
            print(f"Done: {path}")


def print_plan(heading: str, picks: list, length: float) -> None:
    total = sum(p.dur for p in picks)
    print(f"\n{heading}: {len(picks)} cuts, {total:.1f}s")
    print(f"  {'#':>2}  {'clip':<14} {'in':>7} {'out':>7}  {'score':>5}  why")
    for i, p in enumerate(picks, 1):
        if p.clip.kind == "title":
            print(f"  {i:>2}  {p.clip.info['title'] + ' · ' + p.clip.info['subtitle']:<30} {'title':>7} {p.dur:5.1f}s")
            continue
        span = f"{'photo':>7} {p.dur:6.1f}s" if p.clip.kind == "photo" else f"{p.start:7.1f} {p.start + p.src_dur:7.1f}"
        print(f"  {i:>2}  {p.clip.path.name:<14} {span}  {p.score:5.2f}  {p.reasons}")
    if total < length - 0.5:
        print("  (shorter than requested: not enough usable footage)")


if __name__ == "__main__":
    main()
