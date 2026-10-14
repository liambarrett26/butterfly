"""
Synthetic regression test for the gate-then-score pipeline.

Builds three small synthetic clips (cached in tests/.synthetic/), injects fake GPMF
telemetry (a free-fall drop, handling shake + tumble, smooth riding), runs analysis +
scoring + selection with classic CV only (no CLIP, no network), and asserts the gates
behave. Run from the project root:

    python tests/run_synthetic.py

Exit code 0 = all checks passed.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works with or without `pip install -e .`

from butterfly import analyse  # noqa: E402
from butterfly import cli  # noqa: E402
from butterfly import config as C  # noqa: E402
from butterfly.evaluate import evaluate, load_labels  # noqa: E402
from butterfly.photos import find_photos  # noqa: E402
from butterfly import vision  # noqa: E402
from butterfly.score import compute  # noqa: E402
from butterfly.render import fit_graph, render  # noqa: E402
from butterfly.select_moments import Pick, plan_durations, select  # noqa: E402

DATA = Path(__file__).resolve().parent / ".synthetic"
DUR = 30


def make_clips() -> None:
    DATA.mkdir(exist_ok=True)
    ff = ["ffmpeg", "-y", "-v", "error"]
    enc = ["-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac"]
    specs = {
        # sharp, colourful; loud burst 20-22 s
        "GX010061.MP4": ["-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={DUR}",
                         "-f", "lavfi", "-i", f"sine=frequency=440:duration={DUR}",
                         "-filter_complex", "[1:a]volume='if(between(t,20,22),1.0,0.03)':eval=frame[a]",
                         "-map", "0:v", "-map", "[a]"],
        # 0-10 sharp, 10-20 blurred, 20-30 dark
        "GX010062.MP4": ["-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=30:duration={DUR}",
                         "-f", "lavfi", "-i", f"sine=frequency=300:duration={DUR}",
                         "-filter_complex",
                         "[0:v]split=3[a][b][c];[a]trim=0:10,setpts=PTS-STARTPTS[v1];"
                         "[b]trim=10:20,setpts=PTS-STARTPTS,boxblur=20:3[v2];"
                         "[c]trim=20:30,setpts=PTS-STARTPTS,eq=brightness=-0.9[v3];"
                         "[v1][v2][v3]concat=n=3:v=1[v];[1:a]volume=0.05[a]",
                         "-map", "[v]", "-map", "[a]"],
        # different-looking scene, quiet
        "GX010063.MP4": ["-f", "lavfi", "-i", "life=s=320x180:mold=10:r=30:ratio=0.5,scale=640:360",
                         "-t", str(DUR - 5), "-f", "lavfi", "-i", f"sine=frequency=500:duration={DUR - 5}",
                         "-filter_complex", "[1:a]volume=0.05[a]", "-map", "0:v", "-map", "[a]", "-shortest"],
    }
    for name, args in specs.items():
        if not (DATA / name).exists():
            subprocess.run(ff + args + enc + [str(DATA / name)], check=True)
    # phone-style portrait clip: stored 640x360 with a 90° display rotation, as a .MOV
    rot = DATA / "IMG_0001.MOV"
    if not rot.exists():
        tmp = DATA / "_landscape.mp4"
        subprocess.run(ff + ["-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=3"]
                       + enc[:4] + [str(tmp)], check=True)
        subprocess.run(ff + ["-display_rotation", "90", "-i", str(tmp), "-c", "copy", str(rot)], check=True)
        tmp.unlink()
    # optical flow: one textured scene seen three ways, no telemetry (like a phone)
    tex = DATA / "_texture.png"
    flow_specs = {
        "IMG_0002.MOV": "[0:v]crop=640:360:320:180[bg];color=c=red:s=160x160:r=30[o];"
                        "[bg][o]overlay=x='mod(t*150\\,480)':y=100:shortest=1",          # subject moves
        "IMG_0003.MOV": "[0:v]crop=640:360:'100+50*t':180",                               # smooth pan
        "IMG_0004.MOV": "[0:v]crop=640:360:'320+30*sin(t*25)+20*sin(t*41)':'180+20*sin(t*33)+15*sin(t*47)'",  # shaky
    }
    if not all((DATA / n).exists() for n in flow_specs):
        subprocess.run(ff + ["-f", "lavfi", "-i",
                             "nullsrc=s=1280x720,geq=lum='128+50*sin(X/17+3*sin(Y/29))+40*sin(Y/11+2*sin(X/37))'"
                             ":cb=128:cr=128", "-frames:v", "1", str(tex)], check=True)
        for name, graph in flow_specs.items():
            subprocess.run(ff + ["-loop", "1", "-framerate", "30", "-t", "10", "-i", str(tex),
                                 "-filter_complex", graph, "-t", "10"] + enc[:4] + [str(DATA / name)], check=True)
        tex.unlink()


def fake_telemetry(path: Path) -> dict:
    if path.name.startswith("IMG_"):
        return {}                                           # phone-style clip: no telemetry
    rng = np.random.default_rng(0)
    r = 200
    t = np.arange(0, DUR, 1 / r)
    acc = np.tile([9.8, 0, 0], (len(t), 1)) + rng.normal(0, 0.2, (len(t), 3))
    gyro = rng.normal(0, 0.02, (len(t), 3))
    grav = np.tile([0, 1.0, 0], (len(t), 1))
    tg = np.arange(0, DUR, 0.1)
    sp = np.full(len(tg), 0.5)
    if path.name == "GX010061.MP4":
        sp[(tg > 5) & (tg < 15)] = 8.0                       # riding 5-15 s
        gyro[(t > 6) & (t < 14), 2] += 0.6                  # smooth carving turn
        acc[(t > 24) & (t < 24.4)] = [0.4, 0.1, 0.1]        # free-fall drop at 24 s
    if path.name == "GX010062.MP4":
        m = (t > 2) & (t < 8)
        gyro[m] += rng.normal(0, 2.5, (m.sum(), 3))         # handling shake, slow
        grav[m] = [0, 0.2, -0.98]                           # pointing at the ground
    return {"accel": (t, acc), "gyro": (t, gyro), "grav": (t, grav), "speed": (tg, sp)}


def region_q(c, a, b) -> float:
    return float(c.q[int(a / 0.1):int(b / 0.1)].mean())


def cli_outputs() -> dict:
    """Dry-run the CLI on the synthetic folder; outputs must land in .butterfly/."""
    legacy = DATA / ".reel_cache" / "old.npz"     # an old cache must be left alone
    for d in (DATA / ".butterfly", legacy.parent, DATA / "highlights_report"):
        shutil.rmtree(d, ignore_errors=True)       # test artefacts from earlier runs
    legacy.parent.mkdir()
    legacy.write_bytes(b"")
    argv = sys.argv
    sys.argv = ["butterfly", str(DATA), "--dry-run", "--no-vision", "--length", "12"]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main()
    finally:
        sys.argv = argv
    return {
        "cache written to .butterfly/cache/": any((DATA / ".butterfly" / "cache").glob("*.npz")),
        "report written to .butterfly/report/": (DATA / ".butterfly" / "report" / "index.html").exists(),
        "nothing written to the old output paths": (legacy.exists() and len(list(legacy.parent.iterdir())) == 1
                                                    and not (DATA / "highlights_report").exists()),
    }


def rotated_portrait() -> bool:
    """Frames from a ±90°-rotated clip must come out portrait (taller than wide)."""
    _, frames = next(vision.iter_frames(DATA / "IMG_0001.MOV", 2.0, 160, 4))
    return frames.shape[1] > frames.shape[2]


def flow_checks() -> dict:
    """Optical flow on the three no-telemetry clips: subject vs pan vs shake."""
    obj, pan, shaky = clips = [analyse.analyse_clip(DATA / n, None, DATA / ".cache", "test", refresh=True)
                               for n in ("IMG_0002.MOV", "IMG_0003.MOV", "IMG_0004.MOV")]
    compute(clips)
    mid = lambda a: float(a[10:-10].mean())                 # skip the first and last second
    return {
        "moving subject outscores a camera pan": mid(obj.feats["subject_motion"]) > 5 * mid(pan.feats["subject_motion"]) + 0.05,
        "smooth pan is not shake (no gyro)": mid(pan.gates["handling"]) > 0.9,
        "static camera is not shake (no gyro)": mid(obj.gates["handling"]) > 0.9,
        "shaky clip without telemetry penalised": mid(shaky.gates["handling"]) < 0.5,
    }


def render_fill() -> dict:
    """Render a cut of the rotated (portrait) clip: 1920x1080, with a blurred fill, not black bars, at the sides."""
    c = analyse.analyse_clip(DATA / "IMG_0001.MOV", None, DATA / ".cache", "test", vision=False)
    out = DATA / "_fill.mp4"
    with contextlib.redirect_stdout(io.StringIO()):
        render([Pick(c, 0.5, 2.0, 1.0, "")], out, 1080, workers=1)
    w, h = vision.display_size(out)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "1", "-i", str(out), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    out.unlink()
    g = np.frombuffer(raw, np.uint8).reshape(h, w)
    sides = np.concatenate([g[:, : w // 10], g[:, -(w // 10):]], axis=1)
    return {
        "portrait clip renders at 1920x1080": (w, h) == (1920, 1080),
        "portrait clip sides are filled, not black": float(sides.mean()) > 30,
        "16:9 clips skip the fill": "gblur" not in fit_graph(1920, 1080, 3840, 2160),
    }


def label_checks(clips, picks) -> dict:
    """The labels format round-trips, and evaluation agrees with what the gates should do."""
    path = DATA / "_labels.json"
    path.write_text(json.dumps({"version": 1, "labels": [
        {"clip": "GX010061.MP4", "start": 6, "end": 14, "label": "good", "note": "smooth riding"},
        {"clip": "GX010061.MP4", "start": 23.5, "end": 26, "label": "unusable", "note": "drop"},
        {"clip": "GX010062.MP4", "start": 2.5, "end": 7.5, "label": "unusable", "note": "handling"},
        {"clip": "GX010062.MP4", "start": 21, "end": 29, "label": "dull", "note": "dark"},
        {"clip": "GX010063.MP4", "label": "dull"},
    ]}))
    labels = load_labels(path)
    path.unlink()
    r = evaluate(clips, picks, labels)
    try:
        rejected = False
        bad = DATA / "_bad_labels.json"
        bad.write_text(json.dumps([{"clip": "x.MP4", "label": "great"}]))
        load_labels(bad)
    except ValueError:
        rejected = True
    finally:
        bad.unlink()
    return {
        "labels: no violations on the synthetic set": not r["violations"] and not r["missing"],
        "labels: unusable ranges fully gated, never picked": r["rows"]["unusable"]["gated"] > 0.9
                                                             and r["rows"]["unusable"]["picked"] == 0,
        "labels: invalid label rejected": rejected,
    }


PHOTOS = DATA / "photos"


def make_photos() -> None:
    """Five synthetic photos: sharp ×2, blurred, dark, and an EXIF-rotated one with time + offset."""
    from PIL import Image, ImageFilter

    PHOTOS.mkdir(exist_ok=True)

    def texture(seed):
        r = np.random.default_rng(seed)
        a = r.integers(0, 256, (54, 80, 3), dtype=np.uint8)            # blocky detail, upscaled sharply
        return Image.fromarray(a).resize((640, 432), Image.NEAREST).filter(ImageFilter.GaussianBlur(1))

    if not (PHOTOS / "photo_a.jpg").exists():
        im = texture(2)
        exif = Image.Exif()
        exif[0x0112] = 6                                                    # rotate 90° on display -> portrait
        sub = exif.get_ifd(0x8769)
        sub[0x9003] = "2026:10:10 12:35:26"                                 # DateTimeOriginal (local)
        sub[0x9011] = "+02:00"                                              # OffsetTimeOriginal
        im.save(PHOTOS / "photo_a.jpg", exif=exif, quality=92)
        texture(3).save(PHOTOS / "IMG_20261010_140000.jpg", quality=92)    # no EXIF: time from the name
        texture(4).save(PHOTOS / "photo_b.jpg", quality=92)
        texture(5).filter(ImageFilter.GaussianBlur(12)).save(PHOTOS / "photo_blurred.jpg", quality=92)
        Image.fromarray((np.asarray(texture(6)) * 0.08).astype(np.uint8)).save(PHOTOS / "photo_dark.jpg", quality=92)


def photo_checks() -> tuple[dict, list]:
    from datetime import datetime

    make_photos()
    found = find_photos(PHOTOS)
    with contextlib.redirect_stdout(io.StringIO()):
        items = cli.analyse_folder(PHOTOS, [], None, "test", True, True, found)
    compute(items)
    ph = {c.path.name: c for c in items}
    q = lambda n: float(ph[n].q[0])
    r = evaluate(items, [], [{"clip": "photo_b.jpg", "label": "good"},
                             {"clip": "photo_blurred.jpg", "label": "unusable"}])
    return {
        "photos picked up (5)": len(found) == 5 and all(c.kind == "photo" for c in items),
        "sharp photos usable (q > 0.9)": min(q("photo_a.jpg"), q("photo_b.jpg"), q("IMG_20261010_140000.jpg")) > 0.9,
        "blurred photo gated": q("photo_blurred.jpg") < C.GATES["window_min_q"],
        "dark photo penalised (q < 0.7)": q("photo_dark.jpg") < 0.7,
        "EXIF orientation gives a portrait photo": ph["photo_a.jpg"].size[1] > ph["photo_a.jpg"].size[0],
        "EXIF time + offset -> UTC": ph["photo_a.jpg"].created == datetime(2026, 10, 10, 10, 35, 26),
        "file-name time takes the folder's offset": ph["IMG_20261010_140000.jpg"].created == datetime(2026, 10, 10, 12, 0, 0),
        "labels: photo rows, no violations": "good photos" in r["rows"] and not r["violations"],
    }, items


def _gray_frame(path: Path, t: float) -> np.ndarray:
    w, h = vision.display_size(path)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t}", "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w).astype(float)


def photo_reel_checks(videos, photos) -> dict:
    """Photos in selection (share cap, gating, modes, time order) and a rendered photo slot."""
    from butterfly.select_moments import moment

    allc = videos + photos
    compute(allc)
    durations = plan_durations(12, 2.5, None, 0)                       # 5 cuts -> at most 1 photo
    auto = select(allc, durations)
    only = select(allc, durations, photos="only")
    off = select(allc, durations, photos="off")
    in_auto = [p.clip.path.name for p in auto if p.clip.kind == "photo"]
    usable = [c for c in photos if float(c.q[0]) >= C.GATES["window_min_q"]]

    portrait = next(c for c in photos if c.path.name == "photo_a.jpg")
    out = DATA / "_photo_slot.mp4"
    with contextlib.redirect_stdout(io.StringIO()):
        render([Pick(portrait, 0.0, 2.0, 1.0, "")], out, 1080, workers=1)
    w, h = vision.display_size(out)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(out)], capture_output=True, text=True, check=True).stdout)
    first, last = _gray_frame(out, 0.05), _gray_frame(out, 1.9)
    out.unlink()
    sides = np.concatenate([first[:, : w // 10], first[:, -(w // 10):]], axis=1)
    return {
        "photos join the reel, within the 25% share": len(in_auto) == int(C.MAX_PHOTO_SHARE * len(durations)) >= 1,
        "gated photos are never picked": not {"photo_blurred.jpg", "photo_dark.jpg"} & set(
            p.clip.path.name for p in auto + only),
        "--photos only: a photo reel of the usable photos": len(only) == min(len(usable), len(durations))
                                                            and all(p.clip.kind == "photo" for p in only),
        "--photos off: no photos": all(p.clip.kind == "video" for p in off),
        "picks are in time order": all(moment(a) <= moment(b) for a, b in zip(auto, auto[1:])),
        "photo slot renders 1920x1080 at its planned length": (w, h) == (1920, 1080) and abs(dur - 2.0) < 0.1,
        "photo slot zooms (Ken Burns)": float(np.abs(first - last).mean()) > 2.0,
        "portrait photo has a blurred fill, not black bars": float(sides.mean()) > 30,
    }



def dedupe_checks() -> dict:
    """Bursts, Live Photo pairing, clock offsets, photo-vs-video duplicates, device detection."""
    from datetime import datetime, timedelta

    from PIL import Image

    from butterfly.analyse import Clip, device_of
    from butterfly.photos import apply_clock_offsets, live_photo_motion, parse_clock_offset
    from butterfly.select_moments import same_moment

    # bursts: three near-identical shots a second apart, then a different shot
    burst = DATA / "bursts"
    burst.mkdir(exist_ok=True)
    base = np.random.default_rng(7).integers(0, 256, (54, 80, 3), dtype=np.uint8)
    other = np.random.default_rng(8).integers(0, 256, (54, 80, 3), dtype=np.uint8)
    for k, (arr, sec) in enumerate([(base, 0), (base, 1), (base, 2), (other, 3)]):
        noisy = np.clip(arr.astype(int) + np.random.default_rng(20 + k).integers(-6, 7, arr.shape), 0, 255)
        exif = Image.Exif()
        exif.get_ifd(0x8769)[0x9003] = f"2026:10:10 12:00:0{sec}"
        exif.get_ifd(0x8769)[0x9011] = "+02:00"
        Image.fromarray(noisy.astype(np.uint8)).resize((640, 432), Image.NEAREST).save(
            burst / f"shot_{k}.jpg", exif=exif, quality=92)
    with contextlib.redirect_stdout(io.StringIO()):
        shots = cli.analyse_folder(burst, [], None, "test", True, True, find_photos(burst))
    cli.score_folder(shots)
    marked = {c.path.name: c.info.get("burst_of") for c in shots}
    kept = [n for n in ("shot_0.jpg", "shot_1.jpg", "shot_2.jpg") if not marked[n]]

    # Live Photo: IMG_9000.jpg + a 2 s IMG_9000.MOV is one photo; a 6 s IMG_9001.MOV with a photo stays a clip
    live = DATA / "live"
    live.mkdir(exist_ok=True)
    for name, dur in (("IMG_9000", 2), ("IMG_9001", 6)):
        Image.fromarray(base).save(live / f"{name}.jpg")
        if not (live / f"{name}.MOV").exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=30:duration={dur}",
                            "-c:v", "libx264", "-preset", "ultrafast", str(live / f"{name}.MOV")], check=True)
    motion = live_photo_motion(cli.find_videos(live), find_photos(live))

    # clock offsets
    dev, off = parse_clock_offset("TestCam=+01:30")
    c = Clip(Path("x.jpg"), 0.0, datetime(2026, 10, 10, 12), False, 0.0, kind="photo", info={"device": "TestCam"})
    apply_clock_offsets([c], {dev: off})
    try:
        parse_clock_offset("TestCam=soon")
        bad_rejected = False
    except ValueError:
        bad_rejected = True

    # a photo vs a video window: same moment and look -> duplicate; different look or time -> not
    e1 = np.ones(4, np.float32) / 2
    e2 = np.array([1, -1, 1, -1], np.float32) / 2
    t = datetime(2026, 10, 10, 12)
    vid = Clip(Path("v.mp4"), 30.0, t - timedelta(seconds=10), True, 30.0)
    ph = lambda when: Clip(Path("p.jpg"), 0.0, when, False, 0.0, kind="photo")
    window = Pick(vid, 9.0, 2.5, 1.0, "", e1)
    dup = same_moment(Pick(ph(t), 0.0, 2.5, 1.0, "", e1), window)
    diff_look = same_moment(Pick(ph(t), 0.0, 2.5, 1.0, "", e2), window)
    diff_time = same_moment(Pick(ph(t + timedelta(seconds=30)), 0.0, 2.5, 1.0, "", e1), window)

    return {
        "burst of 3 collapses to its best shot": len(kept) == 1 and float(
            next(x for x in shots if x.path.name == kept[0]).score[0]) == max(
            float(x.score[0]) for x in shots if x.path.name != "shot_3.jpg"),
        "a different shot is not part of the burst": not marked["shot_3.jpg"],
        "Live Photo motion (short .MOV, same name) is not a clip": motion == {live / "IMG_9000.MOV"},
        "clock offset corrects that device's times": c.created == datetime(2026, 10, 10, 13, 30) and bad_rejected,
        "photo of a picked video moment is a duplicate": dup and not diff_look and not diff_time,
        "device from QuickTime model / GoPro handler": device_of({"com.apple.quicktime.model": "iPhone 14 Plus"}, [])
            == "iPhone 14 Plus" and device_of({}, [{"tags": {"handler_name": "GoPro H.265"}}]) == "GoPro",
    }


def inout_checks() -> dict:
    """Smart in/out points on a hand-built clip: dodge a quality dip, start on an onset, never overlap."""
    from datetime import datetime

    from butterfly.analyse import Clip
    from butterfly.select_moments import refine

    c = Clip(Path("inout.mp4"), 12.0, datetime(2026, 1, 1), True, 30.0)
    c.q = np.ones(c.n)
    c.q[30:33] = 0.35                               # brief dip at 3.0-3.3 s (still above window_min_q)
    change = np.full(c.n, 1.0)
    change[60:] = 6.0                               # activity steps up at 6.0 s
    c.feats = {"change": change}
    dip, onset = Pick(c, 3.0, 2.0, 1.0, ""), Pick(c, 5.7, 2.0, 1.0, "")
    refine([dip, onset])
    a, b = Pick(c, 1.0, 2.0, 1.0, ""), Pick(c, 3.0, 2.0, 1.0, "")   # adjacent: a would like to move right
    c2 = Clip(Path("inout2.mp4"), 12.0, datetime(2026, 1, 1), True, 30.0)
    c2.q, c2.feats = np.ones(c2.n), {"change": np.ones(c2.n)}
    c2.q[10:13] = 0.35
    a.clip = b.clip = c2
    refine([a, b])
    return {
        "in/out: a cut moves off a quality dip at its start": 3.3 - 1e-6 <= dip.start <= 3.5 + 1e-6,
        "in/out: on flat quality a cut starts on a motion onset": abs(onset.start - 6.0) < 0.11,
        "in/out: cuts of one clip never overlap": a.start + a.dur <= b.start + 1e-6,
    }


def slowmo_checks() -> dict:
    """A 120 fps clip: detected, slowed (only it), and rendered at its slot length from half the source."""
    from butterfly.select_moments import plan_slowmo

    fast = DATA / "fast120.mp4"
    if not fast.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=120:duration=6",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-c:v", "libx264", "-preset", "ultrafast",
                        "-c:a", "aac", "-shortest", str(fast)], check=True)
    c_fast = analyse.analyse_clip(fast, None, DATA / ".cache", "test", refresh=True, vision=False)
    c_slow = analyse.analyse_clip(DATA / "GX010063.MP4", None, DATA / ".cache", "test", vision=False)
    for c in (c_fast, c_slow):
        c.q, c.score = np.ones(c.n), np.linspace(0, 1, c.n)        # later moments score higher
    a, b = Pick(c_fast, 1.0, 2.0, 2.0, ""), Pick(c_slow, 1.0, 2.0, 1.0, "")
    plan_slowmo([a, b])
    out = DATA / "_slow.mp4"
    with contextlib.redirect_stdout(io.StringIO()):
        render([a], out, 1080, workers=1)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(out)], capture_output=True, text=True, check=True).stdout)
    moving = float(np.abs(_gray_frame(out, 0.2) - _gray_frame(out, 1.6)).mean()) > 1.0
    out.unlink()
    return {
        "frame rate detected (120 fps)": abs(c_fast.fps - 120) < 1,
        "only high-frame-rate cuts are slowed": a.speed == C.SLOWMO_SPEED and b.speed == 1.0,
        "slow cut uses half the source, inside its window": abs(a.src_dur - 1.0) < 1e-6 and 1.0 <= a.start
                                                            and a.start + a.src_dur <= 3.0 + 1e-6,
        "slow cut renders at its slot length, not frozen": abs(dur - 2.0) < 0.1 and moving,
    }


def main() -> int:
    make_clips()
    analyse.load_telemetry = fake_telemetry
    clips = [analyse.analyse_clip(DATA / n, None, DATA / ".cache", "test", refresh=True)
             for n in ("GX010061.MP4", "GX010062.MP4", "GX010063.MP4")]
    compute(clips)
    c1, c2, _ = clips
    durations = plan_durations(12, 2.5, None, 0)
    picks = select(clips, durations)
    cap = math.ceil(len(durations) / len(clips)) + C.PICKS_PER_CLIP_SLACK

    photo_results, photo_items = photo_checks()
    checks = {
        "drop detected": len(c1.events["drops"]) == 1,
        "drop window unusable (q < 0.05)": region_q(c1, 23.5, 26) < 0.05,
        "smooth riding usable (q > 0.9)": region_q(c1, 6, 14) > 0.9,
        "handling+tumble penalised (q < 0.3)": region_q(c2, 2.5, 7.5) < 0.3,
        "blur penalised (q < 0.7)": region_q(c2, 11, 19) < 0.7,
        "dark penalised (q < 0.7)": region_q(c2, 21, 29) < 0.7,
        "clean section usable (q > 0.9)": region_q(c2, 8.5, 9.8) > 0.9,
        "reel fills 12 s": abs(sum(p.dur for p in picks) - 12) < 0.1,
        "no pick overlaps the drop window": all(
            not (p.clip is c1 and p.start < 26.9 and p.start + p.dur > 23.0) for p in picks),
        "no pick in the handling section": all(   # 0.2 s tolerance at the region edge
            not (p.clip is c2 and p.start < 7.8 and p.start + p.dur > 2.2) for p in picks),
        "riding moment picked": any(p.clip is c1 and 5 <= p.start + p.dur / 2 <= 15 for p in picks),
        f"no clip takes more than {cap} cuts": all(sum(p.clip is c for p in picks) <= cap for c in clips),
        ".MOV videos are picked up": any(p.name == "IMG_0001.MOV" for p in cli.find_videos(DATA)),
        "rotated clip analysed as portrait": rotated_portrait(),
        **label_checks(clips, picks),
        **photo_results,
        **photo_reel_checks(clips, photo_items),
        **dedupe_checks(),
        **inout_checks(),
        **slowmo_checks(),
        **flow_checks(),
        **render_fill(),
        **cli_outputs(),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("\npicks:", ", ".join(f"{p.clip.path.stem}@{p.start:.1f}" for p in picks))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
