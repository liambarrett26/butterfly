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


class FakeTagger:
    """Stands in for the AST audio tagger: laughter at 15-17 s, wind noise at 2-6 s."""

    def tag(self, path: Path, n: int) -> dict:
        t = np.arange(n) * 0.1
        z = np.zeros(n)
        return {"ev_laughter": np.where((t >= 15) & (t < 17), 0.8, 0.0), "ev_wind": np.where((t >= 2) & (t < 6), 0.9, 0.0),
                "ev_cheering": z, "ev_splash": z, "ev_speech": z}


def audio_event_checks() -> dict:
    c = analyse.analyse_clip(DATA / "GX010063.MP4", None, DATA / ".cache", "test+fake", refresh=True,
                             vision=False, tagger=FakeTagger())
    plain = analyse.analyse_clip(DATA / "GX010063.MP4", None, DATA / ".cache", "test", vision=False)
    compute([c])
    compute([plain])
    laugh, wind = c.parts["laughter"], c.gates["wind"]
    return {
        "laughter raises interest where it happens": laugh[150:170].mean() > 0.5 and laugh[:140].max() == 0,
        "wind noise gates mildly (never below ×0.7)": abs(wind[25:55].mean() - (1 - C.GATES["wind_penalty"])) < 0.01
                                                       and wind.min() >= 1 - C.GATES["wind_penalty"] - 1e-9
                                                       and wind[80:].min() == 1,
        "no tagger: no audio-event features or wind gate": "wind" not in plain.gates
                                                            and not any(k.startswith("ev_") for k in plain.feats),
    }


class FakePeople:
    """Stands in for Apple Vision: a selfie-sized face (5% of the frame) between 10 and 14 s."""

    def __init__(self):
        self.i = 0

    def detect(self, frames) -> dict:
        t = (self.i + np.arange(len(frames))) / 2.0                 # vision frames arrive at 2 fps
        self.i += len(frames)
        on = (t >= 10) & (t < 14)
        return {"face_area": np.where(on, 0.05, 0.0), "n_people": np.where(on, 1.0, 0.0)}


def people_checks() -> dict:
    c = analyse.analyse_clip(DATA / "GX010063.MP4", None, DATA / ".cache", "test+fakepeople", refresh=True,
                             people=FakePeople())
    plain = analyse.analyse_clip(DATA / "GX010063.MP4", None, DATA / ".cache", "test")
    compute([c])
    compute([plain])
    faces = c.parts["faces"]
    return {
        "a face raises interest while it's in view": faces[105:135].mean() > 0.7 and faces[:90].max() < 0.05
                                                     and c.parts["people"][110:130].mean() > 0.05,
        "no detector: no face or people parts": "faces" not in plain.parts and "people" not in plain.parts,
    }


def shot_checks() -> dict:
    """A clip whose CLIP embedding changes at 10 s: two shots, one pick per shot, relaxed to fill."""
    from datetime import datetime

    from butterfly.analyse import Clip
    from butterfly.select_moments import shots

    rng = np.random.default_rng(3)
    a, b = rng.normal(size=64), rng.normal(size=64)
    emb = np.array([(a if i < 20 else b) + rng.normal(scale=0.05, size=64) for i in range(40)])
    c = Clip(Path("scenes.mp4"), 20.0, datetime(2026, 1, 1), True, 30.0)
    c.emb_t, c.emb = np.arange(40) / 2.0, (emb / np.linalg.norm(emb, axis=1, keepdims=True)).astype(np.float16)
    c.q = np.ones(c.n)
    c.score = np.where(np.arange(c.n) < 100, 2.0, 1.0)            # scene A scores twice as high
    c.parts, c.gates = {"x": c.score}, {"q": c.q}
    bounds = shots(c)
    two = select([c], [2.5, 2.5])
    three = select([c], [2.5, 2.5, 2.5])
    side = lambda p: p.start + p.dur / 2 >= 10
    return {
        "shots: one boundary where the scene changes": bounds is not None and len(bounds) == 2
                                                       and abs(bounds[0][1] - 10) <= 1.0,
        "shots: one pick per shot when there's a choice": sorted(map(side, two)) == [False, True],
        "shots: the limit relaxes to fill the reel": len(three) == 3,
    }


def day_checks() -> dict:
    """Local days, a trip reel with day titles, per-day reels, and a rendered title card."""
    from datetime import date, datetime, timedelta

    from butterfly.analyse import Clip
    from butterfly.days import day_of, group_by_day, local_offsets, select_with_titles

    def clip(name, when):
        c = Clip(Path(name), 20.0, when, True, 30.0)
        c.q, c.score = np.ones(c.n), np.ones(c.n)
        c.parts, c.gates = {"x": c.score}, {"q": c.q}
        return c

    d1 = [clip("d1a.mp4", datetime(2026, 2, 18, 15)), clip("d1b.mp4", datetime(2026, 2, 18, 16))]
    d2 = [clip("d2a.mp4", datetime(2026, 2, 19, 23, 30))]            # 23:30 UTC = next day at UTC+2
    allc = d1 + d2
    plus2, _ = local_offsets(allc, "+02:00")
    utc = {id(c): timedelta(0) for c in allc}
    try:
        local_offsets(allc, "two hours")
        bad_tz = False
    except ValueError:
        bad_tz = True
    durations = plan_durations(15, 2.5, None, 0)                       # 6 slots
    reel = select_with_titles(allc, durations, utc)
    kinds = [p.clip.kind for p in reel]
    days = group_by_day(allc, utc)
    per_day = {d: select(items, durations) for d, items in days.items()}

    snap = Clip(Path("snap.jpg"), 0.0, datetime(2026, 2, 21, 12), False, 0.0, kind="photo")
    snap.q, snap.score = np.ones(1), np.ones(1)
    snap.parts, snap.gates = {"x": snap.score}, {"q": snap.q}
    with_photo_day = select_with_titles(allc + [snap], plan_durations(20, 2.5, None, 0),
                                        {**utc, id(snap): timedelta(0)})
    after = [p.clip for p in with_photo_day[[p.clip.info.get("title") for p in with_photo_day].index("Day 4") + 1:]]
    photo_day_ok = after == [snap]

    title = next(p for p in reel if p.clip.kind == "title")
    out = DATA / "_title.mp4"
    with contextlib.redirect_stdout(io.StringIO()):
        render([title], out, 1080, workers=1)
    w, h = vision.display_size(out)
    g = _gray_frame(out, 0.5)
    out.unlink()
    return {
        "days: the time zone decides the local day": day_of(d2[0], plus2) == date(2026, 2, 20) and bad_tz,
        "days: trip reel opens each day with a title": kinds[0] == "title" and kinds.count("title") == 2
            and reel[kinds.index("title", 1)].clip.info["title"] == "Day 2"
            and all(day_of(p.clip, utc) == date(2026, 2, 19) for p in reel[kinds.index("title", 1) + 1:]),
        "days: titles keep the planned length": abs(sum(p.dur for p in reel) - 15) < 0.1,
        "days: split mode keeps each day's cuts to that day": len(per_day) == 2 and all(
            all(p.clip in items for p in per_day[d]) for d, items in days.items()),
        "days: a title card renders (1920x1080, text)": (w, h) == (1920, 1080) and g.max() > 200 and g.mean() < 80,
        "days: a photo-only day gets its photos, not just a title": photo_day_ok,
    }


def review_checks() -> dict:
    """Report buttons, label upserts, and the local review server (run after cli_outputs)."""
    import threading
    import urllib.request

    from butterfly.review import serve, upsert_label

    page = (DATA / ".butterfly" / "report" / "index.html").read_text()
    lab = DATA / ".butterfly" / "labels.json"
    lab.unlink(missing_ok=True)
    upsert_label(lab, {"clip": "GX010061.MP4", "start": 6.0, "end": 8.5, "label": "good"})
    data = upsert_label(lab, {"clip": "GX010061.MP4", "start": 6.1, "end": 8.4, "label": "dull"})   # same cut
    try:
        upsert_label(lab, {"clip": "GX010061.MP4", "label": "great"})
        rejected = False
    except ValueError:
        rejected = True

    server = serve(DATA)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        index_ok = urllib.request.urlopen(base + "/index.html", timeout=5).status == 200
        req = urllib.request.Request(base + "/api/label", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"clip": "GX010062.MP4", "label": "unusable"}).encode())
        posted = json.loads(urllib.request.urlopen(req, timeout=5).read())["ok"]
        served = json.loads(urllib.request.urlopen(base + "/api/labels", timeout=5).read())["labels"]
    finally:
        server.shutdown()
        server.server_close()
    return {
        "report has label buttons for picks and photos": page.count('class="lab"') >= 5 and "data-clip=" in page
                                                         and "/api/label" in page,
        "labels: the same cut is relabelled, not duplicated": len(data["labels"]) == 1
                                                              and data["labels"][0]["label"] == "dull" and rejected,
        "review server: local only, serves the report, saves labels": server.server_address[0] == "127.0.0.1"
            and index_ok and posted and {"clip": "GX010062.MP4", "label": "unusable"} in served,
    }


def ranker_checks() -> dict:
    """Labels that contradict the hand weights: good = colourful, dull = loud (hand favours loud).
    12 labels across 6 clips, enough for the shrinkage (alpha = 12 / 32) to let the evidence show."""
    from datetime import datetime

    from butterfly.analyse import Clip
    from butterfly.ranker import load_weights, train

    rng = np.random.default_rng(5)
    clips = []
    names = [f"rank_{k}.mp4" for k in range(6)]
    for name in names:
        c = Clip(Path(name), 40.0, datetime(2026, 1, 1), True, 30.0)
        t = np.arange(c.n) * 0.1
        c.feats = {"colour": np.where((t >= 5) & (t < 15), 1.0, 0.1) + rng.normal(0, 0.02, c.n),
                   "audio_peak": np.where((t >= 20) & (t < 30), 1.0, 0.1) + rng.normal(0, 0.02, c.n)}
        clips.append(c)
    labels = [{"clip": n, "start": 5, "end": 15, "label": "good"} for n in names] + \
             [{"clip": n, "start": 20, "end": 30, "label": "dull"} for n in names]
    compute(clips)
    hand_order = clips[0].interest[50:150].mean() < clips[0].interest[200:300].mean()   # hand: dull above good
    r = train(clips, labels)
    compute(clips, r["weights"])
    learned_order = clips[0].interest[50:150].mean() > clips[0].interest[200:300].mean()
    folder = DATA / "rankfolder"
    (folder / C.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    (folder / C.OUTPUT_DIR / "ranker.json").write_text(json.dumps(r))
    used, _ = load_weights("auto", folder)
    (folder / C.OUTPUT_DIR / "ranker.json").write_text(json.dumps({**r, "cv_auc_learned": 0.1, "cv_auc_hand": 0.9}))
    refused, _ = load_weights("auto", folder)
    return {
        "ranker: learns colour over loudness from the labels": r["weights"]["colour"] > r["weights"]["audio"]
                                                               and hand_order,
        "ranker: beats the hand weights in cross-validation": r["cv_auc_learned"] > r["cv_auc_hand"],
        "ranker: learned weights reorder interest": learned_order,
        "ranker: auto uses it only when it beats the hand weights": used == r["weights"] and refused is None,
    }


def preset_checks(clips) -> dict:
    """Each preset changes its settings; bike raises riding action; the config is restored afterwards."""
    import copy

    keep = {k: copy.deepcopy(getattr(C, k)) for k in ("POSITIVE_PROMPTS", "NEGATIVE_PROMPTS", "INTEREST_WEIGHTS", "GATES")}

    def restore():
        for k, v in keep.items():
            obj = getattr(C, k)
            obj.clear()
            (obj.extend if isinstance(obj, list) else obj.update)(copy.deepcopy(v))

    ok = True
    try:
        for name, p in C.PRESETS.items():
            C.apply_preset(name)
            ok &= all(x in C.POSITIVE_PROMPTS for x in p["prompts"])
            ok &= all(C.INTEREST_WEIGHTS[k] == v for k, v in p["weights"].items())
            ok &= all(C.GATES[k] == v for k, v in p["gates"].items())
            restore()
        try:
            C.apply_preset("moon")
            unknown = False
        except ValueError:
            unknown = True
        c1 = clips[0]
        compute(clips)
        before = float(c1.parts["action"][60:140].mean())
        C.apply_preset("bike")
        compute(clips)
        after = float(c1.parts["action"][60:140].mean())
    finally:
        restore()
        compute(clips)
    return {
        "presets change their prompts, weights and gates": ok and unknown,
        "bike preset raises riding action": after > before * 1.4 > 0,
        "config restored after presets": C.INTEREST_WEIGHTS == keep["INTEREST_WEIGHTS"] and C.GATES == keep["GATES"],
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
        **audio_event_checks(),
        **people_checks(),
        **shot_checks(),
        **day_checks(),
        **ranker_checks(),
        **preset_checks(clips),
        **flow_checks(),
        **render_fill(),
        **cli_outputs(),
        **review_checks(),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("\npicks:", ", ".join(f"{p.clip.path.stem}@{p.start:.1f}" for p in picks))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
