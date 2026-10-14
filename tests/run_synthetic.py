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
from butterfly.score import compute  # noqa: E402
from butterfly.select_moments import plan_durations, select  # noqa: E402

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


def fake_telemetry(path: Path) -> dict:
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
        **cli_outputs(),
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("\npicks:", ", ".join(f"{p.clip.path.stem}@{p.start:.1f}" for p in picks))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
