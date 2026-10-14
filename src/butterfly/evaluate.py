"""
butterfly-eval — check quality gates and picks against hand labels.

  butterfly-eval ~/Movies/GoPro                    # labels from <folder>/.butterfly/labels.json
  butterfly-eval ~/Movies/GoPro --labels my.json --strict

Runs the same analysis and selection as `butterfly --dry-run`, then reports, per label,
how much is usable (q), how much is gated out, and how much made the reel, plus any
violations: an unusable range that was picked, or a good range that is mostly gated.
The labels file format is described in docs/labels.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config as C
from .cli import analyse_folder, find_videos, load_scorer
from .score import compute
from .select_moments import plan_durations, select

LABELS = ("good", "unusable", "dull")


def load_labels(path: Path) -> list[dict]:
    """Read and validate a labels file -> list of {clip, start?, end?, label, note?}."""
    data = json.loads(Path(path).read_text())
    items = data.get("labels", []) if isinstance(data, dict) else data
    for i, it in enumerate(items):
        if "clip" not in it or it.get("label") not in LABELS:
            raise ValueError(f"labels[{i}]: needs 'clip' and a 'label' in {LABELS}: {it}")
        if ("start" in it) != ("end" in it) or ("start" in it and it["end"] <= it["start"]):
            raise ValueError(f"labels[{i}]: give both 'start' < 'end' (s), or neither for the whole clip: {it}")
    return items


def evaluate(clips: list, picks: list, labels: list[dict]) -> dict:
    """-> {'rows': {label: stats}, 'violations': [str], 'missing': [clip names]}."""
    by_name = {c.path.name: c for c in clips}
    acc = {k: {"items": 0, "seconds": 0.0, "q_sum": 0.0, "gated_s": 0.0, "picked_s": 0.0, "items_picked": 0}
           for k in LABELS}
    violations, missing = [], []
    for it in labels:
        c = by_name.get(it["clip"])
        if c is None:
            missing.append(it["clip"])
            continue
        s, e = float(it.get("start", 0.0)), float(it.get("end", c.duration))
        s, e = max(s, 0.0), min(e, c.duration)
        lo, hi = int(s / C.DT), max(int(e / C.DT), int(s / C.DT) + 1)
        q = c.q[lo:hi]
        secs = (hi - lo) * C.DT
        gated = float((q < C.GATES["window_min_q"]).mean()) if len(q) else 0.0
        picked = sum(max(0.0, min(e, p.start + p.dur) - max(s, p.start)) for p in picks if p.clip is c)
        a = acc[it["label"]]
        a["items"] += 1
        a["seconds"] += secs
        a["q_sum"] += float(q.sum())
        a["gated_s"] += gated * secs
        a["picked_s"] += picked
        a["items_picked"] += picked > 0.3
        where = f"{c.path.name} {s:.1f}–{e:.1f}s" + (f" ({it['note']})" if it.get("note") else "")
        if it["label"] == "unusable" and picked > 0.3:
            violations.append(f"unusable but picked ({picked:.1f}s): {where}")
        if it["label"] == "good" and gated > 0.5:
            violations.append(f"good but {100 * gated:.0f}% gated: {where}")
    rows = {k: {"items": a["items"], "seconds": a["seconds"],
                "mean_q": a["q_sum"] * C.DT / a["seconds"] if a["seconds"] else float("nan"),
                "gated": a["gated_s"] / a["seconds"] if a["seconds"] else float("nan"),
                "picked": a["picked_s"] / a["seconds"] if a["seconds"] else float("nan"),
                "items_picked": a["items_picked"]}
            for k, a in acc.items()}
    return {"rows": rows, "violations": violations, "missing": sorted(set(missing))}


def print_report(result: dict) -> None:
    print(f"\n  {'label':<9} {'items':>5} {'seconds':>8} {'mean q':>7} {'gated':>6} {'picked':>7}  items picked")
    for k, r in result["rows"].items():
        if r["items"]:
            print(f"  {k:<9} {r['items']:>5} {r['seconds']:>8.1f} {r['mean_q']:>7.2f} {100 * r['gated']:>5.0f}% "
                  f"{100 * r['picked']:>6.0f}%  {r['items_picked']}/{r['items']}")
    for m in result["missing"]:
        print(f"  (no clip named {m} in the folder)")
    print(f"\n  {len(result['violations'])} violation(s)")
    for v in result["violations"]:
        print(f"    - {v}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="butterfly-eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--labels", type=Path, help="labels file (default <folder>/.butterfly/labels.json)")
    ap.add_argument("--length", type=float, default=30, help="reel length in seconds (default 30)")
    ap.add_argument("--seg", type=float, default=2.5, help="target seconds per cut (default 2.5)")
    ap.add_argument("--diversity", type=float, default=C.DIVERSITY)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--clip-model", default=C.CLIP_MODEL)
    ap.add_argument("--clip-pretrained", default=C.CLIP_PRETRAINED)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 1 if there are violations")
    args = ap.parse_args()

    folder = args.folder.expanduser()
    labels_path = (args.labels or folder / C.OUTPUT_DIR / "labels.json").expanduser()
    if not labels_path.exists():
        sys.exit(f"No labels file at {labels_path} (format: docs/labels.md)")
    labels = load_labels(labels_path)
    scorer, tag = load_scorer(args.no_vision, args.no_clip, args.clip_model, args.clip_pretrained)
    clips = analyse_folder(folder, find_videos(folder), scorer, tag, args.refresh, not args.no_vision)
    if not clips:
        sys.exit("Nothing to work with.")
    compute(clips)
    picks = select(clips, plan_durations(args.length, args.seg, None, 0), args.diversity)
    result = evaluate(clips, picks, labels)
    print(f"\nLabels: {labels_path} ({len(labels)} entries) · reel {sum(p.dur for p in picks):.1f}s, {len(picks)} cuts")
    print_report(result)
    sys.exit(1 if args.strict and result["violations"] else 0)


if __name__ == "__main__":
    main()
