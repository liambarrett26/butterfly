"""
butterfly-train — learn interest weights from your labels.

  butterfly-train ~/Movies/GoPro        # reads .butterfly/labels.json, writes .butterfly/ranker.json
  butterfly ~/Movies/GoPro --ranker auto        # uses it if it beat the hand-set weights

Each labelled range is cut into windows (and each labelled photo is one sample). Its features
are the unweighted interest parts (CLIP semantics, aesthetics, faces, action, audio …).
An L2-regularised logistic regression learns good (1) against dull or unusable (0). Its
positive coefficients become interest weights, blended with the hand-set ones by how many
labels there are: alpha = items / (items + RANKER_PRIOR). Leave-one-clip-out cross-validation
compares the ranking quality (AUC) of the hand weights with the learned blend.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config as C


def samples(clips: list, labels: list[dict]):
    """-> (raw-part dicts, targets, group ids): windows of RANKER_WINDOW_S every RANKER_STRIDE_S in
    each labelled video range, and one sample per labelled photo."""
    by_name = {c.path.name: c for c in clips}
    feats, ys, groups = [], [], []
    for it in labels:
        c = by_name.get(it["clip"])
        if c is None or not c.raw:
            continue
        y = 1.0 if it["label"] == "good" else 0.0
        if c.kind == "photo":
            feats.append({k: float(v[0]) for k, v in c.raw.items()})
            ys.append(y)
            groups.append(c.path.name)
            continue
        s, e = max(float(it.get("start", 0.0)), 0.0), min(float(it.get("end", c.duration)), c.duration)
        w = min(C.RANKER_WINDOW_S, e - s)
        if w <= 0:
            continue
        for t in np.arange(s, e - w + 1e-6, C.RANKER_STRIDE_S):
            lo, hi = int(t / C.DT), max(int((t + w) / C.DT), int(t / C.DT) + 1)
            feats.append({k: float(v[lo:hi].mean()) for k, v in c.raw.items()})
            ys.append(y)
            groups.append(c.path.name)
    return feats, np.array(ys), np.array(groups)


def _matrix(feats: list[dict], keys: list[str]) -> np.ndarray:
    return np.array([[f.get(k, 0.0) for k in keys] for f in feats], float)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> tuple[np.ndarray, float]:
    """L2-regularised logistic regression by Newton steps -> (coefficients, intercept). Classes are
    weighted equally, so a few dull labels count as much as many good ones."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    sw = np.where(y > 0.5, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1)) * n
    beta = np.zeros(d + 1)
    reg = np.full(d + 1, l2)
    reg[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ beta, -30, 30)))
        grad = Xb.T @ (sw * (p - y)) + reg * beta
        H = (Xb * (sw * p * (1 - p))[:, None]).T @ Xb + np.diag(reg + 1e-9)
        step = np.linalg.solve(H, grad)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta[:-1], float(beta[-1])


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Probability that a random good sample outranks a random bad one (ties count half)."""
    pos, neg = scores[y > 0.5], scores[y <= 0.5]
    if not len(pos) or not len(neg):
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def learn_weights(feats: list[dict], y: np.ndarray, keys: list[str], hand: dict, n_items: int) -> tuple[dict, float]:
    """-> (blended weights for keys, alpha). Learned = positive coefficients on the raw scale,
    scaled to the hand weights' total; blend = (1 - alpha) hand + alpha learned. Parts that never
    vary in the labelled samples (e.g. no laughter on this trip) carry no evidence: they keep
    their hand weight and stay out of the fit."""
    X = _matrix(feats, keys)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    seen = sd > 1e-6
    alpha = n_items / (n_items + C.RANKER_PRIOR)
    out = {k: hand.get(k, 0.0) for k in keys}
    if not seen.any():
        return out, alpha
    coef, _ = fit_logistic((X[:, seen] - mu[seen]) / sd[seen], y, C.RANKER_L2)
    learned = np.clip(coef / sd[seen], 0, None)
    seen_keys = [k for k, s in zip(keys, seen) if s]
    total = sum(hand.get(k, 0.0) for k in seen_keys)
    if learned.sum() > 0:
        learned = learned * (total / learned.sum())
        for k, l in zip(seen_keys, learned):
            out[k] = (1 - alpha) * hand.get(k, 0.0) + alpha * float(l)
    return out, alpha


def train(clips: list, labels: list[dict]) -> dict:
    """Fit on all labels, with leave-one-clip-out AUC for hand vs learned weights."""
    feats, y, groups = samples(clips, labels)
    keys = sorted({k for f in feats for k in f})
    hand = {k: C.INTEREST_WEIGHTS.get(k, 0.0) for k in keys}
    n_items = sum(1 for it in labels if it["clip"] in {c.path.name for c in clips})
    if len(set(y.tolist())) < 2:
        raise ValueError("need both good and dull/unusable labels to train")
    score = lambda w, idx: _matrix([feats[i] for i in idx], keys) @ np.array([w[k] for k in keys])
    held_hand, held_learned, held_y = [], [], []
    for g in sorted(set(groups)):
        test, tr = np.flatnonzero(groups == g), np.flatnonzero(groups != g)
        if len(set(y[tr].tolist())) < 2:
            continue
        w, _ = learn_weights([feats[i] for i in tr], y[tr], keys, hand, n_items)
        held_hand.append(score(hand, test))
        held_learned.append(score(w, test))
        held_y.append(y[test])
    weights, alpha = learn_weights(feats, y, keys, hand, n_items)
    hy = np.concatenate(held_y) if held_y else np.zeros(0)
    return {
        "version": 1,
        "trained": datetime.now().isoformat(timespec="seconds"),
        "items": n_items, "samples": int(len(y)), "alpha": round(alpha, 3),
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "hand_weights": hand,
        "cv_auc_hand": round(auc(np.concatenate(held_hand), hy), 4) if held_y else None,
        "cv_auc_learned": round(auc(np.concatenate(held_learned), hy), 4) if held_y else None,
    }


def load_weights(spec: str, folder: Path) -> tuple[dict | None, str]:
    """--ranker auto|off|FILE -> (weights or None, a one-line note)."""
    if spec == "off":
        return None, ""
    path = folder / C.OUTPUT_DIR / "ranker.json" if spec == "auto" else Path(spec).expanduser()
    if not path.exists():
        return None, "" if spec == "auto" else f"no ranker at {path}"
    r = json.loads(path.read_text())
    better = (r.get("cv_auc_learned") or 0) > (r.get("cv_auc_hand") or 0)
    if spec == "auto" and not better:
        return None, f"ranker not used: it didn't beat the hand weights (AUC {r.get('cv_auc_learned')} vs {r.get('cv_auc_hand')})"
    return r["weights"], f"learned weights from {path.name} ({r['items']} labels, alpha {r['alpha']})"


def main() -> None:
    from .audio_events import load_tagger
    from .cli import analyse_folder, find_videos, load_scorer, score_folder
    from .evaluate import load_labels
    from .people import load_people
    from .photos import find_photos

    ap = argparse.ArgumentParser(prog="butterfly-train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--labels", type=Path, help="labels file (default <folder>/.butterfly/labels.json)")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--preset", choices=sorted(C.PRESETS))
    args = ap.parse_args()
    if args.preset:
        C.apply_preset(args.preset)
    folder = args.folder.expanduser()
    labels_path = (args.labels or folder / C.OUTPUT_DIR / "labels.json").expanduser()
    if not labels_path.exists():
        sys.exit(f"No labels at {labels_path}: label picks with butterfly-review first")
    labels = load_labels(labels_path)
    scorer, tag = load_scorer(False, args.no_clip, C.CLIP_MODEL, C.CLIP_PRETRAINED)
    tag += f"+preset-{args.preset}" if args.preset else ""
    tagger, s1 = load_tagger(True)
    people, s2 = load_people(True)
    clips = analyse_folder(folder, find_videos(folder), scorer, tag + s1 + s2, False, True, find_photos(folder),
                           tagger, people)
    score_folder(clips)
    try:
        r = train(clips, labels)
    except ValueError as e:
        sys.exit(str(e))
    out = folder / C.OUTPUT_DIR / "ranker.json"
    out.write_text(json.dumps(r, indent=2))
    print(f"\nTrained on {r['items']} labels ({r['samples']} samples), alpha {r['alpha']}")
    print(f"  leave-one-clip-out AUC: hand weights {r['cv_auc_hand']}, learned {r['cv_auc_learned']}")
    print("  weights: " + ", ".join(f"{k} {r['hand_weights'][k]:.2f}→{v:.2f}" for k, v in r["weights"].items()
                                   if abs(v - r["hand_weights"][k]) > 0.005 or v > 0))
    print(f"Saved {out}" + ("" if (r["cv_auc_learned"] or 0) > (r["cv_auc_hand"] or 0)
                            else " (--ranker auto won't use it: it doesn't beat the hand weights yet)"))


if __name__ == "__main__":
    main()
