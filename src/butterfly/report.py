"""HTML report: what was picked, why, and per-clip timelines."""

from __future__ import annotations

import html
import subprocess
from pathlib import Path

import numpy as np

from . import config as C
from .score import unusable_fraction, why


def _thumb(p, out: Path) -> bool:
    t = p.start + p.dur / 2
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(p.clip.path),
                        "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4", str(out)],
                       capture_output=True)
    return r.returncode == 0 and out.exists()


def _photo_thumb(path: Path, out: Path) -> bool:
    from .photos import open_photo
    try:
        im, _ = open_photo(path)
        im.thumbnail((480, 480))
        im.save(out, quality=85)
        return True
    except Exception:
        return False


def _timeline(c, picks, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(c.n) * C.DT
    fig, ax = plt.subplots(figsize=(10, 2.2))
    top = max(float(np.max(c.interest)), 0.5)
    ax.fill_between(t, 0, c.q * top, color="#9aa5b1", alpha=0.25, lw=0, label="quality (scaled)")
    ax.plot(t, c.interest, color="#8f8f8f", lw=0.8, label="interest")
    ax.plot(t, c.score, color="#2f6fdb", lw=1.4, label="score")
    for s, e in c.events.get("drops", []):
        ax.axvspan(s, e, color="#d64545", alpha=0.6)
    for p in picks:
        if p.clip is c:
            ax.axvspan(p.start, p.start + p.dur, color="#2e9e5b", alpha=0.25)
    ax.set_xlim(0, max(c.duration, 1))
    ax.set_ylim(0, top * 1.05)
    ax.set_xlabel("seconds")
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


CSS = """
:root{--bg:#fafaf8;--fg:#1d1d1f;--muted:#6b6b70;--card:#fff;--line:#e4e4e0;--accent:#2f6fdb}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#ececec;--muted:#9a9aa0;--card:#1f1f23;--line:#2e2e33}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 12px}
.muted{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.card img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover;background:#000}
.card .b{padding:10px 12px;font-size:13px}.card .t{font-weight:600}
.clip{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:14px}
.clip img{width:100%;height:auto;background:#fff;border-radius:6px}
.card.gated{opacity:.55}.card .bad{color:#d64545;font-weight:600}
.tag{display:inline-block;font-size:12px;padding:1px 8px;border-radius:99px;border:1px solid var(--line);margin-right:6px}
"""


def write_report(clips, picks, folder: Path, out_dir: Path, settings: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for i, p in enumerate(picks, 1):
        img = out_dir / f"pick_{i:02d}.jpg"
        src = img.name if _thumb(p, img) else ""
        cards.append(
            f'<div class="card"><img src="{src}" alt=""><div class="b">'
            f'<div class="t">#{i} · {html.escape(p.clip.path.name)} · {p.start:.1f}–{p.start + p.dur:.1f}s</div>'
            f'<div class="muted">score {p.score:.2f}</div><div>{html.escape(p.reasons)}</div></div></div>')

    photo_cards = []
    for c in sorted((c for c in clips if c.kind == "photo"), key=lambda c: -float(c.score[0])):
        img = out_dir / f"photo_{c.path.stem}.jpg"
        src = img.name if _photo_thumb(c.path, img) else ""
        gated = float(c.q[0]) < C.GATES["window_min_q"]
        when = c.created.strftime("%Y-%m-%d %H:%M") + " UTC"
        photo_cards.append(
            f'<div class="card{" gated" if gated else ""}"><img src="{src}" alt=""><div class="b">'
            f'<div class="t">{html.escape(c.path.name)}</div>'
            f'<div class="muted">q {float(c.q[0]):.2f} · score {float(c.score[0]):.2f} · {html.escape(when)}</div>'
            + ('<div class="bad">gated: not usable</div>' if gated else "")
            + f'<div>{html.escape(why(c, 0, 1))}</div>'
            + (f'<div class="muted">{html.escape("; ".join(c.notes))}</div>' if c.notes else "")
            + '</div></div>')

    sections = []
    for c in (c for c in clips if c.kind == "video"):
        img = out_dir / f"timeline_{c.path.stem}.png"
        _timeline(c, picks, img)
        n_picks = sum(p.clip is c for p in picks)
        tags = [f"{c.duration:.0f}s", f"{n_picks} pick(s)", f"{100 * unusable_fraction(c):.0f}% unusable"]
        if c.events.get("drops"):
            tags.append(f"{len(c.events['drops'])} drop(s)")
        notes = f'<div class="muted">{html.escape("; ".join(c.notes))}</div>' if c.notes else ""
        sections.append(
            f'<div class="clip"><div><b>{html.escape(c.path.name)}</b> '
            + "".join(f'<span class="tag">{html.escape(t)}</span>' for t in tags)
            + f'</div>{notes}<img src="{img.name}" alt="timeline"></div>')

    total = sum(p.dur for p in picks)
    photos_html = (f'<h2>Photos</h2><div class="muted" style="margin-bottom:10px">Best first. '
                   f'Faded = gated (blurred, dark, blown or unusable). Photos join the reel in a later version.</div>'
                   f'<div class="grid">{"".join(photo_cards)}</div>') if photo_cards else ""
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Highlight report</title>
<style>{CSS}</style></head><body><main>
<h1>Highlight report</h1>
<div class="muted">{html.escape(str(folder))} · {sum(c.kind == "video" for c in clips)} clips · {len(photo_cards)} photos · {len(picks)} cuts · {total:.1f}s · {html.escape(settings)}</div>
<h2>Picks</h2><div class="grid">{''.join(cards)}</div>
<h2>Clips</h2>
<div class="muted" style="margin-bottom:10px">Blue = score, grey line = interest, grey fill = quality,
green = picked, red = drop (free-fall).</div>
{''.join(sections)}
{photos_html}
</main></body></html>"""
    index = out_dir / "index.html"
    index.write_text(page)
    return index
