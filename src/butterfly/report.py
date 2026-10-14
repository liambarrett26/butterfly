"""HTML report: what was picked, why, and per-clip timelines."""

from __future__ import annotations

import html
import subprocess
from pathlib import Path

import numpy as np

from . import config as C
from .score import unusable_fraction, why


def _thumb(p, out: Path) -> bool:
    t = p.start + p.src_dur / 2
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


def _label_buttons(clip_name: str, start: float | None = None, end: float | None = None) -> str:
    """good / dull / unusable buttons; butterfly-review saves clicks to the labels file."""
    rng = f' data-start="{start:.2f}" data-end="{end:.2f}"' if start is not None else ""
    return (f'<div class="lab" data-clip="{html.escape(clip_name)}"{rng}>'
            '<button data-l="good" title="good: worth showing">👍</button>'
            '<button data-l="dull" title="dull: usable but not worth showing">👎</button>'
            '<button data-l="unusable" title="unusable: never pick this">🚫</button></div>')


SCRIPT = """
const KEY = "butterfly-labels:" + document.body.dataset.folder;
const served = location.protocol.startsWith("http");
const same = (a, b) => a.clip === b.clip && (a.start === undefined ? b.start === undefined
  : b.start !== undefined && Math.abs(a.start - b.start) < 0.25 && Math.abs(a.end - b.end) < 0.25);
function load() { try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch (e) { return []; } }
function save(list) { try { localStorage.setItem(KEY, JSON.stringify(list)); } catch (e) {} }
function mark(list) {
  document.querySelectorAll(".lab").forEach(el => {
    const e = entry(el, null), hit = list.find(x => same(x, e));
    el.querySelectorAll("button").forEach(b => b.classList.toggle("on", !!hit && hit.label === b.dataset.l));
  });
  const bar = document.getElementById("dl");
  if (bar) bar.hidden = served || !list.length, bar.querySelector("span").textContent = list.length;
}
function entry(el, label) {
  const e = {clip: el.dataset.clip};
  if (el.dataset.start !== undefined) { e.start = +el.dataset.start; e.end = +el.dataset.end; }
  if (label) { e.label = label; e.note = "from the report"; }
  return e;
}
let labels = load();
document.querySelectorAll(".lab button").forEach(b => b.addEventListener("click", async () => {
  const e = entry(b.parentElement, b.dataset.l);
  labels = labels.filter(x => !same(x, e)).concat([e]);
  if (served) {
    try { await fetch("/api/label", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(e)}); }
    catch (err) { alert("Couldn't save the label: is butterfly-review still running?"); }
  } else save(labels);
  mark(labels);
}));
document.getElementById("dlbtn")?.addEventListener("click", () => {
  const blob = new Blob([JSON.stringify({version: 1, labels: labels}, null, 2)], {type: "application/json"});
  const a = Object.assign(document.createElement("a"), {href: URL.createObjectURL(blob), download: "labels.json"});
  a.click();
});
if (served) fetch("/api/labels").then(r => r.json()).then(d => { labels = d.labels || []; mark(labels); }).catch(() => mark(labels));
else mark(labels);
"""


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
    from .select_moments import shots
    for s, _ in (shots(c) or [])[1:]:
        ax.axvline(s, color="#6b6b70", ls=":", lw=1)
    for p in picks:
        if p.clip is c:
            ax.axvspan(p.start, p.start + p.src_dur, color="#2e9e5b", alpha=0.25)
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
table.dev{border-collapse:collapse;font-size:13px}table.dev td{padding:4px 12px 4px 0;border-bottom:1px solid var(--line)}
.lab{margin-top:6px}.lab button{font-size:15px;background:none;border:1px solid var(--line);border-radius:8px;
padding:2px 8px;margin-right:4px;cursor:pointer}.lab button.on{background:var(--accent);border-color:var(--accent)}
#dl{position:fixed;bottom:16px;right:16px;background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px 14px;box-shadow:0 2px 10px rgba(0,0,0,.15)}
.tag{display:inline-block;font-size:12px;padding:1px 8px;border-radius:99px;border:1px solid var(--line);margin-right:6px}
"""


def write_report(clips, picks, folder: Path, out_dir: Path, settings: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for i, p in enumerate(picks, 1):
        img = out_dir / f"pick_{i:02d}.jpg"
        photo = p.clip.kind == "photo"
        if p.clip.kind == "title":
            from .days import title_image
            title_image(p, (480, 270)).save(img, quality=85)
            cards.append(f'<div class="card"><img src="{img.name}" alt=""><div class="b"><div class="t">#{i} · '
                         f'{html.escape(p.clip.info["title"])} · {html.escape(p.clip.info["subtitle"])}</div>'
                         f'<div class="muted">day title · {p.dur:.1f}s</div></div></div>')
            continue
        src = img.name if (_photo_thumb(p.clip.path, img) if photo else _thumb(p, img)) else ""
        span = (f"photo · {p.dur:.1f}s" if photo else f"{p.start:.1f}–{p.start + p.src_dur:.1f}s"
                + (f" · slow ×{p.speed:g}" if p.speed != 1 else ""))
        cards.append(
            f'<div class="card"><img src="{src}" alt=""><div class="b">'
            f'<div class="t">#{i} · {html.escape(p.clip.path.name)} · {span}</div>'
            f'<div class="muted">score {p.score:.2f}</div><div>{html.escape(p.reasons)}</div>'
            + (_label_buttons(p.clip.path.name) if photo else _label_buttons(p.clip.path.name, p.start, p.start + p.src_dur))
            + '</div></div>')

    photo_cards = []
    for c in sorted((c for c in clips if c.kind == "photo"), key=lambda c: -float(c.score[0])):
        img = out_dir / f"photo_{c.path.stem}.jpg"
        src = img.name if _photo_thumb(c.path, img) else ""
        gated = float(c.q[0]) < C.GATES["window_min_q"] or bool(c.info.get("burst_of"))
        when = c.created.strftime("%Y-%m-%d %H:%M") + " UTC"
        photo_cards.append(
            f'<div class="card{" gated" if gated else ""}"><img src="{src}" alt=""><div class="b">'
            f'<div class="t">{html.escape(c.path.name)}</div>'
            f'<div class="muted">q {float(c.q[0]):.2f} · score {float(c.score[0]):.2f} · {html.escape(when)}</div>'
            + (f'<div class="bad">burst: kept {html.escape(c.info["burst_of"])}</div>' if c.info.get("burst_of")
               else '<div class="bad">gated: not usable</div>' if gated else "")
            + f'<div>{html.escape(why(c, 0, 1))}</div>'
            + (f'<div class="muted">{html.escape("; ".join(c.notes))}</div>' if c.notes else "")
            + _label_buttons(c.path.name) + '</div></div>')

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

    devices: dict[str, list] = {}
    for c in clips:
        devices.setdefault(c.info.get("device", "unknown"), []).append(c)
    rows = []
    for dev, items in sorted(devices.items(), key=lambda kv: min(c.created for c in kv[1])):
        t0, t1 = min(c.created for c in items), max(c.created for c in items)
        nv, nph = sum(c.kind == "video" for c in items), sum(c.kind == "photo" for c in items)
        what = ", ".join(x for x in (f"{nv} clip(s)" if nv else "", f"{nph} photo(s)" if nph else "") if x)
        rows.append(f"<tr><td>{html.escape(dev)}</td><td>{what}</td>"
                    f"<td>{t0:%Y-%m-%d %H:%M} – {t1:%Y-%m-%d %H:%M} UTC</td></tr>")
    devices_html = ('<h2>Devices</h2><div class="muted" style="margin-bottom:8px">If one device\'s times look '
                    'hours off from the others, correct its clock with <code>--clock-offset "DEVICE=±HH:MM"</code>.'
                    f'</div><table class="dev">{"".join(rows)}</table>') if len(devices) > 1 else ""
    total = sum(p.dur for p in picks)
    photos_html = (f'<h2>Photos</h2><div class="muted" style="margin-bottom:10px">Best first. '
                   f'Faded = gated (blurred, dark, blown or unusable). Usable photos compete with video for up to a quarter of the cuts.</div>'
                   f'<div class="grid">{"".join(photo_cards)}</div>') if photo_cards else ""
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Highlight report</title>
<style>{CSS}</style></head><body data-folder="{html.escape(str(folder))}"><main>
<h1>Highlight report</h1>
<div class="muted">{html.escape(str(folder))} · {sum(c.kind == "video" for c in clips)} clips · {len(photo_cards)} photos · {len(picks)} cuts · {total:.1f}s · {html.escape(settings)}</div>
<h2>Picks</h2><div class="grid">{''.join(cards)}</div>
<h2>Clips</h2>
<div class="muted" style="margin-bottom:10px">Blue = score, grey line = interest, grey fill = quality,
green = picked, red = drop (free-fall), dotted = shot boundary (at most one pick per shot).</div>
{''.join(sections)}
{photos_html}
{devices_html}
<div class="muted" style="margin-top:24px">Label picks and photos with 👍 good / 👎 dull / 🚫 unusable.
Run <code>butterfly-review {html.escape(str(folder))}</code> to save them straight to
<code>.butterfly/labels.json</code>; they feed <code>butterfly-eval</code> and the learned ranker.</div>
</main>
<div id="dl" hidden><span>0</span> label(s) in this browser ·
<button id="dlbtn">Download labels.json</button> → save it as <code>.butterfly/labels.json</code></div>
<script>{SCRIPT}</script></body></html>"""
    index = out_dir / "index.html"
    index.write_text(page)
    return index
