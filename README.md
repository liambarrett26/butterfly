# 🦋 Butterfly

[![tests](https://github.com/liambarrett26/butterfly/actions/workflows/tests.yml/badge.svg)](https://github.com/liambarrett26/butterfly/actions/workflows/tests.yml)

**Local highlight reels from GoPro footage.**

Butterfly turns a folder of GoPro clips into a short highlight reel on your own
Mac. It reads the telemetry the camera records inside every file (motion,
orientation, GPS), listens to the audio, and looks at the frames with an
on-device vision model. From that it picks the best moments and cuts them
together, optionally on the beat of a song.

Nothing is uploaded. There's no account, no cloud processing and no Wi-Fi
offload. You copy the footage over USB-C or from a card reader, and run it.

> **Status: early development (v0.2).** The pipeline runs end-to-end and passes
> its synthetic test suite. It has **not yet been tuned on real footage**. See
> [Project status](#project-status).

---

## Why

GoPro's Quik app makes automatic highlight reels, but it's mobile-only (the
desktop app was discontinued at the end of 2024). It also works best with
footage uploaded to GoPro's cloud. A modern laptop is more than capable of
doing this locally, and a lot faster.

Butterfly's first prototype scored moments by raw motion and learned the
obvious lesson: dropping the camera is not a highlight. The current design is
built around that.

## How it works

### Gate first, then score

Every 0.1 s of footage gets two separate values:

1. **Quality `q` (0–1): is this footage usable?** Drops, handling shake, blur,
   darkness, blown-out frames, and the lens pointed at the floor all push it
   towards 0.
2. **Interest: is it worth showing?** This comes from what's in the frame (a
   CLIP vision model), how good it looks, *smooth* action, audio peaks and
   water entries. No manual tagging is needed.

```
score(t) = interest(t) × q(t)
```

Motion only counts as interest when it's smooth, intentional motion such as
riding, carving or panning. Erratic shake never adds to it. A candidate cut is
rejected outright if any part of it is unusable, so a great moment that ends
with the camera hitting the sand doesn't make the reel.

### Pipeline

```
 folder of videos (.mp4 .mov .m4v)
        │
        ├── telemetry (GPMF) ─── accel · gyro · gravity · GPS9 · wet-mic
        │     └─ drops (free-fall), handling shake, tumble, smooth turns, speed, water entry
        ├── audio (ffmpeg) ───── loudness peaks above a rolling baseline
        └── vision (2 fps; uses GoPro .LRV proxies when present)
              ├─ classic CV: sharpness, exposure, colourfulness, frame change
              └─ CLIP on the GPU (Apple MPS): interest, aesthetics, "unusable" prompts,
                 and embeddings for diversity
        │
        ▼
 per-clip timelines (cached) → folder-wide normalisation → quality gates × interest
        │
        ▼
 candidate windows → diversity-aware selection (MMR) → chronological order
        │
        ├── optional beat grid from music (librosa)
        ▼
 ffmpeg (VideoToolbox) → highlights.mp4  +  HTML report explaining every pick
```

### Signals

| Signal | Source | Role |
|---|---|---|
| Drop / free-fall | accelerometer magnitude < 3 m/s² for ≥ 80 ms | hard gate, −1 s to +2.5 s around it |
| Handling shake | high-pass gyro, combined with blur / tumble / low speed | gate (softer while riding) |
| Camera shake (no gyro) | optical flow: jitter in the image's camera motion, smooth pans excluded | gate, for footage without telemetry |
| Tumble | gravity vector vs the clip's usual orientation | gate |
| Sharpness, exposure | Laplacian variance, mean luma, clipped pixels | gates |
| CLIP "unusable" | blurry / floor / pocket / finger-on-lens prompts | gate |
| CLIP interest | similarity to "good moment" prompts | interest |
| CLIP aesthetic | "beautiful photo" vs "boring snapshot" | interest |
| Smooth action | GPS speed + low-passed turning, with shake removed | interest |
| Audio peaks | loudness above a 5 s rolling median | interest |
| Water entry | wet-microphone transitions | interest bump |
| Colourfulness | Hasler–Süsstrunk metric | small interest term |

All signals are normalised **across the whole folder**, so a dull clip stays
dull instead of being stretched to look exciting.

## Requirements

- macOS on Apple Silicon (developed on an M3 Pro). Other platforms should work
  on CPU but are untested.
- Python 3.12+
- ffmpeg (`brew install ffmpeg`)
- A GoPro **HERO5 or later**, since earlier models don't record telemetry.
  Developed against a **HERO13 Black**.

## Installation

```bash
git clone https://github.com/<your-username>/butterfly.git
cd butterfly
conda create -n butterfly python=3.12
conda activate butterfly
python -m pip install -e ".[all]"
```

This installs the `butterfly` command. The extras are optional:
`.[vision]` adds the CLIP model (torch, open_clip), `.[music]` adds beat
syncing (librosa), and `.[all]` adds both. Without `vision`, Butterfly runs
with classic computer vision only. Dependencies are pinned to exact versions.

A dedicated environment keeps Butterfly's dependencies apart from everything
else. Use `python -m pip`, not bare `pip`, so the packages land in the same
Python you run Butterfly with. The editable install (`-e`) means edits to
`src/butterfly/config.py` take effect without reinstalling.

The first run downloads the CLIP weights (ViT-B/32, about 600 MB) from Hugging
Face. After that, Butterfly works fully offline.

## Quick start

```bash
# 1. Analyse and plan without rendering; writes an HTML report
butterfly ~/Movies/GoPro --dry-run
open ~/Movies/GoPro/.butterfly/report/index.html

# 2. Render a 30-second reel
butterfly ~/Movies/GoPro

# 3. Or cut it to the beat of a song
butterfly ~/Movies/GoPro --music ~/Music/track.mp3
```

`python -m butterfly …` and `python reel.py …` (from the repository root)
work the same way.

The reel is written to `<folder>/highlights.mp4`. Everything else Butterfly
writes goes in `<folder>/.butterfly/`: the analysis cache (`cache/`), so
re-running with different settings takes seconds, and the HTML report
(`report/`). Delete that folder to start fresh; your footage is never touched.
Versions before 0.2.3 used `.reel_cache/` and `highlights_report/`. Butterfly
ignores those now, and you can delete them.

**Tip:** copy the low-res proxy files (`GL01xxxx.LRV`) from the SD card
alongside the MP4s. Butterfly uses them for the vision pass, which is much
faster than decoding 4K HEVC.

## Options

| Flag | Default | What it does |
|---|---|---|
| `--length` | 30 | Reel length in seconds |
| `--seg` | 2.5 | Target seconds per cut |
| `--music` | — | Song to sync cuts to (every 1/2/4/8 beats) |
| `--music-start` | 0 | Start offset into the song (s) |
| `--clip-audio` | 0.25 | Volume of the original audio under the music (0 = mute) |
| `--res` | 1080 | Output resolution: 1080 or 2160 |
| `--diversity` | 0.3 | 0 = pure score, 1 = maximum visual variety |
| `--no-clip` | off | Classic CV only, no vision model |
| `--no-vision` | off | Telemetry + audio only (fastest) |
| `--clip-model` / `--clip-pretrained` | ViT-B-32 / laion2b | e.g. `MobileCLIP-S2` / `datacompdr` for speed |
| `--refresh` | off | Ignore the analysis cache |
| `--dry-run` | off | Analyse, plan and write the report without rendering |

## Tuning

Everything tunable lives in **`src/butterfly/config.py`**:

- **Prompts** (`POSITIVE_PROMPTS`, `NEGATIVE_PROMPTS`) matter most. Describe
  what good footage from *your* trips looks like, e.g. "snorkelling with
  turtles", "riding a scooter through jungle".
- **Quality gates** (`GATES`): if good but bumpy riding footage is rejected,
  raise `shake_ride_speed` or lower `shake_penalty`.
- **Interest weights** (`INTEREST_WEIGHTS`) balance vision, action, audio and
  water entries.
- **Spread across clips** (`PICKS_PER_CLIP_SLACK`): how many cuts one clip may
  take beyond an even share. Raise it if your best clip deserves more of the
  reel. On a trip where everything looks alike (all turquoise water, say),
  this does more than `DIVERSITY`, because CLIP sees every shot as similar.

The HTML report shows a thumbnail and a "why" for every pick, plus a timeline
per clip with score, quality and drops. It's the main tool for
seeing what to change.

## Repository layout

```
butterfly/
├── src/butterfly/
│   ├── cli.py               `butterfly` command: arguments and orchestration
│   ├── config.py            all weights, thresholds and prompts
│   ├── gpmf.py              GPMF telemetry and GPS9 decoding
│   ├── signals.py           telemetry + audio features on the 0.1 s grid
│   ├── vision.py            frame sampling, classic CV, CLIP scoring
│   ├── analyse.py           per-clip orchestration and cache
│   ├── score.py             normalisation, quality gates, interest
│   ├── select_moments.py    candidate windows, MMR selection, beat plan
│   ├── render.py            ffmpeg cutting, concat, music mix
│   └── report.py            HTML report and timelines
├── tests/
│   └── run_synthetic.py     synthetic regression suite
├── docs/design/             design proposals for upcoming milestones
├── reel.py                  compatibility shim for `python reel.py …`
├── pyproject.toml
└── README.md
```

## Development

Butterfly is built in small, tested iterations. Each roadmap item becomes one
branch and one pull request.

1. **Pick a roadmap item** and open an issue for it with the *Roadmap item*
   template, which asks for a clear definition of done. Report problems with
   the *Bug report* template.
2. **Branch:** `feat/<short-name>`, `fix/<short-name>` or `tune/<short-name>`.
3. **Run the test suite** before every commit:
   ```bash
   python tests/run_synthetic.py      # must exit 0
   ```
   It builds small synthetic clips, injects fake telemetry (a drop, handling
   shake, smooth riding, blur, darkness) and checks the gates and picks. It
   needs no network or model. When you add a signal or gate, add a synthetic
   region and a check for it.
4. **Bump `CACHE_VERSION`** in `src/butterfly/config.py` if you change what the analysis
   computes. Otherwise stale cached features are silently reused.
5. **Update the docs:** tick the roadmap item below, add an entry to the
   [Changelog](#changelog), and update `CLAUDE.md` if any conventions or camera
   facts changed.
6. **Open a PR** describing the change, its before/after effect on a real
   folder where relevant, and the test result. The PR template has the
   checklist.

### Working with Claude Code

`CLAUDE.md` is loaded automatically by Claude Code. It records the
architecture, data conventions, verified facts about HERO13 telemetry (and the
quirks of the `telemetrik` parser), and the prioritised task list. A good way
to work is one roadmap item per session: *"Work on 0.3: verify on real
footage."*

### Versioning

[Semantic versioning](https://semver.org) while pre-1.0: each milestone below
is a minor release (`v0.3.0`, `v0.4.0` …), tagged on `main` when its items are
done. Fixes and tuning in between are patch releases.

## Project status

### Verified

Tested end-to-end on synthetic clips with injected telemetry (21 / 21 checks
passing, also in CI):

| Situation | Expected | Result |
|---|---|---|
| Smooth riding (8 m/s, steady turn) | usable, picked | q = 1.00, picked |
| Loud moment 2 s before a drop | picked, stops before the drop | picked 20.0–22.4 s |
| Free-fall drop | unusable around it | q = 0.00, never picked |
| Handling shake + pointing at the ground | heavily penalised | q = 0.09 |
| Blurred section | penalised | q ≈ 0.5 |
| Dark section | penalised | q < 0.7 |
| Render + report | full-length reel, thumbnails, timelines | ✓ |
| CLI dry run | cache and report in `.butterfly/`, old `.reel_cache/` untouched | ✓ |
| Phone-style `.MOV`, rotated 90° | picked up, analysed as portrait | ✓ |
| No telemetry: handheld shake / smooth pan / static | shake penalised; pan and static not | ✓ |

HERO13 GPMF decoding (all 20 streams, including GPS9) was verified on real
files with a separate probe script.

CLIP with real weights (ViT-B-32, laion2b) was run on one real HERO13 trip:
8 clips, about 5 minutes of floating and boating through mangroves. After
tuning, all 12 picks are usable and the reel draws on 6 of the 8 clips.
Before tuning, it picked murky underwater footage and took 4 cuts each from
two clips.

### Not yet verified

- Trips that look different from that one: snow, cities, biking. The tuning
  so far is based on a single water-heavy trip.
- Gate thresholds against real drops; that trip had none.
- Close-up selfies: CLIP rates them barely above scenery, so they rarely make
  the reel. Face detection (v0.7) is the planned fix.

## Roadmap

### v0.1 — prototype ✅
- [x] Motion + audio + HiLight scoring, beat-synced rendering

### v0.2 — gate-then-score with vision ✅
- [x] Free-fall drop detection with a hard exclusion window
- [x] Handling-shake detector (high-pass gyro × blur / tumble), speed-aware
- [x] Orientation tumble gate from the gravity vector
- [x] Classic CV quality: sharpness, exposure, colourfulness, frame change
- [x] CLIP interest, aesthetic and "unusable" scoring on the MPS GPU
- [x] Smooth-action interest, audio peaks, water entry, HiLights
- [x] Window rejection if any part is unusable
- [x] MMR diversity selection over CLIP embeddings
- [x] Analysis cache, LRV proxy support, HTML report
- [x] Synthetic regression suite

### v0.2.x — repository setup
- [x] `pyproject.toml` with a `butterfly` console command and pinned dependencies
- [x] GitHub Actions running the synthetic suite (`--no-clip`) on every PR
- [x] Licence (MIT)
- [x] Issue and PR templates
- [x] Rename the internal `.reel_cache` / `highlights_*` outputs to `.butterfly/`

### v0.3 — real-footage validation ✅
- [x] Dry-run with real CLIP weights on a real trip; tune `GATES` and prompts
- ~~Confirm the HERO13 HiLight layout with a tagged clip~~ Dropped: Butterfly
  is fully automatic, and HiLight support has been removed
- The labelled real-footage set moved to v0.4, so it can cover more than GoPro

### v0.4 — any video
Design: [docs/design/any-footage.md](docs/design/any-footage.md)
- [x] Pick up `.MOV` / `.m4v` files and honour rotation metadata
- [x] Optical flow: a vision-based camera-shake gate for footage without
  telemetry. Subject motion is measured too, but is off as interest for now:
  on real footage it mostly picks up parallax and water
- [ ] Portrait clips in a landscape reel: a blurred fill instead of black bars
- [ ] A labelled set of real clips (good / unusable / dull) from a mixed
  GoPro + phone folder, as a second test suite

### v0.5 — photos
Design: [docs/design/any-footage.md](docs/design/any-footage.md#part-2-photos-v05)
- [ ] Discover and analyse photos (JPEG, PNG, HEIC): quality, CLIP, EXIF time
  and orientation, with a photos section in the report
- [ ] Photos in the reel: one selection pool with video, a cap on the photo
  share, gentle Ken Burns motion, a blurred fill for portrait shots, and
  beat-aligned durations
- [ ] Bursts and duplicates collapsed, photos of a moment already in a picked
  video dropped, clocks aligned across devices, and Live Photos paired

### v0.6 — motion and timing
- [ ] Smart in/out points: shift cuts off shake and onto motion onsets
- [ ] Slow motion for high-frame-rate clips (≥ 100 fps)

### v0.7 — audio and people
- [ ] Audio event tagging (laughter, cheering, splash; wind as a gate)
- [ ] Faces and people via Apple Vision, on-device

### v0.8 — structure
- [ ] Shot segmentation from embedding change points, at most one pick per shot
- [ ] Multi-day folders: one reel per day, or a trip reel with day titles

### v0.9 — personalisation
- [ ] Thumbs up/down on picks in the HTML report, saved to a labels file
- [ ] A learned ranker over window features, replacing hand-set weights
- [ ] Trip presets (beach, snow, city, bike)

### Later
- [ ] Multi-camera: align two cameras by time or GPS and cut between angles
- [ ] Vertical (9:16) reels

## Known limitations

- Butterfly ignores HiLight tags made on the camera. It picks moments
  automatically, and doesn't depend on a feature specific to one camera
  maker.
- CLIP looks at the centre square of each frame, so action at the far edges of
  a wide shot is under-weighted.
- HyperSmooth stabilises the picture, so gyro shake over-states visible shake.
  That's why the handling gate also requires blur, a big orientation change, or
  low speed.
- The telemetry parser (`telemetrik`) returns raw bytes for some HERO13 streams
  and mislabels camera temperature. Butterfly works around both.

## Changelog

### Unreleased

- Imported the v0.2 code into the repository root; `.gitignore` now covers
  Butterfly's outputs and footage. Install steps use a dedicated conda
  environment, and Pillow is no longer listed (nothing imports it).
- Licence confirmed as MIT.
- Packaged as `butterfly` (`src/butterfly/`, `pyproject.toml`): install with
  `python -m pip install -e ".[all]"` and run `butterfly <folder>`.
  Dependencies are pinned exactly, which raises the minimum to Python 3.12.
  matplotlib is now a core dependency; `vision` and `music` are extras.
  `python reel.py` still works.
- CI: GitHub Actions runs the synthetic test suite on Ubuntu (Python 3.12,
  core dependencies, no torch) on every push and pull request.
- The analysis cache and HTML report now live in `<folder>/.butterfly/cache/`
  and `<folder>/.butterfly/report/`. The reel stays at
  `<folder>/highlights.mp4`. An old `.reel_cache/` is ignored, not deleted.
- Issue templates (bug report, roadmap item) and a pull request template.
- Roadmap: footage from any source (phone videos, other cameras, photos),
  replacing "other action cameras".
- Optical flow (OpenCV, now a core dependency): footage without telemetry,
  such as phone video, gets a camera-shake gate from the image itself. Smooth
  pans don't count as shake. Subject motion is computed but its interest
  weight is 0, because on a real trip it rewarded a boat ride (parallax, the
  bow, water) over people floating. One decode now serves both flow (6 fps)
  and the existing 2 fps metrics. `CACHE_VERSION` is now 6.
- Phone video: `.MOV` and `.m4v` files are now picked up as well as `.mp4`,
  and rotated (portrait) clips are analysed at the right aspect instead of
  squashed. `CACHE_VERSION` is now 5.
- Roadmap reordered: v0.4 is now "any video" (`.MOV`, rotation, optical-flow
  shake gate, portrait fill), and v0.5 is photos, with a design in
  `docs/design/any-footage.md`. The later milestones move up by two.
- Removed HiLight tag support (the `hilight` interest part, its settings, and
  the report markers). Butterfly is fully automatic and shouldn't depend on
  a GoPro-only feature. `CACHE_VERSION` is now 4, so the first run
  re-analyses.
- Tuned on a real trip with CLIP. New prompts reward people floating and
  selfies; a new "murky underwater view" prompt marks cloudy underwater
  footage as unusable; the aesthetic baseline is now "a boring, badly framed
  photo of nothing in particular". A clip may take at most one cut beyond its
  even share of the reel (new `PICKS_PER_CLIP_SLACK`, previously a fixed 2).
  `CACHE_VERSION` is now 3, so the first run re-analyses.

- **0.2.2** (2026-10-13): Project renamed **Butterfly** and set up as a GitHub
  repository; README rewritten with a development workflow and versioned
  roadmap.
- **0.2.1** (2026-09-26): Added `CLAUDE.md` and the synthetic regression suite
  (11 checks). The dark gate now ramps in over ±10 luma around `dark_luma`
  (raised 25 → 40).
- **0.2.0** (2026-09-26): Gate-then-score pipeline, CLIP vision, drop and
  handling detection, MMR diversity, caching, LRV proxies, HTML report.
- **0.1.0** (2026-09-26): Prototype (`gopro_highlights.py`). Motion, audio and
  HiLight scoring with beat-synced rendering.

## Acknowledgements

- [`telemetrik`](https://github.com/kmatzen/telemetrik) for GPMF parsing, and
  GoPro's open [`gpmf-parser`](https://github.com/gopro/gpmf-parser)
  documentation.
- [OpenCLIP](https://github.com/mlfoundations/open_clip) for the vision models.
- [librosa](https://librosa.org) for beat tracking, and [FFmpeg](https://ffmpeg.org).

Butterfly is an independent project and is not affiliated with or endorsed by
GoPro, Inc. GoPro, HERO, HiLight and Quik are trademarks of GoPro, Inc.

## Licence

[MIT](LICENSE) © 2026 Liam Barrett.