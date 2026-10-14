# Design: footage from any source, and photos in the reel

Status: **approved by the owner (2026-10-14)**, with the decisions recorded under
"Decisions" at the end.
Tracks issue #14 (any source) and the v0.4 / v0.5 roadmap milestones.

Butterfly started as a GoPro tool. A holiday folder is usually a mix: GoPro
clips, phone videos (often portrait, often `.MOV`), and photos from the phone
and the GoPro. This document sets out how Butterfly should handle all of
that, and in what order.

## Principles

1. **Gate first, then score, for every source.** Each item gets a usable-ness
   `q` and an `interest`, from whatever evidence that source has. Telemetry
   is extra evidence when it exists, never a requirement.
2. **One selection pool.** Video windows and photos compete in the same
   ranked, diversity-aware selection, so the reel tells one story in time
   order. They aren't two separate reels stitched together.
3. **Video leads, photos punctuate.** A highlight reel is mostly motion.
   Photos fill gaps, mark moments with no video, and add rhythm. Their share
   is capped.
4. **Local only, never touch the originals.** This is unchanged. Photos are
   read; nothing is written next to them except under `.butterfly/`.

## Part 1: any video (v0.4)

### What breaks today

| Assumption | Effect on phone / other-camera video | Fix |
|---|---|---|
| Only `.mp4` is picked up | iPhone `.MOV` and `.m4v` are ignored | Accept `config.VIDEO_EXTS` (`.mp4 .mov .m4v`) |
| Frame size read without the rotation flag | Portrait video is squashed in the vision pass | Read the display rotation, swap width/height for ±90° |
| Drop and handling gates come only from GPMF | Shaky footage isn't penalised at all | Vision-based camera-shake gate from optical flow |
| Render pads everything to 16:9 with black | Portrait clips get wide black bars | Blurred-fill background (below) |
| Clock and timezone assumed consistent | Phone and GoPro items can sort out of order | Normalise to UTC (see Part 2, "Time") |

The first two are a small `fix/` PR and come first.

### Vision-based quality when there's no telemetry

Optical flow (the existing roadmap item, now moved into v0.4) on the 2 fps
analysis frames, upsampled to ~6 fps for flow only:

- **Global flow** (median or robust affine fit) is camera motion. Its
  high-frequency part (frame-to-frame jitter after low-pass) becomes
  `camera_shake`. That's the vision counterpart of the gyro high-pass, and it
  feeds the existing `handling` gate when there's no gyro. When both exist,
  use the gyro, and use vision only as the blur/smear cross-check it already
  is.
- **Residual flow** (flow minus the camera fit) is subject motion. It becomes
  a `subject_motion` interest part. This is "smooth action" for footage
  without GPS speed, and it rewards a moving subject over a panning camera.
- **Drops** without an accelerometer: a burst of extreme global flow plus a
  tumble in the horizon angle. This is weaker evidence than free-fall, so it
  gives a soft gate, not the hard exclusion window.

Folder-wide normalisation stays as it is, but **per source type**: GoPro
footage and phone footage have different shake and sharpness baselines.
Normalising a stabilised phone clip against GoPro p95 values would make it
look calm when it isn't.

### Portrait video in a landscape reel

The default output stays 16:9. Portrait (and other non-16:9) items are shown
over a **blurred, darkened copy of themselves** filling the frame. That's one
ffmpeg filter graph (`split → scale+crop+gblur → overlay`), shared with
photos. A future `--aspect 9:16` option could produce a vertical reel, where
landscape items get the same treatment in reverse.

### Done when

- A mixed folder (GoPro + iPhone video, at least one portrait and one shaky
  clip) gives a sensible plan, report and render.
- There are synthetic checks for a no-telemetry shaky clip (penalised), a
  rotated clip (correct aspect in analysis and render), and `.MOV` discovery.
- The labelled real-footage set (relabelled `good` / `unusable` /
  `dull`) runs on that mixed folder.

## Part 2: photos (v0.5)

### Discovery and decoding

- Extensions: `.jpg .jpeg .png .heic .heif`, via `config.PHOTO_EXTS`.
- Decode with **Pillow**, which also gives EXIF. Pillow becomes a core
  dependency again; it's already installed with matplotlib. HEIC support
  comes from **`pillow-heif`**, a pinned `photos` extra. Without it, HEIC
  files are skipped with one console line saying how to enable them.
- Apply EXIF orientation before anything else.
- GoPro stills (`GOPR*.JPG`) and phone photos are treated the same way.

### Time

Chronological order across devices is the hardest part.

- Videos: `creation_time` (UTC). With GPS, the HERO13 clock matched GPS UTC
  in the clips probed so far.
- Photos: EXIF `DateTimeOriginal` plus `OffsetTimeOriginal` gives UTC. If the
  offset is missing, assume the folder's majority offset, and say so in the
  report.
- **Photos without EXIF are common.** In the owner's example set, 4 of 6
  photos have no EXIF at all (exported or edited copies). Fallback order:
  1. EXIF time and offset;
  2. EXIF time plus the folder's majority offset;
  3. a timestamp in the filename (`IMG_20261010_123526`, `PXL_…`,
     `2026-10-10 12.35.26`);
  4. file modification time, flagged in the report as unreliable.

  An item with no trustworthy time is placed by visual similarity next to
  its closest neighbours in the plan, not by its modification time.
- Devices with a wrong clock: `--clock-offset DEVICE=±HH:MM` (device = EXIF
  `Model`, or "GoPro"), plus a report warning when one device's items cluster
  hours away from everything else.

### Analysis of a photo

A photo becomes an item with **`kind = "photo"`, duration 0, `n = 1`**,
rather than a new class, so the scoring code that works on arrays keeps
working on arrays of length 1.

- Downscale to the vision width, then run the same classic metrics
  (sharpness, luma, blown, colour) and CLIP (semantic, aesthetic, `clip_bad`,
  embedding).
- No telemetry and no audio, so `action` and `audio` are 0.
- **Gates:** `blur`, `dark`, `blown` and `clip_bad` apply as for video; the
  others don't. `edge`, `drop`, `handling` and `tumble` don't exist for
  stills.
- **Normalisation:** photos are normalised against **photos**, because stills
  are sharper and better composed than video frames, and pooling them would
  make every video frame look bad.
- Cached under `.butterfly/cache/` with the same key scheme.

### Bursts and duplicates

People take five near-identical shots. Cluster photos by **time gap ≤ 3 s
and embedding similarity ≥ 0.92** and keep the highest-scoring one per
cluster. Then **dedupe against video:** a photo within ±2 s of a picked video
moment with similarity ≥ 0.9 is dropped, since the video already shows it.
Both thresholds go in `config.py` and get tuned on real folders.

**Live Photos** (iPhone `IMG_1234.HEIC` + `IMG_1234.MOV`, about 3 s) are one
item: the photo. The short `.MOV` isn't treated as a video clip. A later
option could use its 1.5 s of motion in place of the Ken Burns move.

### Selection: one pool

- Each photo is one candidate. Its score is
  `interest × q × config.PHOTO_SCORE_FACTOR`, a factor that puts photos on
  the same scale as video windows. It's tuned so that a great photo beats a
  mediocre video window, but not a great one.
- Same MMR diversity and per-source cap as video, plus
  **`config.MAX_PHOTO_SHARE`** (default 0.25 of the cuts) so the reel stays
  mostly video.
- CLI: `--photos auto|off|only` (default `auto`: photos are used if there are
  any), and `--photo-seconds` (default 2.0). `only` makes a photo slideshow
  reel.

### How photos sit in the reel

- **Placement:** chronological, by UTC time, among the video cuts. Photos
  taken during a video's time span are placed next to that clip's cuts.
- **Duration:** without music, `--photo-seconds`. With music, a photo takes
  one cut slot like a video window, so it lands on the beat grid. Two or more
  consecutive photos can share one slot as a quick-cut montage on single
  beats, which is how phone memory reels make stills feel alive.
- **Motion:** a gentle **Ken Burns** move (zoom 1.00 → 1.08 over the
  duration) via ffmpeg `zoompan`, rendered at 2× output resolution and scaled
  down to avoid zoompan jitter. Direction alternates between consecutive
  photos. Later, once faces exist (v0.7), the zoom heads towards the face.
- **Framing:** a landscape photo fills the frame, with a centre crop of at
  most 10% if its aspect differs. Portrait photos use the same blurred-fill
  treatment as portrait video.
- **Audio:** **music is on by default and optional** (owner decision).
  - With music (the default): the music plays through photos.
  - With music turned off: photo segments get silence, with 0.15 s audio
    crossfades either side so the cut isn't abrupt.
- **Transitions:** hard cuts, like video. An optional short `xfade` into and
  out of photo runs can come later; it's an aesthetic choice for the owner.
- **Render pipeline:** a photo segment is rendered to the same intermediate
  format as video segments (same resolution, fps, pixel format, and a silent
  AAC track), so the existing concat and music mix work unchanged.

### Report

- Photo picks appear in the picks grid with a "photo" badge and their "why".
- A "Photos" section lists all photos as thumbnails with q and score, marks
  the ones collapsed as bursts or duplicates, and warns about clock offsets.

### Tests (synthetic)

- Photos are made with ffmpeg/Pillow: sharp, blurred, dark, a three-shot
  burst, and a portrait one with EXIF orientation.
- Checks:
  - a blurred or dark photo isn't picked;
  - a burst collapses to one photo;
  - photos are placed in time order among video picks;
  - `MAX_PHOTO_SHARE` is respected;
  - a rendered photo segment has the planned duration and output size;
  - EXIF orientation gives a portrait item.

## Phasing

| Step | Milestone | Content |
|---|---|---|
| 1 | v0.4 | `fix/`: `.MOV`/`.m4v` discovery, rotation-aware analysis |
| 2 | v0.4 | Optical flow: `camera_shake` gate, `subject_motion` interest, per-source normalisation |
| 3 | v0.4 | Blurred-fill render for non-16:9 items |
| 4 | v0.4 | Labelled set on a mixed GoPro + phone folder |
| 5 | v0.5 | Photo discovery, EXIF time and orientation, analysis, report section (no render) |
| 6 | v0.5 | Photos in selection and render: share cap, Ken Burns, blurred fill, beat slots |
| 7 | v0.5 | Bursts, video dedupe, clock offsets, Live Photo pairing |

Each step is one issue, one branch and one PR. Step 5 is useful on its own:
the report shows which photos Butterfly would use before any of them reach a
reel.

## Open questions for the owner

Answered on 2026-10-14; see "Decisions" below.

## Decisions (owner, 2026-10-14)

1. **Photo share:** `MAX_PHOTO_SHARE = 0.25` of the cuts.
2. **Motion:** a gentle Ken Burns zoom on every photo by default.
3. **Music:** on by default, and optional. Reels get music unless the user
   turns it off; photos sit under the music. The source is decided (see
   "Music source" below): the user's own song or set if they name one,
   otherwise bundled free tracks chosen by vibe.
4. **HEIC:** `pillow-heif` as a pinned optional `photos` extra.
5. **Vertical reels:** later, not in v0.5.
6. **Test material:** the owner's labelled examples are in
   `example_photos/` in the repo. That folder is excluded via
   `.git/info/exclude` and never committed. It holds six JPEGs:
   `good_landscape`, `good_portrait`, iPhone `good_landscape` and
   `good_portrait`, `landscape_out_of_focus` and `landscape_shaky`. Only the
   two iPhone shots have EXIF. The labels in the filenames are the expected
   outcome: the two bad shots must be gated (blur), and the four good ones
   usable. More material (phone video, a burst, a Live Photo) is still
   wanted.

### Music source (decided 2026-10-14; low priority)

A mix of options (a) and (b):

- **A specific song or set of songs:** the user names it, and Butterfly uses
  it. `--music FILE` works as now; `--music DIR` (new) treats a folder as a
  set and picks the track whose length and tempo best fit the reel.
- **Just a vibe:** Butterfly uses a **bundled library of free tracks**, chosen
  with `--vibe NAME` (e.g. `chill`, `upbeat`), or a default vibe when nothing
  is given.
  - The tracks are **not committed**. A tracked script downloads them, with a
    tracked manifest listing URL, licence, checksum and vibe tags, into a
    local, git-ignored folder. Only the script and manifest live in git.
  - Every track must have a licence that allows use in personal videos. The
    manifest records it, and the script verifies checksums.
  - Downloading is the only network access, it happens on request (like
    model weights on first use), and nothing is ever uploaded.
- **Turning music off:** `--no-music` gives no music; photos then get
  silence with short crossfades (see above).
- **No tracks downloaded yet:** the reel is made without music, with one
  console line saying how to fetch the library.

Which tracks to bundle, and the vibe names, are to be explored later.
