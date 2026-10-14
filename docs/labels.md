# Labels file

Hand labels say what parts of a folder *should* happen. `butterfly-eval`
checks Butterfly against them, and the report's thumbs up/down (v0.9) will
write to the same file.

Default location: `<folder>/.butterfly/labels.json`, inside Butterfly's own
output folder, next to the cache. It describes your footage, so it isn't part
of the repository.

```json
{
  "version": 1,
  "labels": [
    {"clip": "GX010063.MP4", "start": 34.0, "end": 43.0, "label": "unusable", "note": "camera covered"},
    {"clip": "GX010070.MP4", "start": 0.0, "end": 12.0, "label": "good"},
    {"clip": "IMG_4411.MOV", "label": "dull"}
  ]
}
```

- `clip`: the file name in the folder.
- `start` / `end`: seconds on the clip's timeline. Leave both out to label
  the whole clip.
- `label`:
  - `good`: worth showing. It should be usable (not gated) and is a candidate
    for the reel.
  - `unusable`: must never be picked (drops, handling, covered lens, black,
    murk). It should be gated.
  - `dull`: usable but not worth showing; it should rarely be picked.
- `note`: optional free text, shown in violations.

## Running

```bash
butterfly-eval ~/Movies/GoPro                 # same analysis and settings as `butterfly --dry-run`
butterfly-eval ~/Movies/GoPro --strict        # exit 1 on violations (for before/after checks)
```

It prints, per label, the number of items and seconds, mean quality `q`, the
share of time below `window_min_q` (gated), and the share of time that made
the reel (picked). Then it lists **violations**: an `unusable` range that was
picked, or a `good` range that is more than half gated.

Use it before and after any change that affects picks. A good change keeps
`unusable` at 0% picked and `good` mostly ungated.
