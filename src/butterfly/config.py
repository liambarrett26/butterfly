"""All tunable settings live here."""

DT = 0.1                 # analysis timeline resolution (s)
CACHE_VERSION = 10        # bump to invalidate cached analyses
OUTPUT_DIR = ".butterfly"  # Butterfly's folder inside the footage folder: cache/ and report/
PHOTO_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif")  # photos picked up; HEIC needs the photos extra
VIDEO_EXTS = (".mp4", ".mov", ".m4v")  # video files picked up (case-insensitive); GoPro .LRV are proxies, not clips

# ---------------------------------------------------------------- vision
VISION_FPS = 2.0         # frames per second sampled for vision
VISION_WIDTH = 320       # decode width for vision frames
FLOW_FPS = 6.0           # frames per second for optical flow; a multiple of VISION_FPS (one decode serves both)
FLOW_WIDTH = 160         # optical-flow frame width (px); motion is reported in frame-widths per second
FLOW_PAN_S = 1.0         # s; camera motion smoother than this is a deliberate pan or ride, not shake
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_BATCH = 64

# What a good moment looks like. Edit these for your trips — they matter most.
POSITIVE_PROMPTS = [
    "a person jumping into the water",
    "snorkeling underwater with colorful fish",
    "a turtle swimming underwater",
    "surfing a wave",
    "a stunning view of turquoise sea",
    "a beautiful beach with palm trees",
    "a cenote with clear blue water",
    "a waterfall in a lush jungle",
    "riding a bike or scooter along a scenic road",
    "skiing or snowboarding down a snowy mountain",
    "a breathtaking mountain landscape",
    "a sunset over the ocean",
    "friends laughing and having fun together",
    "a close-up selfie of smiling people wearing sunglasses",
    "a smiling selfie of friends on holiday",
    "people floating together in clear turquoise water",
    "a group of people celebrating",
    "an exciting action sports moment",
    "wildlife in nature",
    "ancient ruins in the jungle",
]

# What unusable footage looks like (feeds the quality gate, not interest).
NEGATIVE_PROMPTS = [
    "a blurry photo",
    "a motion-blurred shaky image",
    "a close-up photo of the ground",
    "a photo of the floor",
    "the inside of a pocket or bag",
    "a dark black image",
    "a finger or hand covering the camera lens",
    "water droplets on the camera lens",
    "a plain overexposed white image",
    "a murky underwater view with nothing in it",
]

AESTHETIC_POS = "a beautiful, well composed award-winning travel photograph"
AESTHETIC_NEG = "a boring, badly framed photo of nothing in particular"

# ---------------------------------------------------------------- audio events (optional `audio` extra)
AUDIO_TAG_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"   # AudioSet classifier, on-device
AUDIO_TAG_WIN_S = 2.0    # s of audio per classification…
AUDIO_TAG_HOP_S = 1.0    # …every this many seconds
AUDIO_EVENTS = {         # feature ev_<group> = max probability over these AudioSet classes
    "laughter": ["Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"],
    "cheering": ["Cheering", "Applause", "Whoop"],
    "splash": ["Splash, splatter"],
    "wind": ["Wind", "Wind noise (microphone)"],
    "speech": ["Speech"],
}
EVENT_LO, EVENT_HI = 0.1, 0.5
FACE_LO, FACE_HI = 0.002, 0.03  # largest face as a share of the frame: a selfie face is ~0.05   # event probability where interest starts / is full (absolute: events are rare)

# ---------------------------------------------------------------- interest weights
INTEREST_WEIGHTS = {
    "semantic": 1.0,     # CLIP similarity to POSITIVE_PROMPTS
    "aesthetic": 0.6,    # CLIP beautiful-vs-boring
    "action": 0.8,       # smooth speed + turns (never shake)
    "audio": 0.4,        # loudness peaks above rolling baseline
    "colour": 0.2,       # colourfulness
    "laughter": 0.8,     # audio events (AudioSet probability ramped EVENT_LO..EVENT_HI), absolute bumps
    "cheering": 0.8,
    "splash": 0.6,
    "speech": 0.0,       # measured, off: see the v0.7 PR for the evaluation
    "faces": 0.8,        # largest face's area, ramped FACE_LO..FACE_HI (Apple Vision, optional)
    "people": 0.3,       # people in view (faces or bodies), 1 person = 1/3 … 3+ = full
    # Subject motion (optical flow left after removing camera motion) is computed and cached, but
    # off: on real footage it mostly measures parallax, water and things fixed to the camera (a
    # boat's bow, legs), so it rewarded a boat ride over people floating. Revisit with v0.7 people detection.
    "subject": 0.0,
    "water": 0.8,        # entering water (wet mic transition), absolute bump
}

# ---------------------------------------------------------------- learned ranker (butterfly-train)
RANKER_WINDOW_S = 2.5    # labelled video ranges are cut into windows this long…
RANKER_STRIDE_S = 1.0    # …every this many seconds
RANKER_L2 = 1.0          # ridge penalty on the standardised logistic-regression coefficients
RANKER_PRIOR = 20        # labels' pull: alpha = items / (items + this) blends learned into hand weights

# ---------------------------------------------------------------- quality gates
GATES = {
    "drop_accel": 3.0,        # m/s², |accel| below this = free-fall
    "drop_min_s": 0.08,       # minimum free-fall duration
    "drop_before": 1.0,       # exclude this long before the drop
    "drop_after": 2.5,        # ...and after (tumble, pick-up)
    "shake_penalty": 0.85,    # max reduction from handling shake (0-1)
    "shake_ride_speed": 3.0,  # m/s; above this, shake is treated as a bumpy ride (softer)
    "shake_ride_factor": 0.4, # how much of the shake penalty still applies while riding
    "wind_lo": 0.3,           # wind-noise probability where the (mild) wind gate starts…
    "wind_hi": 0.7,           # …and is full
    "wind_penalty": 0.3,      # max reduction: wind spoils the sound, not the picture
    "cam_shake_lo": 0.06,     # frame-widths/s of jittery camera motion in the image (optical flow)…
    "cam_shake_hi": 0.20,     # …where the vision shake penalty starts / is full. Used only without a gyro.
                              # Stabilised GoPro footage measured p95 0.05–0.10.
    "tumble_deg": 65,         # orientation deviation from the clip's usual attitude
    "tumble_penalty": 0.5,
    "blur_pct": 15,           # sharpness below this folder percentile counts as blurry
    "blur_penalty": 0.5,
    "dark_luma": 40,          # mean luma (0-255); penalty ramps in over ±10 around this
    "dark_penalty": 0.15,
    "clipped_frac": 0.4,      # fraction of blown-out pixels above = overexposed
    "clipped_penalty": 0.5,
    "clip_bad_start": 0.45,   # CLIP negative-prompt prob where penalty starts...
    "clip_bad_full": 0.85,    # ...and where it hits the floor below
    "clip_bad_floor": 0.2,
    "edge_s": 0.8,            # first/last seconds of each clip (button presses)
    "edge_penalty": 0.3,
    "window_min_q": 0.3,      # reject a candidate cut if any moment is below this
}

# ---------------------------------------------------------------- photos
# Photos are scored like one video frame, but against other photos only: stills are sharper
# and better composed than video frames.
PHOTO = {
    "width": 640,        # px; analysis width. Blur separates far better here than at VISION_WIDTH
                         # (owner's examples: bad/good sharpness ratio ~7x at 640 px, ~2x at 320 px)
    "blur_lo": 0.2,      # sharpness / median photo sharpness: at or below this, the blur gate is at its floor…
    "blur_hi": 0.5,      # …at or above this, no blur penalty
    "blur_floor": 0.1,   # blur gate for a clearly blurred photo (below window_min_q, so never used)
}
MAX_PHOTO_SHARE = 0.25   # at most this share of the reel's cuts are photos (owner decision); video leads
PHOTO_SCORE_FACTOR = 0.8 # photo score × this competes with video window scores (photos are normalised separately)
BURST_GAP_S = 3.0        # photos at most this far apart (s)…
BURST_SIM = 0.9          # …whose 32×24 grey signatures correlate at least this much are one burst: keep the best
DEDUPE_S = 2.0           # a photo within this many seconds of a picked video window…
DEDUPE_SIM = 0.9         # …and this CLIP-similar to it shows the same moment: the video wins (needs CLIP)
LIVE_PHOTO_MAX_S = 4.0   # a .MOV this short with the same name as a photo is a Live Photo's motion, not a clip
PHOTO_ZOOM = 0.08        # Ken Burns: zoom 1.00 → 1 + this over a photo's slot (direction alternates)

# ---------------------------------------------------------------- slow motion
SLOWMO_MIN_FPS = 100     # clips shot at this frame rate or faster can play slowed down…
SLOWMO_SPEED = 0.5       # …at this speed (0.5 = half; atempo needs >= 0.5)…
SLOWMO_MAX_CUTS = 2      # …for at most this many of the reel's best cuts

TITLE_DIM = 0.45         # day title cards: brightness of the blurred background frame

# ---------------------------------------------------------------- render
FILL_ASPECT_TOL = 0.02   # items whose display aspect is off the reel's (16:9) by more than this fraction…
FILL_BLUR = 4.0          # …sit on a blurred copy of themselves: gblur sigma (px) at 1/8 output resolution
FILL_DIM = 0.7           # brightness multiplier for that blurred background, so the item stands out

# ---------------------------------------------------------------- selection
DIVERSITY = 0.3          # MMR weight on novelty (0 = pure score, 1 = pure variety)
WINDOW_STRIDE = 0.5      # s between candidate window starts
CANDIDATES_PER_CLIP = 8
INOUT_SHIFT_S = 0.5      # smart in/out: a cut may move up to this far (s) once its length is final…
INOUT_Q_STEP = 0.05      # …first to raise its 5th-percentile q (compared in steps of this size)…
ONSET_WINDOW_S = 0.5     # …then to start where activity (frame change, else loudness) rises over this window…
ONSET_STEP = 0.25        # …compared in steps of this (clip-relative) size; ties keep the smallest shift
SHOT_WINDOW_S = 2.0      # shots: novelty compares the mean CLIP embedding of the previous and next this-many s…
SHOT_NOVELTY = 0.12      # …a peak at or above this (1 − cosine) is a shot boundary (calm footage: ~0.02–0.07)…
SHOT_MIN_S = 3.0         # …and shots are at least this long. One pick per shot (relaxed only to fill the reel)
PICKS_PER_CLIP_SLACK = 1  # cuts a clip may take beyond its even share, ceil(cuts / clips)

# ---------------------------------------------------------------- trip presets (--preset)
# Each adds CLIP prompts and overrides weights / gates for one kind of trip. Applied once at start-up
# (apply_preset), before CLIP encodes the prompts; the cache tag includes the preset.
PRESETS = {
    "beach": {
        "prompts": ["a sunny beach with turquoise water", "people swimming in the sea",
                    "snorkelling in clear water over a reef", "a sunset on the beach"],
        "negative": ["sand or salt water smeared on the camera lens"],
        "weights": {"aesthetic": 0.8, "faces": 0.9},
        "gates": {"clipped_frac": 0.5},                    # bright sand and sea glare aren't a fault
    },
    "snow": {
        "prompts": ["skiing down a snowy slope", "snowboarding through fresh powder",
                    "a snowy mountain panorama", "friends in ski gear on a mountain"],
        "negative": ["snow or fog covering the camera lens"],
        "weights": {"action": 1.2},
        "gates": {"clipped_frac": 0.65, "shake_ride_speed": 5.0},   # white snow; skiing is fast
    },
    "city": {
        "prompts": ["a busy city street", "historic architecture in an old town",
                    "a lively street market", "a city skyline at night"],
        "negative": ["the inside of a pocket while walking"],
        "weights": {"action": 0.3, "faces": 1.0},       # walking pace isn't action; people matter
        "gates": {"dark_luma": 28},                      # evening streets are darker, still usable
    },
    "bike": {
        "prompts": ["riding a bike along a scenic road", "mountain biking on a forest trail",
                    "a cyclist on a winding mountain road", "riding a scooter through the countryside"],
        "negative": ["handlebars filling the whole frame"],
        "weights": {"action": 1.2},
        "gates": {"shake_ride_speed": 2.5, "shake_ride_factor": 0.3},  # bumpy riding is still riding
    },
}


def apply_preset(name: str) -> None:
    """Merge PRESETS[name] into the prompts, weights and gates (call once, before CLIP loads)."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}: choose from {', '.join(PRESETS)}")
    p = PRESETS[name]
    POSITIVE_PROMPTS.extend(x for x in p.get("prompts", []) if x not in POSITIVE_PROMPTS)
    NEGATIVE_PROMPTS.extend(x for x in p.get("negative", []) if x not in NEGATIVE_PROMPTS)
    INTEREST_WEIGHTS.update(p.get("weights", {}))
    GATES.update(p.get("gates", {}))
