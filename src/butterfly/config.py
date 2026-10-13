"""All tunable settings live here."""

DT = 0.1                 # analysis timeline resolution (s)
CACHE_VERSION = 3        # bump to invalidate cached analyses
OUTPUT_DIR = ".butterfly"  # Butterfly's folder inside the footage folder: cache/ and report/

# ---------------------------------------------------------------- vision
VISION_FPS = 2.0         # frames per second sampled for vision
VISION_WIDTH = 320       # decode width for vision frames
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

# ---------------------------------------------------------------- interest weights
INTEREST_WEIGHTS = {
    "semantic": 1.0,     # CLIP similarity to POSITIVE_PROMPTS
    "aesthetic": 0.6,    # CLIP beautiful-vs-boring
    "action": 0.8,       # smooth speed + turns (never shake)
    "audio": 0.4,        # loudness peaks above rolling baseline
    "colour": 0.2,       # colourfulness
    "hilight": 2.5,      # in-camera HiLight tag bump (absolute, not normalised)
    "water": 0.8,        # entering water (wet mic transition), absolute bump
}
HILIGHT_LEAD = 1.0       # tags come just after the moment; centre bump this much earlier
HILIGHT_WIDTH = 1.5      # s, gaussian sigma

# ---------------------------------------------------------------- quality gates
GATES = {
    "drop_accel": 3.0,        # m/s², |accel| below this = free-fall
    "drop_min_s": 0.08,       # minimum free-fall duration
    "drop_before": 1.0,       # exclude this long before the drop
    "drop_after": 2.5,        # ...and after (tumble, pick-up)
    "shake_penalty": 0.85,    # max reduction from handling shake (0-1)
    "shake_ride_speed": 3.0,  # m/s; above this, shake is treated as a bumpy ride (softer)
    "shake_ride_factor": 0.4, # how much of the shake penalty still applies while riding
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

# ---------------------------------------------------------------- selection
DIVERSITY = 0.3          # MMR weight on novelty (0 = pure score, 1 = pure variety)
WINDOW_STRIDE = 0.5      # s between candidate window starts
CANDIDATES_PER_CLIP = 8
