"""Faces and people, on-device, via Apple's Vision framework (macOS; optional `people` extra).

Per frame: the largest face's share of the frame (`face_area`, 0..1) and how many people are in
view (`n_people`: the larger of faces and human bodies). Without Vision (Linux, or the extra
not installed), Butterfly skips these features.
"""

from __future__ import annotations

import io
import sys

import numpy as np


class PeopleDetector:
    def __init__(self):
        import Vision  # noqa: F401  (pyobjc-framework-Vision)
        from Foundation import NSData  # noqa: F401

        self.Vision, self.NSData = Vision, NSData
        print("  faces and people: Apple Vision", file=sys.stderr)

    def _one(self, rgb: np.ndarray) -> tuple[float, int]:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="PNG")
        b = buf.getvalue()
        V = self.Vision
        handler = V.VNImageRequestHandler.alloc().initWithData_options_(self.NSData.dataWithBytes_length_(b, len(b)), None)
        faces, humans = V.VNDetectFaceRectanglesRequest.alloc().init(), V.VNDetectHumanRectanglesRequest.alloc().init()
        ok, _ = handler.performRequests_error_([faces, humans], None)
        if not ok:
            return 0.0, 0
        areas = [r.boundingBox().size.width * r.boundingBox().size.height for r in (faces.results() or [])]
        return float(max(areas, default=0.0)), max(len(areas), len(humans.results() or []))

    def detect(self, frames: np.ndarray) -> dict:
        """frames uint8 [B, H, W, 3] -> {'face_area': [B], 'n_people': [B]}."""
        res = [self._one(f) for f in frames]
        return {"face_area": np.array([r[0] for r in res], float), "n_people": np.array([r[1] for r in res], float)}


def load_people(enabled: bool):
    """-> (PeopleDetector or None, cache-tag suffix). None off macOS or without the extra."""
    if not enabled or sys.platform != "darwin":
        return None, ""
    try:
        return PeopleDetector(), "+vn"
    except ImportError:
        return None, ""
