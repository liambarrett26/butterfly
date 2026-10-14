"""Audio event tagging: laughter, cheering, splash, wind and speech, on the DT grid.

Uses an AudioSet classifier (AST, via Hugging Face transformers) on-device. It's optional
(the `audio` extra): without it, Butterfly skips these features. Weights download from
Hugging Face on first use, like CLIP's, and audio never leaves the machine.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from . import config as C

SR = 16000  # AST's sample rate


class AudioTagger:
    def __init__(self, model_name: str = C.AUDIO_TAG_MODEL):
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.fe = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTForAudioClassification.from_pretrained(model_name).to(self.device).eval()
        ids = {v: int(k) for k, v in self.model.config.id2label.items()}
        self.groups = {g: [ids[c] for c in classes if c in ids] for g, classes in C.AUDIO_EVENTS.items()}
        print(f"  audio events: {model_name.split('/')[-1]} on {self.device}", file=sys.stderr)

    def probabilities(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """Mono float32 audio at SR -> (window centre times, {group: max class probability per window})."""
        win, hop = int(C.AUDIO_TAG_WIN_S * SR), int(C.AUDIO_TAG_HOP_S * SR)
        starts = list(range(0, max(len(x) - win, 0) + 1, hop))
        probs = []
        for i in range(0, len(starts), 16):
            batch = [x[s:s + win] for s in starts[i:i + 16]]
            inp = self.fe(batch, sampling_rate=SR, return_tensors="pt")
            with self.torch.no_grad():
                logits = self.model(**{k: v.to(self.device) for k, v in inp.items()}).logits
            probs.append(self.torch.sigmoid(logits).cpu().numpy())
        p = np.concatenate(probs) if probs else np.zeros((0, 1))
        t = (np.array(starts) + win / 2) / SR
        return t, {g: (p[:, ids].max(axis=1) if len(p) and ids else np.zeros(len(t))) for g, ids in self.groups.items()}

    def tag(self, path: Path, n: int) -> dict:
        """-> {group: probability array on the DT grid} for the clip's audio ({} if it can't be read)."""
        try:
            pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR),
                                  "-f", "f32le", "-"], capture_output=True, check=True).stdout
        except subprocess.CalledProcessError:
            return {}
        x = np.frombuffer(pcm, np.float32)
        if len(x) < SR // 2:
            return {}
        t, groups = self.probabilities(x)
        grid = np.arange(n) * C.DT
        return {f"ev_{g}": np.interp(grid, t, v) for g, v in groups.items()}


def load_tagger(enabled: bool):
    """-> (AudioTagger or None, cache-tag suffix). Quietly None when the audio extra isn't installed."""
    if not enabled:
        return None, ""
    try:
        return AudioTagger(), "+ast"
    except ImportError:
        return None, ""
    except Exception as e:  # weights unavailable offline, etc.
        print(f"  audio events unavailable ({type(e).__name__}: {e})")
        return None, ""
