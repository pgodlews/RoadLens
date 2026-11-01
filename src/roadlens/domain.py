from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Detection:
    object_id: int
    box_xywh: tuple[float, float, float, float]
    score: float | None
    mask: NDArray[np.bool_]

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "box_xywh_normalized": [round(value, 6) for value in self.box_xywh],
            "score": None if self.score is None else round(self.score, 6),
            "foreground_pixels": int(self.mask.sum()),
            "mask_rle": encode_mask(self.mask),
        }


@dataclass(frozen=True)
class FrameSegmentations:
    frame_index: int
    prompt: str
    backend: str
    detections: tuple[Detection, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "prompt": self.prompt,
            "backend": self.backend,
            "model": "sam-3.1",
            "detections": [detection.to_dict() for detection in self.detections],
        }


def encode_mask(mask: NDArray[np.bool_]) -> dict[str, Any]:
    if mask.ndim != 2:
        raise ValueError("Masks must be two-dimensional")
    height, width = mask.shape
    values = mask.astype(bool).ravel()
    if values.size == 0:
        return {"width": width, "height": height, "counts": []}

    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [values.size]))
    counts = np.diff(boundaries)
    if values[0]:
        counts = np.concatenate(([0], counts))
    return {
        "width": int(width),
        "height": int(height),
        "counts": counts.tolist(),
    }


def decode_mask(encoded: dict[str, Any]) -> NDArray[np.bool_]:
    width = int(encoded["width"])
    height = int(encoded["height"])
    counts = [int(count) for count in encoded["counts"]]
    if sum(counts) != width * height:
        raise ValueError("RLE counts do not match the mask dimensions")
    values = np.zeros(width * height, dtype=bool)
    offset = 0
    foreground = False
    for count in counts:
        if foreground:
            values[offset : offset + count] = True
        offset += count
        foreground = not foreground
    return values.reshape(height, width)
