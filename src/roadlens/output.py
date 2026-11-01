import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from roadlens.domain import FrameSegmentations
from roadlens.media import extract_frame

_COLORS = ((239, 68, 68), (34, 197, 94), (59, 130, 246), (234, 179, 8))


def write_frame_result(
    video_path: Path,
    result: FrameSegmentations,
    output_directory: Path,
    *,
    source_frame: Path | None = None,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    frame_path = output_directory / f"frame-{result.frame_index:06d}.png"
    overlay_path = output_directory / f"overlay-{result.frame_index:06d}.png"
    result_path = output_directory / f"result-{result.frame_index:06d}.json"
    if source_frame is not None:
        shutil.copyfile(source_frame, frame_path)
    elif video_path.is_dir():
        shutil.copyfile(video_path / f"{result.frame_index:06d}.png", frame_path)
    else:
        extract_frame(video_path, result.frame_index, frame_path)

    with Image.open(frame_path) as source:
        image = source.convert("RGBA")
        pixels = np.asarray(image).copy()
        for detection in result.detections:
            color = _COLORS[detection.object_id % len(_COLORS)]
            mask = detection.mask
            if mask.shape != pixels.shape[:2]:
                raise ValueError("Mask and source frame dimensions do not match")
            pixels[mask, :3] = (
                0.45 * np.asarray(color) + 0.55 * pixels[mask, :3]
            ).astype(np.uint8)
        Image.fromarray(pixels, "RGBA").convert("RGB").save(overlay_path)

    result_path.write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
