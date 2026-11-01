"""Decode original presentation timestamps and retain selected source frames."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from roadlens.sampling import FrameChoice


def decode_frames(source: Path, start: float, end: float) -> Iterator:
    import av

    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        origin = stream.start_time or 0
        container.seek(int(start / stream.time_base) + origin, stream=stream)
        previous = None
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Source frame has no presentation timestamp")
            seconds = float((frame.pts - origin) * stream.time_base)
            if seconds < start - 1e-6:
                continue
            if seconds >= end:
                break
            if previous is not None and seconds <= previous:
                raise ValueError("Source presentation timestamps are not increasing")
            previous = seconds
            yield seconds, frame


def measure_frames(source: Path, start: float, end: float):
    import cv2

    times, sharpness = [], []
    for seconds, frame in decode_frames(source, start, end):
        small = frame.reformat(
            width=512, height=round(frame.height * 512 / frame.width), format="gray"
        ).to_ndarray()
        # Exclude sky and the Garmin overlay; this is a road-region blur proxy.
        road = small[int(len(small) * 0.45) : int(len(small) * 0.90), 102:410]
        score = float(cv2.Laplacian(road, cv2.CV_64F).var())
        times.append(seconds)
        sharpness.append(score)
    if not times:
        raise ValueError("Requested interval contains no source frames")
    return times, sharpness


def save_selected_frames(
    source: Path,
    choices: list[FrameChoice],
    output: Path,
    width: int = 0,
) -> list[Path]:
    """One sequential decode, PNG evidence without an intermediate lossy video."""
    import cv2

    if not choices:
        return []
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    index = 0
    pending = deque()

    def write(path, pixels):
        if not cv2.imwrite(str(path), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(f"Could not write {path}")

    # Bound queued native images: unbounded futures can exhaust host memory.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for seconds, frame in decode_frames(
            source,
            choices[0].source_seconds,
            choices[-1].source_seconds + 0.1,
        ):
            wanted = choices[index].source_seconds
            if seconds < wanted - 1e-6:
                continue
            if abs(seconds - wanted) > 1e-6:
                raise ValueError(
                    f"Could not recover source PTS {wanted}; got {seconds}"
                )
            if width and width < frame.width:
                height = round(frame.height * width / frame.width)
                frame = frame.reformat(width=width, height=height)
            pixels = frame.to_ndarray(format="bgr24")
            path = output / f"{index:06d}.png"
            pending.append(pool.submit(write, path, pixels))
            if len(pending) >= 8:
                pending.popleft().result()
            paths.append(path)
            index += 1
            if index == len(choices):
                break
        for future in pending:
            future.result()
    if index != len(choices):
        raise ValueError("Video ended before all selected timestamps were recovered")
    return paths


def coverage(choices: list[FrameChoice], native_times: list[float]) -> dict:
    gaps = np.diff([c.distance_m for c in choices])
    return {
        "native_frames": len(native_times),
        "selected_frames": len(choices),
        "max_selected_time_gap_seconds": float(
            max(
                np.diff([c.source_seconds for c in choices]),
                default=0,
            )
        ),
        "integrated_known_distance_m": choices[-1].distance_m,
        "max_known_distance_gap_m": float(max(gaps, default=0)),
        "selected_frames_without_speed": sum(c.speed_mps is None for c in choices),
        "note": "Integrated speed, not surveyed distance; excludes unknown speed.",
    }
