from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from roadlens.domain import FrameSegmentations
from roadlens.telemetry import TelemetrySample


@dataclass(frozen=True)
class Observation:
    object_id: int
    prompt: str
    backend: str
    first_source_seconds: float
    last_source_seconds: float
    best_source_seconds: float
    best_frame_index: int
    recorded_utc: datetime
    score: float | None
    foreground_pixels: int
    latitude: float | None
    longitude: float | None
    location_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "prompt": self.prompt,
            "backend": self.backend,
            "model": "sam-3.1",
            "review_status": "unreviewed",
            "first_source_seconds": self.first_source_seconds,
            "last_source_seconds": self.last_source_seconds,
            "best_source_seconds": self.best_source_seconds,
            "best_frame_index": self.best_frame_index,
            "recorded_utc": self.recorded_utc.isoformat(),
            "score": self.score,
            "foreground_pixels": self.foreground_pixels,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_source": self.location_source,
        }


def build_observations(
    frames: list[FrameSegmentations],
    telemetry: list[TelemetrySample] | tuple[TelemetrySample, ...],
    clip_created_utc: datetime,
    clip_start_seconds: float,
    clip_fps: float,
    location_source: str = "garmin_camera_gps",
) -> list[Observation]:
    if clip_fps <= 0:
        raise ValueError("clip_fps must be positive")
    tracks: dict[tuple[str, int], list[tuple[FrameSegmentations, Any]]] = {}
    for frame in frames:
        for detection in frame.detections:
            key = (frame.backend, detection.object_id)
            tracks.setdefault(key, []).append((frame, detection))

    observations = []
    for (backend, object_id), items in tracks.items():
        best_frame, best = max(
            items,
            key=lambda item: (
                item[1].score if item[1].score is not None else 0,
                int(item[1].mask.sum()),
            ),
        )
        best_source_seconds = (
            clip_start_seconds + best_frame.frame_index / clip_fps
        )
        recorded_utc = clip_created_utc + timedelta(seconds=best_source_seconds)
        nearest = nearest_telemetry(telemetry, recorded_utc)
        observations.append(
            Observation(
                object_id=object_id,
                prompt=best_frame.prompt,
                backend=backend,
                first_source_seconds=clip_start_seconds
                + min(frame.frame_index for frame, _ in items) / clip_fps,
                last_source_seconds=clip_start_seconds
                + max(frame.frame_index for frame, _ in items) / clip_fps,
                best_source_seconds=best_source_seconds,
                best_frame_index=best_frame.frame_index,
                recorded_utc=recorded_utc,
                score=best.score,
                foreground_pixels=int(best.mask.sum()),
                latitude=None if nearest is None else nearest.latitude,
                longitude=None if nearest is None else nearest.longitude,
                location_source=(
                    location_source if nearest is not None else "unavailable"
                ),
            )
        )
    return sorted(observations, key=lambda item: item.best_source_seconds)


def nearest_telemetry(
    samples: list[TelemetrySample] | tuple[TelemetrySample, ...],
    timestamp: datetime,
    tolerance_seconds: float = 2.0,
) -> TelemetrySample | None:
    candidates = [
        sample
        for sample in samples
        if abs((sample.recorded_utc - timestamp).total_seconds()) <= tolerance_seconds
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda sample: abs((sample.recorded_utc - timestamp).total_seconds()),
    )


def write_observations(
    observations: list[Observation], source_video: Path, output_directory: Path
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    serialised = [observation.to_dict() for observation in observations]
    (output_directory / "observations.json").write_text(
        json.dumps(serialised, indent=2) + "\n", encoding="utf-8"
    )
    features = []
    for observation in observations:
        geometry = None
        if observation.latitude is not None and observation.longitude is not None:
            geometry = {
                "type": "Point",
                "coordinates": [observation.longitude, observation.latitude],
            }
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": observation.to_dict(),
            }
        )
    collection = {
        "type": "FeatureCollection",
        "metadata": {
            "source_video": source_video.name,
            "generated_utc": datetime.now(UTC).isoformat(),
            "location_note": (
                "Points are camera GPS positions, not measured hazard coordinates."
            ),
        },
        "features": features,
    }
    (output_directory / "observations.geojson").write_text(
        json.dumps(collection, indent=2) + "\n", encoding="utf-8"
    )
