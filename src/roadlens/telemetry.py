"""Garmin timed-metadata grouping adapted from Ride Studio's telemetry readers."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class TelemetrySample:
    recorded_utc: datetime
    latitude: float
    longitude: float
    speed_mps: float | None
    playback_seconds: float | None


@dataclass(frozen=True)
class TelemetryResult:
    samples: tuple[TelemetrySample, ...]
    rejected_samples: int
    # Camera metadata source; "tesla_sei_unusable" means all fixes were rejected.
    source: str = "garmin"

    @property
    def location_source(self) -> str:
        """Label for observation positions: the camera's GPS, not the defect's."""
        return {"garmin": "garmin_camera_gps", "tesla_sei": "tesla_camera_gps"}.get(
            self.source, "camera_gps"
        )


def read_camera_telemetry(video_path: Path) -> TelemetryResult:
    """Timed GPS from a Garmin Varia clip, or else from Tesla dashcam SEI.

    Garmin clips carry ExifTool-readable timed GPS; Tesla clips carry per-frame
    SEI messages. A clip with neither (for example a Tesla clip recorded while
    parked) returns no samples, and positions are reported as unavailable.
    """
    try:
        garmin = read_garmin_telemetry(video_path)
    except (OSError, ValueError, subprocess.CalledProcessError):
        garmin = TelemetryResult((), 0)
    if garmin.samples:
        return garmin
    from roadlens.tesla import read_tesla_telemetry

    try:
        tesla = read_tesla_telemetry(video_path)
    except (OSError, ValueError, subprocess.CalledProcessError):
        tesla = TelemetryResult((), 0)
    if tesla.samples:
        return tesla
    if tesla.source == "tesla_sei" and tesla.rejected_samples:
        return TelemetryResult((), tesla.rejected_samples, source="tesla_sei_unusable")
    return TelemetryResult((), garmin.rejected_samples, source="none")


def read_garmin_telemetry(video_path: Path) -> TelemetryResult:
    completed = subprocess.run(
        [
            "exiftool",
            "-ee3",
            "-G3",
            "-a",
            "-s",
            "-n",
            "-j",
            "-GPS*",
            "-SampleTime",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(completed.stdout)
    if not data or not isinstance(data[0], dict):
        raise ValueError("ExifTool returned no Garmin metadata")
    return decode_garmin_metadata(data[0])


def decode_garmin_metadata(data: dict[str, str | float]) -> TelemetryResult:
    documents: dict[int, dict[str, str | float]] = {}
    for key, value in data.items():
        group, separator, tag = key.partition(":")
        match = re.fullmatch(r"Doc(\d+)", group)
        if separator and match:
            documents.setdefault(int(match.group(1)), {})[tag] = value

    samples = []
    rejected = 0
    for _, tags in sorted(documents.items()):
        timestamp = _timestamp(tags.get("GPSDateTime"))
        latitude = _finite(tags.get("GPSLatitude"))
        longitude = _finite(tags.get("GPSLongitude"))
        if (
            timestamp is None
            or latitude is None
            or longitude is None
            or not -90 <= latitude <= 90
            or not -180 <= longitude <= 180
            or (latitude == -180 and longitude == -180)
        ):
            rejected += 1
            continue
        samples.append(
            TelemetrySample(
                recorded_utc=timestamp,
                latitude=latitude,
                longitude=longitude,
                speed_mps=_speed(tags.get("GPSSpeed"), tags.get("GPSSpeedRef")),
                playback_seconds=_finite(tags.get("SampleTime")),
            )
        )
    samples.sort(key=lambda sample: sample.recorded_utc)
    return TelemetryResult(tuple(samples), rejected)


def _timestamp(value: str | float | None) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, "%Y:%m:%d %H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def _finite(value: str | float | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _speed(value: str | float | None, reference: str | float | None) -> float | None:
    speed = _finite(value)
    if speed is None or speed < 0:
        return None
    factors = {"M": 0.44704, "K": 1 / 3.6, "N": 0.514444}
    factor = factors.get(str(reference or "M").upper())
    return None if factor is None else speed * factor
