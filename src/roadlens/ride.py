"""Multi-clip Garmin ride indexing and map/time/street segment resolution."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from roadlens.media import probe_video
from roadlens.pmtiles_roads import (
    StreetGeocoder,
    haversine_m,
    point_to_polyline_distance_m,
)
from roadlens.telemetry import TelemetryResult, TelemetrySample, read_garmin_telemetry

HAZARD_PRESETS: tuple[dict[str, str], ...] = (
    {
        "id": "pothole",
        "label": "Pothole",
        "sam_prompt": "pothole",
        "description": "Localized cavity or bowl-shaped asphalt depression.",
    },
    {
        "id": "cracking",
        "label": "Cracking / crazing",
        "sam_prompt": "cracking",
        "description": "Fractured or alligator-cracked road surface.",
    },
    {
        "id": "damaged_asphalt",
        "label": "Damaged / fretted edge",
        "sam_prompt": "damaged asphalt",
        "description": "Broken carriageway edge, fretting, or sunken utility trench.",
    },
    {
        "id": "debris",
        "label": "Road debris",
        "sam_prompt": "road debris",
        "description": "Loose stones, gravel, glass, or spilled load in the lane.",
    },
    {
        "id": "fallen_branch",
        "label": "Fallen tree branch",
        "sam_prompt": "fallen tree branch",
        "description": "Timber, bough, or storm vegetation obstructing the road.",
    },
    {
        "id": "flooding",
        "label": "Standing water / flooding",
        "sam_prompt": "flooded road puddle",
        "description": "Standing water concealing the road surface or blocked gully.",
    },
    {
        "id": "sunken_cover",
        "label": "Sunken drain / ironwork",
        "sam_prompt": "manhole cover drain",
        "description": "Recessed or damaged gully grate or inspection cover.",
    },
)


@dataclass(frozen=True)
class RideTrackPoint:
    clip_name: str
    clip_index: int
    source_seconds: float
    ride_elapsed_seconds: float
    recorded_utc: datetime
    latitude: float
    longitude: float
    speed_mps: float | None


@dataclass(frozen=True)
class ClipIndex:
    clip_index: int
    path: Path
    name: str
    created_utc: datetime
    end_utc: datetime
    duration_seconds: float
    fps: float
    width: int
    height: int
    ride_elapsed_start_seconds: float
    ride_elapsed_end_seconds: float
    distance_m: float
    gps_fix_count: int
    rejected_gps_samples: int
    overlaps_primary_clip: bool
    points: tuple[RideTrackPoint, ...]

    def to_dict(self, include_points: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "clip_index": self.clip_index,
            "name": self.name,
            "path": str(self.path),
            "created_utc": self.created_utc.isoformat(),
            "end_utc": self.end_utc.isoformat(),
            "duration_seconds": round(self.duration_seconds, 3),
            "fps": round(self.fps, 3),
            "width": self.width,
            "height": self.height,
            "ride_elapsed_start_seconds": round(self.ride_elapsed_start_seconds, 3),
            "ride_elapsed_end_seconds": round(self.ride_elapsed_end_seconds, 3),
            "distance_m": round(self.distance_m, 2),
            "gps_fix_count": self.gps_fix_count,
            "rejected_gps_samples": self.rejected_gps_samples,
            "overlaps_primary_clip": self.overlaps_primary_clip,
        }
        if include_points:
            payload["points"] = [
                {
                    "source_seconds": round(pt.source_seconds, 3),
                    "ride_elapsed_seconds": round(pt.ride_elapsed_seconds, 3),
                    "recorded_utc": pt.recorded_utc.isoformat(),
                    "latitude": pt.latitude,
                    "longitude": pt.longitude,
                    "speed_mps": (
                        None if pt.speed_mps is None else round(pt.speed_mps, 3)
                    ),
                }
                for pt in self.points
            ]
        return payload


@dataclass(frozen=True)
class RideIndex:
    clips: tuple[ClipIndex, ...]
    total_elapsed_seconds: float
    total_distance_m: float

    @property
    def primary_points(self) -> list[RideTrackPoint]:
        pts = [
            pt
            for clip in self.clips
            if not clip.overlaps_primary_clip
            for pt in clip.points
        ]
        pts.sort(key=lambda pt: pt.recorded_utc)
        return pts

    def bounds(self) -> tuple[float, float, float, float] | None:
        pts = self.primary_points or [pt for clip in self.clips for pt in clip.points]
        if not pts:
            return None
        lats = [pt.latitude for pt in pts]
        lons = [pt.longitude for pt in pts]
        return (min(lats), min(lons), max(lats), max(lons))

    def to_geojson(self, max_points_per_clip: int = 600) -> dict[str, Any]:
        features: list[dict[str, Any]] = []
        for clip in self.clips:
            if clip.overlaps_primary_clip or not clip.points:
                continue
            step = max(1, math.ceil(len(clip.points) / max_points_per_clip))
            sampled = list(clip.points[::step])
            if sampled[-1] != clip.points[-1]:
                sampled.append(clip.points[-1])
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[pt.longitude, pt.latitude] for pt in sampled],
                    },
                    "properties": {
                        "clip_index": clip.clip_index,
                        "clip_name": clip.name,
                        "duration_seconds": round(clip.duration_seconds, 3),
                        "distance_m": round(clip.distance_m, 2),
                        "ride_elapsed_start_seconds": round(
                            clip.ride_elapsed_start_seconds, 3
                        ),
                        "ride_elapsed_end_seconds": round(
                            clip.ride_elapsed_end_seconds, 3
                        ),
                        "point_meta": [
                            {
                                "s": round(pt.source_seconds, 2),
                                "e": round(pt.ride_elapsed_seconds, 2),
                                "t": pt.recorded_utc.isoformat(),
                                "lat": round(pt.latitude, 6),
                                "lon": round(pt.longitude, 6),
                                "v": (
                                    None
                                    if pt.speed_mps is None
                                    else round(pt.speed_mps, 2)
                                ),
                            }
                            for pt in sampled
                        ],
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_count": len(self.clips),
            "primary_clip_count": sum(
                not clip.overlaps_primary_clip for clip in self.clips
            ),
            "total_elapsed_seconds": round(self.total_elapsed_seconds, 3),
            "total_distance_m": round(self.total_distance_m, 2),
            "bounds": self.bounds(),
            "clips": [clip.to_dict(include_points=False) for clip in self.clips],
        }


@dataclass(frozen=True)
class ResolvedSegment:
    pass_index: int
    query_mode: str
    query_summary: str
    clip_name: str
    clip_path: Path
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    ride_elapsed_start_seconds: float
    ride_elapsed_end_seconds: float
    recorded_start_utc: datetime
    recorded_end_utc: datetime
    distance_m: float
    bearing_deg: float | None
    compass_direction: str | None
    estimated_native_frames: int
    estimated_coarse_frames_1m: int
    street_name: str | None
    place_name: str | None
    center_latitude: float | None
    center_longitude: float | None
    polyline: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_index": self.pass_index,
            "query_mode": self.query_mode,
            "query_summary": self.query_summary,
            "clip_name": self.clip_name,
            "clip_path": str(self.clip_path),
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "ride_elapsed_start_seconds": round(self.ride_elapsed_start_seconds, 3),
            "ride_elapsed_end_seconds": round(self.ride_elapsed_end_seconds, 3),
            "recorded_start_utc": self.recorded_start_utc.isoformat(),
            "recorded_end_utc": self.recorded_end_utc.isoformat(),
            "distance_m": round(self.distance_m, 2),
            "bearing_deg": (
                None if self.bearing_deg is None else round(self.bearing_deg, 1)
            ),
            "compass_direction": self.compass_direction,
            "estimated_native_frames": self.estimated_native_frames,
            "estimated_coarse_frames_1m": self.estimated_coarse_frames_1m,
            "street_name": self.street_name,
            "place_name": self.place_name,
            "center_latitude": self.center_latitude,
            "center_longitude": self.center_longitude,
            "polyline": [[lat, lon] for lat, lon in self.polyline],
        }


def discover_ride_videos(target: Path | Sequence[Path]) -> list[Path]:
    """Locate Garmin MP4 video clips while skipping low-res `.GLV` companions."""
    inputs = [target] if isinstance(target, Path) else list(target)
    found: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        if item.is_dir():
            candidates = sorted(
                path
                for path in item.iterdir()
                if path.is_file()
                and not path.name.startswith(".")
                and path.suffix.lower() == ".mp4"
            )
            for candidate in candidates:
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    found.append(candidate)
        elif item.is_file():
            if item.suffix.lower() == ".glv":
                raise ValueError("Use Garmin .MP4 clips rather than .GLV companions")
            resolved = item.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(item)
        else:
            raise FileNotFoundError(item)
    if not found:
        raise ValueError("No .MP4 ride clips found")
    return found


def index_ride(
    target: Path | Sequence[Path],
    probe_fn: Callable[[Path], dict[str, Any]] = probe_video,
    telemetry_fn: Callable[[Path], TelemetryResult] = read_garmin_telemetry,
    gap_pause_threshold_seconds: float = 5.0,
) -> RideIndex:
    video_paths = discover_ride_videos(target)
    raw_entries: list[
        tuple[datetime, float, Path, dict[str, Any], TelemetryResult]
    ] = []
    for path in video_paths:
        meta = probe_fn(path)
        duration = float(meta["duration"])
        telemetry = telemetry_fn(path)
        if meta.get("creation_time"):
            created = datetime.fromisoformat(
                str(meta["creation_time"]).replace("Z", "+00:00")
            )
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
        elif telemetry.samples:
            first = telemetry.samples[0]
            offset = first.playback_seconds or 0.0
            created = first.recorded_utc - timedelta(seconds=offset)
        else:
            raise ValueError(f"Clip {path.name} has neither creation_time nor GPS time")
        raw_entries.append((created, duration, path, meta, telemetry))

    # Sort chronologically; longer clips come first if start times match so loop
    # clips take precedence over short saved event clips.
    raw_entries.sort(key=lambda entry: (entry[0], -entry[1], entry[2].name))

    primary_intervals: list[tuple[datetime, datetime]] = []
    overlap_flags: list[bool] = []
    for created, duration, _path, _meta, _tel in raw_entries:
        end_utc = created + timedelta(seconds=duration)
        covered = any(
            p_start - timedelta(seconds=2.0) <= created
            and end_utc <= p_end + timedelta(seconds=2.0)
            for p_start, p_end in primary_intervals
        )
        overlap_flags.append(covered)
        if not covered:
            primary_intervals.append((created, end_utc))

    # Build piecewise elapsed-ride-time mapping across primary intervals, pausing
    # elapsed clock across recording gaps > gap_pause_threshold_seconds.
    timeline: list[tuple[datetime, datetime, float]] = []
    elapsed_cursor = 0.0
    prev_end: datetime | None = None
    for p_start, p_end in primary_intervals:
        if prev_end is not None:
            gap = (p_start - prev_end).total_seconds()
            if 0 < gap <= gap_pause_threshold_seconds:
                elapsed_cursor += gap
            elif gap < 0:
                p_start = max(p_start, prev_end)
        span = max(0.0, (p_end - p_start).total_seconds())
        timeline.append((p_start, p_end, elapsed_cursor))
        elapsed_cursor += span
        prev_end = max(prev_end, p_end) if prev_end else p_end

    def utc_to_elapsed(ts: datetime) -> float:
        if not timeline:
            return 0.0
        for idx, (seg_start, seg_end, base_elapsed) in enumerate(timeline):
            if ts <= seg_end or idx == len(timeline) - 1:
                delta = max(0.0, (ts - seg_start).total_seconds())
                return min(
                    base_elapsed + (seg_end - seg_start).total_seconds(),
                    base_elapsed + delta,
                )
        return elapsed_cursor

    clips: list[ClipIndex] = []
    total_distance = 0.0
    for clip_idx, ((created, duration, path, meta, telemetry), is_overlap) in enumerate(
        zip(raw_entries, overlap_flags, strict=True)
    ):
        end_utc = created + timedelta(seconds=duration)
        clip_elapsed_start = utc_to_elapsed(created)
        clip_elapsed_end = utc_to_elapsed(end_utc)
        points: list[RideTrackPoint] = []
        clip_distance = 0.0
        prev_sample: TelemetrySample | None = None
        for sample in telemetry.samples:
            source_sec = (
                sample.playback_seconds
                if sample.playback_seconds is not None
                else (sample.recorded_utc - created).total_seconds()
            )
            source_sec = max(0.0, min(duration, float(source_sec)))
            point_utc = created + timedelta(seconds=source_sec)
            if prev_sample is not None:
                step_m = haversine_m(
                    prev_sample.latitude,
                    prev_sample.longitude,
                    sample.latitude,
                    sample.longitude,
                )
                if step_m < 150.0:
                    clip_distance += step_m
            prev_sample = sample
            points.append(
                RideTrackPoint(
                    clip_name=path.name,
                    clip_index=clip_idx,
                    source_seconds=source_sec,
                    ride_elapsed_seconds=utc_to_elapsed(point_utc),
                    recorded_utc=sample.recorded_utc,
                    latitude=sample.latitude,
                    longitude=sample.longitude,
                    speed_mps=sample.speed_mps,
                )
            )
        if not is_overlap:
            total_distance += clip_distance
        clips.append(
            ClipIndex(
                clip_index=clip_idx,
                path=path,
                name=path.name,
                created_utc=created,
                end_utc=end_utc,
                duration_seconds=duration,
                fps=float(meta.get("fps") or 30.0),
                width=int(meta.get("width") or 0),
                height=int(meta.get("height") or 0),
                ride_elapsed_start_seconds=clip_elapsed_start,
                ride_elapsed_end_seconds=clip_elapsed_end,
                distance_m=clip_distance,
                gps_fix_count=len(points),
                rejected_gps_samples=telemetry.rejected_samples,
                overlaps_primary_clip=is_overlap,
                points=tuple(points),
            )
        )
    return RideIndex(
        clips=tuple(clips),
        total_elapsed_seconds=elapsed_cursor,
        total_distance_m=total_distance,
    )


def parse_ride_offset(spec: str | float | int) -> float:
    """Parse elapsed ride offset (`1200`, `'20m'`, `'20 minutes'`, `'1:05:30'`)."""
    if isinstance(spec, int | float):
        if spec < 0 or not math.isfinite(spec):
            raise ValueError("Ride offset must be nonnegative")
        return float(spec)
    text = spec.strip().lower()
    if not text:
        raise ValueError("Ride offset cannot be empty")
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    colon = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)", text)
    if colon:
        hours = int(colon.group(1) or 0)
        minutes = int(colon.group(2))
        seconds = float(colon.group(3))
        return hours * 3600.0 + minutes * 60.0 + seconds
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|h|minutes?|mins?|min|m|seconds?|secs?|sec|s)"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"Could not parse ride offset: {spec!r}")
    total = 0.0
    for match in matches:
        amount = float(match.group(1))
        unit = match.group(2)[0]
        total += amount * (3600.0 if unit == "h" else 60.0 if unit == "m" else 1.0)
    return total


def parse_wall_clock(
    spec: str,
    reference_utc: datetime,
    tz_name: str = "Europe/London",
) -> datetime:
    """Parse wall-clock time (`'12:30'`, `'3:31 pm'`, or ISO-8601) on the ride date."""
    cleaned = spec.strip()
    if not cleaned:
        raise ValueError("Wall-clock time cannot be empty")
    if "T" in cleaned or re.match(r"^\d{4}-\d{2}-\d{2}", cleaned):
        dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo(tz_name))
        return dt.astimezone(UTC)

    tz = ZoneInfo(tz_name)
    local_ref = reference_utc.astimezone(tz)
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?", cleaned.lower())
    if not match:
        raise ValueError(f"Could not parse wall-clock time: {spec!r}")
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3) or 0)
    meridiem = match.group(4)
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    local_dt = datetime.combine(local_ref.date(), time(hour, minute, second), tzinfo=tz)
    return local_dt.astimezone(UTC)


def _bearing_and_compass(
    points: Sequence[RideTrackPoint],
) -> tuple[float | None, str | None]:
    if len(points) < 2:
        return None, None
    lat1, lon1 = math.radians(points[0].latitude), math.radians(points[0].longitude)
    lat2, lon2 = math.radians(points[-1].latitude), math.radians(points[-1].longitude)
    if abs(lat1 - lat2) < 1e-9 and abs(lon1 - lon2) < 1e-9:
        return None, None
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        dlon
    )
    deg = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
    sectors = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    compass = sectors[int((deg + 22.5) // 45.0) % 8]
    return deg, compass


def _build_segment(
    clip: ClipIndex,
    start_seconds: float,
    end_seconds: float,
    pass_index: int,
    query_mode: str,
    query_summary: str,
    geocoder: StreetGeocoder | None = None,
    explicit_street: str | None = None,
) -> ResolvedSegment:
    lo = max(0.0, min(clip.duration_seconds, start_seconds))
    hi = max(lo + 0.5, min(clip.duration_seconds, end_seconds))
    duration = hi - lo
    sub_points = [pt for pt in clip.points if lo - 1.0 <= pt.source_seconds <= hi + 1.0]
    dist_m = 0.0
    for idx in range(1, len(sub_points)):
        dist_m += haversine_m(
            sub_points[idx - 1].latitude,
            sub_points[idx - 1].longitude,
            sub_points[idx].latitude,
            sub_points[idx].longitude,
        )
    if dist_m == 0.0 and sub_points and sub_points[0].speed_mps:
        dist_m = duration * sub_points[0].speed_mps
    bearing, compass = _bearing_and_compass(sub_points)
    center_lat = center_lon = None
    street_name = explicit_street
    place_name = None
    if sub_points:
        mid = sub_points[len(sub_points) // 2]
        center_lat, center_lon = mid.latitude, mid.longitude
        if geocoder is not None:
            rev = geocoder.reverse_road(center_lat, center_lon)
            if street_name is None:
                street_name = rev.street_name or rev.road_ref
            place_name = rev.place_name
    rec_start = clip.created_utc + timedelta(seconds=lo)
    rec_end = clip.created_utc + timedelta(seconds=hi)
    ratio_lo = lo / clip.duration_seconds if clip.duration_seconds > 0 else 0.0
    ratio_hi = hi / clip.duration_seconds if clip.duration_seconds > 0 else 0.0
    span_elapsed = clip.ride_elapsed_end_seconds - clip.ride_elapsed_start_seconds
    el_lo = clip.ride_elapsed_start_seconds + ratio_lo * span_elapsed
    el_hi = clip.ride_elapsed_start_seconds + ratio_hi * span_elapsed
    est_native = max(1, round(duration * clip.fps))
    est_coarse = (
        max(2, round(dist_m / 1.0) + 1) if dist_m > 0 else max(2, round(duration * 5))
    )
    return ResolvedSegment(
        pass_index=pass_index,
        query_mode=query_mode,
        query_summary=query_summary,
        clip_name=clip.name,
        clip_path=clip.path,
        start_seconds=lo,
        end_seconds=hi,
        duration_seconds=duration,
        ride_elapsed_start_seconds=el_lo,
        ride_elapsed_end_seconds=el_hi,
        recorded_start_utc=rec_start,
        recorded_end_utc=rec_end,
        distance_m=dist_m,
        bearing_deg=bearing,
        compass_direction=compass,
        estimated_native_frames=est_native,
        estimated_coarse_frames_1m=min(est_native, est_coarse),
        street_name=street_name,
        place_name=place_name,
        center_latitude=center_lat,
        center_longitude=center_lon,
        polyline=tuple((pt.latitude, pt.longitude) for pt in sub_points),
    )


def resolve_by_ride_offset(
    ride: RideIndex,
    offset_spec: str | float | int,
    window_seconds: float = 30.0,
    geocoder: StreetGeocoder | None = None,
) -> list[ResolvedSegment]:
    """Resolve `'20 minutes into my ride'` to clip window(s) and street name."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    target = parse_ride_offset(offset_spec)
    if target > ride.total_elapsed_seconds + 1e-6:
        raise ValueError(
            f"Ride offset {target:.1f}s exceeds indexed ride duration "
            f"({ride.total_elapsed_seconds:.1f}s)"
        )
    half = window_seconds / 2.0
    want_lo = max(0.0, target - half)
    want_hi = min(ride.total_elapsed_seconds, target + half)
    candidates = [c for c in ride.clips if not c.overlaps_primary_clip] or list(
        ride.clips
    )
    results: list[ResolvedSegment] = []
    for clip in candidates:
        if (
            clip.ride_elapsed_end_seconds < want_lo
            or clip.ride_elapsed_start_seconds > want_hi
        ):
            continue
        local_lo = max(0.0, want_lo - clip.ride_elapsed_start_seconds)
        local_hi = min(clip.duration_seconds, want_hi - clip.ride_elapsed_start_seconds)
        if local_hi - local_lo < 0.2:
            continue
        results.append(
            _build_segment(
                clip,
                local_lo,
                local_hi,
                pass_index=len(results) + 1,
                query_mode="ride_offset",
                query_summary=f"Ride offset {target:.1f}s (+/-{half:.1f}s)",
                geocoder=geocoder,
            )
        )
    return results


def resolve_by_wall_clock(
    ride: RideIndex,
    clock_spec: str,
    window_seconds: float = 60.0,
    tz_name: str = "Europe/London",
    geocoder: StreetGeocoder | None = None,
) -> list[ResolvedSegment]:
    """Resolve `'around 12:30'` against clip recording timestamps in local/UTC time."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if not ride.clips:
        return []
    half = window_seconds / 2.0
    candidates = [c for c in ride.clips if not c.overlaps_primary_clip] or list(
        ride.clips
    )
    # Try local timezone first, then UTC if local timezone doesn't overlap the ride.
    tried_timezones = [tz_name] if tz_name == "UTC" else [tz_name, "UTC"]
    for candidate_tz in tried_timezones:
        target_utc = parse_wall_clock(
            clock_spec, candidates[0].created_utc, candidate_tz
        )
        want_lo = target_utc - timedelta(seconds=half)
        want_hi = target_utc + timedelta(seconds=half)
        results: list[ResolvedSegment] = []
        for clip in candidates:
            if clip.end_utc < want_lo or clip.created_utc > want_hi:
                continue
            local_lo = max(0.0, (want_lo - clip.created_utc).total_seconds())
            local_hi = min(
                clip.duration_seconds, (want_hi - clip.created_utc).total_seconds()
            )
            if local_hi - local_lo < 0.2:
                continue
            results.append(
                _build_segment(
                    clip,
                    local_lo,
                    local_hi,
                    pass_index=len(results) + 1,
                    query_mode="wall_clock",
                    query_summary=(
                        f"Wall clock {clock_spec} ({candidate_tz}, +/-{half:.0f}s)"
                    ),
                    geocoder=geocoder,
                )
            )
        if results:
            return results
    return []


def resolve_by_corridor(
    ride: RideIndex,
    polylines: Sequence[Sequence[tuple[float, float]]] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    tolerance_m: float = 35.0,
    merge_gap_seconds: float = 8.0,
    padding_seconds: float = 3.0,
    geocoder: StreetGeocoder | None = None,
    query_mode: str = "map_corridor",
    query_summary: str = "Selected map corridor",
    explicit_street: str | None = None,
) -> list[ResolvedSegment]:
    """Find all ride passes (e.g. outbound + return) intersecting a road corridor."""
    if not polylines and bbox is None:
        raise ValueError("Provide polylines or bbox to resolve a corridor")
    candidates = [c for c in ride.clips if not c.overlaps_primary_clip] or list(
        ride.clips
    )
    results: list[ResolvedSegment] = []
    for clip in candidates:
        matching_times: list[float] = []
        for pt in clip.points:
            inside = False
            if bbox is not None:
                min_lat, min_lon, max_lat, max_lon = bbox
                if (
                    min_lat <= pt.latitude <= max_lat
                    and min_lon <= pt.longitude <= max_lon
                ):
                    inside = True
            if not inside and polylines:
                for line in polylines:
                    if (
                        point_to_polyline_distance_m(pt.latitude, pt.longitude, line)
                        <= tolerance_m
                    ):
                        inside = True
                        break
            if inside:
                matching_times.append(pt.source_seconds)
        if not matching_times:
            continue
        intervals: list[list[float]] = []
        for sec in sorted(matching_times):
            if intervals and sec - intervals[-1][1] <= merge_gap_seconds:
                intervals[-1][1] = sec
            else:
                intervals.append([sec, sec])
        for lo, hi in intervals:
            results.append(
                _build_segment(
                    clip,
                    lo - padding_seconds,
                    hi + padding_seconds,
                    pass_index=len(results) + 1,
                    query_mode=query_mode,
                    query_summary=query_summary,
                    geocoder=geocoder,
                    explicit_street=explicit_street,
                )
            )
    return results


def resolve_by_street_name(
    ride: RideIndex,
    street_name: str,
    geocoder: StreetGeocoder,
    tolerance_m: float = 35.0,
    merge_gap_seconds: float = 8.0,
    padding_seconds: float = 3.0,
) -> list[ResolvedSegment]:
    """Resolve `'Eccup Moor Road'` using local PMTiles MVT or injected street index."""
    cleaned = street_name.strip()
    if not cleaned:
        raise ValueError("Street name cannot be empty")
    pts = ride.primary_points or [pt for clip in ride.clips for pt in clip.points]
    if not pts:
        return []
    # Decimate ride points to query nearby map tiles rapidly.
    step = max(1, len(pts) // 200)
    sample_coords = [(pt.latitude, pt.longitude) for pt in pts[::step]]
    roads = geocoder.find_roads(cleaned, sample_coords)
    if not roads:
        return []
    canonical_name = roads[0].name or roads[0].ref or cleaned
    lines = [line for road in roads for line in road.lines]
    return resolve_by_corridor(
        ride,
        polylines=lines,
        tolerance_m=tolerance_m,
        merge_gap_seconds=merge_gap_seconds,
        padding_seconds=padding_seconds,
        geocoder=geocoder,
        query_mode="street_name",
        query_summary=f"Street '{canonical_name}'",
        explicit_street=canonical_name,
    )
