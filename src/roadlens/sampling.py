"""Distance sampling uses source PTS, never sampled-video playback time."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime

import numpy as np

from roadlens.telemetry import TelemetrySample


@dataclass(frozen=True)
class FrameChoice:
    source_seconds: float
    sharpness: float
    distance_m: float
    speed_mps: float | None
    reason: str

    def to_dict(self):
        return asdict(self)


def speed_at_frames(
    times: np.ndarray,
    telemetry: tuple[TelemetrySample, ...],
    created: datetime,
    max_gap: float = 3.0,
) -> np.ndarray:
    """Interpolate valid speed fixes; don't bridge missing/stale intervals."""
    fixes = sorted(
        (
            (
                sample.playback_seconds
                if sample.playback_seconds is not None
                else (sample.recorded_utc - created).total_seconds(),
                sample.speed_mps,
            )
            for sample in telemetry
            if sample.speed_mps is not None
        ),
    )
    valid = [(t, v) for t, v in fixes if np.isfinite(t) and np.isfinite(v) and v >= 0]
    result = np.full(len(times), np.nan)
    if not valid:
        return result
    # Repeated metadata timestamps must not create zero-length interpolation spans.
    unique = dict(valid)
    ts = np.array(sorted(unique))
    vs = np.array([unique[t] for t in ts])
    right = np.searchsorted(ts, times, side="left")
    for i, (t, r) in enumerate(zip(times, right, strict=True)):
        if r < len(ts) and abs(ts[r] - t) < 1e-6:
            result[i] = vs[r]
        elif 0 < r < len(ts) and ts[r] - ts[r - 1] <= max_gap:
            result[i] = np.interp(t, ts[r - 1 : r + 1], vs[r - 1 : r + 1])
    return result


@dataclass(frozen=True)
class MotionSettings:
    """When the vehicle counts as stopped, and what is sampled while it is.

    Below ``stop_speed_mps`` for ``stop_after_seconds`` the vehicle is
    stopped until it exceeds ``resume_speed_mps``: one frame is taken on
    stopping, then one every ``stop_heartbeat_seconds`` (``None``: none).
    Speed below ``stop_speed_mps`` adds no distance, so GPS jitter at a
    standstill is not mistaken for travel. ``stop_speed_mps=None`` disables
    all of this.
    """

    stop_speed_mps: float | None = 0.5
    resume_speed_mps: float = 1.0
    stop_after_seconds: float = 1.0
    stop_heartbeat_seconds: float | None = 10.0

    def __post_init__(self):
        if self.stop_speed_mps is None:
            return
        if self.stop_speed_mps <= 0 or self.resume_speed_mps < self.stop_speed_mps:
            raise ValueError("Stop speed must be positive and below resume speed")
        if self.stop_after_seconds < 0:
            raise ValueError("Stop delay cannot be negative")
        if self.stop_heartbeat_seconds is not None and self.stop_heartbeat_seconds <= 0:
            raise ValueError("Stop heartbeat must be positive, or None to disable")


DEFAULT_MOTION = MotionSettings()
# Behaviour of the 1 November corridor survey: no stop state, a frame at
# least every 0.5 s whatever the speed.
LEGACY_MOTION = MotionSettings(stop_speed_mps=None)
LEGACY_MAX_INTERVAL = 0.5


def stopped_frames(
    times: np.ndarray, speeds: np.ndarray, motion: MotionSettings
) -> np.ndarray:
    """True for frames where the vehicle counts as stopped.

    A stop starts when the current speed and the mean over the last
    ``stop_after_seconds`` are both below ``stop_speed_mps`` (so one jittery
    reading does not cancel it, and pulling away does not re-enter it) and
    ends when speed exceeds ``resume_speed_mps`` or becomes unknown.
    """
    stopped = np.zeros(len(times), dtype=bool)
    if motion.stop_speed_mps is None or not len(times):
        return stopped
    known = np.isfinite(speeds) & (speeds >= 0)
    speed_sum = np.concatenate([[0.0], np.cumsum(np.where(known, speeds, 0))])
    unknown = np.concatenate([[0], np.cumsum(~known)])
    state = False
    for i in range(len(times)):
        if not known[i]:
            state = False
        elif state:
            state = speeds[i] <= motion.resume_speed_mps
        elif (
            speeds[i] < motion.stop_speed_mps
            and times[i] - times[0] >= motion.stop_after_seconds - 1e-9
        ):
            j = int(np.searchsorted(times, times[i] - motion.stop_after_seconds - 1e-9))
            count = i + 1 - j
            mean = (speed_sum[i + 1] - speed_sum[j]) / count
            state = unknown[i + 1] == unknown[j] and mean < motion.stop_speed_mps
        stopped[i] = state
    return stopped


def integrate_distance(
    times: np.ndarray, speeds: np.ndarray, motion: MotionSettings = DEFAULT_MOTION
) -> np.ndarray:
    """Metres travelled at each frame from known speed (trapezoid rule).

    Speed below the stop threshold, and any time while stopped, adds nothing,
    so GPS jitter at a standstill is not mistaken for travel.
    """
    known = np.isfinite(speeds) & (speeds >= 0)
    mean = np.zeros(len(times))
    mean[1:] = np.where(known[1:] & known[:-1], (speeds[1:] + speeds[:-1]) / 2, 0)
    if motion.stop_speed_mps is not None:
        mean[mean < motion.stop_speed_mps] = 0
        mean[stopped_frames(times, speeds, motion)] = 0
    increments = np.zeros(len(times))
    increments[1:] = np.diff(times) * mean[1:]
    return np.cumsum(increments)


def select_frames(
    times: list[float],
    sharpness: list[float],
    speeds: np.ndarray,
    spacing_m: float = 1.0,
    max_interval: float | None = None,
    fallback_interval: float = 0.2,
    sharpness_window: float = 0.05,
    all_frames: bool = False,
    motion: MotionSettings = DEFAULT_MOTION,
) -> list[FrameChoice]:
    """Select around distance targets, with <=0.2 spacing spatial displacement.

    Sharpness is compared only within +/- sharpness_window seconds AND +/-20%
    of spacing around a target. This is approximate spatial sampling, not a
    guarantee of a strict maximum gap; the manifest reports actual gaps.

    ``max_interval`` adds a time floor while speed is known (off by default:
    distance and the stop state cover it); ``fallback_interval`` applies when
    speed is unknown.
    """
    intervals = [spacing_m, fallback_interval]
    if max_interval is not None:
        intervals.append(max_interval)
    if min(intervals) <= 0 or sharpness_window < 0:
        raise ValueError("Sampling intervals must be positive; window nonnegative")
    t = np.asarray(times, dtype=float)
    q = np.asarray(sharpness, dtype=float)
    if len(t) != len(q) or len(t) != len(speeds):
        raise ValueError("Frame times, quality and speeds must have equal lengths")
    if not len(t):
        return []
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Source presentation timestamps must be strictly increasing")
    known = np.isfinite(speeds) & (speeds >= 0)
    distance = integrate_distance(t, np.asarray(speeds, dtype=float), motion)
    stopped = stopped_frames(t, np.asarray(speeds, dtype=float), motion)
    indices = [0]
    reasons = ["native" if all_frames else "start"]
    next_distance = spacing_m
    for i in range(1, len(t)):
        if all_frames:
            indices.append(i)
            reasons.append("native")
            continue
        if stopped[i]:
            heartbeat = motion.stop_heartbeat_seconds
            if not stopped[i - 1]:
                indices.append(i)
                reasons.append("stopped")
            elif heartbeat is not None and t[i] - t[indices[-1]] >= heartbeat - 1e-9:
                indices.append(i)
                reasons.append("stopped_heartbeat")
            continue
        interval = max_interval if known[i] else fallback_interval
        if interval is None:
            interval = np.inf
        due_distance = distance[i] >= next_distance
        if not due_distance and t[i] - t[indices[-1]] < interval - 1e-9:
            continue
        reason = (
            "distance"
            if due_distance
            else ("time_limit" if known[i] else "missing_speed_fallback")
        )
        lo = max(indices[-1] + 1, int(np.searchsorted(t, t[i] - sharpness_window)))
        hi = int(np.searchsorted(t, t[i] + sharpness_window, side="right"))
        candidates = np.arange(lo, hi)
        if due_distance:
            candidates = candidates[
                np.abs(distance[candidates] - distance[i]) <= spacing_m * 0.2
            ]
        if len(candidates):
            chosen = int(candidates[np.argmax(q[candidates])])
            indices.append(chosen)
            reasons.append(reason)
        if due_distance:
            next_distance = (np.floor(distance[i] / spacing_m) + 1) * spacing_m
    if indices[-1] != len(t) - 1:
        indices.append(len(t) - 1)
        reasons.append("end")
    return [
        FrameChoice(
            float(t[i]),
            float(q[i]),
            float(distance[i]),
            float(speeds[i]) if known[i] else None,
            reason,
        )
        for i, reason in zip(indices, reasons, strict=True)
    ]


def refinement_windows(
    detections: list[float],
    start: float,
    end: float,
    padding: float = 2.0,
) -> list[tuple[float, float]]:
    if padding < 0:
        raise ValueError("Refinement padding cannot be negative")
    windows: list[tuple[float, float]] = []
    for t in sorted(detections):
        lo, hi = max(start, t - padding), min(end, t + padding)
        if windows and lo <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(hi, windows[-1][1]))
        else:
            windows.append((lo, hi))
    return windows


def distance_refinement_windows(
    detections: list[float],
    times: list[float],
    speeds: np.ndarray,
    start: float,
    end: float,
    before_m: float = 10.0,
    after_m: float = 15.0,
    fallback_padding: float = 2.0,
    max_seconds: float = 30.0,
    motion: MotionSettings = DEFAULT_MOTION,
) -> list[tuple[float, float]]:
    """Windows from ``before_m`` metres before each hit to ``after_m`` after.

    A time window of +/-2 s covers 27 m at 30 mph but nothing at a red light,
    so windows are set by distance travelled. Where speed is unknown at a hit
    the window falls back to +/- ``fallback_padding`` seconds, and no window
    extends more than ``max_seconds`` either side of its hit.
    """
    if min(before_m, after_m, fallback_padding) < 0 or max_seconds <= 0:
        raise ValueError("Refinement distances and padding cannot be negative")
    t = np.asarray(times, dtype=float)
    v = np.asarray(speeds, dtype=float)
    if not len(t):
        return []
    distance = integrate_distance(t, v, motion)
    known = np.isfinite(v) & (v >= 0)
    windows: list[tuple[float, float]] = []
    for hit in sorted(detections):
        i = min(int(np.searchsorted(t, hit)), len(t) - 1)
        if known[i]:
            lo_i = int(np.searchsorted(distance, distance[i] - before_m, side="left"))
            hi_i = int(np.searchsorted(distance, distance[i] + after_m, side="right"))
            lo, hi = t[min(lo_i, i)], t[max(hi_i - 1, i)]
        else:
            lo, hi = hit - fallback_padding, hit + fallback_padding
        lo = max(start, lo, hit - max_seconds)
        hi = min(end, hi, hit + max_seconds)
        if windows and lo <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(hi, windows[-1][1]))
        else:
            windows.append((float(lo), float(hi)))
    return windows


def refinement_frames(
    times: list[float],
    sharpness: list[float],
    speeds: np.ndarray,
    window: tuple[float, float],
    spacing_m: float = 0.25,
    motion: MotionSettings = DEFAULT_MOTION,
) -> list[FrameChoice]:
    """Dense, stop-aware frames for one refinement window.

    Close to every native frame at cycling speed, one frame (plus heartbeats)
    while stopped, and every native frame where speed is unknown.
    """
    t = np.asarray(times, dtype=float)
    v = np.asarray(speeds, dtype=float)
    idx = np.flatnonzero((t >= window[0] - 1e-9) & (t <= window[1] + 1e-9))
    if not len(idx):
        return []
    choices = select_frames(
        t[idx].tolist(),
        np.asarray(sharpness, dtype=float)[idx].tolist(),
        v[idx],
        spacing_m=spacing_m,
        max_interval=None,
        fallback_interval=1e-9,
        sharpness_window=0,
        motion=motion,
    )
    # Report distance along the whole scan, not from the window start.
    offset = float(integrate_distance(t, v, motion)[idx[0]])
    return [replace(c, distance_m=c.distance_m + offset) for c in choices]

