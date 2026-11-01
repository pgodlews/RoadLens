"""Distance sampling uses source PTS, never sampled-video playback time."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


def select_frames(
    times: list[float],
    sharpness: list[float],
    speeds: np.ndarray,
    spacing_m: float = 1.0,
    max_interval: float = 0.5,
    fallback_interval: float = 0.2,
    sharpness_window: float = 0.05,
    all_frames: bool = False,
) -> list[FrameChoice]:
    """Select around distance targets, with <=0.2 spacing spatial displacement.

    Sharpness is compared only within +/- sharpness_window seconds AND +/-20%
    of spacing around a target. This is approximate spatial sampling, not a
    guarantee of a strict maximum gap; the manifest reports actual gaps.
    """
    if min(spacing_m, max_interval, fallback_interval) <= 0 or sharpness_window < 0:
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
    increments = np.zeros(len(t))
    increments[1:] = np.where(
        known[1:] & known[:-1],
        np.diff(t) * (speeds[1:] + speeds[:-1]) / 2,
        0,
    )
    distance = np.cumsum(increments)
    indices = [0]
    reasons = ["native" if all_frames else "start"]
    next_distance = spacing_m
    for i in range(1, len(t)):
        if all_frames:
            indices.append(i)
            reasons.append("native")
            continue
        interval = max_interval if known[i] else fallback_interval
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
