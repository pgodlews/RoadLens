from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from roadlens.sampling import (
    LEGACY_MAX_INTERVAL,
    LEGACY_MOTION,
    MotionSettings,
    distance_refinement_windows,
    refinement_frames,
    refinement_windows,
    select_frames,
    speed_at_frames,
)
from roadlens.telemetry import TelemetrySample


def test_distance_sampling_adapts_to_car_and_bicycle_speed():
    times = np.arange(0, 10, 1 / 30).tolist()
    bicycle = select_frames(
        times, [1] * len(times), np.full(len(times), 6.0), sharpness_window=0
    )
    car = select_frames(
        times, [1] * len(times), np.full(len(times), 24.0), sharpness_window=0
    )
    assert 59 <= len(bicycle) <= 62
    assert 238 <= len(car) <= 242
    assert bicycle[-1].source_seconds == times[-1]


def test_legacy_sampling_keeps_temporal_coverage_at_stops_and_without_speed():
    times = np.arange(0, 4, 1 / 30).tolist()
    for speed, limit in [(0.0, 0.5), (np.nan, 0.2)]:
        selected = select_frames(
            times,
            [1] * len(times),
            np.full(len(times), speed),
            sharpness_window=0,
            max_interval=LEGACY_MAX_INTERVAL,
            motion=LEGACY_MOTION,
        )
        gap = max(np.diff([c.source_seconds for c in selected]))
        assert gap <= limit + 1 / 30 + 1e-9
        if np.isnan(speed):
            assert selected[1].reason == "missing_speed_fallback"


def test_sharpness_selection_is_local_and_preserves_actual_time():
    times = np.arange(0, 2, 1 / 30).tolist()
    quality = np.ones(len(times))
    quality[16] = 50  # Sharper neighbour of target at 0.5 s.
    quality[20] = 1000  # Too far from the target to be eligible.
    selected = select_frames(times, quality.tolist(), np.full(len(times), 2.0))
    assert selected[1].source_seconds == pytest.approx(16 / 30)
    assert np.all(np.diff([c.source_seconds for c in selected]) > 0)


def test_speed_interpolation_uses_playback_time_and_rejects_gaps():
    created = datetime(2026, 1, 1, tzinfo=UTC)
    samples = tuple(
        TelemetrySample(
            created + timedelta(seconds=100 + t),
            53,
            -1,
            v,
            t,
        )
        for t, v in [(0, 2), (2, 6), (10, 6)]
    )
    speeds = speed_at_frames(np.array([0, 1, 2, 3, 9, 10, 11]), samples, created)
    np.testing.assert_allclose(speeds[:3], [2, 4, 6])
    assert np.isnan(speeds[[3, 4, 6]]).all()


def test_native_mode_keeps_irregular_pts_and_refinement_merges_windows():
    times = [10.0, 10.03, 10.071, 10.12]
    selected = select_frames(times, [1] * 4, np.ones(4), all_frames=True)
    assert [c.source_seconds for c in selected] == times
    assert refinement_windows([11, 12, 19], 10, 20) == [(10, 14), (17, 20)]


def test_invalid_timestamps_fail_instead_of_misplacing_evidence():
    with pytest.raises(ValueError, match="increasing"):
        select_frames([1, 1], [1, 1], np.ones(2))


FPS = 30000 / 1001


def speed_profile(*segments):
    """(seconds, start m/s, end m/s) ramps at the Garmin's frame rate."""
    times, speeds, now = [], [], 0.0
    for duration, v0, v1 in segments:
        n = int(round(duration * FPS))
        times += [now + i / FPS for i in range(n)]
        speeds += [v0 + (v1 - v0) * i / n for i in range(n)]
        now += n / FPS
    return times, np.array(speeds)


TRAFFIC_LIGHTS = ((30, 6.7, 6.7), (3, 6.7, 0), (60, 0, 0), (4, 0, 6.7), (30, 6.7, 6.7))


def reasons(choices):
    return [c.reason for c in choices]


def test_stop_takes_one_frame_then_heartbeats_and_cruise_is_unchanged():
    times, speeds = speed_profile(*TRAFFIC_LIGHTS)
    legacy = select_frames(
        times,
        [1] * len(times),
        speeds,
        max_interval=LEGACY_MAX_INTERVAL,
        motion=LEGACY_MOTION,
    )
    selected = select_frames(times, [1] * len(times), speeds)

    assert reasons(selected).count("stopped") == 1
    assert reasons(selected).count("stopped_heartbeat") == 6
    assert "time_limit" not in reasons(selected)
    assert reasons(legacy).count("time_limit") > 100
    distance = [c for c in selected if c.reason == "distance"]
    assert len(distance) == reasons(legacy).count("distance") - 1
    assert len(selected) < 0.8 * len(legacy)


def test_stop_heartbeat_can_be_turned_off():
    times, speeds = speed_profile(*TRAFFIC_LIGHTS)
    motion = MotionSettings(stop_heartbeat_seconds=None)
    selected = select_frames(times, [1] * len(times), speeds, motion=motion)
    assert reasons(selected).count("stopped") == 1
    assert "stopped_heartbeat" not in reasons(selected)


def test_gps_jitter_at_a_standstill_is_not_distance():
    times, _ = speed_profile((60, 0, 0))
    jitter = np.random.default_rng(0).uniform(0, 0.6, len(times))
    selected = select_frames(times, [1] * len(times), jitter)
    assert selected[-1].distance_m < 0.5
    assert "distance" not in reasons(selected)
    assert len(selected) <= 8


def test_crawling_samples_by_distance_not_time():
    times, speeds = speed_profile((60, 1.0, 1.0))
    selected = select_frames(times, [1] * len(times), speeds)
    assert 59 <= len(selected) <= 62
    assert "stopped" not in reasons(selected)


def test_unknown_speed_ends_a_stop_and_falls_back_to_time():
    times, speeds = speed_profile((5, 0, 0), (2, 0, 0))
    speeds[int(5 * FPS) :] = np.nan
    selected = select_frames(times, [1] * len(times), speeds, sharpness_window=0)
    assert "stopped" in reasons(selected)
    assert reasons(selected).count("missing_speed_fallback") >= 9


def test_motion_settings_reject_inconsistent_speeds():
    with pytest.raises(ValueError, match="below resume"):
        MotionSettings(stop_speed_mps=1.5, resume_speed_mps=1.0)
    with pytest.raises(ValueError, match="heartbeat"):
        MotionSettings(stop_heartbeat_seconds=0)


def test_refinement_by_distance_covers_metres_not_seconds():
    times, speeds = speed_profile((30, 6.7, 6.7))
    windows = distance_refinement_windows([15.0], times, speeds, 0, times[-1])
    ((lo, hi),) = windows
    assert 15.0 - lo == pytest.approx(10 / 6.7, abs=0.05)
    assert hi - 15.0 == pytest.approx(15 / 6.7, abs=0.05)
    frames = refinement_frames(times, [1] * len(times), speeds, windows[0])
    assert frames[0].distance_m == pytest.approx(lo * 6.7, abs=0.3)
    assert 90 <= len(frames) <= 105


def test_refinement_skips_a_stop_inside_the_window():
    times, speeds = speed_profile(*TRAFFIC_LIGHTS)
    hits = [t for t in np.arange(34, 93, 0.5)]
    windows = distance_refinement_windows(hits, times, speeds, 0, times[-1])
    frames = [
        f
        for w in windows
        for f in refinement_frames(times, [1] * len(times), speeds, w)
    ]
    native = sum(
        int(np.sum((np.array(times) >= a) & (np.array(times) <= b))) for a, b in windows
    )
    assert native > 1500
    assert len(frames) < 150
    assert reasons(frames).count("stopped") == 1


def test_refinement_falls_back_to_seconds_without_speed():
    times = np.arange(0, 10, 1 / 30).tolist()
    speeds = np.full(len(times), np.nan)
    windows = distance_refinement_windows([5.0], times, speeds, 0, times[-1])
    assert windows == [(3.0, 7.0)]
    frames = refinement_frames(times, [1] * len(times), speeds, windows[0])
    assert len(frames) == 121
