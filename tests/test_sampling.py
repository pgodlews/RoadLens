from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from roadlens.sampling import refinement_windows, select_frames, speed_at_frames
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


def test_missing_speed_and_stops_keep_temporal_coverage():
    times = np.arange(0, 4, 1 / 30).tolist()
    for speed, limit in [(0.0, 0.5), (np.nan, 0.2)]:
        selected = select_frames(
            times, [1] * len(times), np.full(len(times), speed), sharpness_window=0
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
