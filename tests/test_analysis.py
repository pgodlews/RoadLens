from datetime import UTC, datetime

import numpy as np

from roadlens.analysis import build_observations, nearest_telemetry
from roadlens.domain import Detection, FrameSegmentations
from roadlens.telemetry import TelemetrySample


def sample(second, latitude=53.8755, longitude=-1.5386):
    return TelemetrySample(
        recorded_utc=datetime(2026, 10, 11, 14, 6, second, tzinfo=UTC),
        latitude=latitude,
        longitude=longitude,
        speed_mps=5,
        playback_seconds=None,
    )


def detection(score=0.8):
    return Detection(
        object_id=7,
        box_xywh=(0.1, 0.2, 0.3, 0.2),
        score=score,
        mask=np.ones((4, 5), dtype=bool),
    )


def frame(index, score=0.8):
    return FrameSegmentations(
        frame_index=index,
        prompt="pothole",
        backend="local",
        detections=(detection(score),),
    )


def test_build_observations_uses_best_frame_and_camera_gps():
    created = datetime(2026, 10, 11, 14, 3, 16, tzinfo=UTC)
    telemetry = [sample(second) for second in range(60)]

    observations = build_observations(
        [frame(0, 0.6), frame(5, 0.9)], telemetry, created, 165, 5
    )

    assert len(observations) == 1
    observation = observations[0]
    assert observation.first_source_seconds == 165
    assert observation.last_source_seconds == 166
    assert observation.best_source_seconds == 166
    assert observation.recorded_utc.isoformat() == "2026-10-11T14:06:02+00:00"
    assert observation.latitude == 53.8755
    assert observation.location_source == "garmin_camera_gps"
    assert observation.to_dict()["review_status"] == "unreviewed"


def test_nearest_telemetry_rejects_stale_fix():
    timestamp = datetime(2026, 10, 11, 14, 6, 30, tzinfo=UTC)
    assert nearest_telemetry([sample(20)], timestamp) is None
