import pytest

from roadlens.telemetry import (
    TelemetryResult,
    decode_garmin_metadata,
    read_camera_telemetry,
)


def test_decode_garmin_metadata_filters_no_fix_and_converts_speed():
    result = decode_garmin_metadata(
        {
            "Doc1:GPSDateTime": "2026:10:11 13:50:37Z",
            "Doc1:GPSLatitude": -180,
            "Doc1:GPSLongitude": -180,
            "Doc2:GPSDateTime": "2026:10:11 13:50:38Z",
            "Doc2:GPSLatitude": 53.8755,
            "Doc2:GPSLongitude": -1.5386,
            "Doc2:GPSSpeed": 10,
            "Doc2:GPSSpeedRef": "M",
            "Doc2:SampleTime": 2.5,
            "Main:GPSPosition": "-180 -180",
        }
    )

    assert result.rejected_samples == 1
    assert len(result.samples) == 1
    sample = result.samples[0]
    assert sample.latitude == 53.8755
    assert sample.longitude == -1.5386
    assert sample.speed_mps == 4.4704
    assert sample.playback_seconds == 2.5
    assert sample.recorded_utc.isoformat() == "2026-10-11T13:50:38+00:00"


def test_camera_telemetry_keeps_unusable_tesla_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "roadlens.telemetry.read_garmin_telemetry", lambda p: TelemetryResult((), 2)
    )
    monkeypatch.setattr(
        "roadlens.tesla.read_tesla_telemetry",
        lambda p: TelemetryResult((), 3, source="tesla_sei"),
    )
    result = read_camera_telemetry(tmp_path / "clip.mp4")
    assert not result.samples
    assert result.rejected_samples == 3
    assert result.source == "tesla_sei_unusable"
    assert result.location_source == "camera_gps"


def test_camera_telemetry_keeps_garmin_first(monkeypatch, tmp_path):
    garmin = decode_garmin_metadata(
        {
            "Doc1:GPSDateTime": "2026:10:11 13:50:38Z",
            "Doc1:GPSLatitude": 53.8755,
            "Doc1:GPSLongitude": -1.5386,
        }
    )
    monkeypatch.setattr("roadlens.telemetry.read_garmin_telemetry", lambda p: garmin)

    def unexpected_tesla_read(path):
        pytest.fail("Working Garmin metadata must retain priority")

    monkeypatch.setattr("roadlens.tesla.read_tesla_telemetry", unexpected_tesla_read)
    assert read_camera_telemetry(tmp_path / "clip.mp4") is garmin
