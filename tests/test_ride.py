from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from roadlens.cli import build_parser
from roadlens.pmtiles_roads import (
    PmtilesStreetIndex,
    RoadFeature,
    StaticStreetIndex,
    build_minimal_pmtiles_v3,
    encode_varint,
    latlon_to_zxy,
)
from roadlens.ride import (
    HAZARD_PRESETS,
    discover_ride_videos,
    index_ride,
    parse_ride_offset,
    parse_wall_clock,
    resolve_by_corridor,
    resolve_by_ride_offset,
    resolve_by_street_name,
    resolve_by_wall_clock,
)
from roadlens.telemetry import TelemetryResult, TelemetrySample


def _pb_bytes(field_num: int, payload: bytes) -> bytes:
    return encode_varint((field_num << 3) | 2) + encode_varint(len(payload)) + payload


def _pb_varint(field_num: int, value: int) -> bytes:
    return encode_varint((field_num << 3) | 0) + encode_varint(value)


def _zigzag(value: int) -> int:
    return (value << 1) ^ (value >> 31)


def _build_synthetic_mvt_road_tile() -> bytes:
    # Build an MVT v2 tile with a `roads` LineString and a `places` Point.
    road_geom = b"".join(
        encode_varint(v)
        for v in (
            (1 << 3) | 1,  # MoveTo(1)
            _zigzag(512),
            _zigzag(2048),
            (1 << 3) | 2,  # LineTo(1)
            _zigzag(3072),
            _zigzag(0),
        )
    )
    road_tags = (
        encode_varint(0) + encode_varint(0) + encode_varint(1) + encode_varint(1)
    )
    road_feat = _pb_bytes(2, road_tags) + _pb_varint(3, 2) + _pb_bytes(4, road_geom)
    roads_layer = (
        _pb_bytes(1, b"roads")
        + _pb_bytes(2, road_feat)
        + _pb_bytes(3, b"name")
        + _pb_bytes(3, b"kind")
        + _pb_bytes(4, _pb_bytes(1, b"Demo Moor Road"))
        + _pb_bytes(4, _pb_bytes(1, b"minor_road"))
        + _pb_varint(5, 4096)
    )

    place_geom = b"".join(
        encode_varint(v) for v in ((1 << 3) | 1, _zigzag(2048), _zigzag(2048))
    )
    place_tags = encode_varint(0) + encode_varint(0)
    place_feat = _pb_bytes(2, place_tags) + _pb_varint(3, 1) + _pb_bytes(4, place_geom)
    places_layer = (
        _pb_bytes(1, b"places")
        + _pb_bytes(2, place_feat)
        + _pb_bytes(3, b"name")
        + _pb_bytes(4, _pb_bytes(1, b"Demo Village"))
        + _pb_varint(5, 4096)
    )
    return _pb_bytes(3, roads_layer) + _pb_bytes(3, places_layer)


@pytest.fixture
def synthetic_ride(tmp_path):
    clip1 = tmp_path / "GRMS0010.MP4"
    clip2 = tmp_path / "GRMS0011.MP4"
    overlap_event = tmp_path / "GRMN0009.MP4"
    glv = tmp_path / "GRMS0010.GLV"
    for p in (clip1, clip2, overlap_event, glv):
        p.write_bytes(b"0")

    # 2026-07-15 is BST (UTC+1 in Europe/London): 11:00 UTC == 12:00 BST.
    base_utc = datetime(2026, 7, 15, 11, 0, 0, tzinfo=UTC)

    # Outbound clip1 (600s): rides east from lon -1.560 to -1.500 at lat 53.850
    samples_1 = tuple(
        TelemetrySample(
            recorded_utc=base_utc + timedelta(seconds=t),
            latitude=53.8500,
            longitude=-1.5600 + (t / 600.0) * 0.0600,
            speed_mps=6.5,
            playback_seconds=float(t),
        )
        for t in range(0, 601, 5)
    )
    # Return clip2 (600s, starts 2s after clip1): rides west from -1.500 to -1.560
    start2_utc = base_utc + timedelta(seconds=602)
    samples_2 = tuple(
        TelemetrySample(
            recorded_utc=start2_utc + timedelta(seconds=t),
            latitude=53.8500,
            longitude=-1.5000 - (t / 600.0) * 0.0600,
            speed_mps=6.5,
            playback_seconds=float(t),
        )
        for t in range(0, 601, 5)
    )
    # Overlapping saved event clip inside clip1 (100..160s)
    event_utc = base_utc + timedelta(seconds=100)
    samples_ev = tuple(
        TelemetrySample(
            recorded_utc=event_utc + timedelta(seconds=t),
            latitude=53.8500,
            longitude=-1.5500 + (t / 60.0) * 0.0060,
            speed_mps=6.5,
            playback_seconds=float(t),
        )
        for t in range(0, 61, 5)
    )

    meta_map = {
        clip1.name: {
            "creation_time": base_utc.isoformat(),
            "duration": 600.0,
            "fps": 30.0,
            "width": 2688,
            "height": 2160,
        },
        clip2.name: {
            "creation_time": start2_utc.isoformat(),
            "duration": 600.0,
            "fps": 30.0,
            "width": 2688,
            "height": 2160,
        },
        overlap_event.name: {
            "creation_time": event_utc.isoformat(),
            "duration": 60.0,
            "fps": 30.0,
            "width": 2688,
            "height": 2160,
        },
    }
    tel_map = {
        clip1.name: TelemetryResult(samples_1, 0),
        clip2.name: TelemetryResult(samples_2, 0),
        overlap_event.name: TelemetryResult(samples_ev, 0),
    }
    ride = index_ride(
        tmp_path,
        probe_fn=lambda p: meta_map[p.name],
        telemetry_fn=lambda p: tel_map[p.name],
    )
    geocoder = StaticStreetIndex(
        roads=(
            RoadFeature(
                name="Sample Moor Road",
                kind="minor_road",
                ref=None,
                lines=(((53.8500, -1.5450), (53.8500, -1.5250)),),
            ),
        ),
        places=(("Sampleton", 53.8505, -1.5350),),
    )
    return tmp_path, ride, geocoder


def test_discover_ride_videos_skips_glv_and_indexes_overlapping_event_clips(
    synthetic_ride,
):
    folder, ride, _ = synthetic_ride
    discovered = discover_ride_videos(folder)
    assert [p.name for p in discovered] == [
        "GRMN0009.MP4",
        "GRMS0010.MP4",
        "GRMS0011.MP4",
    ]
    with pytest.raises(ValueError, match="GLV"):
        discover_ride_videos(folder / "GRMS0010.GLV")

    assert len(ride.clips) == 3
    primary = [c for c in ride.clips if not c.overlaps_primary_clip]
    overlapped = [c for c in ride.clips if c.overlaps_primary_clip]
    assert [c.name for c in primary] == ["GRMS0010.MP4", "GRMS0011.MP4"]
    assert [c.name for c in overlapped] == ["GRMN0009.MP4"]
    # 600s + 2s short gap + 600s = 1202s, NOT +60s from overlapping event clip.
    assert ride.total_elapsed_seconds == pytest.approx(1202.0)


def test_resolve_by_ride_offset_and_wall_clock_reverse_geocodes_street(synthetic_ride):
    _, ride, geocoder = synthetic_ride
    assert parse_ride_offset("15 minutes 30 seconds") == pytest.approx(930.0)
    assert parse_ride_offset("15:30") == pytest.approx(930.0)

    # 15m (900s) into ride is ~298s into GRMS0011.MP4, on Sample Moor Road (-1.530)
    by_offset = resolve_by_ride_offset(
        ride, "15 minutes", window_seconds=20.0, geocoder=geocoder
    )
    assert len(by_offset) == 1
    seg = by_offset[0]
    assert seg.clip_name == "GRMS0011.MP4"
    assert seg.start_seconds == pytest.approx(288.0)
    assert seg.end_seconds == pytest.approx(308.0)
    assert seg.street_name == "Sample Moor Road"
    assert seg.place_name == "Sampleton"

    # Base UTC is 11:00 UTC = 12:00 BST. 12:05 BST = 300s into GRMS0010.MP4.
    by_clock = resolve_by_wall_clock(
        ride, "12:05", window_seconds=30.0, tz_name="Europe/London", geocoder=geocoder
    )
    assert len(by_clock) == 1
    assert by_clock[0].clip_name == "GRMS0010.MP4"
    assert by_clock[0].start_seconds == pytest.approx(285.0)
    assert by_clock[0].end_seconds == pytest.approx(315.0)
    assert by_clock[0].street_name == "Sample Moor Road"
    assert parse_wall_clock(
        "12:30 pm", datetime(2026, 7, 15, 11, 0, tzinfo=UTC), "Europe/London"
    ) == datetime(2026, 7, 15, 11, 30, tzinfo=UTC)


def test_resolve_by_street_and_corridor_finds_outbound_and_return_passes(
    synthetic_ride,
):
    _, ride, geocoder = synthetic_ride
    passes = resolve_by_street_name(ride, "Sample Moor Rd", geocoder=geocoder)
    assert len(passes) == 2
    assert passes[0].clip_name == "GRMS0010.MP4"
    assert passes[1].clip_name == "GRMS0011.MP4"
    assert passes[0].compass_direction == "E"
    assert passes[1].compass_direction == "W"
    assert passes[0].street_name == "Sample Moor Road"
    assert passes[0].estimated_coarse_frames_1m < passes[0].estimated_native_frames

    corridor_passes = resolve_by_corridor(
        ride,
        bbox=(53.8495, -1.5450, 53.8505, -1.5250),
        geocoder=geocoder,
    )
    assert len(corridor_passes) == 2
    assert any(p["id"] == "fallen_branch" for p in HAZARD_PRESETS)


def test_pmtiles_street_index_reads_roads_and_places_offline(tmp_path):
    lat, lon = 53.8003, -1.5497
    z, x, y = latlon_to_zxy(lat, lon, 15)
    mvt = _build_synthetic_mvt_road_tile()
    archive_path = tmp_path / "mini.pmtiles"
    archive_path.write_bytes(build_minimal_pmtiles_v3([(z, x, y, mvt)], max_zoom=15))

    index = PmtilesStreetIndex(archive_path)
    roads = index.find_roads("Demo Moor Rd", [(lat, lon)])
    assert len(roads) == 1
    assert roads[0].name == "Demo Moor Road"
    mid_lat, mid_lon = roads[0].lines[0][0]
    rev = index.reverse_road(mid_lat, mid_lon, max_distance_m=100.0)
    assert rev.street_name == "Demo Moor Road"
    assert rev.place_name == "Demo Village"


def test_scan_custom_prompt_propagates_to_observations_and_resume_check(
    tmp_path, monkeypatch
):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("av")
    import numpy as np

    from roadlens.domain import Detection, FrameSegmentations
    from roadlens.scan import run_scan

    video = tmp_path / "branch.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(10):
        writer.write(np.full((48, 64, 3), 90, dtype=np.uint8))
    writer.release()

    created = datetime(2026, 7, 15, 11, 0, tzinfo=UTC)
    monkeypatch.setattr(
        "roadlens.scan.probe_video",
        lambda path: {
            "creation_time": created.isoformat(),
            "duration": 1.0,
            "fps": 10.0,
            "width": 64,
            "height": 48,
        },
    )
    samples = (TelemetrySample(created, 53.85, -1.54, 5.0, 0.0),)
    monkeypatch.setattr(
        "roadlens.scan.read_camera_telemetry", lambda path: TelemetryResult(samples, 0)
    )
    seen_prompts: list[str] = []

    class CustomBackend:
        thresholds = {"output": 0.25, "detection": 0.25, "new_object": 0.35}

        def __init__(self, *args, **kwargs):
            pass

        def segment_video(self, path, prompt):
            seen_prompts.append(prompt)
            for idx, _ in enumerate(sorted(path.glob("*.png"))):
                mask = np.zeros((48, 64), dtype=bool)
                mask[10:20, 10:30] = True
                yield FrameSegmentations(
                    idx,
                    prompt,
                    "local",
                    (Detection(1, (0.2, 0.2, 0.3, 0.2), 0.7, mask),),
                )

    out_dir = tmp_path / "branch-scan"
    args = build_parser().parse_args(
        [
            "scan",
            str(video),
            "--prompt",
            "fallen tree branch",
            "--duration",
            "0.8",
            "--no-refine",
            "--output",
            str(out_dir),
        ]
    )
    manifest = run_scan(args, CustomBackend)
    assert manifest["status"] == "complete"
    assert seen_prompts == ["fallen tree branch"]
    obs = json.loads((out_dir / "observations.json").read_text())
    assert obs[0]["prompt"] == "fallen tree branch"

    args.resume = True
    args.prompt = "pothole"
    with pytest.raises(ValueError, match="same scan settings"):
        run_scan(args, CustomBackend)


def test_build_segment_keeps_narrow_windows_inside_clip_bounds():
    from pathlib import Path

    from roadlens.ride import ClipIndex, _build_segment

    created = datetime(2026, 1, 1, tzinfo=UTC)
    clip = ClipIndex(
        clip_index=0,
        path=Path("clip.mp4"),
        name="clip.mp4",
        created_utc=created,
        end_utc=created + timedelta(seconds=10),
        duration_seconds=10.0,
        fps=30.0,
        width=64,
        height=48,
        ride_elapsed_start_seconds=0.0,
        ride_elapsed_end_seconds=10.0,
        distance_m=0.0,
        gps_fix_count=0,
        rejected_gps_samples=0,
        overlaps_primary_clip=False,
        points=(),
    )
    for start, end in ((9.9, 10.0), (10.0, 10.0), (12.0, 15.0), (0.0, 0.1)):
        seg = _build_segment(clip, start, end, 1, "offset", "test")
        assert 0.0 <= seg.start_seconds < seg.end_seconds <= 10.0
        assert seg.end_seconds - seg.start_seconds == pytest.approx(0.5)
