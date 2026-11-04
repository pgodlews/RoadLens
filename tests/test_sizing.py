import json

import numpy as np
import pytest

from roadlens.domain import encode_mask
from roadlens.sizing import (
    CameraGeometry,
    annotate_run_sizes,
    estimate_size,
    fit_kinematic_geometry,
    geometry_for,
    mask_footprint,
    run_telemetry_source,
)

H, W = 1080, 1344
EXACT = CameraGeometry("test camera", 0.45, 0.85, 1.0, 0, 0, 0)


def ground_rectangle_mask(near_m, length_m, width_m, geometry=EXACT, centre_m=0.0):
    """Project a flat road rectangle through the pinhole model into a mask."""
    focal = geometry.scale * H / geometry.camera_height_m
    horizon = geometry.horizon_fraction * H
    mask = np.zeros((H, W), dtype=bool)
    for row in range(H):
        u = row + 0.5 - horizon
        if u <= 0:
            continue
        z = geometry.scale * H / u
        if near_m <= z <= near_m + length_m:
            half = width_m / 2 * focal / z
            mid = W / 2 + centre_m * focal / z
            mask[row, int(round(mid - half)) : int(round(mid + half))] = True
    return mask


def test_footprint_recovers_ground_rectangle():
    mask = ground_rectangle_mask(near_m=3.0, length_m=0.6, width_m=0.3)
    size = mask_footprint(mask, EXACT)
    assert size["length_m"] == pytest.approx(0.6, rel=0.05)
    assert size["width_m"] == pytest.approx(0.3, rel=0.05)
    assert size["near_distance_m"] == pytest.approx(3.0, rel=0.02)


def test_footprint_rejects_masks_cut_off_or_too_far():
    cut_off = ground_rectangle_mask(near_m=1.0, length_m=1.0, width_m=0.3)
    assert cut_off[-1].any()
    assert mask_footprint(cut_off, EXACT) is None
    far = ground_rectangle_mask(near_m=14.0, length_m=0.6, width_m=0.5)
    assert mask_footprint(far, EXACT) is None


def test_estimate_reports_rounded_size_with_range():
    masks = [
        ground_rectangle_mask(near_m=z, length_m=0.6, width_m=0.3)
        for z in (2.5, 3.0, 4.0)
    ]
    geometry = CameraGeometry("test camera", 0.45, 0.85, 1.0)
    estimate = estimate_size(masks, geometry)
    assert estimate["status"] == "estimated"
    assert estimate["frames_used"] == 3
    assert estimate["length_cm"] == 60
    assert estimate["width_cm"] == 30
    low, high = estimate["length_range_cm"]
    assert low < 60 < high
    assert "depth is not estimated" in estimate["note"]


def test_estimate_explains_when_no_frame_is_usable():
    estimate = estimate_size([np.zeros((H, W), dtype=bool)], EXACT)
    assert estimate["status"] == "unavailable"
    assert "reason" in estimate


def test_kinematic_fit_recovers_horizon_and_scale():
    tracks = []
    for start in (12.0, 12.6):  # near and far edge of one pothole
        track = []
        for step in range(25):
            travel = step * 0.22  # ~6.6 m/s at 30 fps
            row = EXACT.horizon_fraction + EXACT.scale / (start - travel)
            track.append((travel, row))
        tracks.append(track)
    fit = fit_kinematic_geometry(tracks)
    assert fit["horizon_fraction"] == pytest.approx(0.45, abs=0.002)
    assert fit["scale"] == pytest.approx(0.85, rel=0.02)


def test_annotate_run_sizes_uses_every_stored_frame(tmp_path):
    for idx, z in enumerate((2.5, 3.0, 3.5)):
        mask = ground_rectangle_mask(near_m=z, length_m=0.6, width_m=0.3)
        detections = [
            {
                "object_id": 4,
                "box_xywh_normalized": [0.4, 0.6, 0.2, 0.1],
                "mask_rle": encode_mask(mask),
            }
        ]
        (tmp_path / f"result-{idx:06d}.json").write_text(
            json.dumps({"frame_index": idx, "detections": detections})
        )
    observations = [{"object_id": 4, "best_frame_index": 1}]
    (tmp_path / "observations.json").write_text(json.dumps(observations))
    (tmp_path / "observations.geojson").write_text(
        json.dumps({"features": [{"properties": {"object_id": 4}}]})
    )

    annotated = annotate_run_sizes(tmp_path, EXACT)

    assert annotated[0]["size_estimate"]["frames_used"] == 3
    saved = json.loads((tmp_path / "observations.json").read_text())
    assert saved[0]["size_estimate"]["length_cm"] == 60
    feature = json.loads((tmp_path / "observations.geojson").read_text())
    assert feature["features"][0]["properties"]["size_estimate"]["width_cm"] == 30


def test_uncalibrated_camera_is_not_sized_with_garmin_numbers(tmp_path):
    mask = ground_rectangle_mask(near_m=3.0, length_m=0.6, width_m=0.3)
    detection = {
        "object_id": 1,
        "box_xywh_normalized": [0.4, 0.6, 0.2, 0.1],
        "mask_rle": encode_mask(mask),
    }
    (tmp_path / "result-000000.json").write_text(
        json.dumps({"frame_index": 0, "detections": [detection]})
    )
    (tmp_path / "observations.json").write_text(
        json.dumps([{"object_id": 1, "best_frame_index": 0}])
    )

    annotated = annotate_run_sizes(tmp_path, geometry_for("tesla_sei"))

    assert annotated[0]["size_estimate"]["status"] == "unavailable"
    assert "measured height" in annotated[0]["size_estimate"]["reason"]
    assert geometry_for("garmin") is not None


def test_size_command_keeps_tesla_runs_unsized(tmp_path, monkeypatch):
    from roadlens.cli import main

    mask = ground_rectangle_mask(near_m=3.0, length_m=0.6, width_m=0.3)
    detection = {
        "object_id": 1,
        "box_xywh_normalized": [0.4, 0.6, 0.2, 0.1],
        "mask_rle": encode_mask(mask),
    }
    (tmp_path / "result-000000.json").write_text(
        json.dumps({"frame_index": 0, "detections": [detection]})
    )
    (tmp_path / "observations.json").write_text(
        json.dumps([{"object_id": 1, "best_frame_index": 0}])
    )
    (tmp_path / "scan.json").write_text(json.dumps({"telemetry_source": "tesla_sei"}))

    # `roadlens size` used to apply Garmin geometry to any run.
    monkeypatch.setattr("sys.argv", ["roadlens", "size", str(tmp_path)])
    main()

    saved = json.loads((tmp_path / "observations.json").read_text())
    assert saved[0]["size_estimate"]["status"] == "unavailable"


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"scan.json": {"telemetry_source": "tesla_sei"}}, "tesla_sei"),
        ({"observations.json": [{"location_source": "tesla_camera_gps"}]}, "tesla_sei"),
        (
            {
                "observations.json": [
                    {"location_source": "garmin_camera_gps"},
                    {"location_source": "unavailable"},
                ]
            },
            "garmin",
        ),
        ({}, "garmin"),  # runs from before Tesla support
    ],
)
def test_run_telemetry_source(tmp_path, files, expected):
    for name, content in files.items():
        (tmp_path / name).write_text(json.dumps(content))
    assert run_telemetry_source(tmp_path) == expected
