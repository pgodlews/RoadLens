"""Approximate pothole footprint from SAM masks and flat-road camera geometry.

A forward camera at height ``h`` sees a road point at distance ``Z`` on image
row ``y``, with ``Z = h * f / (y - y_horizon)`` for a flat road. Length along
the road needs only the product ``h * f``; width across it needs only ``h``
(``X = h * dx / (y - y_horizon)``), so the focal length itself never appears.

``h * f`` and the horizon row come from a kinematic fit: as the bike moves a
known distance (integrated Garmin speed), the near and far edges of a tracked
SAM mask move down the frame. Results are planimetric estimates with explicit
ranges, never depth and never survey measurements.
"""

from __future__ import annotations

import itertools
import json
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roadlens.domain import decode_mask

METHOD = "flat_road_camera_geometry_v1"
NOTE = (
    "Approximate surface footprint from flat-road camera geometry. "
    "Not a survey measurement; depth is not estimated."
)


@dataclass(frozen=True)
class CameraGeometry:
    """Rows are fractions of image height, so one calibration fits any resolution."""

    name: str
    horizon_fraction: float
    # Camera height x focal length, in metres x image heights.
    scale: float
    camera_height_m: float
    horizon_tolerance: float = 0.012
    scale_tolerance: float = 0.12
    height_tolerance_m: float = 0.1

    def variants(self) -> Iterable[CameraGeometry]:
        """The nominal geometry plus every corner of the tolerance box."""
        yield self
        for dh, ds, dc in itertools.product((-1, 1), repeat=3):
            yield CameraGeometry(
                self.name,
                self.horizon_fraction + dh * self.horizon_tolerance,
                self.scale * (1 + ds * self.scale_tolerance),
                self.camera_height_m + dc * self.height_tolerance_m,
                0,
                0,
                0,
            )


# Garmin Varia Vue under the bars of the rider's Specialized Tarmac SL7 (size L).
# Lens height 0.89 m and focal length 0.81 image heights were measured
# photogrammetrically from the 991 mm wheelbase (docs/ideas-to-explore.md), and
# the horizon sits near 0.47. Independently, a kinematic fit of hosted SAM 3.1
# pothole tracks with Garmin speed (Eccup return pass, 95 edge observations)
# gives a scale of 0.725 at that horizon: focal length 0.814 at 0.89 m.
GARMIN_VARIA_VUE = CameraGeometry(
    name="Garmin Varia Vue, under-bar mount (Tarmac SL7)",
    horizon_fraction=0.47,
    scale=0.89 * 0.81,
    camera_height_m=0.89,
    height_tolerance_m=0.03,
)


def fit_kinematic_geometry(
    edge_tracks: Sequence[Sequence[tuple[float, float]]],
    *,
    horizon_search: tuple[float, float] = (0.2, 0.8),
    step: float = 0.0005,
) -> dict[str, float]:
    """Fit horizon and ``h * f`` from tracked ground edges.

    Each track is ``[(travel_m, row_fraction), ...]`` for one physical edge
    (for example the near edge of one pothole). ``1 / (row - horizon)`` is
    linear in travel with a slope shared by all tracks, so for each candidate
    horizon the rest is a linear least-squares problem.
    """
    tracks = [list(t) for t in edge_tracks if len(t) >= 3]
    if not tracks:
        raise ValueError("At least one edge track with 3+ observations is required")
    travel = np.array([s for t in tracks for s, _ in t], dtype=float)
    rows = np.array([y for t in tracks for _, y in t], dtype=float)
    owner = np.repeat(np.arange(len(tracks)), [len(t) for t in tracks])
    design = np.zeros((len(rows), len(tracks) + 1))
    design[np.arange(len(rows)), owner] = 1.0
    design[:, -1] = -travel
    best: dict[str, float] | None = None
    upper = min(horizon_search[1], rows.min() - 0.01)
    for horizon in np.arange(horizon_search[0], upper, step):
        coef, *_ = np.linalg.lstsq(design, 1.0 / (rows - horizon), rcond=None)
        if coef[-1] <= 0:
            continue
        scale = 1.0 / coef[-1]
        distance = design[:, :-1] @ coef[:-1] * scale - travel
        if (distance <= 0).any():
            continue
        rms = float(np.sqrt(np.mean((horizon + scale / distance - rows) ** 2)))
        if best is None or rms < best["rms_fraction"]:
            best = {
                "horizon_fraction": float(horizon),
                "scale": float(scale),
                "rms_fraction": rms,
                "observations": float(len(rows)),
            }
    if best is None:
        raise ValueError("No physically valid horizon found for these tracks")
    return best


def mask_footprint(
    mask: np.ndarray,
    geometry: CameraGeometry,
    *,
    max_distance_m: float = 12.0,
) -> dict[str, float] | None:
    """Ground length and width of one mask, or ``None`` if it can't be sized.

    Masks cut off by the frame edge, too far away, or too small are rejected,
    because their extent on the road is unknown or dominated by a few pixels.
    """
    height, width = mask.shape
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size < 3 or cols.size < 3 or int(mask.sum()) < 50:
        return None
    if rows[-1] >= height - 1 or cols[0] <= 0 or cols[-1] >= width - 1:
        return None
    horizon = geometry.horizon_fraction * height
    if rows[0] - horizon < 1:
        return None

    def distance(row_edge: float) -> float:
        return geometry.scale * height / (row_edge - horizon)

    far = distance(rows[0])
    near = distance(rows[-1] + 1)
    if far > max_distance_m:
        return None
    # Width per row from the mask's extent in that row; a high percentile
    # ignores single-row spikes along ragged mask edges.
    widths = []
    for row in rows:
        filled = np.flatnonzero(mask[row])
        extent = filled[-1] - filled[0] + 1
        widths.append(geometry.camera_height_m * extent / (row + 0.5 - horizon))
    return {
        "length_m": far - near,
        "width_m": float(np.percentile(widths, 95)),
        "near_distance_m": near,
    }


def estimate_size(
    masks: Sequence[np.ndarray],
    geometry: CameraGeometry = GARMIN_VARIA_VUE,
) -> dict[str, Any]:
    """Median footprint over usable frames, with a range from calibration and spread."""
    per_variant = []
    for variant in geometry.variants():
        sizes = [s for m in masks if (s := mask_footprint(m, variant))]
        per_variant.append(sizes)
    nominal = per_variant[0]
    if not nominal:
        return {
            "method": METHOD,
            "status": "unavailable",
            "reason": (
                "No frame shows the whole candidate on the road within 12 m; "
                "it is cut off by the frame edge, too small, or too far away."
            ),
            "camera": geometry.name,
        }
    estimate: dict[str, Any] = {
        "method": METHOD,
        "status": "estimated",
        "camera": geometry.name,
        "frames_used": len(nominal),
        "nearest_distance_m": round(min(s["near_distance_m"] for s in nominal), 1),
        "note": NOTE,
    }
    for key, label in (("length_m", "length"), ("width_m", "width")):
        values = [s[key] for s in nominal]
        centres = [statistics.median(s[key] for s in v) for v in per_variant if v]
        spread = (
            np.percentile(values, [25, 75]) if len(values) > 1 else [values[0]] * 2
        )
        low = min(min(centres), spread[0])
        high = max(max(centres), spread[1])
        estimate[f"{label}_cm"] = _round_cm(statistics.median(values))
        estimate[f"{label}_range_cm"] = [_round_cm(low), _round_cm(high)]
    estimate["display"] = (
        f"≈ {estimate['length_cm']} × {estimate['width_cm']} cm "
        f"(length {_span(estimate['length_range_cm'])}, "
        f"width {_span(estimate['width_range_cm'])} cm)"
    )
    return estimate


def _span(bounds: list[int]) -> str:
    return str(bounds[0]) if bounds[0] == bounds[1] else f"{bounds[0]}–{bounds[1]}"


def _round_cm(metres: float) -> int:
    # 5 cm steps: finer digits would claim precision the geometry lacks.
    return max(5, int(round(metres * 20)) * 5)


def observation_masks(run_dir: Path, observation: dict[str, Any]) -> list[np.ndarray]:
    """Every stored mask of this candidate: all tracked frames when the run kept
    them (hosted SAM), otherwise the single best-evidence frame (local scan)."""
    object_id = observation.get("object_id")
    masks = []
    for path in sorted(run_dir.glob("result-*.json")):
        data = json.loads(path.read_text())
        for detection in data.get("detections") or []:
            if detection.get("object_id") == object_id and detection.get("mask_rle"):
                masks.append(decode_mask(detection["mask_rle"]))
    if masks:
        return masks
    best = run_dir / f"result-{int(observation.get('best_frame_index', -1)):06d}.json"
    if best.is_file():
        detections = json.loads(best.read_text()).get("detections") or []
        if len(detections) == 1 and detections[0].get("mask_rle"):
            return [decode_mask(detections[0]["mask_rle"])]
    return []


def geometry_for(telemetry_source: str) -> CameraGeometry | None:
    """Calibrated geometry for a camera, or None when none has been measured."""
    return GARMIN_VARIA_VUE if telemetry_source == "garmin" else None


def annotate_run_sizes(
    run_dir: Path, geometry: CameraGeometry | None = GARMIN_VARIA_VUE
) -> list[dict[str, Any]]:
    """Add ``size_estimate`` to every observation (and its GeoJSON feature).

    With no ``geometry`` (an uncalibrated camera) every candidate is marked
    unavailable rather than sized with another camera's numbers.
    """
    path = run_dir / "observations.json"
    observations = json.loads(path.read_text())
    for item in observations:
        if geometry is None:
            item["size_estimate"] = {
                "method": METHOD,
                "status": "unavailable",
                "reason": (
                    "This camera has no measured height and focal length yet, "
                    "so its masks cannot be projected onto the road."
                ),
            }
            continue
        try:
            masks = observation_masks(run_dir, item)
        except (OSError, ValueError, KeyError) as error:
            item["size_estimate"] = {
                "method": METHOD,
                "status": "unavailable",
                "reason": f"Stored mask could not be read: {error}",
                "camera": geometry.name,
            }
            continue
        item["size_estimate"] = estimate_size(masks, geometry)
    path.write_text(json.dumps(observations, indent=2) + "\n", encoding="utf-8")
    geojson_path = run_dir / "observations.geojson"
    if geojson_path.is_file():
        collection = json.loads(geojson_path.read_text())
        sizes = {item.get("object_id"): item["size_estimate"] for item in observations}
        for feature in collection.get("features") or []:
            props = feature.get("properties") or {}
            if props.get("object_id") in sizes:
                props["size_estimate"] = sizes[props["object_id"]]
        geojson_path.write_text(
            json.dumps(collection, indent=2) + "\n", encoding="utf-8"
        )
    return observations
