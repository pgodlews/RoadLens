from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from roadlens.cli import build_parser
from roadlens.pmtiles_roads import (
    build_minimal_pmtiles_v3,
    encode_varint,
    latlon_to_zxy,
)
from roadlens.studio import create_studio_server, render_studio_html
from roadlens.telemetry import TelemetryResult, TelemetrySample


def _pb_bytes(field_num: int, payload: bytes) -> bytes:
    return encode_varint((field_num << 3) | 2) + encode_varint(len(payload)) + payload


def _pb_varint(field_num: int, value: int) -> bytes:
    return encode_varint((field_num << 3) | 0) + encode_varint(value)


def _zigzag(value: int) -> int:
    return (value << 1) ^ (value >> 31)


def _build_synthetic_mvt_road_tile() -> bytes:
    road_geom = b"".join(
        encode_varint(v)
        for v in (
            (1 << 3) | 1,
            _zigzag(512),
            _zigzag(2048),
            (1 << 3) | 2,
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
    return _pb_bytes(3, roads_layer)


def _http_json(
    url: str,
    *,
    method: str = "GET",
    body: dict | bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict]:
    req_headers = dict(headers or {})
    data: bytes | None = None
    if isinstance(body, dict):
        data = json.dumps(body).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    elif isinstance(body, bytes):
        data = body
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


@pytest.fixture
def studio_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from roadlens.pmtiles_roads import PmtilesStreetIndex

    z, x, y = latlon_to_zxy(53.8500, -1.5300, 15)
    pmtiles_path = tmp_path / "demo.pmtiles"
    pmtiles_path.write_bytes(
        build_minimal_pmtiles_v3(
            [(z, x, y, _build_synthetic_mvt_road_tile())], max_zoom=15
        )
    )
    road_line = PmtilesStreetIndex(pmtiles_path).find_roads(
        "Demo Moor Road", [(53.8500, -1.5300)]
    )[0].lines[0]
    (lat0, lon0), (lat1, lon1) = road_line[0], road_line[-1]

    clip_path = tmp_path / "GRMS0010.MP4"
    clip_path.write_bytes(b"fake-mp4-header")
    base_utc = datetime(2026, 7, 15, 11, 0, 0, tzinfo=UTC)
    samples = tuple(
        TelemetrySample(
            recorded_utc=base_utc + timedelta(seconds=t),
            latitude=lat0 + (t / 300.0) * (lat1 - lat0),
            longitude=lon0 + (t / 300.0) * (lon1 - lon0),
            speed_mps=7.0,
            playback_seconds=float(t),
        )
        for t in range(0, 301, 5)
    )

    monkeypatch.setattr(
        "roadlens.ride.probe_video",
        lambda _p: {
            "duration": 300.0,
            "fps": 30.0,
            "width": 2688,
            "height": 2160,
            "creation_time": "2026-07-15T11:00:00Z",
        },
    )
    monkeypatch.setattr(
        "roadlens.ride.read_garmin_telemetry",
        lambda _p: TelemetryResult(samples=samples, rejected_samples=0),
    )

    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "overlay-000000.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (run_dir / "frame-000000.png").write_bytes(b"\x89PNG\r\n\x1a\nframe")
    # Frame 0 shares its result file with another object to exercise box matching.
    (run_dir / "result-000000.json").write_text(
        json.dumps(
            {
                "detections": [
                    {"object_id": 9, "box_xywh_normalized": [0.1, 0.1, 0.1, 0.1]},
                    {"object_id": 0, "box_xywh_normalized": [0.4, 0.6, 0.2, 0.1]},
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "observations.json").write_text(
        json.dumps(
            [
                {
                    "object_id": 0,
                    "prompt": "pothole",
                    "backend": "LocalSamBackend",
                    "first_source_seconds": 40.0,
                    "last_source_seconds": 41.2,
                    "best_source_seconds": 40.5,
                    "best_frame_index": 0,
                    "recorded_utc": "2026-07-15T11:00:40+00:00",
                    "score": 0.77,
                    "foreground_pixels": 12400,
                    "latitude": 53.8500,
                    "longitude": -1.5300,
                    "location_source": "camera_gps",
                    "review_status": "unreviewed",
                },
                {
                    "object_id": 1,
                    "prompt": "pothole",
                    "backend": "LocalSamBackend",
                    "first_source_seconds": 85.0,
                    "last_source_seconds": 85.8,
                    "best_source_seconds": 85.4,
                    "best_frame_index": 1,
                    "recorded_utc": "2026-07-15T11:01:25+00:00",
                    "score": 0.62,
                    "foreground_pixels": 6200,
                    "latitude": 53.8501,
                    "longitude": -1.5280,
                    "location_source": "camera_gps",
                    "review_status": "unreviewed",
                },
            ]
        ),
        encoding="utf-8",
    )

    workspace = tmp_path / "studio-workspace"
    server = create_studio_server(
        workspace_dir=workspace,
        pmtiles_path=pmtiles_path,
        host="127.0.0.1",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield {
            "base_url": base_url,
            "clip_path": clip_path,
            "run_dir": run_dir,
            "workspace": workspace,
            "pmtiles_path": pmtiles_path,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_cli_studio_parser(tmp_path: Path):
    pm = tmp_path / "uk.pmtiles"
    args = build_parser().parse_args(
        ["studio", str(tmp_path / "ride"), "--pmtiles", str(pm), "--port", "8899"]
    )
    assert args.command == "studio"
    assert args.ride == tmp_path / "ride"
    assert args.pmtiles == pm
    assert args.port == 8899


def test_render_studio_html_escapes_initial_path():
    rendered = render_studio_html(Path('/tmp/my"quote<clip>.MP4'))
    assert "&quot;" in rendered
    assert "&lt;clip&gt;" in rendered
    assert "RoadLens Studio" in rendered


def test_studio_html_and_pmtiles_byte_ranges(studio_env):
    base = studio_env["base_url"]
    with urllib.request.urlopen(f"{base}/", timeout=5) as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "Bookended Human-in-the-Loop" in html

    status, presets = _http_json(f"{base}/api/presets")
    assert status == 200
    assert any(p["id"] == "pothole" for p in presets["presets"])

    req = urllib.request.Request(
        f"{base}/__roadlens_osm.pmtiles", headers={"Range": "bytes=0-6"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 206
        assert resp.read() == b"PMTiles"


def test_studio_index_resolve_and_upload(studio_env):
    base = studio_env["base_url"]
    status, indexed = _http_json(
        f"{base}/api/index-ride",
        method="POST",
        body={"path": str(studio_env["clip_path"])},
    )
    assert status == 200
    assert indexed["ride"]["primary_clip_count"] == 1
    assert len(indexed["geojson"]["features"]) == 1

    # Street lookup via local PMTiles MVT index
    status, by_street = _http_json(
        f"{base}/api/resolve",
        method="POST",
        body={"mode": "street", "value": "Demo Moor Road"},
    )
    assert status == 200
    assert len(by_street["passes"]) >= 1
    assert by_street["passes"][0]["street_name"] == "Demo Moor Road"

    # Ride offset lookup ("2m")
    status, by_offset = _http_json(
        f"{base}/api/resolve",
        method="POST",
        body={"mode": "offset", "value": "2m"},
    )
    assert status == 200
    assert len(by_offset["passes"]) == 1

    # Wall clock lookup ("12:02" Europe/London BST == 11:02 UTC)
    status, by_clock = _http_json(
        f"{base}/api/resolve",
        method="POST",
        body={"mode": "clock", "value": "12:02"},
    )
    assert status == 200
    assert len(by_clock["passes"]) == 1

    # Corridor click selection along the indexed clip polyline
    coords = indexed["geojson"]["features"][0]["geometry"]["coordinates"]
    p_start = [coords[2][1], coords[2][0]]
    p_end = [coords[-3][1], coords[-3][0]]
    status, by_corridor = _http_json(
        f"{base}/api/resolve",
        method="POST",
        body={
            "mode": "corridor",
            "polyline": [p_start, p_end],
        },
    )
    assert status == 200
    assert len(by_corridor["passes"]) == 1

    # Raw binary clip upload without deprecated cgi module
    status, uploaded = _http_json(
        f"{base}/api/upload-clip",
        method="POST",
        body=b"synthetic-mp4-stream-payload",
        headers={
            "Content-Type": "application/octet-stream",
            "X-Filename": "GRMS0099.MP4",
        },
    )
    assert status == 200
    saved = Path(uploaded["saved_path"])
    assert saved.name == "GRMS0099.MP4"
    assert saved.read_bytes() == b"synthetic-mp4-stream-payload"


def test_studio_human_review_and_report_workflow(studio_env):
    base = studio_env["base_url"]
    run_dir: Path = studio_env["run_dir"]

    status, run = _http_json(
        f"{base}/api/open-run",
        method="POST",
        body={"path": str(run_dir)},
    )
    assert status == 200
    run_id = run["run_id"]
    assert len(run["observations"]) == 2
    assert run["observations"][0]["review_status"] == "unreviewed"
    # The review pane draws each candidate's own box; no result file means none.
    assert run["observations"][0]["box_xywh_normalized"] == [0.4, 0.6, 0.2, 0.1]
    assert "box_xywh_normalized" not in run["observations"][1]

    # Serve evidence PNG & reject path traversal
    overlay_url = run["observations"][0]["overlay_url"]
    with urllib.request.urlopen(f"{base}{overlay_url}", timeout=5) as resp:
        assert resp.status == 200
        assert resp.read().startswith(b"\x89PNG")

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(
            f"{base}/runs/{run_id}/files/..%2Fobservations.json", timeout=5
        )
    assert exc_info.value.code == 404
    exc_info.value.close()

    # Human confirms #0 and rejects #1
    status, rev0 = _http_json(
        f"{base}/api/runs/{run_id}/review",
        method="POST",
        body={"object_id": 0, "review_status": "confirmed"},
    )
    assert status == 200
    assert rev0["review_status"] == "confirmed"

    status, rev1 = _http_json(
        f"{base}/api/runs/{run_id}/review",
        method="POST",
        body={"object_id": 1, "review_status": "rejected"},
    )
    assert status == 200
    assert rev1["review_status"] == "rejected"

    decisions = json.loads((run_dir / "review-decisions.json").read_text())
    assert decisions == {"0": "confirmed", "1": "rejected"}

    # Re-opening preserves human decisions while keeping raw observations untouched
    _, reopened = _http_json(
        f"{base}/api/open-run",
        method="POST",
        body={"path": str(run_dir)},
    )
    statuses = {o["object_id"]: o["review_status"] for o in reopened["observations"]}
    assert statuses == {0: "confirmed", 1: "rejected"}

    # Draft deterministic summary report and save human edit
    status, drafted = _http_json(
        f"{base}/api/runs/{run_id}/draft-report",
        method="POST",
        body={"vlm_mode": "none"},
    )
    assert status == 200
    assert "RoadLens Hazard Summary Report (Draft)" in drafted["summary_report"]

    edited = drafted["summary_report"] + "\nHuman note: verified cavity on nearside.\n"
    status, saved_res = _http_json(
        f"{base}/api/runs/{run_id}/save-report",
        method="POST",
        body={"summary_report": edited},
    )
    assert status == 200
    assert saved_res["saved"] is True
    assert (run_dir / "summary-report.md").read_text(encoding="utf-8") == edited
