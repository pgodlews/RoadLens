from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from roadlens.cli import build_parser
from roadlens.mcp_server import RoadLensMcpServer, serve_mcp_stdio
from roadlens.pmtiles_roads import (
    PmtilesStreetIndex,
    build_minimal_pmtiles_v3,
    encode_varint,
    latlon_to_zxy,
)
from roadlens.telemetry import TelemetryResult, TelemetrySample
from roadlens.vlm import VlmFinding


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


class _FakeVlmBackend:
    name = "FakeVlmBackend"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def discover_hazards(self, image_path: Path, prompt: str, road_crop: bool = True):
        return [
            VlmFinding(
                hazard_type=prompt,
                classification="hazard_candidate",
                confidence="high",
                description=f"Large {prompt} across nearside lane.",
                box_xywh_normalized=(0.25, 0.55, 0.35, 0.2),
            )
        ]

    def triage_observation(self, raw_frame_path: Path, prompt: str, **kwargs):
        return {
            "classification": "hazard_candidate",
            "confidence": "high",
            "visual_description": f"Depressed asphalt edge consistent with {prompt}.",
            "reasoning_summary": "Distinct shadow and broken tarmac margin.",
            "model": "fake-vlm",
        }

    def draft_summary_report(
        self, observations, *, street_name=None, place_name=None
    ) -> str:
        loc = ", ".join(filter(None, [street_name, place_name])) or "Unnamed corridor"
        return (
            f"# RoadLens Hazard Summary Report (Draft)\n\n"
            f"- Location: {loc}\n"
            f"- Candidates: {len(observations)} (camera GPS, unreviewed)\n"
        )


@pytest.fixture
def mcp_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    clip_path.write_bytes(b"fake-mp4")
    ckpt_path = tmp_path / "sam3.1_multiplex.pt"
    ckpt_path.write_bytes(b"ckpt")

    base_utc = datetime(2026, 7, 15, 11, 20, 0, tzinfo=UTC)  # 12:20 BST
    samples = tuple(
        TelemetrySample(
            recorded_utc=base_utc + timedelta(seconds=t),
            latitude=lat0 + (t / 900.0) * (lat1 - lat0),
            longitude=lon0 + (t / 900.0) * (lon1 - lon0),
            speed_mps=6.0,
            playback_seconds=float(t),
        )
        for t in range(0, 901, 5)
    )
    meta = {
        "duration": 900.0,
        "fps": 30.0,
        "width": 640,
        "height": 480,
        "creation_time": "2026-07-15T11:20:00Z",
    }
    monkeypatch.setattr("roadlens.ride.probe_video", lambda _p: meta)
    monkeypatch.setattr(
        "roadlens.ride.read_garmin_telemetry",
        lambda _p: TelemetryResult(samples=samples, rejected_samples=0),
    )
    monkeypatch.setattr("roadlens.mcp_server.probe_video", lambda _p: meta)
    monkeypatch.setattr(
        "roadlens.mcp_server.read_garmin_telemetry",
        lambda _p: TelemetryResult(samples=samples, rejected_samples=0),
    )
    monkeypatch.setattr(
        "roadlens.mcp_server.create_vlm_backend",
        lambda *_a, **_kw: _FakeVlmBackend(),
    )

    server = RoadLensMcpServer(
        workspace_dir=tmp_path / "mcp-work",
        pmtiles_path=pmtiles_path,
        checkpoint_path=ckpt_path,
    )
    return {
        "server": server,
        "clip_path": clip_path,
        "pmtiles_path": pmtiles_path,
        "ckpt_path": ckpt_path,
        "lat0": lat0,
        "lon0": lon0,
    }


def test_cli_mcp_parser(tmp_path: Path):
    pm = tmp_path / "uk.pmtiles"
    args = build_parser().parse_args(
        ["mcp", "--workspace", str(tmp_path / "w"), "--pmtiles", str(pm)]
    )
    assert args.command == "mcp"
    assert args.pmtiles == pm


def test_mcp_stdio_newline_and_content_length_framing(mcp_fixture, tmp_path: Path):
    # 1. Newline-delimited JSON-RPC + notification ignored cleanly
    req_lines = (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            }
        )
        + "\n"
        + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
        + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        + "\n"
    ).encode("utf-8")
    out_nl = io.BytesIO()
    serve_mcp_stdio(
        workspace_dir=tmp_path / "nl",
        pmtiles_path=mcp_fixture["pmtiles_path"],
        checkpoint_path=mcp_fixture["ckpt_path"],
        stdin=io.BytesIO(req_lines),
        stdout=out_nl,
    )
    responses = [
        json.loads(line)
        for line in out_nl.getvalue().decode("utf-8").splitlines()
        if line.strip()
    ]
    assert len(responses) == 2
    assert responses[0]["result"]["protocolVersion"] == "2024-11-05"
    tool_names = {t["name"] for t in responses[1]["result"]["tools"]}
    assert tool_names == {
        "index_ride_clips",
        "resolve_ride_segment",
        "discover_segment_hazards_vlm",
        "scan_segment_sam",
        "triage_run_candidates",
        "draft_summary_report",
    }

    # 2. Content-Length framed JSON-RPC
    ping_body = json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping"}).encode(
        "utf-8"
    )
    framed_in = io.BytesIO(
        f"Content-Length: {len(ping_body)}\r\n\r\n".encode("ascii") + ping_body
    )
    framed_out = io.BytesIO()
    serve_mcp_stdio(
        workspace_dir=tmp_path / "cl",
        pmtiles_path=mcp_fixture["pmtiles_path"],
        stdin=framed_in,
        stdout=framed_out,
    )
    raw_reply = framed_out.getvalue()
    assert raw_reply.startswith(b"Content-Length:")
    _, json_part = raw_reply.split(b"\r\n\r\n", 1)
    assert json.loads(json_part.decode("utf-8")) == {
        "jsonrpc": "2.0",
        "id": 99,
        "result": {},
    }


def test_mcp_end_to_end_agentic_workflows(
    mcp_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    server: RoadLensMcpServer = mcp_fixture["server"]
    clip_path: Path = mcp_fixture["clip_path"]

    # 1. Index ride clips
    res_index = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "index_ride_clips",
                "arguments": {"path": str(clip_path)},
            },
        }
    )
    assert res_index["result"]["isError"] is False
    indexed = res_index["result"]["structuredContent"]
    assert indexed["ride"]["primary_clip_count"] == 1

    # 2. Resolve all three user scenarios: street, ride offset, and wall clock
    for idx, query in enumerate(
        [
            {"mode": "street", "value": "Demo Moor Road"},
            {"mode": "offset", "value": "10m"},
            {"mode": "clock", "value": "12:30"},
        ],
        start=20,
    ):
        resp = server.handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": idx,
                "method": "tools/call",
                "params": {"name": "resolve_ride_segment", "arguments": query},
            }
        )
        data = resp["result"]["structuredContent"]
        assert data["match_count"] >= 1
        assert data["passes"][0]["street_name"] == "Demo Moor Road"
        assert data["passes"][0]["place_name"] == "Demo Village"

    # 3. Open-vocabulary VLM hazard discovery ("large fallen tree branch")
    monkeypatch.setattr(
        "roadlens.mcp_server.measure_frames",
        lambda _p, s, e: ([s + 1.0, min(e - 0.1, s + 4.0)], [120.0, 145.0]),
    )

    def _fake_save(source: Path, choices, output: Path, width: int = 0):
        output.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, _ in enumerate(choices):
            fp = output / f"{i:06d}.png"
            Image.new("RGB", (160, 120), color=(90, 90, 90)).save(fp)
            paths.append(fp)
        return paths

    monkeypatch.setattr("roadlens.mcp_server.save_selected_frames", _fake_save)

    disc_resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 30,
            "method": "tools/call",
            "params": {
                "name": "discover_segment_hazards_vlm",
                "arguments": {
                    "clip_path": str(clip_path),
                    "start_seconds": 580.0,
                    "duration_seconds": 20.0,
                    "prompt": "large fallen tree branch",
                    "road_crop": False,
                },
            },
        }
    )
    disc = disc_resp["result"]["structuredContent"]
    assert disc["review_status"] == "unreviewed"
    assert disc["frames_with_findings"] == 2
    assert disc["results"][0]["street_name"] == "Demo Moor Road"
    assert (
        disc["results"][0]["findings"][0]["hazard_type"] == "large fallen tree branch"
    )

    # 4. Targeted SAM scan (also ensures noisy stdout inside run_scan does not leak)
    scan_out = tmp_path / "sam-out"

    def _fake_run_scan(ns):
        print("NOISY SAM PROGRESS ON STDOUT")
        ns.output.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (160, 120), color=(80, 80, 80)).save(
            ns.output / "frame-000000.png"
        )
        (ns.output / "result-000000.json").write_text(
            json.dumps(
                {
                    "detections": [
                        {"box_xywh_normalized": [0.3, 0.6, 0.2, 0.15], "score": 0.78}
                    ]
                }
            ),
            encoding="utf-8",
        )
        (ns.output / "observations.json").write_text(
            json.dumps(
                [
                    {
                        "object_id": 0,
                        "prompt": ns.prompt,
                        "backend": "LocalSamBackend",
                        "first_source_seconds": ns.start,
                        "last_source_seconds": ns.start + 1.0,
                        "best_source_seconds": ns.start + 0.5,
                        "best_frame_index": 0,
                        "recorded_utc": "2026-07-15T11:25:00+00:00",
                        "score": 0.78,
                        "foreground_pixels": 4200,
                        "latitude": mcp_fixture["lat0"],
                        "longitude": mcp_fixture["lon0"],
                        "location_source": "camera_gps",
                        "review_status": "unreviewed",
                    }
                ]
            ),
            encoding="utf-8",
        )
        return {"status": "complete"}

    monkeypatch.setattr("roadlens.scan.run_scan", _fake_run_scan)

    scan_resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 40,
            "method": "tools/call",
            "params": {
                "name": "scan_segment_sam",
                "arguments": {
                    "clip_path": str(clip_path),
                    "start_seconds": 300.0,
                    "duration_seconds": 25.0,
                    "prompt": "pothole",
                    "output_dir": str(scan_out),
                },
            },
        }
    )
    scan_data = scan_resp["result"]["structuredContent"]
    assert scan_data["unreviewed_candidate_count"] == 1
    assert scan_data["review_status"] == "unreviewed"

    # 5. VLM triage + summary report draft preserving unreviewed guardrail
    triage_resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 50,
            "method": "tools/call",
            "params": {
                "name": "triage_run_candidates",
                "arguments": {"run_dir": str(scan_out), "vlm": "local"},
            },
        }
    )
    triage_data = triage_resp["result"]["structuredContent"]
    assert triage_data["classification_counts"] == {"hazard_candidate": 1}
    assert triage_data["street_name"] == "Demo Moor Road"
    assert triage_data["review_status"] == "unreviewed"

    report_resp = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 60,
            "method": "tools/call",
            "params": {
                "name": "draft_summary_report",
                "arguments": {"run_dir": str(scan_out), "vlm": "local"},
            },
        }
    )
    report_data = report_resp["result"]["structuredContent"]
    assert "Demo Moor Road, Demo Village" in report_data["summary_report_markdown"]
    assert "roadlens studio" in report_data["studio_review_command"]
