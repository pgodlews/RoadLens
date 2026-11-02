from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest
from PIL import Image

from roadlens.vlm import (
    LocalGlimmerBackend,
    RemoteVlmBackend,
    format_deterministic_summary_report,
    prepare_image_data_url,
    triage_run_directory,
    validate_and_clip_box,
)


def test_validate_and_clip_box_handles_out_of_bounds_and_road_crop():
    # Outbound full-frame box exceeding 1.0 is clipped cleanly to [0, 1].
    clipped = validate_and_clip_box([0.65, 0.70, 0.50, 0.45])
    assert clipped == pytest.approx((0.65, 0.70, 0.35, 0.30))

    # Road-crop [0.42..0.94] vertical coordinate remapping.
    remapped = validate_and_clip_box([0.2, 0.5, 0.3, 0.25], y_offset=0.42, y_scale=0.52)
    assert remapped == pytest.approx((0.2, 0.68, 0.3, 0.13))

    assert validate_and_clip_box([0.5, 0.5, 0.0, 0.2]) is None
    assert validate_and_clip_box([1.2, 0.2, 0.3, 0.3]) is None
    assert validate_and_clip_box("invalid") is None


def test_remote_vlm_redacts_garmin_gps_banner_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("ROADLENS_VLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_API_KEY"):
        RemoteVlmBackend()

    img_path = tmp_path / "frame.png"
    arr = np.full((200, 300, 3), 240, dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

    data_url, _, _ = prepare_image_data_url(
        img_path, road_crop=False, redact_gps_banner=True
    )
    decoded = Image.open(
        io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))
    ).convert("RGB")
    pixels = np.asarray(decoded)
    assert pixels[50, 150].mean() > 200
    assert pixels[-2, 150].mean() < 10  # Bottom Garmin GPS banner is blacked out.

    captured: list[dict] = []

    def fake_transport(base_url, api_key, payload):
        captured.append({"base_url": base_url, "api_key": api_key, "payload": payload})
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "scene_summary": "Rural lane with fallen branch.",
                                "observations": [
                                    {
                                        "hazard_type": "fallen tree branch",
                                        "classification": "hazard_candidate",
                                        "confidence": "high",
                                        "description": "Large bough across left lane.",
                                        "box_xywh_normalized": [0.1, 0.5, 0.95, 0.3],
                                    }
                                ],
                            }
                        )
                    }
                }
            ]
        }

    remote = RemoteVlmBackend(
        api_key="test-key",
        base_url="https://api.meta.ai/v1",
        model="Muse-Glimmer",
        transport=fake_transport,
    )
    res = remote.discover_hazards(
        img_path, hazard_query="fallen tree branch", road_crop=False
    )
    assert res["gps_banner_redacted"] is True
    assert len(res["findings"]) == 1
    assert res["findings"][0]["box_xywh_normalized"] == pytest.approx(
        [0.1, 0.5, 0.9, 0.3]
    )
    assert captured[0]["api_key"] == "test-key"


def test_local_glimmer_triage_preserves_unreviewed_hitl_status(tmp_path):
    overlay = tmp_path / "overlay-000000.png"
    frame = tmp_path / "frame-000000.png"
    Image.fromarray(np.full((120, 160, 3), 128, dtype=np.uint8)).save(overlay)
    Image.fromarray(np.full((120, 160, 3), 140, dtype=np.uint8)).save(frame)
    (tmp_path / "result-000000.json").write_text(
        json.dumps(
            {
                "detections": [
                    {"object_id": 0, "box_xywh_normalized": [0.2, 0.4, 0.3, 0.2]}
                ]
            }
        )
    )
    obs = [
        {
            "object_id": 0,
            "prompt": "pothole",
            "review_status": "unreviewed",
            "best_frame_index": 0,
            "best_source_seconds": 12.4,
            "score": 0.78,
            "latitude": 53.85,
            "longitude": -1.54,
            "location_source": "garmin_camera_gps",
        }
    ]
    (tmp_path / "observations.json").write_text(json.dumps(obs))

    calls: list[dict] = []

    def fake_local_transport(base_url, api_key, payload):
        calls.append(payload)
        if payload.get("response_format"):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "classification": "sealed_repair",
                                    "confidence": "medium",
                                    "hazard_type": "pothole",
                                    "lane_position": "centre",
                                    "visual_description": "Flush dark tarmac patch.",
                                    "uncertainty_note": "Single camera angle.",
                                }
                            )
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": ("## Summary\nCandidate #0 is a sealed repair.")
                    }
                }
            ]
        }

    backend = LocalGlimmerBackend(
        server_url="http://127.0.0.1:9931/v1",
        transport=fake_local_transport,
    )
    summary = triage_run_directory(
        tmp_path,
        backend,
        street_name="Sample Moor Road",
        place_name="Sampleton",
    )
    assert summary["triaged_candidates"] == 1
    assert calls[0]["top_k"] == 64
    assert calls[0]["temperature"] == 1.0

    updated = json.loads((tmp_path / "observations.json").read_text())
    # Crucial bookended HITL check: VLM triage must never auto-confirm or auto-reject!
    assert updated[0]["review_status"] == "unreviewed"
    assert updated[0]["vlm_triage"]["classification"] == "sealed_repair"

    report_text = (tmp_path / "summary-report.md").read_text()
    assert "DRAFT FOR HUMAN REVIEW" in report_text

    det_report = format_deterministic_summary_report(
        updated, street_name="Sample Moor Road", place_name="Sampleton"
    )
    assert "camera GPS" in det_report
    assert "sealed_repair" in det_report
