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
    _detection_box_for_observation,
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


def test_triage_selects_detection_matching_observation_object_id():
    dets = [
        {"object_id": 11, "box_xywh_normalized": [0.1, 0.5, 0.1, 0.1]},
        {"object_id": 22, "box_xywh_normalized": [0.6, 0.6, 0.2, 0.1]},
    ]
    assert _detection_box_for_observation(dets, 22) == [0.6, 0.6, 0.2, 0.1]
    assert _detection_box_for_observation(dets, 11) == [0.1, 0.5, 0.1, 0.1]
    # Unknown ID in a multi-detection frame falls back to the observation box.
    assert _detection_box_for_observation(dets, 33) is None
    assert _detection_box_for_observation([], 11) is None


def test_remote_vlm_uses_documented_endpoint_variable(monkeypatch):
    monkeypatch.setenv("ROADLENS_VLM_API_KEY", "test-key")
    monkeypatch.delenv("ROADLENS_VLM_BASE_URL", raising=False)
    monkeypatch.setenv("ROADLENS_VLM_API_URL", "https://vlm.example.test/v1")
    assert RemoteVlmBackend().base_url == "https://vlm.example.test/v1"

    monkeypatch.delenv("ROADLENS_VLM_API_URL")
    monkeypatch.setenv("ROADLENS_VLM_BASE_URL", "https://legacy.example.test/v1")
    assert RemoteVlmBackend().base_url == "https://legacy.example.test/v1"


class _RecordingBackend:
    def __init__(self):
        self.triaged: list[int] = []
        self.drafted: list[dict] | None = None

    def triage_observation(self, observation, evidence_image_path, **kwargs):
        self.triaged.append(observation["object_id"])
        return {"classification": "uncertain", "confidence": "low"}

    def draft_summary_report(self, observations, **kwargs):
        self.drafted = list(observations)
        return "# Draft\n"


def _write_two_candidate_run(directory):
    for index in (0, 1):
        Image.fromarray(np.full((60, 80, 3), 128, dtype=np.uint8)).save(
            directory / f"frame-{index:06d}.png"
        )
    obs = [
        {"object_id": i, "prompt": "pothole", "review_status": "unreviewed",
         "best_frame_index": i}
        for i in (0, 1)
    ]
    (directory / "observations.json").write_text(json.dumps(obs))


def test_triage_report_reflects_saved_human_review_decisions(tmp_path):
    _write_two_candidate_run(tmp_path)
    (tmp_path / "review-decisions.json").write_text(
        json.dumps({"0": "confirmed", "1": "rejected"})
    )
    backend = _RecordingBackend()
    triage_run_directory(tmp_path, backend)
    assert [o["review_status"] for o in backend.drafted] == ["confirmed", "rejected"]
    # Raw observations stay model output; decisions live in their own file.
    saved = json.loads((tmp_path / "observations.json").read_text())
    assert {o["review_status"] for o in saved} == {"unreviewed"}

    triage_run_directory(tmp_path, backend, draft_with_vlm=False)
    report = (tmp_path / "summary-report.md").read_text()
    assert "1 confirmed · 0 unreviewed · 1 rejected" in report


def test_triage_max_candidates_zero_and_negative(tmp_path):
    _write_two_candidate_run(tmp_path)
    backend = _RecordingBackend()
    summary = triage_run_directory(tmp_path, backend, max_candidates=0)
    assert summary["triaged_candidates"] == 0
    assert backend.triaged == [] and backend.drafted is None
    report = (tmp_path / "summary-report.md").read_text()
    assert "0 confirmed · 0 unreviewed · 0 rejected" in report
    assert "Candidate #" not in report

    triage_run_directory(tmp_path, backend, max_candidates=1)
    assert backend.triaged == [0] and len(backend.drafted) == 1
    with pytest.raises(ValueError, match="nonnegative"):
        triage_run_directory(tmp_path, backend, max_candidates=-1)


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_deterministic_report_uses_the_triage_selection(tmp_path, limit):
    _write_two_candidate_run(tmp_path)
    backend = _RecordingBackend()
    summary = triage_run_directory(
        tmp_path, backend, max_candidates=limit, draft_with_vlm=False
    )
    report = (tmp_path / "summary-report.md").read_text()
    assert f"0 confirmed · {limit} unreviewed · 0 rejected" in report
    assert report.count("### Candidate #") == limit
    assert len(backend.triaged) == summary["triaged_candidates"] == limit
    assert summary["total_observations"] == 2


def test_draft_prompt_separates_human_review_from_model_triage():
    captured: list[dict] = []

    def fake_transport(base_url, api_key, payload):
        captured.append(payload)
        return {"choices": [{"message": {"content": "# DRAFT\nNothing confirmed."}}]}

    backend = LocalGlimmerBackend(
        server_url="http://127.0.0.1:9931/v1", transport=fake_transport
    )
    backend.draft_summary_report(
        [
            {"object_id": 0, "review_status": "unreviewed",
             "vlm_triage": {"classification": "hazard_candidate"}},
            {"object_id": 1, "review_status": "confirmed"},
        ]
    )
    prompt = captured[0]["messages"][0]["content"]
    assert "human-confirmed / hazard_candidate" not in prompt
    assert "only a model opinion" in prompt
    assert '"confirmed": 1' in prompt and '"unreviewed": 1' in prompt
    assert "coordinates are camera GPS positions" in prompt
    assert "Garmin camera GPS" not in prompt


def test_reports_group_candidates_by_their_own_pass_and_road():
    observations = [
        {
            "object_id": index,
            "review_status": "unreviewed",
            "clip_name": clip,
            "pass_index": pass_index,
            "street_name": street,
            "place_name": place,
        }
        for index, pass_index, clip, street, place in (
            (0, 1, "one.mp4", "First Road", "First Place"),
            (1, 2, "two.mp4", "Second Road", "Second Place"),
            (2, 1, "one.mp4", "First Road", "First Place"),
        )
    ]
    report = format_deterministic_summary_report(
        observations, street_name="Wrong fallback road"
    )
    first_heading = "### Pass 1 · First Road, First Place · one.mp4"
    second_heading = "### Pass 2 · Second Road, Second Place · two.mp4"
    first_group, second_group = report.split(first_heading)[1].split(second_heading)
    assert "Candidate #0" in first_group and "Candidate #2" in first_group
    assert "Candidate #1" not in first_group
    assert "Candidate #1" in second_group and "Candidate #0" not in second_group
    assert "Wrong fallback road" not in report

    prompts = []

    def fake_transport(base_url, api_key, payload):
        prompts.append(payload["messages"][0]["content"])
        return {"choices": [{"message": {"content": "# Draft"}}]}

    backend = RemoteVlmBackend(api_key="test-key", transport=fake_transport)
    backend.draft_summary_report(observations)
    prompt = prompts[0]
    context = json.loads(prompt.split("Structured input:\n", 1)[1])
    for original, sanitized in zip(observations, context["items"], strict=True):
        for key in ("clip_name", "pass_index", "street_name", "place_name"):
            assert sanitized[key] == original[key]
    assert "group items by pass_index, clip_name, street_name and place_name" in prompt


def test_remote_vlm_reserves_reasoning_tokens(monkeypatch):
    monkeypatch.delenv("ROADLENS_VLM_MODEL", raising=False)
    monkeypatch.delenv("ROADLENS_VLM_REASONING_TOKENS", raising=False)
    monkeypatch.setenv("ROADLENS_VLM_REASONING_EFFORT", "low")
    payloads = []

    def fake_transport(base_url, api_key, payload):
        payloads.append(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    remote = RemoteVlmBackend(api_key="test-key", transport=fake_transport)
    assert remote.model_name == "muse-spark-1.3"
    assert remote.chat_completion([], max_tokens=1200) == "ok"
    assert payloads[0]["max_tokens"] == 9200
    assert payloads[0]["reasoning_effort"] == "low"


def test_remote_vlm_explains_exhausted_reasoning_budget():
    def fake_transport(base_url, api_key, payload):
        return {"choices": [{"finish_reason": "length", "message": {"content": None}}]}

    remote = RemoteVlmBackend(api_key="test-key", transport=fake_transport)
    with pytest.raises(RuntimeError, match="ROADLENS_VLM_REASONING_TOKENS"):
        remote.chat_completion([])


def test_reports_quote_size_estimates_as_approximate():
    obs = [
        {
            "object_id": 0,
            "prompt": "pothole",
            "review_status": "confirmed",
            "best_source_seconds": 586.6,
            "latitude": 53.87559,
            "longitude": -1.53964,
            "size_estimate": {"status": "estimated", "display": "≈ 65 × 15 cm"},
        },
        {
            "object_id": 1,
            "prompt": "pothole",
            "review_status": "confirmed",
            "size_estimate": {"status": "unavailable", "reason": "cut off"},
        },
    ]
    report = format_deterministic_summary_report(obs)
    assert "- **Approx. size (camera geometry estimate):** ≈ 65 × 15 cm" in report
    assert report.count("Approx. size") == 1
    assert "depth is not estimated" in report

    prompts = []

    def fake_transport(base_url, api_key, payload):
        prompts.append(payload["messages"][0]["content"])
        return {"choices": [{"message": {"content": "Draft"}}]}

    RemoteVlmBackend(api_key="k", transport=fake_transport).draft_summary_report(obs)
    assert '"size_estimate": "\\u2248 65 \\u00d7 15 cm"' in prompts[0]
    assert "never state depth" in prompts[0]
