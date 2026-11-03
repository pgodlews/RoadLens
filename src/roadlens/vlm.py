"""Dual VLM backend (local Muse Glimmer via llama.cpp + remote OpenAI-compatible API).

Supports:
1. Open-vocabulary hazard discovery on road-cropped or full frames with coordinate
   validation/clipping to `[0, 1]`.
2. Candidate triage on SAM 3.1 overlays/frames (separating actionable cavities,
   edge breakups, or obstructions from sealed asphalt patches and organic debris)
   while keeping `review_status == "unreviewed"` for human sign-off.
3. Grounded per-candidate captions and human-editable council/FixMyStreet summary
   reports that never invent physical dimensions or auto-submit externally.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROAD_CROP_TOP = 0.42
ROAD_CROP_BOTTOM = 0.94
GARMIN_BANNER_TOP = 0.935

VALID_CLASSIFICATIONS = frozenset(
    {"hazard_candidate", "sealed_repair", "surface_debris_or_stain", "uncertain"}
)
VALID_CONFIDENCES = frozenset({"low", "medium", "high"})


@dataclass(frozen=True)
class VlmFinding:
    hazard_type: str
    classification: str
    confidence: str
    description: str
    box_xywh_normalized: tuple[float, float, float, float] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hazard_type": self.hazard_type,
            "classification": self.classification,
            "confidence": self.confidence,
            "description": self.description,
            "box_xywh_normalized": (
                None
                if self.box_xywh_normalized is None
                else [round(v, 6) for v in self.box_xywh_normalized]
            ),
        }


def validate_and_clip_box(
    box: Any,
    y_offset: float = 0.0,
    y_scale: float = 1.0,
) -> tuple[float, float, float, float] | None:
    """Validate `[x, y, w, h]`, remap optional vertical crop, and clip to `[0, 1]`."""
    if not isinstance(box, list | tuple) or len(box) != 4:
        return None
    try:
        rx, ry, rw, rh = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (rx, ry, rw, rh)) or rw <= 0 or rh <= 0:
        return None
    fy0 = y_offset + ry * y_scale
    fy1 = y_offset + (ry + rh) * y_scale
    x0 = max(0.0, min(1.0, rx))
    y0 = max(0.0, min(1.0, fy0))
    x1 = max(0.0, min(1.0, rx + rw))
    y1 = max(0.0, min(1.0, fy1))
    if x1 - x0 <= 1e-4 or y1 - y0 <= 1e-4:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def prepare_image_data_url(
    image_path: Path,
    *,
    road_crop: bool = False,
    redact_gps_banner: bool = False,
    max_width: int = 1344,
) -> tuple[str, float, float]:
    """Encode an image as a JPEG data URL, optionally cropping road or masking GPS."""
    with Image.open(image_path) as raw:
        img = raw.convert("RGB")
    width, height = img.size
    y_offset = 0.0
    y_scale = 1.0
    if road_crop:
        top_px = int(round(height * ROAD_CROP_TOP))
        bot_px = int(round(height * ROAD_CROP_BOTTOM))
        img = img.crop((0, top_px, width, bot_px))
        y_offset = ROAD_CROP_TOP
        y_scale = ROAD_CROP_BOTTOM - ROAD_CROP_TOP
    elif redact_gps_banner:
        draw = ImageDraw.Draw(img)
        banner_y = int(round(height * GARMIN_BANNER_TOP))
        draw.rectangle((0, banner_y, width, height), fill=(0, 0, 0))

    if max_width > 0 and img.width > max_width:
        target_h = max(1, round(img.height * max_width / img.width))
        img = img.resize((max_width, target_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", y_offset, y_scale


def build_triage_composite_data_url(
    raw_frame_path: Path,
    box_xywh_normalized: Sequence[float] | None = None,
    *,
    redact_gps_banner: bool = False,
) -> str:
    """Build a non-occluding context + native 1:1 crop image for VLM triage.

    Passing filled mask overlays (`overlay-XXXXXX.png`) obscures interior asphalt
    texture and can mislead a VLM into seeing coloured paint. Instead, draw a
    padded outer rectangle on the unmodified `frame-XXXXXX.png` alongside an
    unmasked 1:1 close-up crop of the candidate region.
    """
    with Image.open(raw_frame_path) as raw:
        img = raw.convert("RGB")
    width, height = img.size
    if redact_gps_banner:
        banner_draw = ImageDraw.Draw(img)
        banner_draw.rectangle(
            (0, int(round(height * GARMIN_BANNER_TOP)), width, height),
            fill=(0, 0, 0),
        )

    clipped = validate_and_clip_box(box_xywh_normalized)
    if clipped is None:
        url, _, _ = prepare_image_data_url(
            raw_frame_path, road_crop=False, redact_gps_banner=redact_gps_banner
        )
        return url

    bx, by, bw, bh = clipped
    x0 = int(round(bx * width))
    y0 = int(round(by * height))
    x1 = int(round((bx + bw) * width))
    y1 = int(round((by + bh) * height))

    # Build unmasked native close-up crop with generous surrounding road context.
    margin_x = max(64, int(round((x1 - x0) * 0.65)))
    margin_y = max(64, int(round((y1 - y0) * 0.65)))
    cx0 = max(0, x0 - margin_x)
    cy0 = max(0, y0 - margin_y)
    cx1 = min(width, x1 + margin_x)
    cy1 = min(int(round(height * GARMIN_BANNER_TOP)), y1 + margin_y)
    if cy1 <= cy0 + 8:
        cy1 = min(height, y1 + margin_y)
    crop = img.crop((cx0, cy0, cx1, cy1))

    # Draw outer indicator rectangle on road-context copy outside the defect edges.
    context_img = img.copy()
    draw = ImageDraw.Draw(context_img)
    pad = max(12, int(round(min(width, height) * 0.008)))
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(width - 1, x1 + pad), min(height - 1, y1 + pad)
    stroke = max(3, int(round(width / 450)))
    draw.rectangle((rx0, ry0, rx1, ry1), outline=(255, 64, 64), width=stroke)

    target_h = 720
    ctx_w = max(1, round(context_img.width * target_h / context_img.height))
    context_resized = context_img.resize((ctx_w, target_h), Image.Resampling.LANCZOS)
    crop_w = max(1, round(crop.width * target_h / max(1, crop.height)))
    crop_w = min(960, max(320, crop_w))
    crop_resized = crop.resize((crop_w, target_h), Image.Resampling.LANCZOS)

    gap = 12
    canvas = Image.new("RGB", (ctx_w + gap + crop_w, target_h), (20, 20, 20))
    canvas.paste(context_resized, (0, 0))
    canvas.paste(crop_resized, (ctx_w + gap, 0))

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(
        f"VLM response did not contain a valid JSON object: {text[:200]!r}"
    )


def _sanitize_classification(raw: Any) -> str:
    val = str(raw or "uncertain").strip().lower()
    if val in VALID_CLASSIFICATIONS:
        return val
    if "repair" in val or "patch" in val:
        return "sealed_repair"
    if "debris" in val or "stain" in val or "manure" in val or "mud" in val:
        return "surface_debris_or_stain"
    if "hazard" in val or "pothole" in val or "defect" in val:
        return "hazard_candidate"
    return "uncertain"


def _sanitize_confidence(raw: Any) -> str:
    val = str(raw or "low").strip().lower()
    return val if val in VALID_CONFIDENCES else "low"


DISCOVERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene_summary": {"type": "string"},
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hazard_type": {"type": "string"},
                    "classification": {
                        "type": "string",
                        "enum": sorted(VALID_CLASSIFICATIONS),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": sorted(VALID_CONFIDENCES),
                    },
                    "description": {"type": "string"},
                    "box_xywh_normalized": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "required": [
                    "hazard_type",
                    "classification",
                    "confidence",
                    "description",
                    "box_xywh_normalized",
                ],
            },
        },
    },
    "required": ["scene_summary", "observations"],
}

TRIAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": sorted(VALID_CLASSIFICATIONS),
        },
        "confidence": {
            "type": "string",
            "enum": sorted(VALID_CONFIDENCES),
        },
        "hazard_type": {"type": "string"},
        "lane_position": {"type": "string"},
        "visual_description": {"type": "string"},
        "uncertainty_note": {"type": "string"},
    },
    "required": [
        "classification",
        "confidence",
        "hazard_type",
        "lane_position",
        "visual_description",
        "uncertainty_note",
    ],
}


class BaseVlmBackend:
    """Shared multimodal prompt/validation logic for local Glimmer and remote APIs."""

    name: str = "base"
    model_name: str = "unknown"
    is_local: bool = False

    def __enter__(self) -> BaseVlmBackend:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        return None

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1200,
    ) -> str:
        raise NotImplementedError

    def discover_hazards(
        self,
        image_path: Path,
        hazard_query: str = "pothole, cracking, road debris, or obstruction",
        *,
        road_crop: bool = True,
        redact_gps_banner: bool | None = None,
    ) -> dict[str, Any]:
        redact = (not self.is_local) if redact_gps_banner is None else redact_gps_banner
        data_url, y_off, y_scale = prepare_image_data_url(
            image_path,
            road_crop=road_crop,
            redact_gps_banner=redact,
        )
        prompt = (
            "You are inspecting a forward bicycle-camera frame for post-ride "
            f"road-hazard discovery. Target query: {hazard_query}. "
            "Report only visible evidence on or obstructing the carriageway. "
            "Carefully distinguish intact sealed tarmac/utility repairs and harmless "
            "mud/manure/stains from open cavities, broken patch edges, flooding, or "
            "obstructions such as fallen branches. Never invent physical depth or "
            "width measurements. Provide normalized [x, y, width, height] boxes in "
            "[0, 1] within this image. Return JSON matching the requested schema."
        )
        raw_text = self.chat_completion(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            json_schema=DISCOVERY_SCHEMA,
        )
        parsed = _extract_json_object(raw_text)
        findings: list[VlmFinding] = []
        for item in parsed.get("observations") or []:
            if not isinstance(item, dict):
                continue
            clipped = validate_and_clip_box(
                item.get("box_xywh_normalized"), y_offset=y_off, y_scale=y_scale
            )
            if clipped is None:
                continue
            findings.append(
                VlmFinding(
                    hazard_type=str(item.get("hazard_type") or hazard_query).strip(),
                    classification=_sanitize_classification(item.get("classification")),
                    confidence=_sanitize_confidence(item.get("confidence")),
                    description=str(item.get("description") or "").strip(),
                    box_xywh_normalized=clipped,
                )
            )
        return {
            "backend": self.name,
            "model": self.model_name,
            "road_crop": road_crop,
            "gps_banner_redacted": redact or road_crop,
            "scene_summary": str(parsed.get("scene_summary") or "").strip(),
            "findings": [f.to_dict() for f in findings],
        }

    def triage_observation(
        self,
        observation: dict[str, Any],
        evidence_image_path: Path,
        *,
        candidate_box_xywh: Sequence[float] | None = None,
        redact_gps_banner: bool | None = None,
    ) -> dict[str, Any]:
        """Classify a SAM candidate frame without altering `review_status`."""
        redact = (not self.is_local) if redact_gps_banner is None else redact_gps_banner
        box = candidate_box_xywh or observation.get("box_xywh_normalized")
        if box is not None:
            data_url = build_triage_composite_data_url(
                evidence_image_path,
                box,
                redact_gps_banner=redact,
            )
            framing_hint = (
                "Left panel shows the road scene with a red outer rectangle around "
                "the SAM 3.1 candidate; right panel shows an unmasked 1:1 close-up "
                "crop of that candidate region. "
            )
        else:
            data_url, _, _ = prepare_image_data_url(
                evidence_image_path,
                road_crop=False,
                redact_gps_banner=redact,
            )
            framing_hint = "Inspect the highlighted candidate region and road surface. "
        concept = observation.get("prompt", "pothole")
        prompt = (
            "You are triaging an unreviewed road-hazard candidate segmented by SAM 3.1 "
            f"for concept '{concept}'. {framing_hint}Classify it as one of:\n"
            "- 'hazard_candidate': open pothole/cavity, sunken/fretted patch edge, "
            "severe cracking, floodwater, or obstruction (e.g. branch/debris).\n"
            "- 'sealed_repair': flush/intact tarmac patch, trench reinstatement, "
            "or tar overbanding without clear open depression.\n"
            "- 'surface_debris_or_stain': mud, leaves, horse droppings, shadow, "
            "or surface stain.\n"
            "- 'uncertain': ambiguous distant or motion-blurred feature.\n"
            "Describe visible appearance and lane position (e.g. nearside/left wheel "
            "track, centre, offside) without inventing cm/mm dimensions. Return JSON."
        )
        raw_text = self.chat_completion(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            json_schema=TRIAGE_SCHEMA,
        )
        parsed = _extract_json_object(raw_text)
        return {
            "backend": self.name,
            "model": self.model_name,
            "gps_banner_redacted": redact,
            "classification": _sanitize_classification(parsed.get("classification")),
            "confidence": _sanitize_confidence(parsed.get("confidence")),
            "hazard_type": str(parsed.get("hazard_type") or concept).strip(),
            "lane_position": str(parsed.get("lane_position") or "unspecified").strip(),
            "visual_description": str(parsed.get("visual_description") or "").strip(),
            "uncertainty_note": str(parsed.get("uncertainty_note") or "").strip(),
        }

    def draft_summary_report(
        self,
        observations: Sequence[dict[str, Any]],
        *,
        street_name: str | None = None,
        place_name: str | None = None,
        query_summary: str | None = None,
    ) -> str:
        """Draft a human-editable report from confirmed/triaged observations."""
        sanitized_items = []
        for obs in observations:
            sanitized_items.append(
                {
                    "object_id": obs.get("object_id"),
                    "prompt": obs.get("prompt"),
                    "review_status": obs.get("review_status", "unreviewed"),
                    "best_source_seconds": obs.get("best_source_seconds"),
                    "sam_score": obs.get("score"),
                    "latitude": obs.get("latitude"),
                    "longitude": obs.get("longitude"),
                    "location_source": obs.get("location_source"),
                    "vlm_triage": obs.get("vlm_triage"),
                    "size_estimate": _size_for_report(obs),
                }
            )
        status_counts = {
            status: sum(item["review_status"] == status for item in sanitized_items)
            for status in ("confirmed", "unreviewed", "rejected")
        }
        context = {
            "street_name": street_name,
            "place_name": place_name,
            "query_summary": query_summary,
            "candidate_count": len(sanitized_items),
            "human_review_counts": status_counts,
            "items": sanitized_items,
        }
        prompt = (
            "Draft a concise, factual Markdown road-defect summary report for a "
            "cyclist to review, edit, and manually copy into a local council or "
            "FixMyStreet report.\n"
            "Strict rules:\n"
            "1. Begin with a clear notice that this is a DRAFT requiring human review "
            "and manual submission (never auto-submitted).\n"
            "2. State clearly that any coordinates are Garmin camera GPS positions at "
            "the moment of capture, not surveyed defect coordinates.\n"
            "3. Never invent physical depth, width, or legal severity ratings. "
            "If an item has a `size_estimate` string, you may quote it verbatim as "
            "'approx. surface footprint (camera geometry estimate)'; never state "
            "depth, and never say it meets an intervention threshold.\n"
            "4. `review_status` is the HUMAN decision; `vlm_triage.classification` is "
            "only a model opinion. Call an item 'confirmed' or 'human-confirmed' only "
            "if its review_status is 'confirmed'. Never merge the two in one heading.\n"
            "5. Group items by review_status first (confirmed, then unreviewed; omit "
            "rejected items except in a count). Within unreviewed items, label the "
            "VLM classification as 'model triage' and list hazard_candidate before "
            "sealed_repair, surface_debris_or_stain, and uncertain.\n\n"
            f"Structured input:\n{json.dumps(context, indent=2)}"
        )
        body = self.chat_completion(
            [{"role": "user", "content": prompt}],
            json_schema=None,
            max_tokens=1600,
        ).strip()
        header = (
            "> **DRAFT FOR HUMAN REVIEW — NOT AUTOMATICALLY SUBMITTED**\n"
            "> Verify all visual evidence, descriptions, and camera GPS locations "
            "before submitting to any highway authority.\n\n"
        )
        if "DRAFT" not in body.upper():
            return header + body
        return body


def _size_for_report(obs: dict[str, Any]) -> str | None:
    size = obs.get("size_estimate") or {}
    return size.get("display") if size.get("status") == "estimated" else None


def format_deterministic_summary_report(
    observations: Sequence[dict[str, Any]],
    *,
    street_name: str | None = None,
    place_name: str | None = None,
    query_summary: str | None = None,
) -> str:
    """Fallback deterministic Markdown report requiring zero VLM/GPU calls."""
    location_label = (
        ", ".join(p for p in (street_name, place_name) if p) or "Unspecified road"
    )
    confirmed = [o for o in observations if o.get("review_status") == "confirmed"]
    unreviewed = [o for o in observations if o.get("review_status") == "unreviewed"]
    rejected = [o for o in observations if o.get("review_status") == "rejected"]

    lines = [
        "# RoadLens Hazard Summary Report (Draft)",
        "",
        "> **DRAFT FOR HUMAN REVIEW — NOT AUTOMATICALLY SUBMITTED**",
        (
            "> Coordinates below are **camera GPS positions** at video capture time, "
            "not surveyed defect coordinates. Sizes are approximate road-surface "
            "footprints from camera geometry, not measurements; depth is not "
            "estimated."
        ),
        "",
        f"- **Road / Area:** {location_label}",
    ]
    if query_summary:
        lines.append(f"- **Inspection Scope:** {query_summary}")
    lines.append(
        f"- **Review Counts:** {len(confirmed)} confirmed · "
        f"{len(unreviewed)} unreviewed · {len(rejected)} rejected"
    )
    lines.append("")

    target_list = confirmed if confirmed else unreviewed
    heading = (
        "## Human-Confirmed Hazards for Reporting"
        if confirmed
        else "## Unreviewed Candidates (Confirm in Review UI Before Reporting)"
    )
    lines.append(heading)
    lines.append("")
    if not target_list:
        lines.append("No confirmed or pending hazard candidates in this selection.")
    for obs in target_list:
        oid = obs.get("object_id")
        concept = obs.get("prompt", "hazard")
        sec = obs.get("best_source_seconds")
        lat, lon = obs.get("latitude"), obs.get("longitude")
        coord = (
            f"{lat:.5f}, {lon:.5f} (camera GPS)"
            if isinstance(lat, int | float) and isinstance(lon, int | float)
            else "unavailable"
        )
        triage = obs.get("vlm_triage") or {}
        desc = (
            triage.get("visual_description")
            or "Visual overlay attached for human inspection."
        )
        lane = triage.get("lane_position")
        cls = triage.get("classification")
        lines.append(f"### Candidate #{oid} — {concept}")
        if sec is not None:
            lines.append(f"- **Clip offset:** {float(sec):.2f} s")
        lines.append(f"- **Camera GPS:** {coord}")
        if cls:
            conf_str = triage.get("confidence", "low")
            lines.append(f"- **VLM triage:** `{cls}` (confidence: `{conf_str}`)")
        if lane:
            lines.append(f"- **Position:** {lane}")
        size = _size_for_report(obs)
        if size:
            lines.append(f"- **Approx. size (camera geometry estimate):** {size}")
        lines.append(f"- **Observation:** {desc}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _post_openai_chat(
    base_url: str,
    api_key: str | None,
    payload: dict[str, Any],
    timeout: float = 180.0,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"VLM HTTP {err.code} from {url}: {detail}") from err


class RemoteVlmBackend(BaseVlmBackend):
    """OpenAI-compatible remote multimodal backend (Meta Model API or custom)."""

    name = "remote"
    is_local = False

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        transport: Callable[[str, str | None, dict[str, Any]], dict[str, Any]]
        | None = None,
    ):
        self._api_key = (
            api_key
            or os.environ.get("ROADLENS_VLM_API_KEY")
            or os.environ.get("MODEL_API_KEY")
        )
        if not self._api_key:
            raise RuntimeError(
                "MODEL_API_KEY or ROADLENS_VLM_API_KEY is required for remote VLM"
            )
        self.base_url = (
            base_url
            or os.environ.get("ROADLENS_VLM_API_URL")
            or os.environ.get("ROADLENS_VLM_BASE_URL")  # Legacy alias.
            or "https://api.meta.ai/v1"
        )
        self.model_name = (
            model or os.environ.get("ROADLENS_VLM_MODEL") or "muse-spark-1.3"
        )
        # Muse Spark reasons before answering and counts those tokens against
        # max_tokens (about 3.3k per triage call in live tests), so reserve room.
        self.reasoning_tokens = int(
            os.environ.get("ROADLENS_VLM_REASONING_TOKENS") or 8000
        )
        self.reasoning_effort = os.environ.get("ROADLENS_VLM_REASONING_EFFORT")
        self._transport = transport or _post_openai_chat

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1200,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens + self.reasoning_tokens,
            "temperature": 0.2,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "roadlens_vlm_output",
                    "schema": json_schema,
                },
            }
        response = self._transport(self.base_url, self._api_key, payload)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("Remote VLM returned no completion choices")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            if choices[0].get("finish_reason") == "length":
                raise RuntimeError(
                    "Remote VLM used its whole token budget before answering; "
                    "raise ROADLENS_VLM_REASONING_TOKENS"
                )
            raise RuntimeError("Remote VLM returned empty message content")
        return content


def _discover_default_glimmer_files() -> tuple[Path | None, Path | None]:
    env_model = os.environ.get("ROADLENS_GLIMMER_MODEL") or os.environ.get(
        "ROADLENS_GLIMMER_GGUF"
    )
    env_mmproj = os.environ.get("ROADLENS_GLIMMER_MMPROJ")
    if env_model and env_mmproj:
        return Path(env_model), Path(env_mmproj)
    hub_roots: list[Path] = []
    for env_key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(env_key):
            hub_roots.append(Path(os.environ[env_key]))
    if os.environ.get("HF_HOME"):
        hub_roots.append(Path(os.environ["HF_HOME"]) / "hub")
    hub_roots.append(Path.home() / ".cache/huggingface/hub")
    candidates = [
        hub / "models--meta-models--Muse-Glimmer-30B-GGUF/snapshots"
        for hub in hub_roots
    ]
    for root in candidates:
        if not root.is_dir():
            continue
        for snap in sorted(root.iterdir()):
            models = sorted(snap.glob("Muse-Glimmer-*.gguf"))
            mmprojs = sorted(snap.glob("mmproj-*.gguf"))
            if models and mmprojs:
                return (
                    Path(env_model) if env_model else models[0],
                    Path(env_mmproj) if env_mmproj else mmprojs[0],
                )
    return (
        Path(env_model) if env_model else None,
        Path(env_mmproj) if env_mmproj else None,
    )


def _discover_llama_server() -> Path | None:
    env_bin = os.environ.get("ROADLENS_LLAMA_SERVER")
    if env_bin:
        return Path(env_bin)
    repo_root = Path(__file__).resolve().parents[2]
    local_bins = sorted(repo_root.glob(".tools/llama-*/llama-*/llama-server"))
    if local_bins:
        return local_bins[-1]
    return None


def _pick_free_localhost_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LocalGlimmerBackend(BaseVlmBackend):
    """Local Muse Glimmer 30B GGUF backend via `llama-server` on `127.0.0.1`."""

    name = "local"
    model_name = "Muse-Glimmer-30B-GGUF"
    is_local = True

    def __init__(
        self,
        *,
        server_url: str | None = None,
        model_path: Path | None = None,
        mmproj_path: Path | None = None,
        llama_server_bin: Path | None = None,
        reasoning_effort: str = "medium",
        ctx_size: int = 8192,
        gpu_layers: int = 99,
        startup_timeout: float = 120.0,
        transport: Callable[[str, str | None, dict[str, Any]], dict[str, Any]]
        | None = None,
    ):
        self._external_url = server_url or os.environ.get("ROADLENS_GLIMMER_SERVER_URL")
        def_model, def_mmproj = _discover_default_glimmer_files()
        self.model_path = model_path or def_model
        self.mmproj_path = mmproj_path or def_mmproj
        self.llama_server_bin = llama_server_bin or _discover_llama_server()
        self.reasoning_effort = reasoning_effort
        self.ctx_size = ctx_size
        self.gpu_layers = gpu_layers
        self.startup_timeout = startup_timeout
        self._transport = transport or _post_openai_chat
        self._process: subprocess.Popen[bytes] | None = None
        self._managed_url: str | None = None

        if not self._external_url:
            if not self.model_path or not self.model_path.is_file():
                raise FileNotFoundError(
                    "Local Muse Glimmer GGUF not found; set ROADLENS_GLIMMER_MODEL "
                    "or ROADLENS_GLIMMER_SERVER_URL"
                )
            if not self.mmproj_path or not self.mmproj_path.is_file():
                raise FileNotFoundError(
                    "Muse Glimmer mmproj GGUF not found; set ROADLENS_GLIMMER_MMPROJ"
                )
            if not self.llama_server_bin or not self.llama_server_bin.is_file():
                raise FileNotFoundError(
                    "llama-server binary not found; set ROADLENS_LLAMA_SERVER "
                    "or ROADLENS_GLIMMER_SERVER_URL"
                )

    def _ensure_server(self) -> str:
        if self._external_url:
            return self._external_url.rstrip("/")
        if (
            self._process is not None
            and self._process.poll() is None
            and self._managed_url
        ):
            return self._managed_url
        assert self.llama_server_bin is not None
        assert self.model_path is not None
        assert self.mmproj_path is not None

        port = _pick_free_localhost_port()
        env = os.environ.copy()
        lib_dirs = [str(self.llama_server_bin.parent)]
        for cudart in sorted(self.llama_server_bin.parent.parent.glob("cudart-*")):
            if cudart.is_dir():
                lib_dirs.append(str(cudart))
        if env.get("LD_LIBRARY_PATH"):
            lib_dirs.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs)

        cmd = [
            str(self.llama_server_bin),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-m",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "-ngl",
            str(self.gpu_layers),
            "-c",
            str(self.ctx_size),
            "--reasoning-effort",
            self.reasoning_effort,
        ]
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        health_url = f"http://127.0.0.1:{port}/health"
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                stderr_tail = ""
                if self._process.stderr:
                    stderr_tail = self._process.stderr.read().decode(
                        "utf-8", errors="replace"
                    )[-500:]
                code = self._process.returncode
                raise RuntimeError(f"llama-server exited early ({code}): {stderr_tail}")
            try:
                with urllib.request.urlopen(health_url, timeout=2.0) as resp:
                    if resp.status == 200:
                        self._managed_url = f"http://127.0.0.1:{port}/v1"
                        return self._managed_url
            except Exception:
                time.sleep(0.4)
        self.close()
        raise TimeoutError("Timed out waiting for local Muse Glimmer llama-server")

    def close(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5.0)
            if self._process.stderr:
                self._process.stderr.close()
            self._process = None
            self._managed_url = None

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1200,
    ) -> str:
        base_url = self._ensure_server()
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "roadlens_glimmer_output",
                    "schema": json_schema,
                },
            }
        response = self._transport(base_url, None, payload)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("Local Muse Glimmer returned no completion choices")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Local Muse Glimmer returned empty message content")
        return content


def create_vlm_backend(name: str = "local", **kwargs: Any) -> BaseVlmBackend:
    if name == "local":
        return LocalGlimmerBackend(**kwargs)
    if name == "remote":
        return RemoteVlmBackend(**kwargs)
    raise ValueError(f"Unknown VLM backend: {name!r}")


def _detection_box_for_observation(
    detections: list[dict[str, Any]], object_id: Any
) -> Sequence[float] | None:
    """Return the box of the detection belonging to `object_id` in a shared frame."""
    for det in detections:
        if object_id is not None and det.get("object_id") == object_id:
            return det.get("box_xywh_normalized")
    # A lone detection without a usable ID is unambiguous; otherwise never guess.
    if len(detections) == 1 and object_id is None:
        return detections[0].get("box_xywh_normalized")
    return None


def triage_run_directory(
    output_dir: Path,
    vlm_backend: BaseVlmBackend,
    *,
    max_candidates: int | None = None,
    street_name: str | None = None,
    place_name: str | None = None,
    draft_with_vlm: bool = True,
) -> dict[str, Any]:
    """Triage SAM observations in `output_dir` and draft `summary-report.md`."""
    if max_candidates is not None and (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or max_candidates < 0
    ):
        raise ValueError("max_candidates must be a nonnegative integer or None")
    obs_path = output_dir / "observations.json"
    observations: list[dict[str, Any]] = json.loads(
        obs_path.read_text(encoding="utf-8")
    )
    selected = (
        observations if max_candidates is None else observations[:max_candidates]
    )
    triaged_count = 0
    for idx, obs in enumerate(selected):
        frame_idx = int(obs.get("best_frame_index", obs.get("object_id", idx)))
        overlay = output_dir / f"overlay-{frame_idx:06d}.png"
        frame = output_dir / f"frame-{frame_idx:06d}.png"
        result_json = output_dir / f"result-{frame_idx:06d}.json"
        box_xywh = None
        if result_json.is_file():
            dets = json.loads(result_json.read_text(encoding="utf-8")).get(
                "detections", []
            )
            box_xywh = _detection_box_for_observation(dets, obs.get("object_id"))
        # Prefer raw `frame-*.png` + outer box so mask fill never hides texture.
        image_path = frame if frame.is_file() else overlay
        if not image_path.is_file():
            continue
        obs["vlm_triage"] = vlm_backend.triage_observation(
            obs, image_path, candidate_box_xywh=box_xywh
        )
        # Preserve bookended HITL invariant: VLM triage never marks confirmed!
        if obs.get("review_status") not in ("confirmed", "rejected"):
            obs["review_status"] = "unreviewed"
        triaged_count += 1

    obs_path.write_text(json.dumps(observations, indent=2) + "\n", encoding="utf-8")
    # Draft from saved human decisions (Studio's `review-decisions.json`), if any.
    decisions_path = output_dir / "review-decisions.json"
    decisions: dict[str, str] = (
        json.loads(decisions_path.read_text(encoding="utf-8"))
        if decisions_path.is_file()
        else {}
    )

    def reviewed(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for idx, obs in enumerate(items):
            key = str(int(obs.get("object_id", idx)))
            status = decisions.get(key, obs.get("review_status"))
            out.append({**obs, "review_status": status})
        return out

    if draft_with_vlm and selected:
        report_md = vlm_backend.draft_summary_report(
            reviewed(selected),
            street_name=street_name,
            place_name=place_name,
        )
    else:
        report_md = format_deterministic_summary_report(
            reviewed(observations),
            street_name=street_name,
            place_name=place_name,
        )
    report_path = output_dir / "summary-report.md"
    report_path.write_text(report_md, encoding="utf-8")
    return {
        "triaged_candidates": triaged_count,
        "total_observations": len(observations),
        "summary_report": str(report_path),
    }
