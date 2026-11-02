"""Stdio Model Context Protocol (MCP) server for RoadLens post-ride investigation.

Exposes structured tools for Muse Glimmer and external coding/assistant agents to:
1. Index multi-clip Garmin ride footage (`index_ride_clips`).
2. Resolve natural-language street, ride-offset ("20m"), wall-clock ("12:30"), or
   corridor queries offline via local PMTiles (`resolve_ride_segment`).
3. Run open-vocabulary VLM hazard discovery on sharp distance-sampled frames
   (`discover_segment_hazards_vlm`, e.g. "large fallen tree branch").
4. Run targeted SAM 3.1 coarse/refined segmentation (`scan_segment_sam`), freeing
   GPU VRAM afterwards so sequential local VLM calls fit on one 24 GB GPU.
5. Triage SAM candidate overlays with non-occluding crops (`triage_run_candidates`).
6. Draft grounded Markdown summary reports (`draft_summary_report`).

Enforces bookended human-in-the-loop: every automated output remains
`review_status: "unreviewed"` until a human confirms it in `roadlens studio`, and no
tool submits anything to councils or external authorities.
"""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import sys
import time
import uuid
from argparse import Namespace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from roadlens.analysis import nearest_telemetry
from roadlens.media import probe_video
from roadlens.pmtiles_roads import PmtilesStreetIndex
from roadlens.ride import (
    HAZARD_PRESETS,
    RideIndex,
    index_ride,
    resolve_by_corridor,
    resolve_by_ride_offset,
    resolve_by_street_name,
    resolve_by_wall_clock,
)
from roadlens.sampling import select_frames, speed_at_frames
from roadlens.scan_media import measure_frames, save_selected_frames
from roadlens.telemetry import read_garmin_telemetry
from roadlens.vlm import (
    create_vlm_backend,
    format_deterministic_summary_report,
    triage_run_directory,
)

SUPPORTED_PROTOCOL_VERSIONS = ("2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL_VERSION = "2024-11-05"

HUMAN_REVIEW_NOTICE = (
    "All candidates remain `review_status: unreviewed` and coordinates reflect "
    "Garmin camera GPS (`camera_gps`), not surveyed defect positions. A human must "
    "visually confirm overlays in `roadlens studio` before exporting council GeoJSON."
)


def _release_torch_vram() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


MCP_TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "index_ride_clips",
        "description": (
            "Index a Garmin .MP4 clip or multi-clip ride directory, deduplicating "
            "overlapping saved event clips and returning duration, distance, clip "
            "windows, and built-in hazard presets."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to a Garmin .MP4 file or ride folder.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "resolve_ride_segment",
        "description": (
            "Resolve a natural-language street name ('Eccup Moor Road'), ride offset "
            "('20m' into the ride), local wall-clock time ('12:30'), or lat/lon "
            "corridor polyline into exact clip timestamps and street/place names "
            "100% offline using local OpenStreetMap PMTiles."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["street", "offset", "clock", "corridor"],
                    "description": "Segment lookup mode.",
                },
                "value": {
                    "type": "string",
                    "description": (
                        "Street name (mode=street), ride offset like '20m' or "
                        "'00:20:00' (mode=offset), or wall clock like '12:30' "
                        "(mode=clock)."
                    ),
                },
                "polyline": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "description": "List of [lat, lon] points when mode=corridor.",
                },
                "ride_path": {
                    "type": "string",
                    "description": "Optional clip/folder path if not yet indexed.",
                },
                "window_seconds": {
                    "type": "number",
                    "description": "Half-width window in seconds for offset/clock.",
                },
                "timezone_name": {
                    "type": "string",
                    "default": "Europe/London",
                    "description": "IANA timezone for wall-clock queries.",
                },
                "include_polyline": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include GPS track coordinates per pass.",
                },
            },
            "required": ["mode"],
        },
    },
    {
        "name": "discover_segment_hazards_vlm",
        "description": (
            "Extract distance-sampled sharp frames from a resolved clip window and "
            "use local Muse Glimmer or a remote OpenAI-compatible VLM to look for "
            "open-vocabulary hazards (e.g. 'large fallen tree branch', 'flooding', "
            "'pothole'). Findings are marked unreviewed for human sign-off."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "clip_path": {"type": "string"},
                "start_seconds": {"type": "number"},
                "duration_seconds": {"type": "number", "default": 15.0},
                "prompt": {
                    "type": "string",
                    "description": "Hazard description to search for.",
                },
                "vlm": {
                    "type": "string",
                    "enum": ["local", "remote"],
                    "default": "local",
                },
                "spacing_m": {"type": "number", "default": 4.0},
                "max_frames": {"type": "integer", "default": 8},
                "road_crop": {"type": "boolean", "default": True},
                "output_dir": {"type": "string"},
            },
            "required": ["clip_path", "start_seconds", "prompt"],
        },
    },
    {
        "name": "scan_segment_sam",
        "description": (
            "Run targeted SAM 3.1 distance-sampled video segmentation on a resolved "
            "clip window (`--prompt` e.g. 'pothole' or 'fallen tree branch'), "
            "saving candidate overlays and unreviewed `observations.json`."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "clip_path": {"type": "string"},
                "start_seconds": {"type": "number"},
                "duration_seconds": {"type": "number"},
                "prompt": {"type": "string", "default": "pothole"},
                "spacing_m": {"type": "number", "default": 1.0},
                "refine": {"type": "boolean", "default": True},
                "output_dir": {"type": "string"},
            },
            "required": ["clip_path", "start_seconds", "duration_seconds"],
        },
    },
    {
        "name": "triage_run_candidates",
        "description": (
            "Run local Muse Glimmer or remote VLM triage on a SAM scan output "
            "directory using non-occluding frame + 1:1 crop composites to separate "
            "hazard candidates from sealed asphalt repairs and surface debris while "
            "preserving `review_status: unreviewed`."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_dir": {"type": "string"},
                "vlm": {
                    "type": "string",
                    "enum": ["local", "remote"],
                    "default": "local",
                },
                "max_candidates": {"type": "integer"},
                "street_name": {"type": "string"},
                "place_name": {"type": "string"},
            },
            "required": ["run_dir"],
        },
    },
    {
        "name": "draft_summary_report",
        "description": (
            "Generate or refresh a grounded human-editable Markdown summary report "
            "(`summary-report.md`) for a scan directory using `local` Muse Glimmer, "
            "`remote` VLM, or deterministic `template`."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_dir": {"type": "string"},
                "vlm": {
                    "type": "string",
                    "enum": ["local", "remote", "template"],
                    "default": "local",
                },
                "street_name": {"type": "string"},
                "place_name": {"type": "string"},
            },
            "required": ["run_dir"],
        },
    },
)


class RoadLensMcpServer:
    """Dependency-free JSON-RPC 2.0 MCP server for RoadLens."""

    def __init__(
        self,
        workspace_dir: Path,
        pmtiles_path: Path | None = None,
        checkpoint_path: Path | None = None,
    ):
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        resolved_pmtiles = pmtiles_path or (
            Path(os.environ["ROADLENS_PMTILES"])
            if os.environ.get("ROADLENS_PMTILES")
            else None
        )
        if resolved_pmtiles is not None and not resolved_pmtiles.is_file():
            raise FileNotFoundError(resolved_pmtiles)
        self.pmtiles_path = resolved_pmtiles
        self.street_index = (
            PmtilesStreetIndex(resolved_pmtiles)
            if resolved_pmtiles is not None
            else None
        )
        self.checkpoint_path = checkpoint_path or (
            Path(os.environ["ROADLENS_SAM_CHECKPOINT"])
            if os.environ.get("ROADLENS_SAM_CHECKPOINT")
            else None
        )
        self.ride: RideIndex | None = None
        self.ride_source: Path | None = None

    def _ensure_ride(self, ride_path: str | None) -> RideIndex:
        if ride_path:
            target = Path(ride_path).expanduser().resolve()
            if self.ride is None or self.ride_source != target:
                self.ride = index_ride(target)
                self.ride_source = target
        if self.ride is None:
            raise ValueError(
                "No ride indexed yet. Call `index_ride_clips` first or pass "
                "`ride_path`."
            )
        return self.ride

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        # Redirect any stray stdout prints inside library helpers to stderr so
        # stdio JSON-RPC framing is never corrupted.
        with contextlib.redirect_stdout(sys.stderr):
            if name == "index_ride_clips":
                return self._tool_index_ride(arguments)
            if name == "resolve_ride_segment":
                return self._tool_resolve_segment(arguments)
            if name == "discover_segment_hazards_vlm":
                return self._tool_discover_vlm(arguments)
            if name == "scan_segment_sam":
                return self._tool_scan_sam(arguments)
            if name == "triage_run_candidates":
                return self._tool_triage_run(arguments)
            if name == "draft_summary_report":
                return self._tool_draft_report(arguments)
        raise KeyError(f"Unknown tool: {name!r}")

    def _tool_index_ride(self, args: dict[str, Any]) -> dict[str, Any]:
        target = Path(str(args["path"])).expanduser().resolve()
        self.ride = index_ride(target)
        self.ride_source = target
        return {
            "indexed_path": str(target),
            "ride": self.ride.to_dict(),
            "hazard_presets": list(HAZARD_PRESETS),
        }

    def _tool_resolve_segment(self, args: dict[str, Any]) -> dict[str, Any]:
        ride = self._ensure_ride(args.get("ride_path"))
        mode = str(args["mode"])
        tz_name = str(args.get("timezone_name") or "Europe/London")
        include_polyline = bool(args.get("include_polyline", False))

        if mode == "street":
            if self.street_index is None:
                raise ValueError(
                    "Street-name resolution requires `--pmtiles` or `ROADLENS_PMTILES`."
                )
            passes = resolve_by_street_name(
                ride,
                str(args.get("value") or ""),
                geocoder=self.street_index,
            )
        elif mode == "offset":
            passes = resolve_by_ride_offset(
                ride,
                args.get("value", 0),
                window_seconds=float(args.get("window_seconds", 30.0)),
                geocoder=self.street_index,
            )
        elif mode == "clock":
            passes = resolve_by_wall_clock(
                ride,
                str(args.get("value") or ""),
                tz_name=tz_name,
                window_seconds=float(args.get("window_seconds", 60.0)),
                geocoder=self.street_index,
            )
        elif mode == "corridor":
            raw_line = args.get("polyline") or []
            line = [(float(pt[0]), float(pt[1])) for pt in raw_line]
            passes = resolve_by_corridor(
                ride,
                polylines=[line],
                tolerance_m=float(args.get("tolerance_m", 45.0)),
                geocoder=self.street_index,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode!r}")

        serialized = []
        for seg in passes:
            item = seg.to_dict()
            if not include_polyline:
                item.pop("polyline", None)
            serialized.append(item)
        return {
            "match_count": len(serialized),
            "passes": serialized,
            "human_in_the_loop_note": HUMAN_REVIEW_NOTICE,
        }

    def _tool_discover_vlm(self, args: dict[str, Any]) -> dict[str, Any]:
        clip_path = Path(str(args["clip_path"])).expanduser().resolve()
        start_s = max(0.0, float(args["start_seconds"]))
        dur_s = max(1.0, float(args.get("duration_seconds", 15.0)))
        prompt = str(args["prompt"]).strip()
        vlm_mode = str(args.get("vlm") or "local")
        spacing_m = max(0.5, float(args.get("spacing_m", 4.0)))
        max_frames = max(1, int(args.get("max_frames", 8)))
        road_crop = bool(args.get("road_crop", True))

        out_dir = (
            Path(str(args["output_dir"])).expanduser().resolve()
            if args.get("output_dir")
            else self.workspace_dir
            / f"vlm-discovery-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        )
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        meta = probe_video(clip_path)
        end_s = min(float(meta["duration"]), start_s + dur_s)
        created = (
            datetime.fromisoformat(str(meta["creation_time"]).replace("Z", "+00:00"))
            if meta.get("creation_time")
            else datetime.now(tz=UTC)
        )
        telemetry = read_garmin_telemetry(clip_path)
        times, sharpness = measure_frames(clip_path, start_s, end_s)
        speeds = speed_at_frames(np.asarray(times), telemetry.samples, created)
        choices = select_frames(
            times,
            sharpness,
            speeds,
            spacing_m=spacing_m,
            max_interval=max(1.0, spacing_m / 4.0),
        )
        if len(choices) > max_frames:
            step = math.ceil(len(choices) / max_frames)
            choices = choices[::step][:max_frames]

        saved_paths = save_selected_frames(clip_path, choices, frames_dir)
        frame_reports: list[dict[str, Any]] = []
        with create_vlm_backend(vlm_mode) as backend:
            for choice, frame_path in zip(choices, saved_paths, strict=True):
                recorded_utc = created + timedelta(seconds=choice.source_seconds)
                fix = nearest_telemetry(telemetry.samples, recorded_utc)
                street_name = None
                place_name = None
                if fix is not None and self.street_index is not None:
                    rev = self.street_index.reverse_road(fix.latitude, fix.longitude)
                    street_name = rev.street_name
                    place_name = rev.place_name
                discovery = backend.discover_hazards(
                    frame_path, prompt, road_crop=road_crop
                )
                frame_reports.append(
                    {
                        "frame_path": str(frame_path),
                        "source_seconds": round(choice.source_seconds, 3),
                        "recorded_utc": recorded_utc.isoformat(),
                        "latitude": None if fix is None else fix.latitude,
                        "longitude": None if fix is None else fix.longitude,
                        "location_source": (
                            "unavailable" if fix is None else "camera_gps"
                        ),
                        "street_name": street_name,
                        "place_name": place_name,
                        "review_status": "unreviewed",
                        "scene_summary": discovery.get("scene_summary", ""),
                        "findings": list(discovery.get("findings") or []),
                    }
                )

        matched_frames = [fr for fr in frame_reports if fr["findings"]]
        summary_payload = {
            "clip_path": str(clip_path),
            "start_seconds": round(start_s, 3),
            "end_seconds": round(end_s, 3),
            "prompt": prompt,
            "vlm_backend": vlm_mode,
            "inspected_frames": len(frame_reports),
            "frames_with_findings": len(matched_frames),
            "output_dir": str(out_dir),
            "review_status": "unreviewed",
            "human_in_the_loop_note": HUMAN_REVIEW_NOTICE,
            "results": frame_reports,
        }
        (out_dir / "vlm-discovery.json").write_text(
            json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
        )
        return summary_payload

    def _tool_scan_sam(self, args: dict[str, Any]) -> dict[str, Any]:
        from roadlens.scan import run_scan

        if self.checkpoint_path is None:
            raise ValueError(
                "SAM 3.1 scan requires `--checkpoint` or `ROADLENS_SAM_CHECKPOINT`."
            )
        clip_path = Path(str(args["clip_path"])).expanduser().resolve()
        start_s = max(0.0, float(args["start_seconds"]))
        dur_s = max(1.0, float(args["duration_seconds"]))
        prompt = str(args.get("prompt") or "pothole").strip()
        spacing_m = max(0.25, float(args.get("spacing_m", 1.0)))
        refine = bool(args.get("refine", True))

        out_dir = (
            Path(str(args["output_dir"])).expanduser().resolve()
            if args.get("output_dir")
            else self.workspace_dir
            / f"sam-scan-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        )
        scan_ns = Namespace(
            command="scan",
            video=clip_path,
            prompt=prompt,
            checkpoint=self.checkpoint_path,
            start=start_s,
            duration=dur_s,
            sampling="distance",
            spacing=spacing_m,
            max_interval=0.5,
            sharpness_window=0.05,
            width=0,
            chunk_frames=64,
            overlap_frames=16,
            grounding_batch_size=4,
            confidence=0.25,
            detection_threshold=0.25,
            new_object_threshold=0.35,
            refine_padding=2.0,
            no_refine=not refine,
            resume=False,
            plan_only=False,
            output=out_dir,
        )
        try:
            manifest = run_scan(scan_ns)
        finally:
            _release_torch_vram()

        obs = json.loads((out_dir / "observations.json").read_text(encoding="utf-8"))
        return {
            "output_dir": str(out_dir),
            "status": manifest.get("status"),
            "prompt": prompt,
            "unreviewed_candidate_count": len(obs),
            "candidates_preview": obs[:20],
            "review_status": "unreviewed",
            "human_in_the_loop_note": HUMAN_REVIEW_NOTICE,
        }

    def _tool_triage_run(self, args: dict[str, Any]) -> dict[str, Any]:
        run_dir = Path(str(args["run_dir"])).expanduser().resolve()
        vlm_mode = str(args.get("vlm") or "local")
        max_candidates = (
            int(args["max_candidates"])
            if args.get("max_candidates") is not None
            else None
        )
        street_name = args.get("street_name")
        place_name = args.get("place_name")

        obs_before = json.loads(
            (run_dir / "observations.json").read_text(encoding="utf-8")
        )
        if not street_name and obs_before and self.street_index is not None:
            first_geo = next(
                (
                    o
                    for o in obs_before
                    if o.get("latitude") is not None and o.get("longitude") is not None
                ),
                None,
            )
            if first_geo is not None:
                rev = self.street_index.reverse_road(
                    float(first_geo["latitude"]), float(first_geo["longitude"])
                )
                street_name = street_name or rev.street_name
                place_name = place_name or rev.place_name

        with create_vlm_backend(vlm_mode) as backend:
            summary = triage_run_directory(
                run_dir,
                backend,
                max_candidates=max_candidates,
                street_name=street_name,
                place_name=place_name,
            )

        obs_after = json.loads(
            (run_dir / "observations.json").read_text(encoding="utf-8")
        )
        counts: dict[str, int] = {}
        for item in obs_after:
            cls = (item.get("vlm_triage") or {}).get("classification")
            if cls:
                counts[cls] = counts.get(cls, 0) + 1

        return {
            **summary,
            "classification_counts": counts,
            "street_name": street_name,
            "place_name": place_name,
            "review_status": "unreviewed",
            "human_in_the_loop_note": HUMAN_REVIEW_NOTICE,
        }

    def _tool_draft_report(self, args: dict[str, Any]) -> dict[str, Any]:
        run_dir = Path(str(args["run_dir"])).expanduser().resolve()
        vlm_mode = str(args.get("vlm") or "local")
        street_name = args.get("street_name")
        place_name = args.get("place_name")

        obs = json.loads((run_dir / "observations.json").read_text(encoding="utf-8"))
        decisions_path = run_dir / "review-decisions.json"
        if decisions_path.is_file():
            decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
            for idx, item in enumerate(obs):
                oid = str(int(item.get("object_id", idx)))
                if oid in decisions:
                    item["review_status"] = decisions[oid]

        if not street_name and obs and self.street_index is not None:
            first_geo = next(
                (
                    o
                    for o in obs
                    if o.get("latitude") is not None and o.get("longitude") is not None
                ),
                None,
            )
            if first_geo is not None:
                rev = self.street_index.reverse_road(
                    float(first_geo["latitude"]), float(first_geo["longitude"])
                )
                street_name = street_name or rev.street_name
                place_name = place_name or rev.place_name

        if vlm_mode in ("local", "remote"):
            with create_vlm_backend(vlm_mode) as backend:
                report_md = backend.draft_summary_report(
                    obs, street_name=street_name, place_name=place_name
                )
        elif vlm_mode == "template":
            report_md = format_deterministic_summary_report(
                obs, street_name=street_name, place_name=place_name
            )
        else:
            raise ValueError(f"Unsupported vlm mode: {vlm_mode!r}")

        report_path = run_dir / "summary-report.md"
        report_path.write_text(report_md, encoding="utf-8")
        pmtiles_arg = (
            str(self.pmtiles_path) if self.pmtiles_path else "<path/to/tiles.pmtiles>"
        )
        return {
            "summary_report_path": str(report_path),
            "summary_report_markdown": report_md,
            "studio_review_command": f"roadlens studio --pmtiles {pmtiles_arg}",
            "report_review_command": (
                f"roadlens report {run_dir} --pmtiles {pmtiles_arg}"
            ),
            "human_in_the_loop_note": HUMAN_REVIEW_NOTICE,
        }

    def handle_jsonrpc(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        msg_id = message.get("id")

        # JSON-RPC notifications (no `id`) do not expect a response.
        if msg_id is None:
            return None

        if method == "initialize":
            params = message.get("params") or {}
            requested = str(params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION)
            negotiated = (
                requested
                if requested in SUPPORTED_PROTOCOL_VERSIONS
                else DEFAULT_PROTOCOL_VERSION
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": negotiated,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "roadlens-mcp", "version": "0.2.0"},
                    "instructions": HUMAN_REVIEW_NOTICE,
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": list(MCP_TOOLS)},
            }

        if method == "tools/call":
            params = message.get("params") or {}
            tool_name = str(params.get("name") or "")
            tool_args = dict(params.get("arguments") or {})
            if not any(t["name"] == tool_name for t in MCP_TOOLS):
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32602,
                        "message": f"Unknown tool: {tool_name!r}",
                    },
                }
            try:
                payload = self.execute_tool(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(payload, indent=2),
                            }
                        ],
                        "structuredContent": payload,
                        "isError": False,
                    },
                }
            except Exception as err:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {"error": f"{type(err).__name__}: {err}"}, indent=2
                                ),
                            }
                        ],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method!r}"},
        }


def _read_stdio_message(stream: BinaryIO) -> tuple[dict[str, Any] | None, bool]:
    """Read one JSON-RPC message (either Content-Length framed or newline JSON)."""
    while True:
        line = stream.readline()
        if not line:
            return None, False
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith(b"content-length:"):
            length = int(stripped.split(b":", 1)[1].strip())
            while True:
                hdr = stream.readline()
                if not hdr or hdr in (b"\r\n", b"\n"):
                    break
            body = stream.read(length)
            if not body:
                return None, True
            return json.loads(body.decode("utf-8")), True
        return json.loads(stripped.decode("utf-8")), False


def _write_stdio_message(
    stream: BinaryIO, payload: dict[str, Any], content_length_framed: bool
) -> None:
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if content_length_framed:
        header = f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii")
        stream.write(header + encoded)
    else:
        stream.write(encoded + b"\n")
    stream.flush()


def serve_mcp_stdio(
    workspace_dir: Path,
    pmtiles_path: Path | None = None,
    checkpoint_path: Path | None = None,
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
) -> None:
    server = RoadLensMcpServer(
        workspace_dir=workspace_dir,
        pmtiles_path=pmtiles_path,
        checkpoint_path=checkpoint_path,
    )
    in_stream = stdin if stdin is not None else sys.stdin.buffer
    out_stream = stdout if stdout is not None else sys.stdout.buffer
    while True:
        msg, framed = _read_stdio_message(in_stream)
        if msg is None:
            break
        response = server.handle_jsonrpc(msg)
        if response is not None:
            _write_stdio_message(out_stream, response, framed)
