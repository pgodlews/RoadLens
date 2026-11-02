from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from roadlens.analysis import build_observations, write_observations
from roadlens.backends import create_backend
from roadlens.media import prepare_analysis_clip, probe_video
from roadlens.output import write_frame_result
from roadlens.report import serve_report
from roadlens.telemetry import read_garmin_telemetry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roadlens", description="Post-ride analysis for Garmin footage"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    segment = commands.add_parser("segment", help="Segment a concept in a video")
    segment.add_argument("video", type=Path)
    segment.add_argument("--prompt", required=True)
    segment.add_argument("--backend", choices=("local", "remote"), default="local")
    segment.add_argument(
        "--checkpoint",
        type=Path,
        default=os.environ.get("ROADLENS_SAM_CHECKPOINT"),
    )
    segment.add_argument("--confidence", type=float, default=0.5)
    segment.add_argument("--max-frames", type=int)
    segment.add_argument("--output", type=Path, required=True)

    analyze = commands.add_parser(
        "analyze", help="Create GPS-linked hazard observations from a Garmin clip"
    )
    analyze.add_argument("video", type=Path)
    analyze.add_argument("--prompt", default="pothole")
    analyze.add_argument("--backend", choices=("local", "remote"), default="local")
    analyze.add_argument(
        "--checkpoint",
        type=Path,
        default=os.environ.get("ROADLENS_SAM_CHECKPOINT"),
    )
    analyze.add_argument("--confidence", type=float, default=0.5)
    analyze.add_argument("--start", type=float, default=0)
    analyze.add_argument("--duration", type=float, default=10)
    analyze.add_argument("--fps", type=float, default=5)
    analyze.add_argument("--output", type=Path, required=True)

    scan = commands.add_parser(
        "scan",
        help="Distance-sampled local scan with native-frame candidate refinement",
    )
    scan.add_argument("video", type=Path)
    scan.add_argument("--prompt", default="pothole", help="Hazard concept prompt")
    scan.add_argument(
        "--checkpoint", type=Path, default=os.environ.get("ROADLENS_SAM_CHECKPOINT")
    )
    scan.add_argument("--start", type=float, default=0)
    scan.add_argument("--duration", type=float, default=10)
    scan.add_argument("--sampling", choices=("distance", "all"), default="distance")
    scan.add_argument(
        "--spacing", type=float, default=1.0, help="Target metres per sample"
    )
    scan.add_argument("--max-interval", type=float, default=0.5)
    scan.add_argument("--sharpness-window", type=float, default=0.05)
    scan.add_argument(
        "--width", type=int, default=0, help="0 preserves source resolution"
    )
    scan.add_argument("--chunk-frames", type=int, default=64)
    scan.add_argument("--overlap-frames", type=int, default=16)
    scan.add_argument("--grounding-batch-size", type=int, default=4)
    scan.add_argument("--confidence", type=float, default=0.25, help="Output filter")
    scan.add_argument("--detection-threshold", type=float, default=0.25)
    scan.add_argument("--new-object-threshold", type=float, default=0.35)
    scan.add_argument("--refine-padding", type=float, default=2.0)
    scan.add_argument("--no-refine", action="store_true")
    scan.add_argument(
        "--resume", action="store_true", help="Reuse completed scan chunks"
    )
    scan.add_argument(
        "--plan-only", action="store_true", help="Measure sampling without SAM"
    )
    scan.add_argument("--output", type=Path, required=True)

    telemetry = commands.add_parser(
        "telemetry", help="Extract timed Garmin GPS from a video"
    )
    telemetry.add_argument("video", type=Path)
    telemetry.add_argument("--output", type=Path, required=True)

    report = commands.add_parser("report", help="Serve the local OSM review report")
    report.add_argument("output", type=Path)
    report.add_argument("--pmtiles", type=Path, required=True)
    report.add_argument("--host", default="127.0.0.1")
    report.add_argument("--port", type=int, default=8765)

    triage = commands.add_parser(
        "triage",
        help="Run VLM triage and draft a human-review summary report",
    )
    triage.add_argument("output", type=Path)
    triage.add_argument("--vlm", choices=("local", "remote"), default="local")
    triage.add_argument("--max-candidates", type=int)
    triage.add_argument("--street-name")
    triage.add_argument("--place-name")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "scan":
        from roadlens.scan import run_scan

        if not args.plan_only and args.checkpoint is None:
            raise ValueError("--checkpoint is required for scan inference")
        run_scan(args)
    elif args.command == "segment":
        backend = create_backend(
            args.backend,
            checkpoint_path=args.checkpoint,
            confidence_threshold=args.confidence,
        )
        count = 0
        detections = 0
        for result in backend.segment_video(
            args.video, args.prompt, max_frames=args.max_frames
        ):
            write_frame_result(args.video, result, args.output)
            count += 1
            detections += len(result.detections)
            print(
                f"frame={result.frame_index} detections={len(result.detections)}",
                flush=True,
            )
        print(
            f"Saved {count} frame result(s), {detections} detection(s) to {args.output}"
        )
    elif args.command == "analyze":
        metadata = probe_video(args.video)
        if not metadata["creation_time"]:
            raise ValueError("The Garmin video has no recording creation time")
        clip_created = datetime.fromisoformat(
            metadata["creation_time"].replace("Z", "+00:00")
        )
        telemetry = read_garmin_telemetry(args.video)
        working_clip = args.output / "analysis.mp4"
        prepare_analysis_clip(
            args.video,
            working_clip,
            args.start,
            args.duration,
            args.fps,
        )
        backend = create_backend(
            args.backend,
            checkpoint_path=args.checkpoint,
            confidence_threshold=args.confidence,
        )
        frames = []
        for result in backend.segment_video(working_clip, args.prompt):
            frames.append(result)
            if result.detections:
                write_frame_result(working_clip, result, args.output)
            print(
                f"frame={result.frame_index} detections={len(result.detections)}",
                flush=True,
            )
        observations = build_observations(
            frames, telemetry.samples, clip_created, args.start, args.fps
        )
        write_observations(observations, args.video, args.output)
        print(f"Saved {len(observations)} unreviewed observation(s) to {args.output}")
    elif args.command == "telemetry":
        result = read_garmin_telemetry(args.video)
        payload = {
            "samples": [
                {
                    "recorded_utc": sample.recorded_utc.isoformat(),
                    "latitude": sample.latitude,
                    "longitude": sample.longitude,
                    "speed_mps": sample.speed_mps,
                    "playback_seconds": sample.playback_seconds,
                }
                for sample in result.samples
            ],
            "rejected_samples": result.rejected_samples,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"Saved {len(result.samples)} GPS fixes; "
            f"rejected {result.rejected_samples} invalid sample(s)"
        )
    elif args.command == "report":
        serve_report(args.output, args.pmtiles, args.host, args.port)
    elif args.command == "triage":
        from roadlens.vlm import create_vlm_backend, triage_run_directory

        with create_vlm_backend(args.vlm) as vlm_backend:
            summary = triage_run_directory(
                args.output,
                vlm_backend,
                max_candidates=args.max_candidates,
                street_name=args.street_name,
                place_name=args.place_name,
            )
        print(
            f"Triaged {summary['triaged_candidates']}/{summary['total_observations']} "
            f"unreviewed candidate(s); draft report: {summary['summary_report']}"
        )


if __name__ == "__main__":
    main()
