from __future__ import annotations

import argparse
import os
from pathlib import Path

from roadlens.backends import create_backend
from roadlens.output import write_frame_result


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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "segment":
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


if __name__ == "__main__":
    main()
