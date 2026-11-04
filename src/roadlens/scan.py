"""Auditable coarse-to-fine local SAM scans, independent of pothole map pins."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import time
from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from roadlens.analysis import Observation, nearest_telemetry, write_observations
from roadlens.domain import Detection, FrameSegmentations, decode_mask
from roadlens.media import probe_video
from roadlens.output import write_frame_result
from roadlens.sampling import (
    DEFAULT_MOTION,
    LEGACY_MAX_INTERVAL,
    LEGACY_MOTION,
    MotionSettings,
    distance_refinement_windows,
    refinement_frames,
    refinement_windows,
    select_frames,
    speed_at_frames,
)
from roadlens.scan_media import coverage, measure_frames, save_selected_frames
from roadlens.sizing import annotate_run_sizes, geometry_for
from roadlens.telemetry import read_camera_telemetry


def box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0,
        min(ay + ah, by + bh) - max(ay, by),
    )
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def evidence_quality(pixels, detection) -> tuple[float, dict]:
    """Auditable heuristic: confidence, crop sharpness, size and clipping."""
    import cv2

    height, width = pixels.shape[:2]
    x, y, w, h = detection.box_xywh
    x0, y0 = max(0, int(x * width)), max(0, int(y * height))
    x1, y1 = (
        min(width, math.ceil((x + w) * width)),
        min(
            height,
            math.ceil((y + h) * height),
        ),
    )
    crop = pixels[y0:y1, x0:x1]
    sharpness = 0.0
    if crop.size:
        if max(crop.shape[:2]) > 512:
            scale = 512 / max(crop.shape[:2])
            crop = cv2.resize(crop, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clipped = min(x, y) <= 0.005 or max(x + w, y + h) >= 0.995
    area = int(detection.mask.sum())
    score = detection.score if detection.score is not None else 0.0
    quality = score * math.log1p(area) * math.log1p(sharpness)
    if clipped:
        quality *= 0.25
    return quality, {
        "rank": quality,
        "crop_sharpness": sharpness,
        "touches_frame_boundary": clipped,
        "mask_pixels": area,
        "method": "score * log1p(mask_pixels) * log1p(crop_sharpness); boundary x0.25",
    }


class TrackRegistry:
    """Join tracks only using matching boxes at identical source timestamps.

    This joins overlap and coarse/fine duplicates without claiming that nearby
    camera GPS positions identify the same physical defect. Other splits remain
    separate unreviewed candidates.
    """

    def __init__(self):
        self.parents: list[int] = []
        self.seen: dict[float, list[tuple[int, tuple]]] = {}
        self.tracks: dict[int, dict] = {}

    def root(self, identity):
        while self.parents[identity] != identity:
            self.parents[identity] = self.parents[self.parents[identity]]
            identity = self.parents[identity]
        return identity

    def add(self, local_ids, result, choice, image, pass_name):
        import cv2

        pixels = cv2.imread(str(image)) if result.detections else None
        timestamp = round(choice.source_seconds, 9)
        for detection in result.detections:
            if detection.object_id not in local_ids:
                identity = len(self.parents)
                self.parents.append(identity)
                local_ids[detection.object_id] = identity
            identity = self.root(local_ids[detection.object_id])
            matches = [
                self.root(other)
                for other, box in self.seen.get(timestamp, [])
                if box_iou(box, detection.box_xywh) >= 0.5
            ]
            for other in matches:
                self.parents[self.root(other)] = identity
            self.seen.setdefault(timestamp, []).append((identity, detection.box_xywh))
            quality, details = evidence_quality(pixels, detection)
            track = self.tracks.setdefault(
                identity,
                {
                    "times": set(),
                    "best": None,
                    "quality": -1,
                },
            )
            track["times"].add(timestamp)
            # SAM 3.1 scores a remembered-but-undetected object -10000; such a
            # frame is never evidence, even when a flat crop makes quality -0.0.
            placeholder = detection.score is not None and detection.score < 0
            if not placeholder and quality > track["quality"]:
                track.update(
                    quality=quality,
                    best={
                        "choice": choice,
                        "image": image,
                        "detection": detection,
                        "pass": pass_name,
                        "details": details,
                    },
                )

    def merged(self):
        merged = {}
        for identity, track in self.tracks.items():
            root = self.root(identity)
            combined = merged.setdefault(
                root,
                {
                    "times": set(),
                    "quality": -1,
                    "best": None,
                },
            )
            combined["times"].update(track["times"])
            if track["quality"] > combined["quality"]:
                combined.update(quality=track["quality"], best=track["best"])
        return sorted(merged.values(), key=lambda t: min(t["times"]))


def run_pass(
    backend,
    choices,
    paths,
    name,
    output,
    registry,
    chunk_size,
    overlap,
    prompt: str = "pothole",
):
    detections, timings = [], []
    completed = {}
    chunk_path = output / "chunks.jsonl"
    if chunk_path.exists():
        receipts = []
        for line in chunk_path.read_text().splitlines():
            try:
                receipts.append((line, json.loads(line)))
            except json.JSONDecodeError:
                continue  # A torn receipt means the chunk did not finish: rerun it.
        completed = {row["offset"]: row for _, row in receipts if row["pass"] == name}
        # Rewrite without the torn line so the next receipt is not glued onto it.
        temporary = chunk_path.with_suffix(".tmp")
        temporary.write_text("".join(f"{line}\n" for line, _ in receipts))
        temporary.replace(chunk_path)
    cached = {}
    ledger_path = output / "frames.jsonl"
    if ledger_path.exists():
        kept = []
        for line in ledger_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # An interrupted final write is not a completed result.
            if row["pass"] != name or row["chunk_offset"] in completed:
                kept.append(line)
            if row["pass"] == name and row["chunk_offset"] in completed:
                cached.setdefault(row["chunk_offset"], []).append(row)
        temporary = ledger_path.with_suffix(".tmp")
        temporary.write_text("\n".join(kept) + ("\n" if kept else ""))
        temporary.replace(ledger_path)
    with (output / "frames.jsonl").open("a", encoding="utf-8") as ledger:
        for offset in range(0, len(choices), chunk_size - overlap):
            subset = choices[offset : offset + chunk_size]
            local_ids = {}
            if offset in completed:
                records = cached.get(offset, [])
                if sorted(row["frame_index"] for row in records) != list(
                    range(len(subset))
                ):
                    raise ValueError("Completed chunk ledger is missing or duplicated")
                for row in records:
                    result = FrameSegmentations(
                        row["frame_index"],
                        row["prompt"],
                        row["backend"],
                        tuple(
                            Detection(
                                d["object_id"],
                                tuple(d["box_xywh_normalized"]),
                                d["score"],
                                decode_mask(d["mask_rle"]),
                            )
                            for d in row["detections"]
                        ),
                    )
                    choice = subset[result.frame_index]
                    if abs(row["source_seconds"] - choice.source_seconds) > 1e-6:
                        raise ValueError("Cached chunk timestamps differ from the plan")
                    registry.add(
                        local_ids,
                        result,
                        choice,
                        paths[offset + result.frame_index],
                        name,
                    )
                    if result.detections:
                        detections.append(choice.source_seconds)
                timings.append(completed[offset])
                print(
                    f"Reused {name} chunk {offset}: {len(records)} frames", flush=True
                )
                if offset + chunk_size >= len(choices):
                    break
                continue
            started = time.perf_counter()
            inference_seconds = 0.0
            count = 0
            processed = set()
            with tempfile.TemporaryDirectory(prefix="roadlens-") as temporary:
                directory = Path(temporary)
                for i, path in enumerate(paths[offset : offset + chunk_size]):
                    (directory / f"{i:06d}.png").symlink_to(path.resolve())
                with closing(backend.segment_video(directory, prompt)) as iterator:
                    while True:
                        before = time.perf_counter()
                        try:
                            result = next(iterator)
                        except StopIteration:
                            inference_seconds += time.perf_counter() - before
                            break
                        inference_seconds += time.perf_counter() - before
                        if result.frame_index in processed or not (
                            0 <= result.frame_index < len(subset)
                        ):
                            raise ValueError(
                                "SAM returned a duplicate or invalid frame index"
                            )
                        processed.add(result.frame_index)
                        choice = subset[result.frame_index]
                        serialised = result.to_dict()
                        for raw, saved in zip(
                            result.detections, serialised["detections"], strict=True
                        ):
                            saved["box_xywh_normalized"] = list(raw.box_xywh)
                            saved["score"] = raw.score
                        ledger.write(
                            json.dumps(
                                {
                                    "pass": name,
                                    "chunk_offset": offset,
                                    "source_seconds": choice.source_seconds,
                                    **serialised,
                                }
                            )
                            + "\n"
                        )
                        registry.add(
                            local_ids,
                            result,
                            choice,
                            paths[offset + result.frame_index],
                            name,
                        )
                        if result.detections:
                            detections.append(choice.source_seconds)
                        count += 1
                ledger.flush()
                if processed != set(range(len(subset))):
                    raise ValueError("SAM did not return every requested frame")
            timing = {
                "pass": name,
                "offset": offset,
                "frames": count,
                "inference_seconds": inference_seconds,
                "wall_seconds": time.perf_counter() - started,
            }
            timings.append(timing)
            with (output / "chunks.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(timing) + "\n")
            print(
                f"{name} chunk {offset}: {count} frames, "
                f"{inference_seconds:.1f}s inference, "
                f"{len(set(detections))} positive source frames so far",
                flush=True,
            )
            if offset + chunk_size >= len(choices):
                break
    return detections, timings


def _source_signature(video: Path) -> dict:
    stat = video.stat()
    return {
        "path": str(video.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def sampling_settings(args) -> dict:
    """Frame-efficiency settings, with defaults for callers that omit them."""
    if getattr(args, "legacy_sampling", False):
        return {
            "max_interval": LEGACY_MAX_INTERVAL,
            "motion": LEGACY_MOTION,
            "refine_mode": "time",
        }
    default = DEFAULT_MOTION
    stop_speed = getattr(args, "stop_speed", default.stop_speed_mps)
    heartbeat = getattr(args, "stop_heartbeat", default.stop_heartbeat_seconds)
    return {
        "max_interval": getattr(args, "max_interval", None) or None,
        "motion": MotionSettings(
            stop_speed_mps=stop_speed or None,
            resume_speed_mps=getattr(args, "resume_speed", default.resume_speed_mps),
            stop_after_seconds=getattr(args, "stop_after", default.stop_after_seconds),
            stop_heartbeat_seconds=heartbeat or None,
        ),
        "refine_mode": getattr(args, "refine_mode", "distance"),
    }


def run_scan(args, backend_factory=None):
    prompt = getattr(args, "prompt", "pothole")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Scan prompt cannot be empty")
    if args.resume:
        previous = json.loads((args.output / "scan.json").read_text())
        settings = {
            key: value
            for key, value in vars(args).items()
            if key not in ("checkpoint", "video", "output", "resume")
        }
        if any(
            previous["settings"].get(k, "pothole" if k == "prompt" else None) != v
            for k, v in settings.items()
        ):
            raise ValueError("Resume requires the same scan settings")
        if previous["source_video"] != args.video.name:
            raise ValueError("Resume requires the same source video")
        if previous["status"] == "complete":
            # Never return a completed manifest for a missing or replaced source.
            if not args.video.is_file():
                raise FileNotFoundError(args.video)
            if "source_signature" not in previous:
                # Older manifests cannot prove the file is the one scanned.
                raise ValueError(
                    "Saved run has no source signature; rescan into a new directory"
                )
            if previous["source_signature"] != _source_signature(args.video):
                raise ValueError("Source file changed since the saved run")
            return previous
    elif args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use an empty output directory to avoid mixing runs")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        return _run_scan(args, backend_factory)
    except BaseException as error:
        path = args.output / "scan.json"
        manifest = json.loads(path.read_text()) if path.exists() else {}
        manifest.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise


def _run_scan(args, backend_factory=None):
    from roadlens.backends.local import LocalSamBackend

    if args.start < 0 or args.duration <= 0 or args.width < 0:
        raise ValueError("Start/width must be nonnegative and duration positive")
    if args.chunk_frames < 2 or not 0 <= args.overlap_frames < args.chunk_frames:
        raise ValueError("Require chunk_frames >= 2 and 0 <= overlap < chunk_frames")
    started = time.perf_counter()
    previous = (
        json.loads((args.output / "scan.json").read_text()) if args.resume else None
    )
    metadata = probe_video(args.video)
    if not metadata["creation_time"]:
        raise ValueError("Source has no creation time for Garmin alignment")
    created = datetime.fromisoformat(metadata["creation_time"].replace("Z", "+00:00"))
    end = min(args.start + args.duration, metadata["duration"])
    if args.start >= end:
        raise ValueError("Requested interval is outside the source video")
    telemetry = read_camera_telemetry(args.video)
    print("Measuring source-frame sharpness and presentation timestamps...", flush=True)
    times, quality = measure_frames(args.video, args.start, end)
    speeds = speed_at_frames(np.asarray(times), telemetry.samples, created)
    sampling = sampling_settings(args)
    motion = sampling["motion"]
    choices = select_frames(
        times,
        quality,
        speeds,
        spacing_m=args.spacing,
        max_interval=sampling["max_interval"],
        sharpness_window=args.sharpness_window,
        all_frames=args.sampling == "all",
        motion=motion,
    )
    manifest = {
        "status": "preparing",
        "source_video": args.video.name,
        "source_metadata": metadata,
        "source_signature": _source_signature(args.video),
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in ("checkpoint", "video", "output")
        },
        "coverage": coverage(choices, times),
        "frame_efficiency": {
            "max_interval_seconds": sampling["max_interval"],
            "motion": asdict(motion),
            "refine_mode": sampling["refine_mode"],
        },
        "native_intervals_over_spacing": int(
            np.sum(
                np.diff(times) * speeds[1:] > args.spacing,
            )
        ),
        "native_frames_without_speed": int(np.isnan(speeds).sum()),
        "telemetry_source": telemetry.source,
        "coarse_frames": [c.to_dict() for c in choices],
        "timestamp_basis": "decoded PTS minus video stream start_time",
        "gps_time_basis": "MP4 creation_time + source PTS; camera positions only",
    }
    repository = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode == 0:
        manifest["code_revision"] = revision.stdout.strip()
        manifest["tracked_code_modified"] = (
            subprocess.run(
                ["git", "diff", "--quiet", "HEAD"],
                cwd=repository,
                check=False,
            ).returncode
            != 0
        )
    manifest_path = args.output / "scan.json"
    if previous:
        if (
            previous.get("source_signature", manifest["source_signature"])
            != (manifest["source_signature"])
        ):
            raise ValueError("Source file changed since the saved run")
        if previous["source_metadata"] != metadata or [
            c["source_seconds"] for c in previous["coarse_frames"]
        ] != [c.source_seconds for c in choices]:
            raise ValueError("Source metadata or sampling changed since the saved run")
        manifest["resume_history"] = previous.get("resume_history", []) + [
            {
                key: previous.get(key)
                for key in (
                    "status",
                    "code_revision",
                    "preparation_seconds",
                    "model_load_seconds",
                    "wall_seconds",
                    "error",
                )
            }
        ]
        manifest["timing_note"] = (
            "Wall time covers latest attempt; inference totals include reused chunks."
        )

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    print(
        f"Selected {len(choices)} of {len(times)} native frames; "
        f"{choices[-1].distance_m:.1f} m of integrated known speed",
        flush=True,
    )
    if args.plan_only:
        manifest.update(status="planned", wall_seconds=time.perf_counter() - started)
        save_manifest()
        return manifest
    paths = [args.output / "coarse" / f"{i:06d}.png" for i in range(len(choices))]
    if not previous or not all(path.is_file() for path in paths):
        paths = save_selected_frames(
            args.video, choices, args.output / "coarse", args.width
        )
    manifest["preparation_seconds"] = time.perf_counter() - started
    factory = backend_factory or LocalSamBackend
    loading = time.perf_counter()
    backend = factory(
        args.checkpoint,
        args.confidence,
        detection_threshold=args.detection_threshold,
        new_object_threshold=args.new_object_threshold,
        grounding_batch_size=args.grounding_batch_size,
    )
    manifest.update(
        model_load_seconds=time.perf_counter() - loading,
        thresholds=backend.thresholds,
        status="running",
    )
    save_manifest()
    prompt = getattr(args, "prompt", "pothole").strip()
    registry = TrackRegistry()
    positive, timings = run_pass(
        backend,
        choices,
        paths,
        "coarse",
        args.output,
        registry,
        args.chunk_frames,
        args.overlap_frames,
        prompt=prompt,
    )
    by_distance = sampling["refine_mode"] == "distance"
    if args.no_refine or args.sampling == "all":
        windows = []
    elif by_distance:
        windows = distance_refinement_windows(
            positive,
            times,
            speeds,
            args.start,
            end,
            before_m=getattr(args, "refine_before", 10.0),
            after_m=getattr(args, "refine_after", 15.0),
            fallback_padding=args.refine_padding,
            motion=motion,
        )
    else:
        windows = refinement_windows(positive, args.start, end, args.refine_padding)
    manifest["refinement_windows"] = windows
    save_manifest()
    native_choices = select_frames(times, quality, speeds, all_frames=True)
    for index, (lo, hi) in enumerate(windows):
        fine = (
            refinement_frames(
                times,
                quality,
                speeds,
                (lo, hi),
                spacing_m=getattr(args, "refine_spacing", 0.25),
                motion=motion,
            )
            if by_distance
            else [c for c in native_choices if lo <= c.source_seconds <= hi]
        )
        fine_paths = [
            args.output / f"fine-{index:03d}" / f"{i:06d}.png" for i in range(len(fine))
        ]
        if not previous or not all(
            path.is_file() and path.stat().st_size > 0 for path in fine_paths
        ):
            fine_paths = save_selected_frames(
                args.video,
                fine,
                args.output / f"fine-{index:03d}",
                args.width,
            )
        manifest.setdefault("fine_frames", []).append(
            {
                "window": index,
                "frames": [c.to_dict() for c in fine],
            }
        )
        save_manifest()
        _, fine_timings = run_pass(
            backend,
            fine,
            fine_paths,
            f"fine-{index:03d}",
            args.output,
            registry,
            args.chunk_frames,
            args.overlap_frames,
            prompt=prompt,
        )
        timings.extend(fine_timings)
    observations, details = [], []
    # A track seen only with SAM 3.1's placeholder score (see TrackRegistry.add)
    # has no evidence frame to show.
    tracks = registry.merged()
    evidenced = [track for track in tracks if track["best"] is not None]
    for identity, track in enumerate(evidenced):
        best = track["best"]
        choice = best["choice"]
        recorded = created + timedelta(seconds=choice.source_seconds)
        gps = nearest_telemetry(telemetry.samples, recorded)
        detection = replace(best["detection"], object_id=identity)
        frame = FrameSegmentations(identity, prompt, "local", (detection,))
        write_frame_result(args.video, frame, args.output, source_frame=best["image"])
        observations.append(
            Observation(
                identity,
                prompt,
                "local",
                min(track["times"]),
                max(track["times"]),
                choice.source_seconds,
                identity,
                recorded,
                detection.score,
                int(detection.mask.sum()),
                gps.latitude if gps else None,
                gps.longitude if gps else None,
                telemetry.location_source if gps else "unavailable",
            )
        )
        details.append(
            {
                "object_id": identity,
                "best_pass": best["pass"],
                "supporting_source_frames": len(track["times"]),
                "quality": best["details"],
            }
        )
    write_observations(observations, args.video, args.output)
    annotate_run_sizes(args.output, geometry_for(telemetry.source))
    (args.output / "evidence-selection.json").write_text(json.dumps(details, indent=2))
    manifest.update(
        status="complete",
        chunks=timings,
        observations=len(observations),
        tracks_without_valid_score=len(tracks) - len(evidenced),
        wall_seconds=time.perf_counter() - started,
        total_inference_seconds=sum(t["inference_seconds"] for t in timings),
        inferred_frames_including_overlap=sum(t["frames"] for t in timings),
        note="Unreviewed candidates; recall and exact hazard locations are unverified.",
    )
    save_manifest()
    print(
        f"Completed in {manifest['wall_seconds']:.1f}s: "
        f"{len(observations)} unreviewed candidates",
        flush=True,
    )
    return manifest
