from __future__ import annotations

import asyncio
import math
import os
import sys
import tempfile
from collections import defaultdict
from collections.abc import AsyncIterable, AsyncIterator, Iterator, Sequence
from contextlib import aclosing
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from roadlens.domain import Detection, FrameSegmentations
from roadlens.media import cut_frame_range, probe_video

# Hosted SAM 3.1 stops adding objects once a session tracks 16. On 3 November a
# 180 s window found nothing new after its 16th object, while the last 100 s
# of it on their own found seven more, including a rider-reported pothole. Long
# clips therefore go up as overlapping chunks, and tracks crossing a chunk
# boundary are re-linked by mask overlap in the shared frames.
OBJECT_CAP = 16
CHUNK_FRAMES = 200
CHUNK_OVERLAP = 5
MAX_CONCURRENT_CHUNKS = 3
LINK_IOU = 0.3


class RemoteSamBackend:
    name = "remote"

    def __init__(self, api_key: str | None = None, *, legacy_stitch: bool = False):
        self._api_key = api_key or os.environ.get("MODEL_API_KEY")
        self._legacy_stitch = legacy_stitch
        if not self._api_key:
            raise RuntimeError("MODEL_API_KEY is required for the remote backend")

    def segment_video(
        self,
        video_path: Path,
        prompt: str,
        max_frames: int | None = None,
    ) -> Iterator[FrameSegmentations]:
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        if not prompt.strip():
            raise ValueError("The segmentation prompt cannot be empty")
        metadata = probe_video(video_path)
        width, height = metadata["width"], metadata["height"]
        source_total = math.ceil(metadata["duration"] * metadata["fps"] - 1e-6)
        total = source_total
        if max_frames is not None:
            total = min(total, max_frames)
        chunk_frames = int(
            os.environ.get("ROADLENS_REMOTE_CHUNK_FRAMES") or CHUNK_FRAMES
        )
        spans = chunk_spans(total, chunk_frames, CHUNK_OVERLAP)
        if len(spans) == 1:
            if total < source_total:
                with tempfile.TemporaryDirectory(prefix="roadlens-remote-") as tmp:
                    path = Path(tmp) / "chunk-000.mp4"
                    cut_frame_range(video_path, path, 0, total)
                    result = asyncio.run(self._request(path, prompt))
            else:
                result = asyncio.run(self._request(video_path, prompt))
            frames = list(
                self._convert_result(result, prompt, width, height, max_frames)
            )
            _warn_if_capped(frames, 0, len(spans))
            yield from frames
            return

        with tempfile.TemporaryDirectory(prefix="roadlens-remote-") as tmp:
            paths = []
            for index, (start, count) in enumerate(spans):
                path = Path(tmp) / f"chunk-{index:03d}.mp4"
                cut_frame_range(video_path, path, start, count)
                paths.append(path)
            results = asyncio.run(self._request_all(paths, prompt))
        chunks = []
        for index, (result, (_, count)) in enumerate(zip(results, spans, strict=True)):
            frames = list(self._convert_result(result, prompt, width, height, count))
            _warn_if_capped(frames, index, len(spans))
            chunks.append(frames)
        yield from stitch_chunks(
            chunks,
            [start for start, _ in spans],
            CHUNK_OVERLAP,
            legacy_stitch=self._legacy_stitch,
        )

    async def _request_all(self, paths: Sequence[Path], prompt: str) -> list[Any]:
        gate = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)

        async def one(path: Path) -> Any:
            async with gate:
                return await self._request(path, prompt)

        return await asyncio.gather(*(one(path) for path in paths))

    async def _request(self, video_path: Path, prompt: str) -> Any:
        from meta_sam_parser import parse_responses_stream, video_segmentation_format
        from openai import AsyncOpenAI

        async with AsyncOpenAI(
            base_url="https://api.meta.ai/v1", api_key=self._api_key
        ) as client:
            uploaded = await client.files.create(
                file=(video_path.name, video_path.read_bytes(), "video/mp4"),
                purpose="user_data",
            )
            try:
                stream = await client.responses.create(
                    model="sam-3.1",
                    stream=True,
                    metadata={"include_confidence": "true"},
                    input=[
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_video", "file_id": uploaded.id},
                            ],
                        }
                    ],
                )
                async with stream, aclosing(_stream_events(stream)) as events:
                    parsed = parse_responses_stream(
                        events, video_segmentation_format()
                    )
                    async with parsed:
                        return await parsed.final_result()
            finally:
                # Uploaded files never expire by default; keep ride footage off
                # Meta's storage once segmentation is done.
                await client.files.delete(uploaded.id)

    @staticmethod
    def _convert_result(
        result: Any,
        prompt: str,
        width: int,
        height: int,
        max_frames: int | None,
    ) -> Iterator[FrameSegmentations]:
        from meta_sam_parser import decode_mask_to_raster

        if result.outcome.status != "completed":
            reason = getattr(result.outcome, "reason", "unknown")
            raise RuntimeError(f"Remote segmentation was incomplete: {reason}")
        errors = [item for item in result.diagnostics if item.severity == "error"]
        if errors:
            messages = "; ".join(item.message for item in errors)
            raise RuntimeError(
                f"Remote segmentation returned invalid records: {messages}"
            )

        boxes = {}
        for record in result.records:
            if record.kind == "box" and record.frame is not None:
                frame_index = record.frame.frame_index
                boxes[(frame_index, record.object_id)] = (
                    record.left / width,
                    record.top / height,
                    (record.right - record.left) / width,
                    (record.bottom - record.top) / height,
                )

        frames: dict[int, list[Detection]] = defaultdict(list)
        for record in result.records:
            if record.kind != "mask" or record.frame is None:
                continue
            frame_index = record.frame.frame_index
            if max_frames is not None and frame_index >= max_frames:
                continue
            mask_width = record.mask.width
            mask_height = record.mask.height
            raster = np.fromiter(
                decode_mask_to_raster(record.mask),
                dtype=np.uint8,
                count=mask_width * mask_height,
            ).reshape(mask_height, mask_width)
            # The raster is box-local: it starts at the box's top-left corner.
            left = int(record.bounds.left)
            top = int(record.bounds.top)
            right = min(left + mask_width, width)
            bottom = min(top + mask_height, height)
            if left < 0 or top < 0 or right <= left or bottom <= top:
                raise ValueError("Remote mask lies outside the video frame")
            mask = np.zeros((height, width), dtype=bool)
            cropped = raster[: bottom - top, : right - left]
            mask[top:bottom, left:right] = cropped.astype(bool)
            if not mask.any():
                continue
            box = boxes.get((frame_index, record.object_id))
            if box is None:
                box = (
                    left / width,
                    top / height,
                    mask_width / width,
                    mask_height / height,
                )
            frames[frame_index].append(
                Detection(
                    object_id=int(record.object_id),
                    box_xywh=box,
                    score=record.confidence,
                    mask=mask,
                )
            )

        for frame_index in sorted(frames):
            yield FrameSegmentations(
                frame_index=frame_index,
                prompt=prompt,
                backend=RemoteSamBackend.name,
                detections=tuple(frames[frame_index]),
            )


def chunk_spans(total: int, chunk_frames: int, overlap: int) -> list[tuple[int, int]]:
    """Equal ``(start, count)`` chunks covering ``total`` frames, each sharing
    ``overlap`` frames with the next."""
    if total <= 0:
        raise ValueError("The clip has no frames")
    if chunk_frames <= overlap:
        raise ValueError("Chunks must be longer than their overlap")
    if total <= chunk_frames:
        return [(0, total)]
    count = math.ceil((total - overlap) / (chunk_frames - overlap))
    step = math.ceil((total - overlap) / count)
    return [(i * step, min(step + overlap, total - i * step)) for i in range(count)]


def stitch_chunks(
    chunks: Sequence[Sequence[FrameSegmentations]],
    starts: Sequence[int],
    overlap: int,
    *,
    legacy_stitch: bool = False,
) -> Iterator[FrameSegmentations]:
    """Join per-chunk results into one clip, with object ids unique across it.

    Shared frames retain both chunks' detections, except same-frame mask
    duplicates. Matching tracks keep their ids beyond the overlap. Results
    are merged before emission so each source frame is yielded once, in order.
    ``legacy_stitch`` restores dropping all later-chunk overlap frames,
    including objects that the previous chunk's session cap may have hidden.
    """
    next_id = 0
    shared: dict[int, tuple[Detection, ...]] = {}
    combined: dict[int, FrameSegmentations] = {}
    for index, (frames, start) in enumerate(zip(chunks, starts, strict=True)):
        best: dict[int, tuple[float, int]] = {}
        for frame in frames:
            if index == 0 or frame.frame_index >= overlap:
                continue
            for detection in frame.detections:
                for earlier in shared.get(start + frame.frame_index, ()):
                    iou = _mask_iou(detection.mask, earlier.mask)
                    if iou >= LINK_IOU and iou > best.get(detection.object_id, (0,))[0]:
                        best[detection.object_id] = (iou, earlier.object_id)
        ids = {local: linked for local, (_, linked) in best.items()}
        next_start = starts[index + 1] if index + 1 < len(starts) else None
        earlier_frames = shared
        shared = {}
        for frame in frames:
            if legacy_stitch and index > 0 and frame.frame_index < overlap:
                continue
            global_index = start + frame.frame_index
            earlier_detections = earlier_frames.get(global_index, ())
            detections = []
            for detection in frame.detections:
                if not legacy_stitch and any(
                    _mask_iou(detection.mask, earlier.mask) >= LINK_IOU
                    for earlier in earlier_detections
                ):
                    continue
                if detection.object_id not in ids:
                    ids[detection.object_id] = next_id
                    next_id += 1
                detections.append(
                    replace(detection, object_id=ids[detection.object_id])
                )
            if not legacy_stitch and global_index in combined:
                detections = [*combined[global_index].detections, *detections]
            if next_start is not None and global_index >= next_start:
                shared[global_index] = tuple(detections)
            merged = replace(
                frame, frame_index=global_index, detections=tuple(detections)
            )
            if legacy_stitch:
                yield merged
            else:
                combined[global_index] = merged
        if not legacy_stitch:
            for frame_index in sorted(combined):
                if next_start is None or frame_index < next_start:
                    yield combined.pop(frame_index)
            shared = {
                frame_index: frame.detections
                for frame_index, frame in combined.items()
            }


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _warn_if_capped(
    frames: Sequence[FrameSegmentations], index: int, count: int
) -> None:
    objects = {d.object_id for frame in frames for d in frame.detections}
    if len(objects) >= OBJECT_CAP:
        print(
            f"Warning: hosted SAM tracked {len(objects)} objects in chunk "
            f"{index + 1}/{count}, its apparent limit; later objects in that "
            "chunk may be missing. Lower ROADLENS_REMOTE_CHUNK_FRAMES.",
            file=sys.stderr,
        )


async def _stream_events(stream: AsyncIterable[Any]) -> AsyncIterator[dict[str, Any]]:
    completed_lanes = set()
    async for event in stream:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        lane = (
            data.get("item_id"),
            data.get("output_index"),
            data.get("content_index"),
        )
        if data.get("type") == "response.output_text.done":
            if lane not in completed_lanes:
                completed_lanes.add(lane)
                yield data
            continue
        yield data
        part = data.get("part")
        if (
            data.get("type") == "response.content_part.done"
            and lane not in completed_lanes
            and isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ):
            completed_lanes.add(lane)
            yield {
                "type": "response.output_text.done",
                "item_id": data.get("item_id"),
                "output_index": data.get("output_index"),
                "content_index": data.get("content_index"),
                "text": part["text"],
            }
