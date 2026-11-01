from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from contextlib import aclosing
from pathlib import Path
from typing import Any

import numpy as np

from roadlens.domain import Detection, FrameSegmentations
from roadlens.media import probe_video


class RemoteSamBackend:
    name = "remote"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("MODEL_API_KEY")
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
        result = asyncio.run(self._request(video_path, prompt))
        yield from self._convert_result(
            result,
            prompt,
            metadata["width"],
            metadata["height"],
            max_frames,
        )

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
            stream = await client.responses.create(
                model="sam-3.1",
                stream=True,
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
                parsed = parse_responses_stream(events, video_segmentation_format())
                async with parsed:
                    return await parsed.final_result()

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
            left = int(record.left)
            top = int(record.top)
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
                    score=getattr(record, "score", None),
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
