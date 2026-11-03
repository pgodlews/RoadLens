"""Latency and throughput of hosted SAM 3.1 (Meta Model API) for a live stream.

Image mode sends one request per frame as a base64 JPEG, first one at a time
at several resolutions, then with several requests in flight. Video mode
uploads short clips as files and times them from upload start to final
result; each upload is deleted afterwards.

Billed: $2.50 per 1,000 images and $0.20 per 1,000 video frames (minimum 50
frames per request). The defaults send about 270 images and three clips.

    MODEL_API_KEY=... python hosted_sam.py --frames frames/ --video ride.mp4 \\
        --video-start 584 --out result.json
"""

import argparse
import asyncio
import base64
import io
import json
import os
import statistics
import subprocess
import tempfile
import time
from contextlib import aclosing
from pathlib import Path

from meta_sam_parser import (
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)
from openai import AsyncOpenAI
from PIL import Image

from roadlens.backends.remote import _stream_events

URL = "https://api.meta.ai/v1"


def jpeg_data_url(path: Path, size: tuple[int, int]) -> tuple[str, int]:
    buffer = io.BytesIO()
    Image.open(path).convert("RGB").resize(size, Image.BILINEAR).save(
        buffer, "JPEG", quality=85
    )
    data = buffer.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(data).decode(), len(data)


def image_part(url: str) -> dict:
    return {"type": "input_image", "image_url": url}


async def segment(client, prompt, part, fmt):
    """Seconds to the final result and to the first streamed event, and scores."""
    start = time.perf_counter()
    stream = await client.responses.create(
        model="sam-3.1",
        stream=True,
        metadata={"include_confidence": "true"},
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}, part],
            }
        ],
    )
    first = None

    async def timed(events):
        nonlocal first
        async for event in events:
            if first is None:
                first = time.perf_counter() - start
            yield event

    async with stream, aclosing(timed(_stream_events(stream))) as events:
        parsed = parse_responses_stream(events, fmt)
        async with parsed:
            result = await parsed.final_result()
    scores = [
        round(r.confidence, 3)
        for r in result.records
        if r.kind == "mask" and r.confidence is not None
    ]
    return time.perf_counter() - start, first, scores


def summary(latencies: list[float]) -> dict:
    latencies = sorted(latencies)
    return {
        "median_s": round(statistics.median(latencies), 2),
        "p90_s": round(latencies[int(0.9 * (len(latencies) - 1))], 2),
        "min_s": round(latencies[0], 2),
    }


def parse_size(text: str) -> tuple[int, int]:
    w, h = (int(v) for v in text.split("x"))
    return w, h


async def sequential_runs(client, frames, prompt, inputs):
    runs = []
    for size in inputs:
        urls = [jpeg_data_url(f, parse_size(size)) for f in frames]
        fmt = image_segmentation_format()
        await segment(client, prompt, image_part(urls[0][0]), fmt)  # warm-up
        latencies, firsts, found = [], [], []
        for url, _ in urls:
            seconds, first, scores = await segment(
                client, prompt, image_part(url), image_segmentation_format()
            )
            latencies.append(seconds)
            firsts.append(first)
            found.append((len(scores), max(scores) if scores else None))
        run = {
            "mode": "image_sequential",
            "input": size,
            "jpeg_kb": round(statistics.mean(n for _, n in urls) / 1024),
            **summary(latencies),
            "first_event_median_s": round(statistics.median(firsts), 2),
            "fps": round(1 / statistics.median(latencies), 2),
            "detections": found,
        }
        print(json.dumps(run), flush=True)
        runs.append(run)
    return runs


async def concurrent_runs(client, frames, prompt, size, levels, requests):
    urls = [jpeg_data_url(f, parse_size(size))[0] for f in frames]
    runs = []
    for in_flight in levels:
        gate = asyncio.Semaphore(in_flight)

        async def one(i, gate=gate):
            async with gate:
                return await segment(
                    client,
                    prompt,
                    image_part(urls[i % len(urls)]),
                    image_segmentation_format(),
                )

        start = time.perf_counter()
        results = await asyncio.gather(*(one(i) for i in range(requests)))
        wall = time.perf_counter() - start
        run = {
            "mode": "image_concurrent",
            "input": size,
            "in_flight": in_flight,
            "requests": requests,
            "wall_s": round(wall, 1),
            "throughput_fps": round(requests / wall, 2),
            **summary([r[0] for r in results]),
        }
        print(json.dumps(run), flush=True)
        runs.append(run)
    return runs


async def video_runs(client, source, start_s, prompt, lengths, fps, width):
    runs = []
    for seconds in lengths:
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-ss", str(start_s), "-t", str(seconds), "-i", str(source),
                    "-vf", f"fps={fps},scale={width}:-2", "-an",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-y", str(clip),
                ],
                check=True,
            )  # fmt: skip
            begin = time.perf_counter()
            uploaded = await client.files.create(
                file=("clip.mp4", clip.read_bytes(), "video/mp4"),
                purpose="user_data",
            )
            upload_s = time.perf_counter() - begin
            try:
                segment_s, first, _ = await segment(
                    client,
                    prompt,
                    {"type": "input_video", "file_id": uploaded.id},
                    video_segmentation_format(),
                )
            finally:
                await client.files.delete(uploaded.id)
        run = {
            "mode": "video_chunk",
            "seconds": seconds,
            "frames": round(seconds * fps),
            "width": width,
            "upload_s": round(upload_s, 2),
            "segment_s": round(segment_s, 2),
            "first_event_s": round(first, 2),
            "end_to_end_s": round(upload_s + segment_s, 2),
            "realtime_factor": round((upload_s + segment_s) / seconds, 2),
        }
        print(json.dumps(run), flush=True)
        runs.append(run)
    return runs


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-start", type=float, default=0.0)
    parser.add_argument("--prompt", default="pothole")
    parser.add_argument(
        "--inputs", nargs="+", default=["2688x2160", "1344x1080", "672x540"]
    )
    parser.add_argument("--concurrent-input", default="672x540")
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[4, 8, 16, 32, 48]
    )
    parser.add_argument("--concurrent-requests", type=int, default=48)
    parser.add_argument("--video-seconds", type=float, nargs="+", default=[1, 2, 5])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frames = sorted(args.frames.glob("*.png"))
    key = os.environ["MODEL_API_KEY"]
    async with AsyncOpenAI(base_url=URL, api_key=key) as client:
        runs = await sequential_runs(client, frames, args.prompt, args.inputs)
        runs += await concurrent_runs(
            client,
            frames,
            args.prompt,
            args.concurrent_input,
            args.concurrency,
            args.concurrent_requests,
        )
        runs += await video_runs(
            client,
            args.video,
            args.video_start,
            args.prompt,
            args.video_seconds,
            10,
            1280,
        )
    args.out.write_text(json.dumps({"runs": runs}, indent=1) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
