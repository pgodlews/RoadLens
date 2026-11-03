import json
import subprocess
from pathlib import Path
from typing import Any


def probe_video(video_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=width,height,r_frame_rate:"
                "format=duration:format_tags=creation_time"
            ),
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(completed.stdout)
    stream = data["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration": float(data["format"]["duration"]),
        "fps": float(numerator) / float(denominator),
        "creation_time": data["format"].get("tags", {}).get("creation_time"),
    }


def prepare_analysis_clip(
    source: Path,
    output: Path,
    start_seconds: float,
    duration_seconds: float,
    fps: float,
    width: int = 1280,
) -> None:
    if start_seconds < 0 or duration_seconds <= 0 or fps <= 0 or width <= 0:
        raise ValueError("Clip timing, frame rate, and width must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(start_seconds),
            "-t",
            str(duration_seconds),
            "-i",
            str(source),
            "-vf",
            f"fps={fps},scale={width}:-2",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-y",
            str(output),
        ],
        check=True,
    )


def cut_frame_range(
    video_path: Path, output_path: Path, start_frame: int, frame_count: int
) -> None:
    """Re-encode frames ``[start_frame, start_frame + frame_count)`` exactly."""
    if start_frame < 0 or frame_count <= 0:
        raise ValueError("Frame range must start at 0 or later and be non-empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    end_frame = start_frame + frame_count
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"trim=start_frame={start_frame}:end_frame={end_frame},"
            "setpts=PTS-STARTPTS",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-y",
            str(output_path),
        ],
        check=True,
    )


def extract_frame(video_path: Path, frame_index: int, output_path: Path) -> None:
    if frame_index < 0:
        raise ValueError("frame_index cannot be negative")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-frames:v",
            "1",
            "-y",
            str(output_path),
        ],
        check=True,
    )
