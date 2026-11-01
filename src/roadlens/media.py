import json
import subprocess
from pathlib import Path


def probe_video(video_path: Path) -> dict[str, float | int]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
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
    }


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
