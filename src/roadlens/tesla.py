"""Tesla dashcam telemetry from SEI messages embedded in the H.264 stream.

Tesla firmware 2025.44.25+ on HW3 and later writes one protobuf message per
video frame while the car is in gear (none while parked): GPS, speed, heading,
gear and Autopilot state. The message layout follows ``dashcam.proto`` in
github.com/teslamotors/dashcam; this is an independent, dependency-free
decoder for it. Timing comes from the MP4 ``creation_time`` (UTC clip start)
plus each frame's presentation timestamp.
"""

from __future__ import annotations

import bisect
import json
import struct
import subprocess
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from roadlens.media import probe_video
from roadlens.telemetry import TelemetryResult, TelemetrySample

SOURCE = "tesla_sei"

# Field number -> (name, wire type) from dashcam.proto (SeiMetadata).
_FIELDS = {
    1: ("version", 0),
    2: ("gear_state", 0),
    3: ("frame_seq_no", 0),
    4: ("vehicle_speed_mps", 5),
    5: ("accelerator_pedal_position", 5),
    6: ("steering_wheel_angle", 5),
    7: ("blinker_on_left", 0),
    8: ("blinker_on_right", 0),
    9: ("brake_applied", 0),
    10: ("autopilot_state", 0),
    11: ("latitude_deg", 1),
    12: ("longitude_deg", 1),
    13: ("heading_deg", 1),
    14: ("linear_acceleration_mps2_x", 1),
    15: ("linear_acceleration_mps2_y", 1),
    16: ("linear_acceleration_mps2_z", 1),
}
GEARS = {0: "park", 1: "drive", 2: "reverse", 3: "neutral"}
AUTOPILOT = {0: "none", 1: "self_driving", 2: "autosteer", 3: "tacc"}


def _varint(data: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, i
        shift += 7
        if shift > 63:
            raise ValueError("Varint too long")


def decode_sei_metadata(data: bytes) -> dict[str, Any]:
    """Decode one SeiMetadata protobuf message (proto3 defaults omitted)."""
    out: dict[str, Any] = {}
    i = 0
    try:
        while i < len(data):
            key, i = _varint(data, i)
            field, wire = key >> 3, key & 7
            if wire == 0:
                value, i = _varint(data, i)
            elif wire == 1:
                (value,) = struct.unpack_from("<d", data, i)
                i += 8
            elif wire == 5:
                (value,) = struct.unpack_from("<f", data, i)
                i += 4
            elif wire == 2:
                length, i = _varint(data, i)
                i += length
                continue
            else:
                raise ValueError(f"Unsupported protobuf wire type {wire}")
            name = _FIELDS.get(field, (None,))[0]
            if name is not None:
                out[name] = value
    except (IndexError, struct.error) as error:
        raise ValueError("Truncated SEI metadata") from error
    if i != len(data):
        raise ValueError("Truncated SEI metadata")
    return out


def _unescape(data: bytes) -> bytes:
    """Remove H.264 emulation-prevention bytes (00 00 03 -> 00 00)."""
    out = bytearray()
    zeros = 0
    for byte in data:
        if zeros >= 2 and byte == 0x03:
            zeros = 0
            continue
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


def sei_payload(nal: bytes) -> bytes | None:
    """Tesla protobuf bytes from an SEI user-data-unregistered NAL, or None.

    After the NAL header, payload type (5) and payload size, Tesla writes a
    run of 0x42 bytes, a 0x69 byte, the protobuf message and the RBSP
    trailing byte.
    """
    if len(nal) < 4 or nal[0] & 0x1F != 6 or nal[1] != 5:
        return None
    body = _unescape(nal[2:])
    i = 0
    while i < len(body) and body[i] == 0xFF:  # payload size, 0xFF-extended
        i += 1
    i += 1
    start = i
    while i < len(body) and body[i] == 0x42:
        i += 1
    if i == start or i + 1 >= len(body) or body[i] != 0x69:
        return None
    return body[i + 1 : -1]


def _mdat_range(fp) -> tuple[int, int]:
    fp.seek(0, 2)
    end = fp.tell()
    fp.seek(0)
    while fp.tell() + 8 <= end:
        start = fp.tell()
        size, kind = struct.unpack(">I4s", fp.read(8))
        header = 8
        if size == 1:
            size, header = struct.unpack(">Q", fp.read(8))[0], 16
        elif size == 0:
            size = end - start
        if size < header:
            raise ValueError("Invalid MP4 atom size")
        if kind == b"mdat":
            return start + header, start + size
        fp.seek(start + size)
    raise ValueError("MP4 has no mdat atom")


def iter_sei(video_path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """(file offset, decoded message) for every Tesla SEI NAL in the video."""
    with open(video_path, "rb") as fp:
        lo, hi = _mdat_range(fp)
        fp.seek(lo)
        while fp.tell() + 5 <= hi:
            offset = fp.tell()
            (size,) = struct.unpack(">I", fp.read(4))
            if size < 2 or offset + 4 + size > hi:
                break
            head = fp.read(2)
            if head[0] & 0x1F == 6 and head[1] == 5:
                payload = sei_payload(head + fp.read(size - 2))
                if payload:
                    try:
                        yield offset, decode_sei_metadata(payload)
                    except ValueError:
                        pass
            fp.seek(offset + 4 + size)


def packet_times(video_path: Path) -> tuple[list[int], list[float]]:
    """File offsets and source-relative presentation times of video packets."""
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time,pos:stream=start_time",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(completed.stdout)
    origin = float((data.get("streams") or [{}])[0].get("start_time") or 0)
    rows = sorted(
        (int(p["pos"]), float(p["pts_time"]) - origin)
        for p in data.get("packets", [])
        if p.get("pos") not in (None, "N/A") and p.get("pts_time") not in (None, "N/A")
    )
    return [p for p, _ in rows], [t for _, t in rows]


def read_tesla_telemetry(video_path: Path) -> TelemetryResult:
    """Per-frame GPS and speed samples from a Tesla dashcam clip."""
    metadata = probe_video(video_path)
    if not metadata.get("creation_time"):
        raise ValueError("Tesla clip has no creation time")
    start = datetime.fromisoformat(metadata["creation_time"].replace("Z", "+00:00"))
    positions, times = packet_times(video_path)
    samples, rejected = [], 0
    for offset, meta in iter_sei(video_path):
        index = bisect.bisect_right(positions, offset) - 1
        if index < 0:
            rejected += 1
            continue
        latitude = meta.get("latitude_deg", 0.0)
        longitude = meta.get("longitude_deg", 0.0)
        if (latitude == 0 and longitude == 0) or not (
            -90 <= latitude <= 90 and -180 <= longitude <= 180
        ):
            rejected += 1
            continue
        speed = meta.get("vehicle_speed_mps")
        seconds = times[index]
        samples.append(
            TelemetrySample(
                recorded_utc=start + timedelta(seconds=seconds),
                latitude=latitude,
                longitude=longitude,
                # Reversing reports negative speed; distance sampling needs
                # how far the camera moved, not which way.
                speed_mps=abs(speed) if speed is not None else 0.0,
                playback_seconds=seconds,
            )
        )
    samples.sort(key=lambda sample: sample.recorded_utc)
    return TelemetryResult(tuple(samples), rejected, source=SOURCE)
