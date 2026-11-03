import struct
from datetime import UTC, datetime, timedelta

import pytest

from roadlens import tesla
from roadlens.telemetry import TelemetryResult, TelemetrySample, read_camera_telemetry


def varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def sei_message(lat, lon, speed, heading=90.0, gear=1, seq=7):
    """SeiMetadata protobuf bytes, encoded field by field as in dashcam.proto."""
    return b"".join(
        [
            varint(1 << 3) + varint(1),
            varint(2 << 3) + varint(gear),
            varint(3 << 3) + varint(seq),
            varint(4 << 3 | 5) + struct.pack("<f", speed),
            varint(11 << 3 | 1) + struct.pack("<d", lat),
            varint(12 << 3 | 1) + struct.pack("<d", lon),
            varint(13 << 3 | 1) + struct.pack("<d", heading),
            # An unknown length-delimited field must be skipped, not fatal.
            varint(99 << 3 | 2) + varint(3) + b"abc",
        ]
    )


def escape(data):
    """Insert H.264 emulation-prevention bytes."""
    out, zeros = bytearray(), 0
    for byte in data:
        if zeros >= 2 and byte <= 3:
            out.append(3)
            zeros = 0
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


def sei_nal(message):
    body = b"\x42" * 3 + b"\x69" + message
    return bytes([0x06, 0x05, len(body)]) + escape(body) + b"\x80"


def test_decode_sei_metadata_reads_tesla_fields():
    decoded = tesla.decode_sei_metadata(sei_message(53.8, -1.55, 6.5))
    assert decoded["latitude_deg"] == 53.8
    assert decoded["longitude_deg"] == -1.55
    assert decoded["vehicle_speed_mps"] == pytest.approx(6.5)
    assert decoded["gear_state"] == 1
    assert decoded["frame_seq_no"] == 7
    with pytest.raises(ValueError, match="Truncated"):
        tesla.decode_sei_metadata(sei_message(53.8, -1.55, 6.5)[:-5])


def test_sei_payload_removes_emulation_prevention_bytes():
    # Zero latitude bytes force 00 00 0x sequences that H.264 must escape.
    message = sei_message(0.0, -1.55, 0.0)
    nal = sei_nal(message)
    assert b"\x00\x00\x03" in nal
    assert tesla.sei_payload(nal) == message
    assert tesla.sei_payload(b"\x65\x88\x84\x00") is None  # a picture slice


def write_clip(path, messages):
    """Minimal MP4: ftyp plus an mdat of length-prefixed SEI and slice NALs."""
    units, offsets = [], []
    position = 8 + 16 + 8  # ftyp (16 bytes) then the mdat header
    for message in messages:
        for nal in (sei_nal(message), b"\x65\x88\x84\x21\x00"):
            if nal[0] == 0x06:
                offsets.append(position)
            units.append(struct.pack(">I", len(nal)) + nal)
            position += 4 + len(nal)
    mdat = b"".join(units)
    path.write_bytes(
        struct.pack(">I4s", 16, b"ftyp")
        + b"isom\x00\x00\x00\x00"
        + struct.pack(">I4s", 8, b"free")
        + struct.pack(">I4s", 8 + len(mdat), b"mdat")
        + mdat
    )
    return offsets


def test_read_tesla_telemetry_times_each_frame(tmp_path, monkeypatch):
    clip = tmp_path / "2026-10-24_09-20-49-front.mp4"
    offsets = write_clip(
        clip,
        [
            sei_message(53.8, -1.55, 6.0),
            sei_message(0.0, 0.0, 6.0),  # no GPS fix yet
            sei_message(53.8001, -1.5501, -0.04),  # reversing / standstill
        ],
    )
    monkeypatch.setattr(
        tesla,
        "probe_video",
        lambda path: {"creation_time": "2026-10-24T08:20:49.000000Z"},
    )
    monkeypatch.setattr(
        tesla, "packet_times", lambda path: (offsets, [0.0, 1 / 36, 2 / 36])
    )

    result = tesla.read_tesla_telemetry(clip)

    assert result.source == "tesla_sei"
    assert result.rejected_samples == 1
    assert [s.playback_seconds for s in result.samples] == [0.0, 2 / 36]
    start = datetime(2026, 10, 24, 8, 20, 49, tzinfo=UTC)
    assert result.samples[1].recorded_utc == start + timedelta(seconds=2 / 36)
    assert result.samples[0].speed_mps == pytest.approx(6.0)
    assert result.samples[1].speed_mps == pytest.approx(0.04)
    assert result.location_source == "tesla_camera_gps"


def test_camera_telemetry_prefers_garmin_then_tesla(monkeypatch, tmp_path):
    clip = tmp_path / "clip.mp4"
    sample = TelemetrySample(datetime(2026, 1, 1, tzinfo=UTC), 53, -1, 5, 0.0)
    garmin = TelemetryResult((sample,), 0)
    tesla_result = TelemetryResult((sample,), 0, source="tesla_sei")
    empty = TelemetryResult((), 2)

    monkeypatch.setattr("roadlens.telemetry.read_garmin_telemetry", lambda p: garmin)
    monkeypatch.setattr(tesla, "read_tesla_telemetry", lambda p: tesla_result)
    assert read_camera_telemetry(clip).source == "garmin"

    monkeypatch.setattr("roadlens.telemetry.read_garmin_telemetry", lambda p: empty)
    assert read_camera_telemetry(clip).source == "tesla_sei"

    monkeypatch.setattr(tesla, "read_tesla_telemetry", lambda p: empty)
    parked = read_camera_telemetry(clip)
    assert parked.source == "none"
    assert not parked.samples
    assert parked.location_source == "camera_gps"
