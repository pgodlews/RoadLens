import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from roadlens.cli import build_parser
from roadlens.domain import Detection, FrameSegmentations
from roadlens.sampling import FrameChoice
from roadlens.scan import TrackRegistry, run_pass, run_scan, sampling_settings
from roadlens.scan_media import measure_frames, save_selected_frames
from roadlens.telemetry import TelemetryResult, TelemetrySample


@pytest.fixture
def video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("av")
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    rng = np.random.default_rng(1)
    for _ in range(30):
        writer.write(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def test_decoder_round_trip_keeps_nonzero_source_timestamps(video, tmp_path):
    times, sharpness = measure_frames(video, 0.7, 1.4)
    assert times == pytest.approx([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    selected = [FrameChoice(times[i], sharpness[i], 0, None, "test") for i in [0, 3, 6]]
    paths = save_selected_frames(video, selected, tmp_path / "frames")
    assert len(paths) == 3
    assert all(p.is_file() for p in paths)


def test_scan_audits_empty_frames_refines_and_exports_source_times(
    video,
    tmp_path,
    monkeypatch,
):
    created = datetime(2026, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(
        "roadlens.scan.probe_video",
        lambda path: {
            "creation_time": created.isoformat(),
            "duration": 3,
            "fps": 10,
            "width": 64,
            "height": 48,
        },
    )
    samples = tuple(
        TelemetrySample(created + timedelta(seconds=t), 53, -1, 4, t) for t in range(4)
    )
    monkeypatch.setattr(
        "roadlens.scan.read_camera_telemetry", lambda path: TelemetryResult(samples, 0)
    )

    class Backend:
        thresholds = {"output": 0.25, "detection": 0.25, "new_object": 0.35}

        def __init__(self, *args, **kwargs):
            pass

        def segment_video(self, path, prompt):
            for index, _ in enumerate(sorted(path.glob("*.png"))):
                mask = np.zeros((48, 64), dtype=bool)
                mask[20:28, 25:35] = True
                detections = (
                    ()
                    if index == 0
                    else (Detection(1, (25 / 64, 20 / 48, 10 / 64, 8 / 48), 0.8, mask),)
                )
                yield FrameSegmentations(index, prompt, "local", detections)

    output = tmp_path / "scan"
    args = build_parser().parse_args(
        [
            "scan",
            str(video),
            "--start",
            "0.7",
            "--duration",
            "1.5",
            "--output",
            str(output),
            "--chunk-frames",
            "8",
            "--overlap-frames",
            "2",
        ]
    )
    result = run_scan(args, Backend)
    assert result["status"] == "complete"
    assert result["refinement_windows"]
    assert result["frame_efficiency"]["refine_mode"] == "distance"
    assert result["coverage"]["selection_reasons"]["start"] == 1
    ledger = [
        json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()
    ]
    assert any(not frame["detections"] for frame in ledger)
    assert any(frame["pass"].startswith("fine") for frame in ledger)
    assert all(frame["source_seconds"] >= 0.7 for frame in ledger)
    observations = json.loads((output / "observations.json").read_text())
    assert len(observations) == 1
    assert observations[0]["best_source_seconds"] >= 0.7
    assert observations[0]["latitude"] == 53
    assert (output / "overlay-000000.png").is_file()
    with pytest.raises(ValueError, match="empty output"):
        run_scan(args, Backend)

    # Simulate interruption after completed receipts, plus a partial next chunk.
    manifest_path = output / "scan.json"
    previous = json.loads(manifest_path.read_text())
    previous["status"] = "interrupted"
    manifest_path.write_text(json.dumps(previous))
    with (output / "frames.jsonl").open("a") as file:
        file.write('{"pass":"coarse","chunk_offset":999,"frame_index":0}\n')
        file.write('{"unfinished":')

    class NoMoreInference(Backend):
        def segment_video(self, path, prompt):
            raise AssertionError("Completed chunks should be replayed, not inferred")

    monkeypatch.setattr(
        "roadlens.scan.save_selected_frames",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("Existing coarse/fine PNGs should be reused on resume")
        ),
    )
    args.resume = True
    resumed = run_scan(args, NoMoreInference)
    assert resumed["status"] == "complete"
    assert resumed["observations"] == result["observations"]
    assert resumed["total_inference_seconds"] == result["total_inference_seconds"]
    assert len(resumed["resume_history"]) == 1
    assert json.loads((output / "observations.json").read_text()) == observations
    # A completed resume still validates that the source is the same file.
    assert run_scan(args, NoMoreInference)["status"] == "complete"
    original = video.read_bytes()
    video.write_bytes(original + b"\0")
    with pytest.raises(ValueError, match="Source file changed"):
        run_scan(args, NoMoreInference)
    video.unlink()
    with pytest.raises(FileNotFoundError):
        run_scan(args, NoMoreInference)
    video.write_bytes(original)
    args.spacing = 2
    with pytest.raises(ValueError, match="same scan settings"):
        run_scan(args, NoMoreInference)
    assert json.loads(manifest_path.read_text())["status"] == "complete"


def test_track_registry_only_merges_with_shared_visual_evidence(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), np.full((48, 64, 3), 128, dtype=np.uint8))
    registry = TrackRegistry()
    mask = np.ones((48, 64), dtype=bool)

    def add(time, box, identities):
        frame = FrameSegmentations(
            0, "pothole", "local", (Detection(0, box, 0.8, mask),)
        )
        registry.add(
            identities, frame, FrameChoice(time, 1, 1, 1, "test"), path, "test"
        )

    add(10, (0.1, 0.1, 0.2, 0.2), {})
    add(10, (0.1, 0.1, 0.2, 0.2), {})  # Same PTS and box: overlap duplicate.
    add(10, (0.6, 0.6, 0.2, 0.2), {})  # Same PTS, different defect.
    add(11, (0.1, 0.1, 0.2, 0.2), {})  # No shared PTS: don't assume identity.
    assert len(registry.merged()) == 3


def test_retried_interrupted_chunk_does_not_duplicate_ledger_rows(tmp_path):
    class Backend:
        def __init__(self, fail_after=None):
            self.fail_after = fail_after

        def segment_video(self, path, prompt):
            for index, _ in enumerate(sorted(path.glob("*.png"))):
                if self.fail_after is not None and index == self.fail_after:
                    raise RuntimeError("simulated crash mid-chunk")
                yield FrameSegmentations(index, prompt, "local", ())

    choices = [FrameChoice(t, 1, t, 1, "test") for t in range(6)]
    paths = [tmp_path / f"{i}.png" for i in range(6)]
    for path in paths:
        path.touch()

    def run(backend):
        return run_pass(
            backend, choices, paths, "coarse", tmp_path, TrackRegistry(), 4, 1
        )

    # First attempt: chunk 0 completes, chunk 3 crashes after writing 2 rows.
    calls = iter([Backend(), Backend(fail_after=2)])

    class Flaky:
        def segment_video(self, path, prompt):
            return next(calls).segment_video(path, prompt)

    with pytest.raises(RuntimeError, match="simulated crash"):
        run(Flaky())
    # Retry completes the interrupted chunk; a further resume replays both.
    run(Backend())
    run(Backend(fail_after=0))  # Would raise if any chunk were re-inferred.
    ledger = (tmp_path / "frames.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in ledger]
    keys = [(row["chunk_offset"], row["frame_index"]) for row in rows]
    assert len(keys) == len(set(keys)) == 4 + 3


def test_incomplete_backend_output_cannot_pass_as_complete_coverage(tmp_path):
    class Backend:
        def segment_video(self, path, prompt):
            yield FrameSegmentations(0, prompt, "local", ())

    choices = [FrameChoice(t, 1, t, 1, "test") for t in (0, 1)]
    with pytest.raises(ValueError, match="every requested frame"):
        run_pass(
            Backend(),
            choices,
            [tmp_path / "a.png", tmp_path / "b.png"],
            "coarse",
            tmp_path,
            TrackRegistry(),
            4,
            1,
        )


def test_failure_manifest_does_not_claim_a_completed_scan(tmp_path):
    args = build_parser().parse_args(
        [
            "scan",
            str(tmp_path / "missing.mp4"),
            "--start",
            "-1",
            "--output",
            str(tmp_path / "failed"),
        ]
    )
    with pytest.raises(ValueError, match="nonnegative"):
        run_scan(args)
    manifest = json.loads((args.output / "scan.json").read_text())
    assert manifest["status"] == "failed"


def test_frame_efficiency_flags_default_on_and_can_be_turned_off():
    parser = build_parser()
    base = ["scan", "clip.mp4", "--output", "out"]

    default = sampling_settings(parser.parse_args(base))
    assert default["max_interval"] is None
    assert default["motion"].stop_speed_mps == 0.5
    assert default["motion"].stop_heartbeat_seconds == 10.0
    assert default["refine_mode"] == "distance"

    legacy = sampling_settings(parser.parse_args([*base, "--legacy-sampling"]))
    assert legacy["max_interval"] == 0.5
    assert legacy["motion"].stop_speed_mps is None
    assert legacy["refine_mode"] == "time"

    piecemeal = sampling_settings(
        parser.parse_args(
            [*base, "--stop-speed", "0", "--stop-heartbeat", "0"]
            + ["--refine-mode", "time", "--max-interval", "0.5"]
        )
    )
    assert piecemeal["motion"].stop_speed_mps is None
    assert piecemeal["motion"].stop_heartbeat_seconds is None
    assert piecemeal["refine_mode"] == "time"
    assert piecemeal["max_interval"] == 0.5


def test_frame_efficiency_defaults_apply_to_callers_without_the_flags():
    from argparse import Namespace

    settings = sampling_settings(Namespace())
    assert settings["motion"].stop_speed_mps == 0.5
    assert settings["refine_mode"] == "distance"
    assert sampling_settings(Namespace(legacy_sampling=True))["refine_mode"] == "time"

