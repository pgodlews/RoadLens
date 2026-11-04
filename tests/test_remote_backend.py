import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from meta_sam_parser import (
    FrameReference,
    SegmentationBoxRecord,
    SegmentationMask,
    SegmentationMaskBounds,
    SegmentationMaskIdentity,
    SegmentationMaskRecord,
)

from roadlens.backends.remote import RemoteSamBackend, chunk_spans, stitch_chunks
from roadlens.domain import Detection, FrameSegmentations


def test_convert_remote_result_places_cropped_mask(monkeypatch):
    import meta_sam_parser

    monkeypatch.setattr(
        meta_sam_parser, "decode_mask_to_raster", lambda mask: iter([1, 0])
    )
    # Real parser records, so the test breaks if the live record shape changes.
    frame = FrameReference(frame_index=4)
    box = SegmentationBoxRecord(
        order=0, object_id="2", left=1, top=1, right=3, bottom=2, frame=frame
    )
    mask = SegmentationMaskRecord(
        order=1,
        object_id="2",
        identity=SegmentationMaskIdentity(
            media="video", frame_index=4, object_id="2"
        ),
        revision=1,
        mask=SegmentationMask(encoding="lossless", payload="~", width=2, height=1),
        bounds=SegmentationMaskBounds(left=1, top=1, right=3, bottom=2),
        frame=frame,
        confidence=0.75,
    )
    result = SimpleNamespace(
        outcome=SimpleNamespace(status="completed"), diagnostics=[], records=[box, mask]
    )

    frames = list(RemoteSamBackend._convert_result(result, "pothole", 4, 3, None))

    assert len(frames) == 1
    assert frames[0].backend == "remote"
    detection = frames[0].detections[0]
    assert detection.object_id == 2
    assert detection.score == 0.75
    assert detection.box_xywh == (0.25, 1 / 3, 0.5, 1 / 3)
    assert detection.mask.tolist() == [
        [False, False, False, False],
        [False, True, False, False],
        [False, False, False, False],
    ]


def test_convert_remote_result_rejects_incomplete_response():
    result = SimpleNamespace(
        outcome=SimpleNamespace(status="failed", reason="refused"),
        diagnostics=[],
        records=[],
    )

    try:
        list(RemoteSamBackend._convert_result(result, "pothole", 4, 3, None))
    except RuntimeError as error:
        assert "refused" in str(error)
    else:
        raise AssertionError("Incomplete remote response was accepted")


def _frame(index, *objects):
    """Frame with one 4x4 mask per (object_id, column) pair."""
    detections = []
    for object_id, column in objects:
        mask = np.zeros((4, 4), dtype=bool)
        mask[:, column] = True
        detections.append(
            Detection(object_id=object_id, box_xywh=(0, 0, 1, 1), score=0.7, mask=mask)
        )
    return FrameSegmentations(
        frame_index=index,
        prompt="pothole",
        backend="remote",
        detections=tuple(detections),
    )


def test_chunk_spans_cover_clip_with_overlap():
    assert chunk_spans(150, 200, 5) == [(0, 150)]
    spans = chunk_spans(1806, 200, 5)
    assert all(count <= 200 for _, count in spans)
    assert spans[0][0] == 0
    assert spans[-1][0] + spans[-1][1] == 1806
    for (start, count), (next_start, _) in itertools.pairwise(spans):
        assert start + count - next_start == 5


def test_stitch_links_tracks_across_chunks_and_keeps_ids_unique():
    first = [_frame(0, (0, 0)), _frame(8, (0, 1), (1, 3))]
    # Local id 0 in the second chunk overlaps first-chunk object 1 in the
    # shared frame 8 (local 0); local id 1 is new.
    second = [_frame(0, (0, 3)), _frame(3, (0, 3), (1, 2))]

    frames = list(stitch_chunks([first, second], [0, 8], overlap=2))

    assert [f.frame_index for f in frames] == [0, 8, 11]
    assert [d.object_id for d in frames[1].detections] == [0, 1]
    assert [d.object_id for d in frames[2].detections] == [1, 2]


def test_stitch_keeps_overlap_only_object_and_merges_source_frames():
    first = [_frame(8, (0, 0)), _frame(9, (0, 0))]
    second = [_frame(0, (0, 3)), _frame(1, (0, 3))]

    frames = list(stitch_chunks([first, second], [0, 8], overlap=2))

    assert [f.frame_index for f in frames] == [8, 9]
    assert [[d.object_id for d in f.detections] for f in frames] == [[0, 1], [0, 1]]
    assert all(f.detections[1].mask[:, 3].all() for f in frames)


def test_stitch_drops_only_same_frame_linked_duplicates():
    first = [_frame(8, (0, 0)), _frame(10, (0, 0))]
    second = [_frame(0, (2, 0)), _frame(1, (2, 1)), _frame(3, (2, 2))]

    frames = list(stitch_chunks([first, second], [0, 8], overlap=2))

    assert [f.frame_index for f in frames] == [8, 9, 10, 11]
    assert all(len(f.detections) == 1 for f in frames)
    assert all(f.detections[0].object_id == 0 for f in frames)
    assert frames[1].detections[0].mask[:, 1].all()


def test_stitch_legacy_drops_all_later_overlap_frames():
    first = [_frame(0, (0, 0)), _frame(8, (0, 1), (1, 3))]
    second = [_frame(0, (0, 3), (1, 2)), _frame(1, (2, 0)), _frame(3, (0, 3), (3, 2))]

    frames = list(stitch_chunks([first, second], [0, 8], overlap=2, legacy_stitch=True))

    assert [f.frame_index for f in frames] == [0, 8, 11]
    assert [[d.object_id for d in f.detections] for f in frames] == [
        [0],
        [0, 1],
        [1, 2],
    ]


def test_stitch_links_overlap_only_object_across_a_third_chunk():
    first = [_frame(8, (0, 0))]
    second = [_frame(0, (1, 3)), _frame(8, (1, 3))]
    third = [_frame(0, (2, 3)), _frame(2, (2, 3))]

    frames = list(stitch_chunks([first, second, third], [0, 8, 16], overlap=2))

    assert [f.frame_index for f in frames] == [8, 16, 18]
    assert [[d.object_id for d in f.detections] for f in frames] == [[0, 1], [1], [1]]


@pytest.mark.parametrize("max_frames", [None, 10, 180, 200])
def test_single_chunk_cuts_only_when_frame_limit_truncates_clip(
    monkeypatch, tmp_path, max_frames
):
    from roadlens.backends import remote

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"original")
    monkeypatch.setenv("ROADLENS_REMOTE_CHUNK_FRAMES", "200")
    monkeypatch.setattr(
        remote,
        "probe_video",
        lambda path: {"width": 4, "height": 4, "duration": 18.0, "fps": 10.0},
    )
    cuts = []

    def fake_cut(src, dst, start, count):
        cuts.append((src, dst, start, count))
        dst.write_bytes(b"limited")

    monkeypatch.setattr(remote, "cut_frame_range", fake_cut)
    requested = []

    async def fake_request(self, path, prompt):
        requested.append(path)
        assert path.read_bytes() == (b"limited" if max_frames == 10 else b"original")
        return object()

    monkeypatch.setattr(RemoteSamBackend, "_request", fake_request)
    limits = []

    def fake_convert(result, prompt, width, height, limit):
        limits.append(limit)
        return iter([_frame(0, (0, 0))])

    monkeypatch.setattr(RemoteSamBackend, "_convert_result", staticmethod(fake_convert))

    frames = list(
        RemoteSamBackend(api_key="test").segment_video(video, "pothole", max_frames)
    )

    assert len(frames) == 1
    assert limits == [max_frames]
    if max_frames == 10:
        assert cuts == [(video, requested[0], 0, 10)]
        assert requested[0] != video
        assert not requested[0].exists()
    else:
        assert cuts == []
        assert requested == [video]


@pytest.mark.parametrize("legacy_stitch", [False, True])
def test_backend_passes_legacy_stitch_to_chunk_merging(
    monkeypatch, tmp_path, legacy_stitch
):
    from roadlens.backends import remote

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    monkeypatch.setenv("ROADLENS_REMOTE_CHUNK_FRAMES", "100")
    monkeypatch.setattr(
        remote,
        "probe_video",
        lambda path: {"width": 4, "height": 4, "duration": 18.0, "fps": 10.0},
    )
    monkeypatch.setattr(remote, "cut_frame_range", lambda *args: None)

    async def fake_request(self, path, prompt):
        return path.name

    monkeypatch.setattr(RemoteSamBackend, "_request", fake_request)
    monkeypatch.setattr(
        RemoteSamBackend,
        "_convert_result",
        staticmethod(
            lambda result, *args: iter(
                [_frame(88, (0, 0))]
                if result == "chunk-000.mp4"
                else [_frame(0, (0, 3))]
            )
        ),
    )

    backend = RemoteSamBackend(api_key="test", legacy_stitch=legacy_stitch)
    frames = list(backend.segment_video(video, "pothole"))

    assert [f.frame_index for f in frames] == [88]
    assert len(frames[0].detections) == (1 if legacy_stitch else 2)


@pytest.mark.parametrize("legacy_stitch", [False, True])
def test_analyze_parser_exposes_legacy_stitch(legacy_stitch):
    from roadlens.cli import build_parser

    arguments = ["analyze", "clip.mp4", "--output", "results"]
    if legacy_stitch:
        arguments.append("--legacy-stitch")

    assert build_parser().parse_args(arguments).legacy_stitch is legacy_stitch


def test_long_clip_is_segmented_in_chunks(monkeypatch, tmp_path):
    from roadlens.backends import remote

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    monkeypatch.setenv("MODEL_API_KEY", "test")
    monkeypatch.setenv("ROADLENS_REMOTE_CHUNK_FRAMES", "100")
    monkeypatch.setattr(
        remote,
        "probe_video",
        lambda path: {"width": 4, "height": 4, "duration": 18.0, "fps": 10.0},
    )
    cuts = []
    monkeypatch.setattr(
        remote,
        "cut_frame_range",
        lambda src, dst, start, count: cuts.append((start, count)),
    )

    async def fake_request(self, path, prompt):
        return path.name

    monkeypatch.setattr(RemoteSamBackend, "_request", fake_request)
    monkeypatch.setattr(
        RemoteSamBackend,
        "_convert_result",
        staticmethod(lambda result, *args: iter([_frame(10, (0, 0))])),
    )

    frames = list(RemoteSamBackend().segment_video(video, "pothole"))

    assert cuts == [(0, 93), (88, 92)]
    assert [f.frame_index for f in frames] == [10, 98]
    assert [f.detections[0].object_id for f in frames] == [0, 1]


@pytest.mark.parametrize("legacy_stitch", [False, True])
def test_create_backend_forwards_legacy_stitch(monkeypatch, legacy_stitch):
    from roadlens.backends import create_backend

    monkeypatch.setenv("MODEL_API_KEY", "test")
    backend = create_backend("remote", legacy_stitch=legacy_stitch)
    assert backend._legacy_stitch is legacy_stitch
