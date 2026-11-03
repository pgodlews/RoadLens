import itertools
from types import SimpleNamespace

import numpy as np
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
