from types import SimpleNamespace

from meta_sam_parser import (
    FrameReference,
    SegmentationBoxRecord,
    SegmentationMask,
    SegmentationMaskBounds,
    SegmentationMaskIdentity,
    SegmentationMaskRecord,
)

from roadlens.backends.remote import RemoteSamBackend


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
