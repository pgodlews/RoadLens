from types import SimpleNamespace

from roadlens.backends.remote import RemoteSamBackend


def test_convert_remote_result_places_cropped_mask(monkeypatch):
    import meta_sam_parser

    monkeypatch.setattr(
        meta_sam_parser, "decode_mask_to_raster", lambda mask: iter([1, 0])
    )
    frame = SimpleNamespace(frame_index=4)
    box = SimpleNamespace(
        kind="box",
        frame=frame,
        object_id="2",
        left=1,
        top=1,
        right=3,
        bottom=2,
    )
    mask = SimpleNamespace(
        kind="mask",
        frame=frame,
        object_id="2",
        left=1,
        top=1,
        mask=SimpleNamespace(width=2, height=1),
        score=0.75,
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
