from types import SimpleNamespace

import numpy as np

from roadlens.backends.local import LocalSamBackend, _start_multiplex_session


def test_convert_frame_normalises_meta_output_and_skips_empty_masks():
    outputs = {
        "out_obj_ids": np.array([3, 4]),
        "out_probs": np.array([0.9, 0.8]),
        "out_boxes_xywh": np.array([[0.1, 0.2, 0.3, 0.4], [0, 0, 0, 0]]),
        "out_binary_masks": np.array(
            [[[False, True], [False, True]], [[False, False], [False, False]]]
        ),
    }

    result = LocalSamBackend._convert_frame(7, outputs, "pothole")

    assert result.frame_index == 7
    assert result.backend == "local"
    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.object_id == 3
    assert detection.score == 0.9
    assert detection.box_xywh == (0.1, 0.2, 0.3, 0.4)
    assert detection.mask.sum() == 2


def test_multiplex_session_adapter_omits_unsupported_offload_argument():
    calls = []

    class Model:
        def init_state(self, **kwargs):
            calls.append(kwargs)
            return {"state": 1}

    predictor = SimpleNamespace(
        model=Model(), async_loading_frames=True, _all_inference_states={}
    )
    response = _start_multiplex_session(predictor, "clip.mp4", session_id="test")

    assert response == {"session_id": "test"}
    assert calls == [
        {
            "resource_path": "clip.mp4",
            "offload_video_to_cpu": False,
            "async_loading_frames": True,
        }
    ]
    assert predictor._all_inference_states["test"]["state"] == {"state": 1}


def test_propagation_keeps_revised_first_frame_and_closes_session(tmp_path):
    calls = []
    empty = {
        "out_obj_ids": [],
        "out_probs": [],
        "out_boxes_xywh": [],
        "out_binary_masks": [],
    }
    revised = {
        "out_obj_ids": [0],
        "out_probs": [0.8],
        "out_boxes_xywh": [[0.1, 0.1, 0.2, 0.2]],
        "out_binary_masks": [np.ones((4, 4), dtype=bool)],
    }

    class Predictor:
        def handle_request(self, request):
            calls.append(request["type"])
            if request["type"] == "start_session":
                return {"session_id": "test"}
            return {"outputs": empty}

        def handle_stream_request(self, request):
            yield {"frame_index": 0, "outputs": revised}
            yield {"frame_index": 1, "outputs": empty}

    backend = LocalSamBackend.__new__(LocalSamBackend)
    backend._predictor = Predictor()
    frames = list(backend.segment_video(tmp_path, "pothole"))
    assert len(frames) == 2
    assert len(frames[0].detections) == 1
    assert calls[-1] == "close_session"
