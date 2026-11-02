from types import SimpleNamespace

import numpy as np

from roadlens.backends.local import (
    LocalSamBackend,
    _start_multiplex_session,
    _wrap_clear_all_points_in_frame,
)


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


def test_multiplex_clear_adapter_promotes_remaining_object_conditioning_frame():
    resets = []

    def fake_clear(
        inference_state,
        frame_idx,
        obj_id,
        need_output=True,
        preserve_user_refined=False,
    ):
        obj_idx = inference_state["obj_id_to_idx"][obj_id]
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        out = inference_state["output_dict"]["cond_frame_outputs"].pop(frame_idx, None)
        if out is not None:
            inference_state["output_dict"]["non_cond_frame_outputs"][frame_idx] = out
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].discard(
            frame_idx
        )
        if not inference_state["output_dict"]["cond_frame_outputs"]:
            resets.append(True)
            inference_state["output_dict"]["non_cond_frame_outputs"].clear()
            for store in ("point_inputs_per_obj", "mask_inputs_per_obj"):
                for per_obj in inference_state[store].values():
                    per_obj.clear()

    wrapped = _wrap_clear_all_points_in_frame(fake_clear)
    state = {
        "obj_ids": [10, 20],
        "obj_id_to_idx": {10: 0, 20: 1},
        "point_inputs_per_obj": {0: {}, 1: {}},
        "mask_inputs_per_obj": {0: {0: "mask0"}, 1: {15: "mask15"}},
        "output_dict": {
            "cond_frame_outputs": {0: {"frame": 0}},
            "non_cond_frame_outputs": {15: {"frame": 15}},
        },
        "output_dict_per_obj": {
            0: {"cond_frame_outputs": {0: "o0"}, "non_cond_frame_outputs": {}},
            1: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {15: "o15"}},
        },
        "consolidated_frame_inds": {
            "cond_frame_outputs": {0},
            "non_cond_frame_outputs": {15},
        },
        "first_ann_frame_idx": 0,
    }

    wrapped(state, 0, 10, need_output=False)
    assert resets == []
    assert set(state["output_dict"]["cond_frame_outputs"]) == {15}
    assert state["consolidated_frame_inds"]["cond_frame_outputs"] == {15}
    assert state["output_dict_per_obj"][1]["cond_frame_outputs"] == {15: "o15"}
    assert state["mask_inputs_per_obj"][1] == {15: "mask15"}
    assert state["first_ann_frame_idx"] == 15

    # Removing the final object still resets tracking state cleanly.
    wrapped(state, 15, 20, need_output=False)
    assert resets == [True]
    assert state["output_dict"]["cond_frame_outputs"] == {}
