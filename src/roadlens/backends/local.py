from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np

from roadlens.domain import Detection, FrameSegmentations


class LocalSamBackend:
    name = "local"

    def __init__(
        self,
        checkpoint_path: Path,
        confidence_threshold: float = 0.5,
        detection_threshold: float | None = None,
        new_object_threshold: float | None = None,
        grounding_batch_size: int = 4,
    ):
        if grounding_batch_size < 1:
            raise ValueError("Grounding batch size must be positive")
        for value in (confidence_threshold, detection_threshold, new_object_threshold):
            if value is not None and not 0 <= value <= 1:
                raise ValueError("SAM thresholds must be between 0 and 1")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

        import torch
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        if not torch.cuda.is_available():
            raise RuntimeError("The local backend requires a CUDA-capable GPU")

        self._predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint_path),
            use_fa3=False,
            compile=False,
            default_output_prob_thresh=confidence_threshold,
        )
        # SAM 3.1's shared base passes a state-offload argument its multiplex
        # model does not accept. Keep the compatibility adapter in RoadLens.
        self._predictor.start_session = MethodType(
            _start_multiplex_session, self._predictor
        )
        # In dynamic multiplex tracking, new objects added to an existing bucket
        # on an already-tracked frame store their initial mask in
        # non_cond_frame_outputs. If all earlier cond_frame_outputs objects are
        # later removed, upstream clear_all_points_in_frame resets the entire
        # bucket and crashes on the next frame ("No points are provided").
        tracker_model = self._predictor.model.tracker.model
        tracker_model.clear_all_points_in_frame = _wrap_clear_all_points_in_frame(
            tracker_model.clear_all_points_in_frame
        )
        if detection_threshold is not None:
            self._predictor.model.score_threshold_detection = detection_threshold
            self._predictor.model.image_only_det_thresh = detection_threshold
        if new_object_threshold is not None:
            self._predictor.model.new_det_thresh = new_object_threshold
        self._predictor.model.batched_grounding_batch_size = grounding_batch_size

    @property
    def thresholds(self) -> dict[str, float]:
        model = self._predictor.model
        return {
            "output": self._predictor.default_output_prob_thresh,
            "detection": model.score_threshold_detection,
            "new_object": model.new_det_thresh,
            "image_only": model.image_only_det_thresh,
        }

    def segment_video(
        self,
        video_path: Path,
        prompt: str,
        max_frames: int | None = None,
    ) -> Iterator[FrameSegmentations]:
        if not video_path.is_file() and not video_path.is_dir():
            raise FileNotFoundError(video_path)
        if not prompt.strip():
            raise ValueError("The segmentation prompt cannot be empty")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be at least 1")

        response = self._predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": str(video_path),
                "offload_video_to_cpu": True,
            }
        )
        session_id = response["session_id"]
        yielded: set[int] = set()
        try:
            response = self._predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt,
                }
            )
            if max_frames == 1:
                yield self._convert_frame(0, response["outputs"], prompt)
                return

            for response in self._predictor.handle_stream_request(
                request={"type": "propagate_in_video", "session_id": session_id}
            ):
                frame_index = int(response["frame_index"])
                if max_frames is not None and frame_index >= max_frames:
                    break
                if frame_index not in yielded:
                    yield self._convert_frame(frame_index, response["outputs"], prompt)
                    yielded.add(frame_index)
        finally:
            self._predictor.handle_request(
                request={"type": "close_session", "session_id": session_id}
            )

    @staticmethod
    def _convert_frame(
        frame_index: int, outputs: dict[str, Any], prompt: str
    ) -> FrameSegmentations:
        object_ids = _numpy(outputs["out_obj_ids"])
        probabilities = _numpy(outputs["out_probs"])
        boxes = _numpy(outputs["out_boxes_xywh"])
        masks = _numpy(outputs["out_binary_masks"])
        detections = []
        for index, object_id in enumerate(object_ids):
            mask = np.asarray(masks[index]).squeeze().astype(bool)
            if mask.ndim != 2 or not mask.any():
                continue
            score = probabilities[index]
            detections.append(
                Detection(
                    object_id=int(object_id),
                    box_xywh=tuple(float(value) for value in boxes[index]),
                    score=None if score is None else float(score),
                    mask=mask,
                )
            )
        return FrameSegmentations(
            frame_index=frame_index,
            prompt=prompt,
            backend=LocalSamBackend.name,
            detections=tuple(detections),
        )


def _start_multiplex_session(
    predictor,
    resource_path,
    session_id=None,
    offload_video_to_cpu=False,
    offload_state_to_cpu=False,
):
    if offload_state_to_cpu:
        raise ValueError("SAM 3.1 multiplex does not support state CPU offload")
    init_kwargs = {
        "resource_path": resource_path,
        "offload_video_to_cpu": offload_video_to_cpu,
    }
    if hasattr(predictor, "async_loading_frames"):
        init_kwargs["async_loading_frames"] = predictor.async_loading_frames
    inference_state = predictor.model.init_state(**init_kwargs)
    session_id = session_id or str(uuid.uuid4())
    predictor._all_inference_states[session_id] = {
        "state": inference_state,
        "session_id": session_id,
        "start_time": time.time(),
        "last_use_time": time.time(),
    }
    return {"session_id": session_id}


def _wrap_clear_all_points_in_frame(original_clear):
    def wrapped(
        inference_state,
        frame_idx,
        obj_id,
        need_output=True,
        preserve_user_refined=False,
    ):
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id)
        cond_outputs = inference_state["output_dict"]["cond_frame_outputs"]
        non_cond_outputs = inference_state["output_dict"]["non_cond_frame_outputs"]
        if obj_idx is not None and set(cond_outputs) == {frame_idx}:
            other_on_frame = any(
                k != obj_idx
                and (
                    frame_idx in inference_state["point_inputs_per_obj"].get(k, {})
                    or frame_idx in inference_state["mask_inputs_per_obj"].get(k, {})
                )
                for k in range(len(inference_state["obj_ids"]))
            )
            if not other_on_frame:
                remaining_frames = {
                    f
                    for k in range(len(inference_state["obj_ids"]))
                    for store in ("point_inputs_per_obj", "mask_inputs_per_obj")
                    for f in inference_state[store].get(k, {})
                    if not (k == obj_idx and f == frame_idx) and f in non_cond_outputs
                }
                if remaining_frames:
                    promoted = min(remaining_frames)
                    cond_outputs[promoted] = non_cond_outputs.pop(promoted)
                    consolidated = inference_state["consolidated_frame_inds"]
                    consolidated["non_cond_frame_outputs"].discard(promoted)
                    consolidated["cond_frame_outputs"].add(promoted)
                    for per_obj in inference_state["output_dict_per_obj"].values():
                        if promoted in per_obj["non_cond_frame_outputs"]:
                            per_obj["cond_frame_outputs"][promoted] = per_obj[
                                "non_cond_frame_outputs"
                            ].pop(promoted)
        result = original_clear(
            inference_state,
            frame_idx,
            obj_id,
            need_output=need_output,
            preserve_user_refined=preserve_user_refined,
        )
        if (
            cond_outputs
            and inference_state.get("first_ann_frame_idx") not in cond_outputs
        ):
            inference_state["first_ann_frame_idx"] = min(cond_outputs)
        return result

    return wrapped


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
