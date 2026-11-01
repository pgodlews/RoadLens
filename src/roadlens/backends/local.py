from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from roadlens.domain import Detection, FrameSegmentations


class LocalSamBackend:
    name = "local"

    def __init__(self, checkpoint_path: Path, confidence_threshold: float = 0.5):
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

    def segment_video(
        self,
        video_path: Path,
        prompt: str,
        max_frames: int | None = None,
    ) -> Iterator[FrameSegmentations]:
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        if not prompt.strip():
            raise ValueError("The segmentation prompt cannot be empty")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be at least 1")

        response = self._predictor.handle_request(
            request={"type": "start_session", "resource_path": str(video_path)}
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
            yield self._convert_frame(0, response["outputs"], prompt)
            yielded.add(0)
            if max_frames == 1:
                return

            for response in self._predictor.handle_stream_request(
                request={"type": "propagate_in_video", "session_id": session_id}
            ):
                frame_index = int(response["frame_index"])
                if max_frames is not None and frame_index >= max_frames:
                    break
                if frame_index not in yielded:
                    yield self._convert_frame(
                        frame_index, response["outputs"], prompt
                    )
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


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)
