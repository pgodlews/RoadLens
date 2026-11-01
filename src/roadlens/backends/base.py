from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from roadlens.domain import FrameSegmentations


class SegmentationBackend(Protocol):
    name: str

    def segment_video(
        self,
        video_path: Path,
        prompt: str,
        max_frames: int | None = None,
    ) -> Iterator[FrameSegmentations]: ...
