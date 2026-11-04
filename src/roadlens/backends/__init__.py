from pathlib import Path

from roadlens.backends.base import SegmentationBackend


def create_backend(
    name: str,
    checkpoint_path: Path | None = None,
    confidence_threshold: float = 0.5,
    legacy_stitch: bool = False,
) -> SegmentationBackend:
    if name == "local":
        if checkpoint_path is None:
            raise ValueError("--checkpoint is required for the local backend")
        from roadlens.backends.local import LocalSamBackend

        return LocalSamBackend(checkpoint_path, confidence_threshold)
    if name == "remote":
        from roadlens.backends.remote import RemoteSamBackend

        return RemoteSamBackend(legacy_stitch=legacy_stitch)
    raise ValueError(f"Unknown backend: {name}")
