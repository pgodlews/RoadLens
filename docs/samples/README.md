# Garmin evaluation samples

These four frames were extracted from Piotr Godlewski's Garmin Varia Vue recording on Eccup Moor Road on 11 October 2026. They are included intentionally for hackathon evaluation and inspection; the source video and telemetry remain private and excluded from Git.

- `eccup-control.png`: comparatively smooth road surface used as a negative control.
- `eccup-repair.png`: a visible repaired or discoloured asphalt area; not confirmed as a pothole.
- `eccup-cracking.png`: visible cracked/damaged asphalt; not confirmed as a pothole.
- `eccup-known-location.png`: frame from the user-reported pothole location, extracted at the nearest outbound GPS fix (1.23 m away, 158 seconds into `GRMS0010.MP4`).

The filenames describe visible appearance, not verified ground-truth hazard labels. Camera GPS and speed overlays are retained so the exact model inputs can be inspected. No detection or severity claim is implied.
