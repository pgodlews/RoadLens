# Garmin evaluation samples

These six frames were extracted from Piotr Godlewski's Garmin Varia Vue recording on Eccup Moor Road on 11 October 2026. They are included intentionally for hackathon evaluation and inspection; the source video and telemetry remain private and excluded from Git.

- `eccup-control.png`: comparatively smooth road surface used as a negative control.
- `eccup-repair.png`: a visible repaired or discoloured asphalt area; not confirmed as a pothole.
- `eccup-cracking.png`: visible cracked/damaged asphalt; not confirmed as a pothole.
- `eccup-known-location.png`: frame from the first user-reported pothole location, extracted at the nearest outbound GPS fix (1.23 m away, 158 seconds into `GRMS0010.MP4`).
- `eccup-second-outbound.png`: second reported area, outbound (1.82 m from the supplied coordinate, 295 seconds into `GRMS0010.MP4`).
- `eccup-second-return.png`: second reported area, return pass (3.35 m from the supplied coordinate, 483 seconds into `GRMS0012.MP4`).
- `tarmac-sl7-camera-geometry.jpg`: EXIF-stripped side-profile photogrammetric reference diagram of the Specialized Tarmac SL7 (Size L / 56 cm) with under-bar Garmin Varia Vue mount (`H_cam = 890 mm`, `ΔX_axle = -50.5 mm`, cross-checked across wheelbase, `700×28c` tyre diameter, and `140 mm` rear rotor; see `../ideas-to-explore.md`).
- `eccup-pothole-size-estimate.jpg`: EXIF-stripped, GPS-redacted ground-plane projection of return-pass SAM 3.1 candidate `#74` (`~13 cm × ~59 cm ± 15%` contour footprint, `~587 cm²` planimetric area, `1.43–2.02 m` ahead of lens, with `3.05 m` lane-width sanity check and `10 × 10 cm` bird's-eye metric plan inset; approximate camera-derived estimate, unreviewed).

The filenames describe visible appearance, not verified ground-truth hazard labels. Camera GPS and speed overlays are retained so the exact model inputs can be inspected. No detection or severity claim is implied.
