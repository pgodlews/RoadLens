# Ideas to Explore (Nice-to-Have Backlog)

Items in this document are optional future explorations, **not** required core deliverables. Any size estimate added in future work must be explicitly labelled as an approximate camera-derived estimate with an uncertainty range, never a surveyed measurement or guaranteed millimetre depth.

## 1. Kinematic Ground-Plane Sizing from SAM 3.1 Multi-Frame Tracks + Garmin Speed

- **Priority:** Nice-to-have
- **Motivation:** During native `30 fps` refinement (`fine-*` windows in `roadlens scan`), SAM 3.1 already tracks candidate masks across `10–60+` consecutive frames while Garmin telemetry provides forward ground speed `v` (`speed_mps`).
- **Approach:**
  1. Estimate the road vanishing point (horizon pitch) over the segment from background road-plane optical flow.
  2. Combine inter-frame camera forward travel (`v * dt`, e.g. `~25 cm` per frame at `7.5 m/s` and `30 fps`) with the tracked mask's vertical motion in image space to recover local metres-per-pixel scale on the road plane without requiring a physical calibration target.
  3. Project the best-evidence SAM 3.1 mask polygon onto the local ground plane to report an approximate planimetric **width × length band** (e.g. `~35–45 cm wide × 50–65 cm long`) alongside explicit quality gates (minimum tracked frames, stable speed fix, non-boundary mask).
- **Guardrails:** Pure NumPy/OpenCV (`0 MiB` extra GPU VRAM). Do not claim millimetre cavity depth from planar ground projection; optionally pair with Muse Glimmer contextual size-band checks (`<15 cm`, `15–40 cm`, `>40 cm`) anchored against visible UK road markings (`100 mm` lines, ~`400–450 mm` ironwork) and rim/shadow severity cues.

## 2. Calibrated Ground-Plane Homography / Inverse Perspective Mapping (IPM)

- **Priority:** Nice-to-have (single-frame fallback when multi-frame tracking is short or speed is unavailable)
- **Motivation:** Some detections appear cleanly in only a few frames or near stops/junctions where kinematic scale recovery is noisy.
- **Approach:**
  1. Accept an optional rider camera preset (handlebar mount height above tarmac, e.g. `~0.95–1.05 m`, plus Garmin camera horizontal FOV) combined with auto-estimated horizon pitch.
  2. Compute a ground-plane homography mapping image coordinates `(u, v)` below the horizon to flat-road coordinates `(X, Z)` in `<1 ms` per candidate.
  3. Derive approximate width, length, and surface area (`m^2`) directly from the SAM 3.1 mask contour, reporting wider uncertainty bounds to reflect road camber, gradient, and pitching on rough tarmac.
