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
  1. Accept an optional rider camera preset (handlebar mount height above tarmac + effective horizontal/vertical FOV) combined with auto-estimated horizon pitch.
  2. Compute a ground-plane homography mapping image coordinates `(u, v)` below the horizon to flat-road coordinates `(X, Z)` in `<1 ms` per candidate.
  3. Derive approximate width, length, and surface area (`m^2`) directly from the SAM 3.1 mask contour, reporting wider uncertainty bounds to reflect road camber, gradient, and pitching on rough tarmac.
- **Reference Bike Calibration (Specialized Tarmac SL7 Size L / 56 cm, Under-Bar Out-Front Mount):**
  - Photogrammetric scale verified across three independent bike references (`991 mm` wheelbase, `678 mm` `700×28c` outer tyre diameter, and `140 mm` rear disc rotor -> `2.49–2.53 mm/px` consistent to `<1.5%`):
    - **Camera lens height above road (`H_cam`):** `0.89 m ± 0.02 m` (`894 mm` unloaded; `~890 mm` with `~4 mm` rider tyre sag — `555 mm` perpendicular above the axle line plus `335 mm` loaded tyre radius; ~`10 cm` lower than a naive top-of-hoods guess because of the aero stack and under-bar mount).
    - **Longitudinal lens offset:** `~0.05 m` (`5 cm`) behind the front thru-axle (`~0.39 m` behind the front tyre's leading edge).
    - **Effective `2688×2160` (`5:4` EIS-stabilized) intrinsics (cross-checked via kinematic travel `v * dt`):** `fy / height ≈ 0.81 ± 0.03` (`fx / width ≈ 0.65`, effective `HFOV ≈ 75°`, `VFOV ≈ 63°`, typical pitch `~2°` downward with horizon near `y ≈ 0.47`). Cross-checking this homography on a rural single-track lane recovered a `3.05 m` carriageway width (matching standard `2.75–3.1 m` UK single-track lanes) and sized an elongated nearside cavity at `~15 cm × ~59 cm`.

![Specialized Tarmac SL7 Size L Garmin Varia Vue camera geometry](samples/tarmac-sl7-camera-geometry.jpg)
