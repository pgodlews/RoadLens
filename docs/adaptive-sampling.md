# Distance-based scanning experiment

## Decision record — 1 November 2026

The first review found that previous pothole experiments used selected stills and
short excerpts, not a complete survey of Eccup Moor Road. Three stills were chosen
using the nearest camera GPS fix to user-supplied map pins. Those pins are useful
references, not ground truth or reliable selectors of the best forward-camera view.

We will evaluate a scan independent of pothole pins, using the full road corridor
with margins. The initial hypothesis is one sample per metre of travel, selecting
a sharp nearby frame, followed by native-frame-rate examination around candidates.
One metre is an experimental default, not a measured optimum.

The nominal inference rate is speed in metres/second divided by spacing in metres.
At one-metre spacing this is about 6 fps for this cyclist, 13.4 fps at 30 mph, and
26.8 fps at 60 mph. Source frame rate limits achievable spatial coverage. We must
report gaps rather than silently claim one-metre coverage when speed is too high.

Selection must preserve source presentation timestamps: sampled-video frame index
divided by an artificial playback rate is not the original recording time. Speed
integration is approximate; stale or missing telemetry needs a logged time-based
fallback. Stationary footage needs a maximum sampling interval.

Sharpness will be measured on a road-focused image region within a limited temporal
neighbourhood. It is a blur heuristic, not a pothole detector. Candidate refinement
must include frames before detection as well as after it. Empty first-pass results
cannot establish that the road is defect-free: a full-frame comparison and manual
review remain necessary to assess missed defects.

The review also found that RoadLens's `--confidence` adjusts SAM's output filter,
while the installed multiplex builder independently sets detection filtering to
0.4 and new-object admission to 0.65. Earlier low-output-threshold experiments did
not demonstrate equivalently low internal detection thresholds.

Implementation, tests, measured timings and negative results will be recorded in
subsequent commits. Private video, telemetry and generated run artifacts stay out
of Git; published documentation will contain aggregate results and methodology.

## Implementation checkpoint

Added `roadlens scan` with these experimental defaults:

- One-metre distance targets; sharpness selection within 0.05 seconds and 0.2 m
  of the target frame. Actual selected gaps are reported, not assumed to be 1 m.
- A 0.5-second time limit at low speed, and 0.2-second fallback when speed cannot
  be interpolated across valid fixes no more than three seconds apart. Source
  frames may introduce rounding and the sharpness window may extend these gaps.
- Native resolution PNG inputs and decoded source presentation timestamps.
- 64-frame SAM sessions with 16-frame overlap, followed by native-frame passes
  within two seconds either side of first-pass detections. Overlapping refinement
  windows are merged. SAM still performs its internal input resizing.
- Output/detection/new-object thresholds of 0.25/0.25/0.35, explicitly recorded.
  These settings favour candidate discovery and may increase false positives;
  SAM's other tracking/confirmation heuristics remain enabled.
- Evidence ranking combines confidence, defect-crop sharpness, mask area and a
  penalty for touching the frame boundary. This heuristic needs human review.
- Track merging requires overlapping boxes at the same original timestamp.
  Tracks without shared-frame evidence remain separate; GPS proximity alone
  cannot prove identity. Repeated passes are not automatically deduplicated.

The frame ledger includes empty outputs. The manifest records all selected source
timestamps, estimated distance gaps, missing-speed counts, configuration, model
load time, inference time and wall time. Partial runs never receive `complete`
status. Full-frame mode uses the same machinery for a controlled comparison.

The initial 49-frame GPU trial failed with CUDA out-of-memory using SAM's upstream
16-frame grounding batch, even with 1280-pixel evidence. The next trial reduces
grounding to four frames and retains native source resolution. This is separate
from the 64-frame tracking-session length. Source-frame storage is offloaded to
CPU memory.

A further backend correction preserves propagation's revised frame-zero result.
Previously, the initial prompt output was emitted and the propagated frame-zero
output was discarded. This could discard detections established with video context.

At this implementation checkpoint, 26 tests pass and Ruff passes. Tests include
real encoded-video timestamp/extraction round trips and a fake-inference end-to-end
scan covering refinement, empty results, evidence export and overlap deduplication.
They verify plumbing, not real-world pothole recall. GPU benchmarks follow below.
