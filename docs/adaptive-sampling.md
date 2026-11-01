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
