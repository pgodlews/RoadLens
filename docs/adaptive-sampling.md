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

```mermaid
flowchart LR
    V[Source video: original timestamps] --> Q[Measure road-region sharpness]
    G[Garmin speed over time] --> S[Select near distance targets]
    Q --> S
    S --> C[SAM detection and tracking]
    C --> L[Record every result, including empty frames]
    C --> W[Merge candidate time windows]
    W --> F[SAM on every original frame in those windows]
    F --> E[Rank and preserve evidence]
    E --> R[Camera GPS and human review]
```

The nominal inference rate is speed in metres/second divided by spacing in metres.
At one-metre spacing this is about 6 fps for this cyclist, 13.4 fps at 30 mph, and
26.8 fps at 60 mph. Source frame rate limits achievable spatial coverage. We must
report gaps rather than silently claim one-metre coverage when speed is too high.

| Example speed | 0.5 m spacing | 1 m spacing | 2 m spacing |
|---|---:|---:|---:|
| 13 mph cycling | 11.6 fps | 5.8 fps | 2.9 fps |
| 30 mph driving | 26.8 fps | 13.4 fps | 6.7 fps |
| 60 mph driving | 53.6 fps | 26.8 fps | 13.4 fps |

These are required sampling rates, not model throughput or detection accuracy.
For a 30 fps source, the 0.5 m target at 60 mph is unattainable: retain all frames
and report the larger spatial gaps. There is no separate hardcoded bike/car mode.
A fixed every-Nth-frame rule would change spatial coverage with speed. A global
"sharpest frame" rule could skip road sections; quality selection stays local.

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

## First measured comparison

On the RTX 3090, `GRMS0012.MP4` from 480–486 s contains 180 native frames
and approximately 48.1 m of travel. This diagnostic section deliberately includes
the earlier positive example; it is not an unbiased road-wide evaluation.
Both runs used native 2688×2160 evidence, the thresholds above, four-frame grounding
batches, 64-frame sessions and **eight-frame overlap** (the initial trial setting).

| Method | First-pass frames | Inferences including overlap/refinement | SAM time | Total wall time | Unreviewed tracks |
|---|---:|---:|---:|---:|---:|
| One metre + refinement | 49 | 230 | 49.3 s | 122.3 s | 7 |
| Every native frame | 180 | 204 | 43.0 s | 107.8 s | 7 |

The distance-only first pass took 11.8 s of SAM time and produced detections at
21 sampled timestamps. Its merged refinement interval covered 5.52 of the six
seconds, so the adaptive workflow performed more total work than the full-frame
run. This is an expected failure mode on densely damaged or patchy road.
Matching track counts do not establish matching defects, recall, or precision.
Visual review of the adaptive contact sheet shows several repaired/patchy regions;
these are **not seven confirmed potholes**.

A subsequent direct comparison did find that all seven selected evidence frames
had identical source timestamps, scores and masks between the two runs (mask IoU
1.0 for each matched pair). Thus the adaptive method reproduced this full-frame
baseline's chosen evidence on this one section, at higher runtime. This is model
agreement, not agreement with manually labelled potholes.

Model loading took about 12.5 s per run. Preparing native input frames took 12.3 s
for the sampled run and 36.5 s for the full-frame run. Total wall time also includes
fine-frame extraction, mask serialization and evidence export. These single runs
are illustrative, not statistically stable performance measurements.

After this comparison, PNG writing was parallelized using four workers with a
bounded eight-frame queue. This preserves native pixels while reducing serial
encoding overhead; subsequent wall times are not a controlled comparison with
these first runs. Final defaults use 16-frame overlap to provide more context
around SAM's 15-frame hot-start period. Coverage checks now reject incomplete or
duplicate backend frame output, manifests record the code revision and failure
status, and 28 tests pass.

## Road-wide scan observations (in progress)

The outbound survey covers 150–360 s of `GRMS0010.MP4`; the return covers
420 s to the end of `GRMS0012.MP4`. These include margins around the previously
identified road corridor. Neither scan accepts or uses pothole pins for frame
selection. The outbound plan selects 1,381 of 6,294 original frames over about
1,380 m of integrated speed. All original frames have usable interpolated speed;
the largest selected spatial gap is 1.48 m, illustrating why "one metre" is a
target rather than a strict coverage guarantee.

During inspection of the outbound first-pass ledger, a candidate at 159.027 s
segments a small road-edge defect in the same scene as the earlier reported-location
still at 158 s, which had no SAM detection. This is useful evidence that a nearby
video view can succeed where a selected still failed. It does not isolate the
benefit of sampling: internal thresholds, input handling and temporal context also
changed. The candidate remains unreviewed; generated images remain local.
