# Development log

## 1 November 2026

RoadLens started from an empty Git repository for the Meta Global AI Developer Hackathon. The initial commit added the MIT license, privacy exclusions, and project scope. Subsequent commits record the implementation in working increments.

### Prior work and attribution

The Garmin timed-metadata grouping and ExifTool extraction approach was adapted from [Ride Studio](https://github.com/pgodlews/ride-studio), also Copyright (c) 2026 Piotr Godlewski and MIT licensed. See `NOTICE` and `LICENSE-RIDE-STUDIO`. The RoadLens backend abstraction, SAM integration, analysis workflow, observation model, report server, and review interface were developed for this project.

### Source footage and telemetry

- Located Garmin Varia Vue front footage and Varia rear footage on the local NAS. Source media and telemetry remain excluded from Git.
- Verified that the front MP4s contain timed Garmin GPS documents decodable with ExifTool. The first inspected rear clip had no usable fixes; this does not establish that all rear clips lack telemetry.
- Used the supplied Eccup Moor Road GPS corridor to locate the outbound section in `GRMS0010.MP4` at approximately 168–348 seconds and the return in `GRMS0012.MP4` at approximately 434–600 seconds.
- The GLV companions inspected are low-resolution videos, not the primary timed-GPS source used by RoadLens.

### SAM 3.1 environment

- Found the complete 3.3 GB `facebook/sam3.1` multiplex checkpoint in the local Hugging Face cache; no checkpoint download was needed.
- Selected a local RTX 3090 backend for prototyping and a separate Meta Model API backend for the hosted demo.
- Added runtime dependencies required by SAM's actual inference import graph, including setuptools, einops, video decoding, COCO tools, and supporting libraries.
- Disabled FlashAttention 3 on the Ampere RTX 3090.
- Added a RoadLens-local adapter for an upstream SAM 3.1 session API mismatch: the shared predictor passes `offload_state_to_cpu`, while the multiplex model does not accept it. A unit test covers the adapter.
- Confirmed successful checkpoint loading and real GPU inference. The model log reports missing rotary-frequency keys; control inference subsequently produced valid masks.

### Real-footage findings

Experiments used a short Eccup Moor Road excerpt, not synthetic output.

| Prompt or input | Result | Interpretation |
|---|---:|---|
| `pothole`, first excerpt, threshold 0.5 | 0 detections | No detection; not evidence of absence. |
| `pothole`, damage excerpt, threshold 0.25 | 0 detections | Text prompting did not find the visible repair/damage. |
| `road surface damage`, `road damage`, `pothole in asphalt`, `asphalt patch`, threshold 0.1 | 0 detections each | Broader wording did not resolve the issue. |
| `pothole`, threshold 0.01 | 0 detections | Lowering the threshold did not produce a candidate. |
| Positive point on visible damage | 0 retained masks | Point-guided damage segmentation remains unproven. |
| `car`, threshold 0.01 | 1 mask, score 0.855 | Valid control detection and plausible mask. |
| `road`, threshold 0.01 | 1 mask, score 0.934 | Valid control detection covering the road surface. |

These results verify the local inference path but **do not establish pothole detection accuracy**. No pothole recall, precision, or physical measurement claim is made. The next detection experiment should use a separate candidate-discovery stage (Muse when credits are available, or a validated detector) followed by SAM segmentation, with manually labelled positive and negative Garmin examples.

### Hosted API status

The remote backend uses `sam-3.1` and Meta's official streaming parser. Its result conversion is tested with mocked records. No live hosted request has been made; it awaits the user's hackathon API key and credits.

### End-to-end pipeline and map review

- Ran `roadlens analyze` on four seconds of original Garmin footage at five sampled frames per second. SAM tracked one `road` object through all 20 frames; RoadLens selected the highest-scoring frame and linked it to the nearest embedded Garmin GPS fix.
- Exported `observations.json` and GeoJSON with the source offset, timestamp, backend, score, mask area, review status, and `garmin_camera_gps` provenance. The observation is explicitly unreviewed and is not presented as a confirmed hazard.
- Added a localhost review interface with evidence overlays, confirm/reject states, reviewed GeoJSON download, and OpenStreetMap display through the user's local United Kingdom PMTiles archive.
- Verified the report server with real HTTP requests: JavaScript syntax passed, full-archive HEAD returned the correct size, and a 16-byte range request returned `206 Partial Content` with bytes identical to the source archive.
- At this checkpoint, 16 unit tests and Ruff lint pass.

### Muse Glimmer candidate-discovery evaluation

- Published six user-authorised Garmin evaluation frames, including negative control, repair/discolouration, visible damage, and two user-supplied pothole locations in outbound and return directions. The repository remained private at publication time.
- Ran Meta's official 17 GB Muse Glimmer GGUF and vision projector locally through llama.cpp build 12032 on the RTX 3090.
- Full-frame Glimmer inference found the clearer damage at the second return location but missed the first reported pothole. Road-only crops improved sensitivity, producing damaged-asphalt, cracking, and possible-pothole candidates at both second-location views. The smooth control remained negative.
- Glimmer boxes were sometimes coarse and one exceeded the frame, so integration must validate coordinates and retain human review. No accuracy percentage is claimed from this small qualitative set.
- Raw responses, annotated outputs, method, and limitations are documented in `docs/glimmer-evaluation.md` and `docs/evaluation`.

### SAM 3.1 published-image comparison

- Ran all six published images through local SAM 3.1 with `pothole`, `cracking`, and `damaged asphalt` prompts at threshold 0.1, for 18 completed image/prompt combinations.
- The second return image produced one `pothole` mask with score 0.7460. The 964-pixel mask tightly covers the isolated mid-distance defect. All other combinations returned no detections.
- This is the first confirmed useful pothole segmentation in the evaluation, but it does not establish reliable general detection. The result and comparison with Glimmer's broader candidate boxes are recorded in `docs/sam-image-evaluation.md`.

### Independent video coverage and distance sampling

- Reviewed saved experiments and found no complete frame-by-frame pothole survey.
  Nearest-fix selection at user-supplied map pins could choose a poor forward-camera
  view, even if a pin were accurate. New scans use road-wide time intervals with
  margins instead of pothole-location pins.
- Added `roadlens scan`: speed-integrated distance sampling, bounded local sharpness
  selection, exact source presentation timestamps, overlapping SAM sessions and
  native-frame refinement around candidates. It exports per-frame empty results,
  timing/coverage manifests, reviewable observations and best-frame evidence.
- Exposed SAM's separate detection/new-object thresholds; the earlier output-only
  confidence sweep did not lower these internal admission thresholds. Corrected
  handling of propagated frame zero, which can differ from the initial prompt.
- A 49-frame trial exhausted GPU memory at the upstream 16-frame grounding batch.
  Four-frame batches completed successfully on the RTX 3090.
- Compared distance sampling plus refinement with every native frame on six seconds
  of the return footage. The adaptive run took 122.3 s versus 107.8 s for full-frame
  processing because refinement covered almost the entire diagnostic section.
  Both yielded seven unreviewed tracks; visual inspection includes repaired/patchy
  regions, so this is not seven confirmed potholes or a precision/recall result.
- Parallelized lossless PNG writing after observing significant preprocessing cost.
  Added failure manifests and coverage checks; 28 tests and Ruff pass.
- Detailed methods, settings, limitations and subsequent road-wide results are
  maintained in `docs/adaptive-sampling.md`; aggregate diagnostic timings are in
  `docs/evaluation/adaptive-sampling-results.json`.
