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

The remote backend uses `sam-3.1` and Meta's official streaming parser. Its result conversion is tested with mocked records. No live hosted request has been made; it awaits the user's hackathon API key and credits. (Superseded: see 3 November.)

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
- Parallelized lossless PNG writing (`a7baa61`) after observing significant
  preprocessing cost, but the outbound road-wide scan subsequently stalled while
  exporting native PNGs for the first refinement window (exact cause unproven).
  Reverted PNG export to the reliable serial implementation, added verified
  `--resume` replay from completed chunk/frame ledgers (`e1d49cb`), and resumed
  without repeating the 29 completed coarse GPU chunks.
- Completed the full outbound Eccup Moor Road corridor scan (`GRMS0010.MP4`,
  150–360 s, ~1,380 m): 5,753 frame inferences across 29 coarse and 66
  native-rate refinement chunks in 1,164.7 s of SAM time, yielding 85 unreviewed
  candidate tracks. Visual inspection confirmed useful detections around both
  previously missed outbound reported-pothole scenes (`~160 s` and `~290–299 s`),
  alongside false-positive tarmac repair patches, organic surface debris, and
  unmerged track splits.
- Diagnosed and fixed an upstream Meta SAM 3.1 dynamic-multiplex tracking crash
  (`RuntimeError: No points are provided; please add points first`) encountered
  during `fine-003` of the return corridor scan (`ef900c7`): when dynamic
  multiplex reuses a bucket on an already-tracked frame and later removes all
  earlier objects conditioned on `cond_frame_outputs`, `clear_all_points_in_frame`
  previously wiped the live object's tracking state. Added a surgical adapter in
  `src/roadlens/backends/local.py` with unit tests and reused existing `fine-*`
  PNG exports on `--resume`.
- Completed the full return Eccup Moor Road corridor scan (`GRMS0012.MP4`,
  420–600.6 s, ~1,314 m, including real end-of-clip `missing_speed_fallback`
  coverage): 5,754 frame inferences across 28 coarse and 66 native-rate
  refinement chunks in 1,204.9 s of SAM time, yielding 86 unreviewed candidate
  tracks. Visual inspection confirmed sharp cavity/edge detections at both
  reported return locations (`#74` at `586.59 s` score `0.770` and `#28`–`#33`
  across `480.3–483.9 s`), while demonstrating that sealed asphalt patches can
  score up to `0.808`—confirming that confidence filtering alone cannot replace
  multimodal triage and human review.
- Evaluated coarse-spacing sensitivity (`0.5 m`–`5.0 m` vs `all_native`) across
  both corridors. Detailed methods, settings, limitations and road-wide results
  are maintained in `docs/adaptive-sampling.md`; aggregate sanitized metrics are
  in `docs/evaluation/adaptive-sampling-results.json` and
  `docs/evaluation/eccup-corridor-survey.json`.
- Built multi-clip Garmin ride indexing (`src/roadlens/ride.py`), parameterized
  `roadlens scan --prompt`, and a dependency-free offline PMTiles v3 MVT street
  and place index (`src/roadlens/pmtiles_roads.py`, `a519de5`). Verified against
  a 12-clip, 25.2 km Garmin ride and `united-kingdom.pmtiles` that `.GLV`
  companions and overlapping saved event clips are skipped/deduplicated and that
  street-name, ride-offset (`20m`), wall-clock (`12:30`), and map-corridor
  queries resolve completely offline in milliseconds.
- Implemented dual VLM triage and grounded summary-report drafting
  (`src/roadlens/vlm.py`, `roadlens triage`, `a5bf5f2`), supporting both local
  **Muse Glimmer 30B** (`llama-server` auto-managed on `127.0.0.1` with clean
  VRAM release) and remote OpenAI-compatible multimodal endpoints (with default
  burned-in GPS-banner masking on remote images).
- **Negative result & fix during real GPU validation of Muse Glimmer:** passing
  filled blue SAM mask overlays (`overlay-*.png`) to Muse Glimmer obscured
  interior asphalt texture, causing Glimmer to misclassify real pothole `#74` on
  the return corridor as blue surface paint (`surface_debris_or_stain`).
  Switching `build_triage_composite_data_url` to pair the unmodified
  `frame-*.png` (with a padded outer red rectangle so zero candidate pixels are
  tinted) alongside an unmasked 1:1 crop yielded 3/3 accurate separation on real
  return-corridor examples (`#74` pothole -> `hazard_candidate`, `#54`
  resurfaced patch -> `sealed_repair`, `#51` organic stain ->
  `surface_debris_or_stain`) while preserving `review_status: "unreviewed"`.
- Added the interactive map-first web workflow (`src/roadlens/studio.py`,
  `roadlens studio`, `72d1af0`) and the stdio Model Context Protocol server plus
  investigator skill (`src/roadlens/mcp_server.py`, `roadlens mcp`,
  `skills/roadlens-investigator/SKILL.md`), enforcing bookended human-in-the-loop
  scoping at the start and human visual confirmation before council GeoJSON
  export at the end.

## 2 November 2026

### Code review fixes

An independent code review found defects in the review guardrails and the
agent workflow. Each fix below has a regression test (53 tests pass with
`-W error`; Ruff clean).

- **MCP VLM discovery:** `discover_segment_hazards_vlm` crashed against the real
  backend because it treated the `discover_hazards` dictionary as a list. The
  test double now mirrors the production return contracts.
- **Standalone report export:** `roadlens report` exported every candidate,
  including unreviewed and rejected ones. The default download is now
  human-confirmed-only (disabled until something is confirmed), with a
  separately labelled full-audit export, matching Studio.
- **Standalone report state:** browser-stored decisions were keyed by URL path,
  so confirming candidate `0` in one run could pre-confirm candidate `0` in
  another. Decisions are now keyed by run directory and candidate identity, and
  observation text is HTML-escaped before rendering.
- **Triage evidence box:** when several detections shared a result frame, every
  candidate was triaged with the first detection's box. Triage now selects the
  detection matching the observation's `object_id`.
- **Completed `--resume`:** a completed scan was returned without checking that
  the source video still existed or was unchanged; both are now validated.
  A reported duplicate-ledger issue on retried chunks did not reproduce against
  the actual code, and a regression test now locks that behaviour in.
- **Remote VLM endpoint:** the documented `ROADLENS_VLM_API_URL` was ignored in
  favour of an undocumented variable, which could send frames to the default
  endpoint. The documented name is now honoured (old name kept as an alias).
- **Studio triage reports:** drafted summaries ignored saved review decisions.
  `triage_run_directory` now drafts from `review-decisions.json` while leaving
  raw model observations unchanged.
- **Smaller fixes:** narrow segment windows can no longer extend past the clip
  end, and `max_candidates` rejects negative values and treats `0` as "triage
  none".

## 3 November 2026

### First live Meta Model API run

The hackathon API credits arrived, so both hosted paths were run against real
Eccup footage for the first time ([data](evaluation/remote-api-eccup-2026-11-03.json)).
The live run found two defects that the mocked tests could not.

- **Model names:** the API lists `sam-3.1` and the `muse-spark` family. Muse
  Glimmer is open-weights only, so the remote VLM default changed from the
  guessed `Muse-Glimmer` to `muse-spark-1.3`.
- **Reasoning budget:** Muse Spark reasons before answering and counts those
  tokens against `max_tokens`. A triage call used about 3,300 reasoning tokens,
  so the 1,200-token limit returned an empty answer. The remote backend now
  reserves 8,000 extra tokens (`ROADLENS_VLM_REASONING_TOKENS`), can pass
  `reasoning_effort`, and explains a budget exhausted before the answer.
- **Hosted SAM mask records:** the real parser places masks through
  `record.bounds` and reports `record.confidence`; the backend read
  non-existent `left`/`top`/`score` attributes and crashed. The unit test now
  builds the parser's real dataclasses, and fails on the old code.
- **Confidence and storage:** requests now ask for per-object confidence
  (`include_confidence`), and the uploaded clip is deleted from Meta's file
  storage once segmentation finishes, since uploads never expire by default.

Results, both on the 2 November return-pass window:

- **Hosted SAM 3.1** (`roadlens analyze --backend remote`, 584–596 s at 10 fps)
  returned exactly two objects in 34 seconds: the two rider-reported potholes at
  586.6 s (confidence 0.74) and 594.6 s (0.71). Overlays show both masks on the
  defects. The local scan of the same 12 seconds, with a lower threshold and
  30 fps refinement, had returned 8 candidates.
- **Muse Spark 1.3 triage** of the 9 local candidates took 3 minutes. Both
  reported potholes were `hazard_candidate` (one at high confidence). Spark
  agreed with local Glimmer on 7 of 9; the two disagreements (#4, #5) swap
  `sealed_repair` and `hazard_candidate`, exactly the borderline cases left to
  human review. Every `review_status` stayed `unreviewed`, and Spark's drafted
  report kept human decisions and model opinions separate.
- **Fully hosted Studio run:** with Segment set to "Meta API · hosted" and
  Triage to Muse Spark, the whole 36-second return pass (564–600 s) was
  scanned and triaged in about 2 minutes. It returned 3 candidates: the two
  reported potholes and the open cavity at 600.4 s that the local scan also
  found (#8), all triaged `hazard_candidate` and left `unreviewed`. Meta's file
  storage was empty afterwards.
- **JSON schema:** structured output is supported. Without it, Spark invented
  its own field names and two triage fields came back empty.

This is one window on one road, not a precision or recall measurement. At the
published prices the run cost well under a dollar (about $0.02 per triage call
and $0.20 per 1,000 SAM video frames).

### Why hosted SAM returns fewer candidates

Hosted SAM 3.1 found 2–3 objects where the local scan found 8, so the same
12-second window was sent three ways ([data](evaluation/remote-api-eccup-2026-11-03.json)):

| Hosted input | Objects | Time |
|---|---|---|
| Native 2688×2160, 30 fps | 3 (both reported potholes plus a right-side feature near local #3) | 93 s |
| 1280 px wide, 30 fps | 2 (both reported potholes) | 39 s |
| Native, 10 fps | 2 (both reported potholes) | 42 s |

Every hosted object scored 0.70–0.76, and across the full pass the hosted run
returned exactly the local candidates that scored above 0.7 (#0, #6, #8). The
API has no threshold parameter, so its fixed cut-off, apparently near 0.7, is
most of the difference; the local scan deliberately keeps detections from
0.25–0.35 because a sealed patch scored 0.808 and only a human can settle the
borderline ones. Resolution and frame rate add a little recall on top.

### Approximate size estimation

Every scan now estimates each candidate's footprint on the road
(`src/roadlens/sizing.py`, `roadlens size`). A flat-road pinhole model maps a
mask row to distance, `Z = h·f / (y − y_horizon)`, so length along the road
needs `h·f` and width across it needs only the camera height `h`.

- **Calibration:** the default is the Garmin Varia Vue under the bars of the
  rider's Tarmac SL7, measured photogrammetrically on 2 November (lens height
  0.89 m, focal length 0.81 image heights, horizon near 0.47). As an
  independent check, hosted SAM tracks of both reported potholes were fitted
  against distance travelled from integrated Garmin speed (95 edge
  observations). At the measured horizon the tracks give a focal length of
  0.814, within 1 % of the photogrammetric value.
- **Estimate:** the median over every stored mask of the candidate (all
  tracked frames for hosted runs, the best-evidence frame for local scans),
  rounded to 5 cm, with a range from the calibration tolerances and the
  frame-to-frame spread. Masks cut off by the frame edge, or more than 12 m
  away, are not sized.
- **Results:** the reported pothole is ≈ 60 × 15 cm from the local mask and
  ≈ 65 × 15 cm from hosted masks; the hand-annotated estimate from 2 November
  was ≈ 59 × 13 cm. The reported broken patch is ≈ 155 × 50 cm locally and
  ≈ 170 × 55 cm hosted ([data](evaluation/size-estimation-eccup-2026-11-03.json)).
- **Guardrails:** sizes are labelled as approximate road-surface footprints in
  Studio, exports and both report paths; depth is never estimated, and drafted
  reports may not claim an intervention threshold is met. The model assumes a
  flat road and a level bike, so raised obstructions (a branch) are not sized
  meaningfully. No tape-measure ground truth exists yet.

### Every detection in the evidence frame

A frame with several potholes showed only the selected candidate's box. Studio
now draws the frame's other detections as dashed boxes: labelled with their
candidate number for hosted runs, which store every detection per frame, and
taken from the frame log for local scans. Duplicates, specks and frame-filling
merged masks are skipped. Studio's multi-pass merge also renumbers each copied
detection so masks stay matched to merged candidates.

### Hosted SAM stops adding objects after 16

A sweep of the Eccup corridor with hosted SAM 3.1 (10 fps, 1280 px)
exposed a limit. One 180-second session over the return pass produced 16
tracked objects, the last appearing at 488.7 s, then nothing new for the
remaining 112 s. The last 100 s, sent alone, gave 7 candidates there,
including the rider-reported pothole at 586.6 s.

Clips longer than 200 frames now go up as overlapping chunks, three at a
time. Each chunk shares 5 frames with the next, and a track keeps its id
across a boundary when its masks overlap (IoU ≥ 0.3) in the shared frames. A
warning is printed if a chunk still reaches 16 objects
(`ROADLENS_REMOTE_CHUNK_FRAMES` sets the chunk length).

| Window | One session | Chunked |
|---|---|---|
| Return pass, 420–600.6 s | 16 | 25, including 586.6 s |
| Outbound pass, 150–360 s | 13 | 15 |

Every single-session candidate is also found chunked, except one outbound
object tracked for 34 s, which chunking replaces with shorter tracks. Size
estimates of candidates found both ways differ by at most 15 cm, on the
larger patches.
All candidates remain unreviewed
([data](evaluation/hazard-sweep-eccup-2026-11-03.json)).

The API documentation states the limit as "up to 16 objects per frame", but
the capped session stopped adding objects after 16 in total, while no frame
held more than two at once.

### Other static hazards on the same corridor

RoadLens takes any text prompt, but this footage has no clear example of a
static hazard other than potholes. Five other prompts were run over both
corridor windows to see what the model does when the target is rare or
absent. All hits were checked by eye from the frames.

| Prompt | Hits (out / back) | What the hits were |
|---|---|---|
| manhole cover | 1 / 12 | Potholes, patches and a dark stain; no manhole cover in any hit |
| drain grate | 1 / 3 | Potholes and patches; no drain grate in any hit |
| standing water | 0 / 0 | Nothing (dry day) |
| road debris | 0 / 0 | Nothing |
| cracked asphalt | 2 / 1 | The whole road surface |

- **Absent concepts stay quiet:** standing water and road debris returned
  nothing, so they do not invent hazards on a clean, dry lane. Their recall is
  untested, because there is no footage of either.
- **Look-alike concepts land on potholes:** with no manhole or grate in view,
  the closest-looking round dark depressions pass the hosted cut-off. The
  candidates are real surface defects under the wrong name, which is why each
  one goes to triage and a person instead of being trusted by label.
- **Diffuse concepts are not candidates:** "cracked asphalt" segments the
  whole carriageway. Area-type defects need a different output, such as the
  share of each frame affected, rather than one tracked object.

Showing another hazard class properly needs footage that contains one.

### Real-time limits: edge versus hosted

RoadLens is built for post-ride review, but the same detector could run on a
live camera on a robot or a vehicle. Ten native-resolution frames around both
reported potholes were used to measure per-frame text-prompt detection
without tracking. A live camera delivers raw frames, so video decoding was
left out of the timed loop
([data](evaluation/realtime-edge-vs-hosted-2026-11-03.json),
[scripts](../benchmarks/realtime/README.md)).

**Edge.** The SAM 3 image model ran on a Jetson AGX Orin 64 GB (MAXN, bf16)
and, for comparison, on the RTX 3090:

| Model input | Orin AGX | RTX 3090 | Detections vs 1008 px |
|---|---|---|---|
| 1008 px (native) | 2.2 fps, 450 ms | 7.0 fps | reference |
| 672 px | 4.3 fps, 232 ms | 13.8 fps | same, except the most distant frame |
| 336 px | 7.5 fps, 133 ms | 29.4 fps | several lost, weaker scores |

- **Camera resolution barely matters:** SAM 3 resizes every frame to its own
  input size, so a 672 × 540 camera frame was only 1–3 % faster than
  2688 × 2160. Shrinking the model input is what pays, and 672 px kept the
  detections.
- **Compute-bound:** the Orin GPU ran at 99 % and the GPU+SoC rail peaked at
  42 W. `torch.compile` gained only 3–4 %; TensorRT, a smaller distilled
  model, or a cheap detector that wakes SAM only on likely frames are the
  remaining options, none tested.

**Hosted.** The same frames went to `sam-3.1` on the Meta Model API from a
home connection:

| Mode | Throughput | Delay per result | Price |
|---|---|---|---|
| Image, 1 request at a time, 672 × 540 JPEG | 1.1 fps | 0.9 s | $2.50 per 1,000 |
| Image, 1 at a time, 2688 × 2160 JPEG | 0.55 fps | 1.8 s | $2.50 per 1,000 |
| Image, 16 in flight | 11.6 fps | 0.9 s | $2.50 per 1,000 |
| Image, 48 in flight | 29.7 fps | 1.1 s | $2.50 per 1,000 |
| Video, 5 s clip | 0.55 s of video per second | 9.1 s after the clip ends | $0.20 per 1,000 frames |

- **Image mode scales with requests in flight** and reached camera rate at
  48, well inside the 3,000 requests per minute limit, but every result is
  about a second old and costs 12.5 times as much per frame as video mode:
  about $36 an hour at 4 frames per second.
- **Upload size is the one place resolution matters:** a full-resolution
  JPEG doubles the delay, and 30 fps of 672 px JPEGs needs about 15 Mbit/s of
  uplink.
- **Video mode is not real time:** it takes finished clips only. A 1 s clip
  took 4.9 s end to end, mostly upload and request overhead, and long clips
  were processed at about 4 frames per second, roughly 2.6 times slower than
  10 fps footage. Sending three chunks at once did not speed up a 180 s
  window, which suggests per-account throttling. It suits uploading a ride
  as it happens, with candidates ready minutes after it ends.

**What this means by vehicle.** RoadLens samples about one frame per metre,
and a pothole is large enough to detect over roughly the last 15–20 m. On
the Orin at 672 px:

| Vehicle | Speed | Metres between frames | Metres travelled per inference |
|---|---|---|---|
| Warehouse robot, forklift | 2 m/s | 0.5 | 0.5 |
| Tractor, site machinery | 4 m/s | 0.9 | 0.9 |
| Bicycle (this footage) | 6.7 m/s | 1.6 | 1.5 |
| Car at 70 mph | 31 m/s | 7.2 | 7.2 |

Slow machinery gets the one-frame-per-metre density on the Orin, with under
a metre travelled before each result, which is enough to react. A bicycle
gets a look every 1.6 m, enough to log hazards live. A motorway car gets
about two looks at a pothole and moves 31 m during a hosted round trip,
which might do for coarse logging but not for reacting; it would need
several times the compute.
