# RoadLens

RoadLens turns Garmin cycling-camera footage into reviewable observations of road hazards. It is a **post-ride analysis tool**, not a live warning system.

The initial goal is an end-to-end workflow that imports a clip, detects and segments candidate hazards with Meta models, preserves video evidence, and exports structured results for review and mapping.

## Status

Initial development for the Meta Global AI Developer Hackathon. The repository contains local and remote SAM 3.1 backends, Garmin timed-GPS extraction, segmentation overlays, and GPS-linked JSON/GeoJSON observations. Detection quality is being validated against real Eccup Moor Road footage.

## Requirements

- Python 3.12, FFmpeg/FFprobe, and ExifTool.
- Local backend: a CUDA GPU and the SAM 3.1 multiplex checkpoint.
- Remote backend: `MODEL_API_KEY` and Meta Model API access. Remote response conversion is unit-tested with mocked records; the live API has not been exercised yet.

## Setup

```bash
uv sync --extra local --extra remote --extra dev
export ROADLENS_SAM_CHECKPOINT=/path/to/sam3.1_multiplex.pt
```

Analyse a ten-second section of a Garmin ride:

```bash
uv run roadlens analyze /path/to/GRMS0010.MP4 \
  --prompt pothole --backend local --start 243 --duration 10 \
  --output artifacts/eccup
```

The output includes the downsampled analysis clip, overlays for frames with detections, `observations.json`, and `observations.geojson`. Observations remain `unreviewed`; their points represent **camera GPS**, not measured hazard coordinates.

Use `--backend remote` to select the hosted Meta API instead. No remote request is made during tests.

Scan by distance, then inspect candidate neighbourhoods at the native frame rate:

```bash
uv run roadlens scan /path/to/GRMS0010.MP4 \
  --start 150 --duration 210 --spacing 1 \
  --output artifacts/eccup-distance
```

This local-only experimental workflow selects a sharp nearby frame approximately
every metre using integrated Garmin speed. It retains source timestamps and native
resolution, uses overlapping SAM chunks, and revisits two seconds before and after
each detected candidate. `--sampling all` provides a full-frame comparison;
`--plan-only` measures sampling without loading SAM. `--width 1280` explicitly
trades evidence resolution for lower storage and processing cost.

`scan.json` records settings, selected timestamps, coverage and timings;
`frames.jsonl` records every inference result, including empty frames. The existing
`report` command displays the chosen evidence and camera GPS observations. Use a
new output directory for every run. One metre and the detection thresholds are
experimental settings, not validated optimal values. See the
[experiment journey](docs/adaptive-sampling.md) for methods and measured results.

Extract only telemetry:

```bash
uv run roadlens telemetry /path/to/GRMS0010.MP4 \
  --output artifacts/telemetry.json
```

Serve the review interface with a local OpenStreetMap PMTiles archive:

```bash
uv run roadlens report artifacts/eccup \
  --pmtiles /path/to/united-kingdom.pmtiles
```

Open `http://127.0.0.1:8765`. The report displays observation markers and segmentation evidence, supports confirm/reject review, and downloads reviewed GeoJSON. Leaflet and PMTiles JavaScript load from pinned CDN URLs; map data comes from the local archive.

Run checks:

```bash
uv run pytest
uv run ruff check .
```

## Privacy and licensing

Source clips, ride telemetry, credentials, and generated analysis artifacts are excluded from Git. Do not commit private location data or footage containing identifiable people or vehicles.

RoadLens source code is available under the MIT License. Meta SAM 3.1 and its checkpoints are governed separately by the [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE). Prior-work attribution is recorded in `NOTICE`.

## Development history

Development started from an empty repository on 1 November 2026. Commit history records the implementation as it is built and validated during the hackathon. See the [development log](docs/development-log.md) for real-footage experiments, negative results, prior-work attribution, and hosted-API verification status.
