# RoadLens

RoadLens turns Garmin cycling-camera footage into human-reviewed road-hazard evidence and summary reports. It is a **post-ride analysis workflow**, not a live warning system or an automated council spammer.

RoadLens implements a **bookended human-in-the-loop** pattern:
1. **Human scopes at the start:** drop or index a ride folder/clip on an offline OpenStreetMap basemap (`roadlens studio`)—or ask an assistant via MCP (`roadlens mcp`) by street name, elapsed ride offset (*"20 minutes into my ride"*), or wall-clock time (*"around 12:30"*)—and pick a hazard preset (`pothole`, `cracking`, `debris`, `flooding`, `fallen tree branch`) or custom prompt.
2. **Meta SAM 3.1 + VLM assist in the middle:** distance-sampled (`~1 m`) + native-rate refined SAM 3.1 segments candidate hazards, and local **Muse Glimmer** (`llama.cpp`) or a remote OpenAI-compatible VLM triages candidates using non-occluding frame + 1:1 crop composites (`hazard_candidate`, `sealed_repair`, `surface_debris_or_stain`, `uncertain`) and drafts a grounded Markdown summary report.
3. **Human confirms at the end:** every model detection stays `review_status: "unreviewed"` until a human inspects the visual overlay, confirms or rejects each candidate, optionally edits the summary report, and exports **only human-confirmed** GeoJSON. Coordinates are explicitly labelled as **Garmin camera GPS** (`camera_gps`), never surveyed defect coordinates, and nothing is submitted automatically to local authorities.

## Status

Built for the Meta Global AI Developer Hackathon:
- **Bidirectional SAM 3.1 corridor survey completed and audited** on real Eccup Moor Road Garmin footage (~2.7 km across outbound and return passes; see [docs/adaptive-sampling.md](docs/adaptive-sampling.md) and [docs/evaluation/eccup-corridor-survey.json](docs/evaluation/eccup-corridor-survey.json)).
- **Offline PMTiles v3 MVT street-name resolution & multi-clip Garmin ride indexing** (`src/roadlens/pmtiles_roads.py`, `src/roadlens/ride.py`) verified offline against `united-kingdom.pmtiles` with zero external API calls.
- **Dual VLM triage & summary reporting** (`src/roadlens/vlm.py`): `LocalGlimmerBackend` (`Muse-Glimmer-30B` via `llama-server`) empirically verified on an RTX 3090; `RemoteVlmBackend` and remote SAM API conversion are unit-tested with mocked HTTP responses (with default burned-in GPS-banner masking on remote VLM calls).
- **Interactive web workflow (`roadlens studio`)** and **stdio MCP server (`roadlens mcp` + `skills/roadlens-investigator/SKILL.md`)** ready for human-guided and agent-assisted investigation.

## Requirements

- Python 3.12, FFmpeg/FFprobe, and ExifTool.
- Local OpenStreetMap `.pmtiles` archive (Protomaps v3/v4 MVT vector tiles) for map display and offline street lookup.
- Local SAM 3.1 backend: a CUDA GPU and `sam3.1_multiplex.pt`.
- Optional local VLM backend: `llama-server` (`llama.cpp`) plus `Muse-Glimmer-30B` GGUF + multimodal projector (`ROADLENS_GLIMMER_GGUF`, `ROADLENS_GLIMMER_MMPROJ`). On a single 24 GB GPU, RoadLens runs SAM 3.1 and Muse Glimmer sequentially, releasing CUDA cache between stages.
- Optional remote backends: `MODEL_API_KEY` (SAM remote API) and `ROADLENS_VLM_API_URL` / `ROADLENS_VLM_MODEL` / `ROADLENS_VLM_API_KEY` (OpenAI-compatible multimodal VLM).

## Setup

```bash
uv sync --extra local --extra remote --extra dev
export ROADLENS_SAM_CHECKPOINT=/path/to/sam3.1_multiplex.pt
export ROADLENS_PMTILES=/path/to/united-kingdom.pmtiles
```

## Workflows

### 1. Interactive Post-Ride Studio (Recommended)

Overlay multi-clip Garmin GPS tracks on local OpenStreetMap tiles, click two points along a road fragment (or search by street name, ride offset, or wall-clock time), run targeted SAM 3.1 + VLM triage, confirm/reject candidate overlays, edit the drafted Markdown report, and export confirmed GeoJSON:

```bash
uv run roadlens studio /path/to/garmin-ride-folder \
  --pmtiles /path/to/united-kingdom.pmtiles \
  --workspace artifacts/studio
```

Open `http://127.0.0.1:8766`. You can also drag-and-drop a `.MP4` clip into the browser or open any existing `scan` / `analyze` directory for review.

### 2. Distance-Sampled Local SAM 3.1 Scan (`roadlens scan`)

Select a sharp nearby frame approximately every metre using integrated Garmin speed, retain native source resolution and presentation timestamps, and revisit `±2 s` around coarse candidates at full native frame rate:

```bash
uv run roadlens scan /path/to/GRMS0010.MP4 \
  --prompt pothole \
  --start 150 --duration 210 --spacing 1 \
  --output artifacts/eccup-distance
```

- `--prompt` accepts any text concept (default `pothole`, e.g. `"fallen tree branch"`, `"flooding"`).
- `--no-refine` runs only the coarse `~1 m` pass; `--sampling all` compares against every native frame; `--plan-only` measures distance sampling without loading SAM; `--resume` replays completed chunks from `scan.json` / `frames.jsonl`.
- One metre spacing and the default thresholds (`0.25` / `0.35`) are experimental settings, not validated optimal values; see [docs/adaptive-sampling.md](docs/adaptive-sampling.md).

### 3. VLM Candidate Triage & Draft Summary Report (`roadlens triage`)

Run local Muse Glimmer (`--vlm local`) or a remote OpenAI-compatible VLM (`--vlm remote`) over an existing scan directory to attach `vlm_triage` classifications (`hazard_candidate`, `sealed_repair`, `surface_debris_or_stain`, `uncertain`) and write `summary-report.md` while keeping `review_status: "unreviewed"`:

```bash
uv run roadlens triage artifacts/eccup-distance \
  --vlm local \
  --street-name "Eccup Moor Road"
```

### 4. Agentic MCP Server & Investigator Skill (`roadlens mcp`)

Expose RoadLens over stdio JSON-RPC (Model Context Protocol) for Muse Glimmer, Claude Code, Antigravity, or other MCP hosts:

```bash
uv run roadlens mcp \
  --workspace artifacts/mcp \
  --pmtiles /path/to/united-kingdom.pmtiles
```

Available MCP tools: `index_ride_clips`, `resolve_ride_segment` (`street`, `offset`, `clock`, `corridor`), `discover_segment_hazards_vlm`, `scan_segment_sam`, `triage_run_candidates`, and `draft_summary_report`. Companion agent instructions and guardrails live in [skills/roadlens-investigator/SKILL.md](skills/roadlens-investigator/SKILL.md).

### 5. Lower-Level Clip & Review Commands

Fixed-FPS clip pass (`local` or `remote` backend):

```bash
uv run roadlens analyze /path/to/GRMS0010.MP4 \
  --prompt pothole --backend local --start 243 --duration 10 \
  --output artifacts/eccup
```

Extract timed Garmin GPS telemetry only:

```bash
uv run roadlens telemetry /path/to/GRMS0010.MP4 \
  --output artifacts/telemetry.json
```

Serve the lightweight single-run OSM review page:

```bash
uv run roadlens report artifacts/eccup \
  --pmtiles /path/to/united-kingdom.pmtiles
```

## Checks

```bash
uv run pytest
uv run ruff check .
```

## Privacy and licensing

Source clips, ride telemetry, credentials, and generated analysis artifacts are excluded from Git. Do not commit private location data or footage containing identifiable people or vehicles.

RoadLens source code is available under the MIT License. Meta SAM 3.1 and its checkpoints are governed separately by the [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE). Prior-work attribution is recorded in `NOTICE`.

## Development history

Development started from an empty repository on 1 November 2026. Commit history records the implementation as it is built and validated during the hackathon. See the [development log](docs/development-log.md) for real-footage experiments, negative results, prior-work attribution, and hosted-API verification status, and [ideas to explore](docs/ideas-to-explore.md) for optional future enhancements (such as kinematic and homography-based pothole width/length estimation).
