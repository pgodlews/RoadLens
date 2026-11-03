---
name: roadlens-investigator
description: Investigate post-ride Garmin cycling footage by street name, ride offset, wall-clock time, or map corridor using RoadLens MCP tools, SAM 3.1, and local Muse Glimmer or remote VLM triage while enforcing bookended human-in-the-loop review.
---

# RoadLens Investigator Skill

Use this skill when a rider wants to locate, inspect, triage, or summarize road hazards (such as potholes, surface breakups, flooding, debris, or a fallen tree branch) from Garmin camera footage without manually scrubbing entire ride videos.

## MCP Server Configuration

Run the built-in stdio MCP server (`roadlens mcp`) with your local OpenStreetMap PMTiles v3 archive and optional SAM 3.1 checkpoint:

```json
{
  "mcpServers": {
    "roadlens": {
      "command": "/path/to/RoadLens/.venv/bin/roadlens",
      "args": [
        "mcp",
        "--workspace", "artifacts/mcp",
        "--pmtiles", "/path/to/united-kingdom.pmtiles",
        "--checkpoint", "/path/to/sam3.1_multiplex.pt"
      ],
      "env": {
        "ROADLENS_GLIMMER_GGUF": "/path/to/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
        "ROADLENS_GLIMMER_MMPROJ": "/path/to/mmproj-Muse-Glimmer-30B-BF16.gguf"
      }
    }
  }
}
```

For remote VLM inference instead of local Muse Glimmer, set `MODEL_API_KEY` (Meta Model API, `muse-spark-1.3` by default) or point `ROADLENS_VLM_API_URL`, `ROADLENS_VLM_MODEL` and `ROADLENS_VLM_API_KEY` at another OpenAI-compatible endpoint, then pass `"vlm": "remote"` to VLM tools. Remote calls automatically mask the bottom Garmin burned-in GPS banner by default.

## Core Investigation Workflows

### 1. Street-Name Query
Example: *"Find me potholes on Eccup Moor Road."*

1. Call `index_ride_clips` on the Garmin clip or ride folder (automatically skips `.GLV` companion files and deduplicates overlapping saved event clips).
2. Call `resolve_ride_segment` with `mode: "street"` and `value: "Eccup Moor Road"`.
   - Note whether the rider traversed the street once or multiple times (e.g. outbound and return passes) and check `estimated_coarse_frames_1m` before launching heavy scans.
3. Call `scan_segment_sam` on each relevant pass with `prompt: "pothole"` (`refine: true` for best native-frame boundary evidence, or `refine: false` for a fast ~1 m coarse pass).
4. Call `triage_run_candidates` (`vlm: "local"` or `"remote"`) to classify candidates into `hazard_candidate`, `sealed_repair`, `surface_debris_or_stain`, or `uncertain` using non-occluding raw frame + 1:1 close-up crops.
5. Call `draft_summary_report` and direct the rider to confirm/reject candidates in `roadlens studio`.

### 2. Elapsed Ride-Offset Query (Unknown Street)
Example: *"I spotted a pothole 20 minutes into my ride, don't know the street name."*

1. Call `index_ride_clips` on the ride directory.
2. Call `resolve_ride_segment` with `mode: "offset"`, `value: "20m"`, and `window_seconds: 30` (widen to `60` if the rider's timing estimate is approximate).
   - Report the offline reverse-geocoded `street_name`, `place_name`, `clip_name`, and exact clip timestamps back to the rider.
3. Run `scan_segment_sam` (`prompt: "pothole"`) followed by `triage_run_candidates` and `draft_summary_report`.

### 3. Wall-Clock & Open-Vocabulary Hazard Query (Unknown Street)
Example: *"I passed a large fallen tree branch on the street around 12:30, don't know the street name."*

1. Call `index_ride_clips` on the ride directory.
2. Call `resolve_ride_segment` with `mode: "clock"`, `value: "12:30"`, `timezone_name: "Europe/London"`, and `window_seconds: 60`.
   - Report the offline reverse-geocoded `street_name` and `place_name`.
3. For open-vocabulary obstructions or scene-level hazards (`"large fallen tree branch"`, `"flooded lane"`, `"broken glass"`), call `discover_segment_hazards_vlm` (`spacing_m: 4.0`, `max_frames: 8`, `road_crop: false` when branches/obstructions may extend above the lower road crop) for rapid visual discovery.
4. Optionally follow up with `scan_segment_sam` (`prompt: "fallen tree branch"`) if pixel-level mask overlays and multi-frame tracking are desired, then `draft_summary_report`.

## Hardware & Civic Guardrails (Mandatory)

1. **Bookended Human-in-the-Loop:**
   - Human scopes the segment/query at the start and explicitly confirms overlays + edits the summary report at the end.
   - **Never** present `review_status: "unreviewed"` candidates or `vlm_triage: "hazard_candidate"` tags as confirmed physical defects.
   - **Never** submit reports automatically to local councils, highway authorities, or FixMyStreet. Always instruct the user to review and export confirmed GeoJSON via `roadlens studio --pmtiles <tiles.pmtiles>`.
2. **Scientific & Positional Honesty:**
   - Always state that coordinates are Garmin camera GPS positions (`camera_gps`), not surveyed defect coordinates.
   - Never invent physical width, depth, or severity measurements from monocular footage. You may quote a candidate's `size_estimate.display` as an approximate camera-geometry footprint; never present it as measured, never infer depth from it, and never claim it meets a council intervention threshold.
3. **Single-GPU Sequential Execution:**
   - On a single 24 GB GPU (e.g. RTX 3090), never run `scan_segment_sam` (`LocalSamBackend`) and `LocalGlimmerBackend` (`llama-server`) concurrently. Run SAM first; `scan_segment_sam` releases CUDA cache before returning so VLM triage can safely load next.
