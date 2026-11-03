# Notes for coding agents

Read [ARCHITECTURE.md](ARCHITECTURE.md) first. These are the rules and habits
that keep the repository consistent.

## Setup and checks

```bash
uv sync --extra local --extra remote --extra dev
uv run pytest -q -W error
uv run ruff check .
```

- Run both checks before every commit. Warnings are errors in tests.
- Run `ruff format` only on files you wrote; the repository is not
  reformatted wholesale.
- Local SAM needs `ROADLENS_SAM_CHECKPOINT` and a CUDA GPU; tests do not.
- Secrets live in `.env` (git-ignored). Load it with
  `set -a && . ./.env && set +a`, and never print or commit a key.

## Hard rules

- **Never commit ride data:** no video, `.fit`, `.gpx` or other telemetry,
  raw GPS tracks, local storage paths, or anything under `artifacts/`.
- **Keep the rider's home out:** publish only footage and GPS from the Eccup
  corridor used in the docs; nothing from the start or end of a ride, and no
  near-home footage sent to hosted APIs.
- **Tesla dashcam clips** live on the rider's NAS and carry GPS for every
  drive, including home. Keep them and anything derived from them local
  unless the rider approves a specific stretch.
- **Never mark a candidate confirmed or rejected in code,** and never submit
  anything to a council or other service. Model output stays `unreviewed`.
- **Delete hosted uploads** (`DELETE /v1/files/{id}`) once a request is done.
- **Report results honestly:** no recall or precision claims from a single
  road, sizes described as approximate, and failures documented alongside
  successes.

## Conventions

- Match the surrounding code: naming, comment density, and small functions
  with docstrings that explain why.
- Keep source timestamps (PTS) as the time base; never derive times from
  re-encoded playback.
- New behaviour that changes results gets a flag, a sensible default and a
  way back to the old behaviour (see `--legacy-sampling`).
- Tests use the real types of external libraries (for example
  `meta_sam_parser` records) rather than guessed stand-ins; mocks that guessed
  field names hid two bugs in the first live API run.
- `STUDIO_HTML` in `studio.py` is a plain triple-quoted string: it must not
  contain backslashes or triple quotes.
- On a single 24 GB GPU, never run local SAM and local Muse Glimmer at the
  same time.

## Documentation

- Record decisions and measurements as a dated section in
  `docs/development-log.md`, with data in `docs/evaluation/*.json`.
- Update the README's results and "What is verified" table when a capability
  is first run on real footage.
- Keep the MCP tool descriptions and
  `skills/roadlens-investigator/SKILL.md` in step with CLI behaviour.

## Hosted Meta Model API facts

- `sam-3.1` returns objects above an apparent 0.7 cut-off with no threshold
  setting, and a session tracks at most 16 objects in total, so long clips are
  chunked.
- `muse-spark-1.3` spends about 3,300 reasoning tokens before answering;
  reserve them in `max_tokens`. Use JSON-schema `response_format`.
- Pricing: images $2.50 per 1,000, video $0.20 per 1,000 frames (minimum 50
  per request).
