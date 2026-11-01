# RoadLens

RoadLens turns Garmin cycling-camera footage into reviewable observations of road hazards. It is a **post-ride analysis tool**, not a live warning system.

The initial goal is an end-to-end workflow that imports a clip, detects and segments candidate hazards with Meta models, preserves video evidence, and exports structured results for review and mapping.

## Status

Initial development for the Meta Global AI Developer Hackathon. The detection approach will be validated against real Garmin footage before the architecture is finalised.

## Privacy and licensing

Source clips, ride telemetry, credentials, and generated analysis artifacts are excluded from Git. Do not commit private location data or footage containing identifiable people or vehicles.

RoadLens source code is available under the MIT License. Meta SAM 3.1 and its checkpoints are governed separately by the [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE).

## Development history

Development started from an empty repository on 1 November 2026. Commit history records the implementation as it is built and validated during the hackathon.
