# Muse Glimmer road-hazard evaluation

## Method

On 1 November 2026, Meta's official `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` and `mmproj-Muse-Glimmer-30B-Q4_K_M.gguf` were run locally through llama.cpp build 12032 with CUDA on an RTX 3090. Both files came from `meta-models/Muse-Glimmer-30B-GGUF` (Apache 2.0).

The six published Garmin frames in `docs/samples` were evaluated with a consistent structured-output prompt. The model was asked to identify only visible road-surface hazards, distinguish repairs and discolouration, provide normalized boxes, and avoid inferring dimensions or severity. Three reported-location examples and the three initial examples were also tested using a crop covering the road area (42–94% of frame height). Reasoning strength was `medium`; sampling followed Meta's recommended temperature 1.0, top-p 0.95, and top-k 64.

Raw model responses are preserved in `docs/evaluation`. These are a small qualitative evaluation, not a representative accuracy benchmark.

## Full-frame results

| Sample | Glimmer output |
|---|---|
| Control | No observations. |
| Repair/discolouration | No observation; correctly noted the lighter patch without confirming damage. |
| Cracking/damage | No observations; missed visible surface distress. |
| First reported pothole | No observations; missed the visible damage. |
| Second location, outbound | Low-confidence damaged-asphalt candidate. |
| Second location, return | Medium-confidence damaged asphalt and low-confidence possible pothole. |

## Road-only crop results

| Sample | Glimmer output |
|---|---|
| Control | No observations. |
| Repair/discolouration | Low-confidence damaged-asphalt candidate, while acknowledging repair/staining uncertainty. |
| Cracking/damage | No observations. |
| First reported pothole | No observations. |
| Second location, outbound | Two damaged-asphalt candidates and one cracking candidate. |
| Second location, return | Medium-confidence possible pothole, medium-confidence damaged asphalt, and low-confidence cracking. |

## Assessment

Road-only cropping improved sensitivity and box specificity. Glimmer also remained conservative on the smooth control and uncertain repair, which is desirable for a candidate-review workflow.

Important limitations:

- It missed the first user-reported pothole in both full-frame and crop tests, so recall is inconsistent.
- Some boxes are coarse regions rather than object boundaries. One outbound full-frame box extended beyond the image; coordinates require validation and clipping.
- A textual `pothole` label and categorical confidence are not verified severity or ground truth.
- Adjacent video frames are not independent examples. No precision, recall, or accuracy percentage should be inferred from this sample.

The annotated full-frame outputs for the second location are included as `glimmer-second-outbound-annotated.png` and `glimmer-second-return-annotated.png`.

**Recommendation:** Glimmer is worth integrating as a candidate-discovery stage on road-focused crops. Its output needs coordinate validation, segmentation/tracking or human review, and evaluation over a larger labelled set. It is not yet suitable as a standalone pothole detector.
