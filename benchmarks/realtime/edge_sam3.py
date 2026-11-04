"""Per-frame SAM 3 text-prompt detection speed on a local GPU.

Every frame is detected independently (no tracking), as a live camera stream
needs: upload, resize to the model size, image backbone, prompt decoder and
mask upsampling back to the camera resolution. The text prompt is encoded
once and reused. Frames are decoded beforehand: a live camera delivers raw
frames, so video decoding is not part of the measured loop.

Run from a directory containing the ``sam3`` package (or with it installed):

    python edge_sam3.py --checkpoint sam3.pt --frames frames/ --out result.json
"""

import argparse
import gc
import inspect
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path.cwd()))


def build(size: int, checkpoint: str, compile_backbone: bool):
    """SAM 3 image model with ``size`` px input instead of the built-in 1008."""
    import sam3.model_builder as builder
    import torch

    source = inspect.getsource(builder).replace("1008", str(size))
    namespace = dict(builder.__dict__)
    exec(compile(source, builder.__file__, "exec"), namespace)
    model = namespace["build_sam3_image_model"](
        checkpoint_path=None, load_from_HF=False, compile=compile_backbone
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    weights = {
        k.replace("detector.", ""): v
        for k, v in state.items()
        # RoPE tables are fixed functions of the input size; keep the model's
        # own when it was built for another size.
        if "detector" in k and not (size != 1008 and k.endswith("freqs_cis"))
    }
    missing, _ = model.load_state_dict(weights, strict=False)
    missing = [k for k in missing if not k.endswith("freqs_cis")]
    if missing:
        raise RuntimeError(f"Missing weights: {missing[:5]}")
    return model


def run_input(size, input_size, originals, detect, rounds):
    """Keep output tensors local and measure memory for just this configuration."""
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    w, h = (int(v) for v in input_size.split("x"))
    frames = [
        torch.from_numpy(np.asarray(im.resize((w, h), Image.BILINEAR)))
        .permute(2, 0, 1)
        .contiguous()
        for im in originals
    ]
    for frame in frames[:3]:  # warm-up
        detect(frame)
    torch.cuda.synchronize()
    times, found = [], []
    for round_index in range(rounds):
        for frame in frames:
            start = time.perf_counter()
            state = detect(frame)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
            if round_index == 0:
                scores = state["scores"].float().cpu().tolist()
                best = round(max(scores), 3) if scores else None
                found.append({"detections": len(scores), "max_score": best})
            del state
    times.sort()
    run = {
        "model_size": size,
        "input": input_size,
        "median_ms": round(statistics.median(times), 1),
        "p90_ms": round(times[int(0.9 * (len(times) - 1))], 1),
        "fps": round(1000 / statistics.median(times), 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "per_frame": found,
    }
    return run


def run_model(size, originals, args, dtype):
    """Release the model, processor and prompt closure before the next size."""
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor

    gc.collect()
    torch.cuda.empty_cache()
    model = build(size, args.checkpoint, args.compile)
    processor = Sam3Processor(
        model, resolution=size, confidence_threshold=args.threshold
    )
    autocast = torch.autocast("cuda", dtype=dtype, enabled=dtype is not None)
    runs = []
    with torch.inference_mode(), autocast:
        text = model.backbone.forward_text([args.prompt], device="cuda")

        def detect(frame):
            state = processor.set_image(frame)
            state["backbone_out"].update(text)
            state["geometric_prompt"] = model._get_dummy_prompt()
            return processor._forward_grounding(state)

        for input_size in args.inputs:
            run = run_input(size, input_size, originals, detect, args.rounds)
            runs.append(run)
            print(json.dumps({k: v for k, v in run.items() if k != "per_frame"}))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1008, 672, 336])
    parser.add_argument(
        "--inputs", nargs="+", default=["2688x2160", "1344x1080", "672x540"]
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prompt", default="pothole")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch

    names = sorted(p.name for p in args.frames.glob("*.png"))
    originals = [Image.open(args.frames / n).convert("RGB") for n in names]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]
    results = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "compile": args.compile,
        "prompt": args.prompt,
        "threshold": args.threshold,
        "frames": names,
        "runs": [],
    }
    for size in args.sizes:
        results["runs"].extend(run_model(size, originals, args, dtype))
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
