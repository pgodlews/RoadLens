# Real-time benchmarks

Two scripts measure whether RoadLens' text-prompt detector keeps up with a
live camera. Results and discussion are in the
[development log](../../docs/development-log.md#real-time-limits-edge-versus-hosted)
and the [data file](../../docs/evaluation/realtime-edge-vs-hosted-2026-11-03.json).

Both read a folder of `.png` frames, decoded beforehand, because a live camera
delivers raw frames rather than compressed video.

## Edge: `edge_sam3.py`

Times the SAM 3 image model on one frame at a time with the prompt encoded once,
at model input sizes of 1008, 672 and 336 px and several camera resolutions.

```bash
python edge_sam3.py --checkpoint sam3.pt --frames frames/ --out result.json
```

Add `--compile` to try `torch.compile`. On a Jetson AGX Orin (JetPack 6.2,
Python 3.10), these steps got it running:

- Install `torch==2.10.0` and `torchvision==0.25.0` from
  `https://pypi.jetson-ai-lab.io/jp6/cu126/+simple`, plus the `sam3` package
  and its dependencies (`einops`, `pycocotools`, `psutil`, `timm`, `ftfy`,
  `iopath`, and Triton 3.6 from the same index).
- That torch wheel needs `libcudss.so.0`. Install `nvidia-cudss-cu12<1`, then
  uninstall the `nvidia-cublas-cu12` and `nvidia-cuda-nvrtc-cu12` it pulls in:
  they shadow the Jetson libraries and fail with `CUBLAS_STATUS_ALLOC_FAILED`.
  Add the cuDSS `lib` folder to `LD_LIBRARY_PATH`.
- For `--compile`, set `CPATH=/usr/local/cuda/include` and
  `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.

## Hosted: `hosted_sam.py`

Times `sam-3.1` on the Meta Model API: image requests one at a time at
several JPEG sizes, image requests with 4 to 48 in flight, and short video
clips from upload to result. Uploads are deleted after use.

```bash
MODEL_API_KEY=... python hosted_sam.py --frames frames/ --video ride.mp4 \
  --video-start 584 --out result.json
```

The defaults send about 270 images and three short clips, roughly $0.70 at
$2.50 per 1,000 images and $0.20 per 1,000 video frames.
