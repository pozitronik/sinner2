# Investigation backlog: performance, hardware, models

Ideas surfaced while reviewing the inference hot paths. Not committed work —
a backlog to validate and prioritize. Each item notes rough payoff, effort,
and the code it touches (paths/lines are accurate as of when this was written;
re-check before acting).

**Before investing in any perf item:** get a profile first (`py-spy dump` /
`py-spy record` on a realtime session, and an `nsys`/timing pass on a batch
run). The items below are ranked *hypotheses* about where the wall-clock goes
(GPU inference? PNG codec? decode? the GFPGAN lock?). The TensorRT + fp16 pair
is low-risk enough to just try; validate the rest against a profile.

## Current state (baseline)

What the pipeline already does well, so we don't redo it:

- Realtime decode is overlapped with inference via the async `ReaderPool`
  (dispatcher submits non-blocking, workers await futures).
- The swapper ONNX session is **shared** across workers (ORT handles concurrent
  `.run()`); only the non-thread-safe GFPGAN enhancer is per-worker
  (`PerWorkerProcessor`).
- The source face **embedding is computed once** at setup and cached
  (`face_swapper.py` setup, ~L114), not recomputed per frame.
- Batch is **processor-major**: each stage runs the whole frame range through
  one resident processor before the next stage — model stays warm.
- Detection has a **frame-interval cache** seam already
  (`face_analyser.py` ~L66–87, `FaceSwapperParams.detection_interval`).

## 1. Performance

### High payoff

- **TensorRT execution provider** for the swapper + detector. Today the ORT
  session is created with only a providers list — no `session_options`, fp16,
  IO binding, CUDA graph, or cudnn algo search (`face_swapper.py` ~L70–79,
  `face_analyser.py` ~L30–38). inswapper_128 + SCRFD/buffalo_l are fixed-shape
  small-conv models → typically 1.5–3× over CUDA EP. Cost: first-run engine
  build (cache it with `trt_engine_cache_enable`). Slots into the existing
  providers list (`["TensorrtExecutionProvider", "CUDAExecutionProvider",
  "CPUExecutionProvider"]`). **Likely the single biggest realtime lever.**

- ~~**FP16 for GFPGAN.**~~ **✅ DONE.** `FaceEnhancerParams.fp16` (default on,
  CUDA only) half()s the GFPGAN generator and wraps `restorer.enhance` in
  `torch.autocast("cuda", float16)` (`face_enhancer.py`). Halves the generator's
  per-worker VRAM + tensor-core convs; toggle in the realtime + batch UIs.

- **Batch the enhancer in batch mode.** Batch stages call `process()` one frame
  at a time (`stage.py` ~L286 feed loop, ~L351 `_process_write`) — no batched
  tensor inference. Because staging is processor-major the model is already
  resident for the whole stage, so a `process_batch(frames)` path (especially
  for GFPGAN) raises GPU occupancy with no model-reload cost. Needs a
  `process_batch` seam on the processor protocol.

### Medium payoff

- **Skip the PNG round-trip in batch.** Every stage writes frames to disk as
  PNG/JPEG and the next stage + ffmpeg re-read them (`stage.py` ~L116,
  `driver.py` ~L186, `video_encoder.py` reads `%08d.{ext}`). PNG codec is
  CPU-heavy and serialized against GPU work. For the **final** encode, pipe raw
  BGR frames into ffmpeg stdin (`-f rawvideo -pix_fmt bgr24 -i -`) to skip both
  the codec and the disk read. Keep the disk cache for resume; consider an
  in-memory handoff between stages with disk spill only on pause. (Related to
  the deferred pre-extract/preview-reuse work.)

- **IO binding for the swapper.** Each `swapper.get()` ships numpy in/out →
  host↔device copy per face per frame (`face_swapper.py` ~L124). `io_binding` +
  pinned memory keeps the frame on-device across the chain. Catch: insightface's
  `INSwapper.get()` hides the session, so we'd call the underlying session
  directly to bind buffers. More invasive than the EP swap.

- **Detection downscale / interval.** `det_size` hardcoded `(640, 640)` on the
  full-res frame every frame at `detection_interval=1` (`face_analyser.py`
  ~L35, ~L78). For static talking-head video, detect every 2–3 frames and reuse
  the box (machinery already exists) — gated so multi-worker realtime doesn't
  reuse a stale box across the wrong frame.

### Lower / situational

- **NVDEC GPU decode.** Both readers are CPU-only
  (`cv2_video_target_reader.py`, `video_target_reader.py` ffmpeg subprocess).
  For 4K targets, decode can become the long pole. `ffmpeg -hwaccel cuda` or
  PyNvVideoCodec offloads it — but decode already overlaps inference, so only
  worth it when a profile shows decode-bound. Measure first.

- **ORT session tuning** (cheap to try alongside TensorRT): graph optimization
  level ALL, intra/inter-op thread counts, cudnn conv algo search.

## 2. Intel AI Boost (NPU) as a backend

Technically reachable, not worth it as a primary backend.

- **Path:** ONNX Runtime → Intel NPU via the **OpenVINO EP**
  (`onnxruntime-openvino`, `device_type="NPU"`). The provider list is plain
  `list[str]` (`config/execution.py` ~L30, `DEFAULT_ONNX_PROVIDERS`), so
  `"OpenVINOExecutionProvider"` would slot in — but it's **not** a clean
  extension point today (hardcoded strings, no registry). Adding it cleanly =
  a small provider-selection refactor (also unblocks any future EP).

- **Why not primary:** the NPU is a low-power sustained-inference part
  (Meteor Lake ~11 TOPS, Lunar Lake ~48 TOPS INT8), well below a discrete GPU.
  Limited op set + static-shape preference; GFPGAN (a `.pth` generative model)
  almost certainly won't fully offload (ops fall back to CPU; would need ONNX/IR
  export). Good utilization basically requires INT8 quantization → quality cost.

- **Where it could earn a place:** offload the **face detector** (small,
  quantization-tolerant) to the NPU to free the GPU for swap+enhance — fits the
  per-processor execution-profile design (detector on OpenVINO/NPU, swapper +
  enhancer on CUDA). File as an experiment, not a priority, especially when an
  NVIDIA GPU is present.

## 3. Other models / processors

**Already shipped since this section was written:** CodeFormer (second enhancer,
fidelity knob), the alternative swappers (ReSwapper / Ghost / SimSwap / UniFace,
with a model selector), occlusion-aware masking (BiSeNet / ParseNet), and the
Real-ESRGAN whole-frame upscaler. The remaining open items below are GPEN /
RestoreFormer++, the SwinIR/HAT-class upscalers, and the adjacent processors.

The processor protocol (`setup → process → release`) + per-processor execution
profiles make this a clean plugin surface: a new model ≈ a new processor class +
a model entry. Roughly in value order:

### Face enhancers (drop-in alternatives to GFPGAN — same slot)

- **CodeFormer** — best quality-per-effort add. Fidelity/quality knob (`w`)
  trading identity vs restoration; often beats GFPGAN on heavily-degraded faces.
  ONNX available. Making the enhancer a *choice* (GFPGAN / CodeFormer / GPEN) is
  probably the cleanest first feature since the slot already exists.
- **GPEN** (BFR-512/1024/2048), **RestoreFormer++** — higher-res / transformer
  restorers for more detail than GFPGAN-512.

### General (whole-frame) upscalers — a processor type sinner1 had, sinner2 lacks

- **Real-ESRGAN** (x2/x4 + anime variant) — standard general SR; pairs as the
  background upsampler behind the face enhancer. Gives a non-face enhancement
  path.
- **SwinIR / HAT / BSRGAN / SCUNet** (denoise) and the community ESRGAN-arch
  models (OpenModelDB: 4x-UltraSharp, etc.) — all share the ESRGAN interface;
  support one → support the catalog.

### Biggest quality leap — occlusion-aware masking (new processor)

- **Face-parsing / occluder mask** (BiSeNet face-parsing, or facefusion's
  face-occluder model) so the swap respects hair/hands/glasses/mics crossing the
  face. Runs alongside the swapper and modulates paste-back. Single highest
  visual-quality improvement; small model. (facefusion is a good reference.)

### Alternative face swappers — RISK CAVEAT

- Openly-licensed options: **SimSwap-512**, **GHOST / GHOST-2**, **hififace**,
  **uniface**, **blendswap**, and **Reswapper** (recent open inswapper-quality
  reimplementation). facefusion is the reference for ONNX swappers + weights.
- **Caution:** the higher-res "inswapper 256/512" weights in circulation are
  unofficial/leaked, dubious provenance + license. Given this is a deepfake tool
  with a responsible-use stance, stick to openly-licensed swappers; do not
  bundle leaked ones.

### Adjacent processor ideas (more scope)

- **Color/lighting harmonization** to match the swapped face to target lighting
  (beyond histogram correction).
- **Frame interpolation** (RIFE / FILM) as a post-stage for smoother/slow-mo.
- **Faster detectors** as options — YOLOv8-face, SCRFD variants, MediaPipe for a
  CPU/NPU-friendly path (ties to the NPU experiment above).

## Suggested sequencing

1. **Speed:** ~~GFPGAN fp16~~ (✅ done) → **TensorRT EP + engine cache** is now
   the biggest remaining realtime lever; pair with the cheap ORT session tuning.
2. **Quality:** ~~enhancer-as-a-choice (CodeFormer) + occlusion masking~~
   (✅ done) → GPEN / RestoreFormer++ for higher-detail restoration.
3. **Hardware:** treat the provider list as the seam to later experiment with
   OpenVINO/NPU (detector offload).
