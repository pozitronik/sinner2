# Models / processors — to-do

Ranked queue of model-related features. Status: ✅ done · 🚧 in progress · ⬜ queued.

## Done
- ✅ **Face swap** — inswapper_128.
- ✅ **Face enhance** — GFPGANv1.4.
- ✅ **Rotation compensation** — upright tilted faces before swap/enhance (3D-pose +
  re-detect proven best). Shared by swapper + enhancer.
- ✅ **General upscaler** — Real-ESRGAN (general-x4v3 / x4plus / x2plus), tiling +
  fp16, lazy confirmed download. Model-selector + lazy-download is now a reusable
  template for any ESRGAN-family model.

## Done (cont.)
- ✅ **Occlusion-aware face masking (v1)** — masks the swap to the facial-skin
  region (facexlib BiSeNet parse) so hair / glasses / hat / neck / boundary keep
  the original. Confirmed lazy download of `parsing_bisenet.pth`. Follow-up:
  arbitrary occluders (a hand over the cheek, parsed as skin) need a dedicated
  occluder/XSeg model composed with this mask.
- ✅ **CodeFormer as a second enhancer** — Model selector (GFPGAN / CodeFormer) in
  the FaceEnhancer group with a fidelity `w` knob. ONNX (facefusion-assets
  `codeformer.onnx`, scalar `weight` input verified), per-face align→restore→paste,
  shared thread-safe session. Confirmed lazy download; Upscale/Fidelity rows
  enable per model. Wired through realtime + batch + edit dialog.
- ✅ **Alternative face-swap models** — Selector in the FaceSwapper group:
  inswapper_128 (default) + ReSwapper-128 (drop-in via insightface INSwapper) +
  Ghost 1/2/3, SimSwap-256, UniFace-256 (facefusion-style, `GenericOnnxSwapper`).
  Per-model contract (template/size/mean/std/source-embedding/de-norm) ported
  VERBATIM from facefusion master; ghost/simswap pull a crossface embedding
  converter companion. All URLs verified; lazy confirmed download (declined →
  revert to inswapper). Avoids the leaked inswapper-256/512. Rotation + occlusion
  work unchanged (generic backend exposes insightface's `.get()` signature).
  NOTE: 256px models validated structurally (stub-session unit tests) — needs a
  real-footage quality pass; SimSwap is CC-BY-NC.

## Queued (high value)
- ⬜ **More upscaler models** — nearly free now (register URL + arch): 4x-UltraSharp,
  anime SR, plus denoise/deblur/JPEG-artifact (SCUNet, NAFNet — sibling arches).

## Queued (worth it)
- ⬜ **Color / lighting harmonization** — match swapped-face color to target lighting
  beyond histogram correction. Small, cheap, more realistic.
- ⬜ **Frame interpolation** — RIFE / FILM as a batch post-stage (smooth / slow-mo).
- ⬜ **Faster/better detectors** — YOLOv8-face / SCRFD variants; better keypoints also
  help the swap + rotation comp. (Ties to the NPU experiment in ideas.md.)

## Infra
- ⬜ **Models management view** — see installed/available models, download/remove, pick
  variants. Natural now that there's a registry + multiple optional models + confirmed
  downloads.

See `docs/ideas.md` for the broader perf/hardware backlog.
