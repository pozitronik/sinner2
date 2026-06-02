# sinner2

[![CI](https://github.com/pozitronik/sinner2/actions/workflows/ci.yml/badge.svg)](https://github.com/pozitronik/sinner2/actions/workflows/ci.yml)
[![License: LGPL v3](https://img.shields.io/badge/license-LGPLv3-blue.svg)](LICENSE)

A face-swapping tool with a realtime preview GUI and a batch processing queue.

sinner2 is a ground-up rewrite of [sinner](https://github.com/pozitronik/sinner),
which itself began as a rework of [s0md3v/roop](https://github.com/s0md3v/roop).
It takes a face from a single source image, places it onto an image or video
target, and can optionally restore facial detail with a face enhancer and
upscale the result. Several swap and enhancer models are selectable. The
interface is built around a video-player surface: you load a source and a
target, watch the result update as the target plays, and adjust the processing
parameters while it runs.

It is built for personal and technical use rather than as a finished product.
Read [Responsible use](#responsible-use) before you start.

## Features

- **Realtime preview.** The chain (face swap → optional enhance → optional
  upscale) runs frame by frame while the target plays, so parameter changes are
  visible at once.
- **Selectable models per stage.** The face swapper offers `inswapper_128`
  (default) plus alternatives (ReSwapper, Ghost, SimSwap, UniFace); the enhancer
  offers GFPGAN (with a half-precision option) or CodeFormer (with a fidelity
  control). Non-default weights download on demand, with confirmation.
- **Whole-frame upscaler.** An optional Real-ESRGAN super-resolution stage after
  the face work, with tile-size and half-precision controls.
- **Rotation compensation.** For faces tilted in-plane past a threshold, the
  swapper and enhancer upright a crop, re-detect, process, then composite it
  back — fixing the smearing detectors produce on rolled faces.
- **Occlusion-aware masking.** Restricts the swap to the facial-skin region (via
  a face-parsing model) so hair, glasses, hats, and the jaw boundary keep their
  original pixels.
- **Detection overlays.** A toggle draws detected faces (box, keypoints,
  sex/age/score/pose) on the preview; a linked overlay shows original-vs-swapped
  thumbnails for each face.
- **Processing scale.** Optionally downscale frames before the chain for speed;
  the output is the reduced resolution.
- **Batch queue.** Capture the current source, target, and settings as a job,
  queue several jobs, and run them one at a time. Jobs can be paused, resumed,
  and canceled; per-job progress and throughput are shown.
- **Resumable batch runs.** Rendered frames are cached to disk, so a paused or
  interrupted job continues from where it stopped instead of restarting.
- **Two execution strategies, one set of processors.** Realtime mode is
  frame-major (low latency). Batch mode is processor-major: each stage runs over
  all frames before the next, which keeps the model resident and the GPU busy.
- **Per-processor execution settings.** The swapper, enhancer, and upscaler each
  have their own ONNX Runtime providers / Torch device and worker count, so you
  can tune their parallelism or place them on different hardware.
- **Optional TensorRT acceleration.** Selecting the TensorRT provider compiles a
  GPU-specific engine for the face swap, cached to disk so the one-time build is
  paid once. The win is modest at the safe fp32 precision (~1.3× on inswapper;
  fp16 is faster but corrupts the swap, so it's off by default). Requires the
  TensorRT runtime (the installer offers it); falls back to CUDA when it's
  unavailable.
- **Responsive session switching.** Changing the source or target rebuilds the
  session on a background thread, so the UI stays responsive while the old models
  unload and the new ones load. Disabling a stage frees its model from memory.
- **Source and target libraries.** Thumbnail browsers with drag-and-drop, a
  configurable set of accepted file types, and per-panel zoom and sort.
- **Audio playback** for video targets during preview.
- **On-demand model download.** Required models are fetched on first launch;
  optional ones download the first time you enable the feature that needs them.

## Requirements

- Windows or Linux. (Apple Silicon is recognized by the installer but is not yet
  packaged with a launcher.)
- An NVIDIA GPU with a current driver is recommended. A CPU-only build works but
  is much slower.
- Python 3.12. The installer provisions it for you; a manual setup needs it
  already installed.
- `ffmpeg` on `PATH` for encoded video output in batch mode. Without it, video
  jobs fall back to writing an image sequence.

The two required model files are downloaded on first run (under 1 GB total).
Optional models download later, on demand, when you first enable the feature
that uses them.

## Installation

The installer detects your hardware, selects the matching build, sets up an
isolated environment, and writes a launcher. It uses
[uv](https://docs.astral.sh/uv/) and installs it for you if it is missing.

- Linux / WSL: `bash install.sh`
- Windows: run `install.bat` (double-click, or run it from a terminal)

It offers these builds and recommends one based on your GPU and driver:

| Build     | When to use it                              |
|-----------|---------------------------------------------|
| `cuda`    | NVIDIA GPU, driver 525 or newer (CUDA 12.8) |
| `cuda118` | Older NVIDIA GPU or driver (CUDA 11.8)      |
| `cpu`     | No GPU, or as a fallback                    |
| `mac-arm` | Apple Silicon                               |

If a GPU build is selected but the driver is missing or too old, the installer
explains what to install and lets you recheck, switch to the CPU build, or stop.
On a GPU build it also offers to install **TensorRT** (an optional ~2 GB
download) for a modestly faster swap; you can decline and add it later with
`uv pip install -e ".[tensorrt]"`. When it finishes, it writes `run.sh` (Linux)
or `run.bat` (Windows).

Running the installer again on an existing install opens a menu to repair,
switch build, run the checks, reinstall, or uninstall.

### Checking the install

`bash doctor.sh` (or `doctor.bat`) re-verifies the environment: Python version,
PyTorch and CUDA, the ONNX Runtime CUDA provider, and that the package imports.
Run it whenever something looks wrong.

### Manual installation

To set things up by hand:

1. Install Python 3.12 and [uv](https://docs.astral.sh/uv/).
2. Create the environment: `uv venv --python 3.12`.
3. Install PyTorch for your hardware from the matching index. For CUDA 12.8:
   `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`.
4. Install the application: `uv pip install -e ".[cuda,gui]"` (use `cpu` in place
   of `cuda` for a CPU build).
5. On a GPU build, reinstall the GPU ONNX Runtime so it takes precedence:
   `uv pip install onnxruntime-gpu --reinstall --no-deps`.

The installer automates these steps. It also patches a `basicsr` import that
breaks with current torchvision: if you see an `ImportError` for
`torchvision.transforms.functional_tensor`, change that import to
`torchvision.transforms.functional` in the installed `basicsr` package.

## Running

The installer writes a launcher on the machine where it runs; these are not
committed to the repository:

- Linux: `./run.sh`
- Windows: `run.bat`

With the package installed, `python -m sinner2.gui` starts the application from
any environment.

## Usage

1. Load a source face (an image) and a target (an image or video) from the
   Sources and Targets tabs, or with the pickers.
2. The preview shows the swapped result. Adjust the face swapper and enhancer in
   the Settings tab; changes apply to the running preview.
3. To process and save, add the current setup to the queue with the plus button
   on the transport bar, then open the Batch tab and start the queue.

Batch output is written next to the target as `{source}+{target}.mp4` by
default. The output folder and per-job format can be changed in the job
settings.

## Models

Two models are required and downloaded on first launch if missing:

- `inswapper_128.onnx` — the default face swap.
- `GFPGANv1.4.pth` — the default face enhancer.

Other models are optional and download the first time you enable the feature
that uses them (with a confirmation prompt):

- Alternative swappers: ReSwapper, Ghost, SimSwap, UniFace (and the crossface
  embedding converters some of them need).
- CodeFormer, a second face enhancer.
- Real-ESRGAN upscaler weights.
- Face-parsing models for occlusion masking (BiSeNet / ParseNet).

Disabling a stage unloads its model from GPU memory. To install any model by
hand, put the file in the models directory. Set `SINNER2_MODELS_DIR` to choose
that directory; otherwise a default location is used.

The optional models are third-party and carry their own licenses — for example,
SimSwap is released for non-commercial use only. Check the upstream terms before
relying on one.

## Configuration

Settings are stored in a JSON file and kept between runs. These environment
variables override the default locations:

| Variable                | Purpose                                 |
|-------------------------|-----------------------------------------|
| `SINNER2_SETTINGS_PATH` | Path to the settings file               |
| `SINNER2_MODELS_DIR`    | Directory holding the model files       |
| `SINNER2_CACHE_DIR`     | Directory for the processed-frame cache |

## Updating

`bash update.sh` (or `update.bat`) checks GitHub releases for a newer version
and, if there is one, updates the checkout and re-syncs dependencies. The
application version is taken from the Git tag.

## Development

```sh
git clone https://github.com/pozitronik/sinner2
cd sinner2
uv venv --python 3.12
uv pip install -e ".[cpu,gui,dev]"
```

- Run the tests: `pytest`
- Lint: `flake8 .`
- Type-check: `mypy`

Continuous integration runs the lint and test jobs on every push and pull
request. Pushing a `vX.Y.Z` tag builds the distribution and opens a draft
release; the version is derived from the tag. Slow end-to-end tests that need
real models and video are excluded by default and run with `pytest -m slow`.

## Responsible use

sinner2 produces synthetic media. Use it only on images of people who have
agreed to it, and only for lawful purposes. Do not use it to impersonate,
defraud, or harass anyone, to produce sexual content involving a person without
their consent, or to create material that is illegal where you live. You are
responsible for what you make with it.

## License

sinner2 is licensed under the GNU Lesser General Public License v3.0 (LGPLv3).
See [LICENSE](LICENSE) for the LGPLv3 terms and [COPYING](COPYING) for the GNU
GPLv3 that it incorporates.

## Acknowledgements

- [s0md3v/roop](https://github.com/s0md3v/roop), where this line of work started.
- [InsightFace](https://github.com/deepinsight/insightface) for the inswapper model.
- [GFPGAN](https://github.com/TencentARC/GFPGAN) and
  [CodeFormer](https://github.com/sczhou/CodeFormer) for face enhancement.
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for upscaling and
  [facexlib](https://github.com/xinntao/facexlib) for face parsing.
- [ReSwapper](https://github.com/somanchiu/ReSwapper) and the
  [FaceFusion assets](https://github.com/facefusion/facefusion-assets) that host
  several of the optional ONNX swap and restoration models.
- [ONNX Runtime](https://onnxruntime.ai/), [PySide6](https://doc.qt.io/qtforpython/),
  and [uv](https://docs.astral.sh/uv/).
