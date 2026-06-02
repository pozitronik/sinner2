# sinner2 code audit — findings & ordered todo list

Generated 2026-06-02 by a 14-subsystem multi-agent review with per-finding adversarial verification (53 raw findings → 45 confirmed, 8 refuted). Ranked by severity × blast-radius; all items below survived an independent skeptic re-reading the cited code.

**Progress (2026-06-02):** ranks **1–11 FIXED** — the P0, all nine P1s, and the systemic worker-error amplifier (#11) — each with a failing-test-first (TDD) regression test and a separate commit; full suite green (1208 passed). Remaining: ranks 12–45 (P2 + P3).

| # | Priority | Category | Title |
|---|----------|----------|-------|
| 1 | P0-critical | error-handling | App shutdown cancels the running batch task and wipes its entire resumable frame cache (data loss) |
| 2 | P1-high | api-misuse | PNG output crashes every session/task: JPEG-scale quality (1-100) passed to PNGImageWriter which requires 0-9 |
| 3 | P1-high | concurrency-race | FaceSwapper.process() does not snapshot backend handles, violating the documented release()/process() race contract (whole-executor stop) |
| 4 | P1-high | concurrency-race | set_chain publishes the new chain to workers before setup() completes (workers run process() on un-set-up processors) |
| 5 | P1-high | concurrency-race | set_roots()/set_paths() replace the library model without cancelling in-flight folder scans; stale folder contents repopulate the new grid |
| 6 | P1-high | correctness-bug | Enhancer/upscaler params missing from processed-frame cache key -> stale cached frames served |
| 7 | P1-high | resource-leak | CodeFormer shared ONNX session evicted/leaked: N per-worker FaceEnhancer wrappers share one cached session with no refcount |
| 8 | P1-high | concurrency-race | Double-click opens the edit dialog on the RUNNING batch task, racing the queue's store writer and corrupting resume state |
| 9 | P1-high | correctness-bug | EOF-shrunk frame total not restored on resume -> multi-stage AUTO batch tasks hard-fail with misleading 'frames missing' |
| 10 | P1-high | correctness-bug | cuda118 installer silently broken: unpinned onnxruntime-gpu reinstall overrides CUDA-11.8 pin with a CUDA-12 build (silent CPU fallback) |
| 11 | P2-medium | error-handling | Worker error sets the global stop_event, silently killing the whole executor on a single transient bad frame and leaking GPU state |
| 12 | P2-medium | error-handling | BoundedWriteExecutor swallows real write failures and counts them as completed -> silent frame loss, corrupted metrics |
| 13 | P2-medium | concurrency-race | Detection QThread can be destroyed while still running during close (wait(2000) times out on first model load) |
| 14 | P2-medium | correctness-bug | Switching audio backend mid-session silently kills audio (no media reload, no state restore) |
| 15 | P2-medium | correctness-bug | Failed async source/target swap leaves audio paused while video keeps playing (A/V desync) |
| 16 | P2-medium | resource-leak | Shutdown racing a completing swap leaks the new write executor + frame store |
| 17 | P2-medium | resource-leak | OcclusionMasker torch (BiSeNet) model VRAM not freed on FaceSwapper.release() |
| 18 | P2-medium | error-handling | Per-frame worker exceptions silently swallowed in batch stage; deterministic failures misreported as 'frames missing' |
| 19 | P2-medium | correctness-bug | settings.save() is non-atomic and load() silently swallows corruption -> crash/power-loss truncates settings.json, then defaults overwrite it (preference loss) |
| 20 | P2-medium | performance | FFmpeg reader forks a fresh ffmpeg process per trailing frame when nb_frames over-counts |
| 21 | P2-medium | error-handling | smoke.py setup-failure handling is dead code; a model-load failure hangs for the full 300s timeout |
| 22 | P2-medium | performance | Rotated display re-allocates a full-resolution rotated pixmap on every paint |
| 23 | P3-low | correctness-bug | Shared thumbnail generator's in-flight dedup permanently blanks duplicate tiles across the two libraries |
| 24 | P3-low | correctness-bug | Editing a batch task on a machine missing a requested ONNX provider silently drops that provider |
| 25 | P3-low | concurrency-race | Detection probe uses providers fixed at construction; a live providers change can rebuild the shared detector on the stale EP list |
| 26 | P3-low | correctness-bug | Non-atomic per-frame image writes leave truncated non-zero files that frame_ok accepts and the encoder ships |
| 27 | P3-low | concurrency-race | ReaderPool.read_async/shutdown TOCTOU can enqueue a future that is never resolved or cancelled (latent) |
| 28 | P3-low | correctness-bug | latest_index_at_or_below can return an invalidated index, stalling the playback fallback for one tick |
| 29 | P3-low | performance | Session-start enforce_size_cap can evict the cache dir the session is about to reuse (no protect passed) |
| 30 | P3-low | correctness-bug | enforce_size_cap over-deletes (can wipe the whole cache) when protect is non-empty: protected bytes counted in total but never subtracted (latent) |
| 31 | P3-low | correctness-bug | on_frame callback receives fallback-frame pixels paired with the timeline index, not the frame's own index |
| 32 | P3-low | error-handling | Model-download teardown blocks the GUI thread on an unbounded thread.wait() |
| 33 | P3-low | performance | thread.wait(5000) on the GUI thread can stall close for up to 5s/job when a scan worker is stuck in os.scandir |
| 34 | P3-low | error-handling | _update_settings swallows all exceptions, leaving in-memory settings stale and hiding persistence failures |
| 35 | P3-low | api-misuse | get_onnx_session caches by path only while get_insightface_swap_model caches by (path, providers): wrong-EP session for a future second consumer (latent) |
| 36 | P3-low | performance | Native-size probe opens the target media file synchronously on every keystroke in the target path field |
| 37 | P3-low | concurrency-race | _wait_for_tensorrt_build has no re-entrancy guard: a swap completing during a build stacks a second dialog + polling timer |
| 38 | P3-low | correctness-bug | Metrics write_fps/drop_fps rate trackers are not reset when the overlay is re-shown (first reading is a long-window smear) |
| 39 | P3-low | correctness-bug | Persisted last_completed_frame uses out-of-order done count; GUI can show progress past unwritten gaps |
| 40 | P3-low | performance | Per-progress-tick allocation of a throwaway _StepTracker via setdefault |
| 41 | P3-low | correctness-bug | ffmpeg -shortest can truncate rendered video to source-audio length, dropping trailing frames |
| 42 | P3-low | maintainability | TensorRT-specific provider tooltip is permanently clobbered by mark_providers_failed() on every launch |
| 43 | P3-low | error-handling | _driver_gate loops on unrecognized input with no guidance and crashes on EOF (no abort branch) |
| 44 | P3-low | maintainability | Incorrect rationale for cross-thread read of worker._task in _on_completed (false thread-exit invariant) |
| 45 | P3-low | resource-leak | ThumbnailGenerator.submit() leaves cancelled-before-start Paths in _inflight at shutdown (bookkeeping-only) |

---

## Details

### 1. [P0-critical] App shutdown cancels the running batch task and wipes its entire resumable frame cache (data loss)

- **Category:** error-handling &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/batch/queue.py:127-139 (stop); /mnt/c/life/sinner/sinner2/src/sinner2/batch/driver.py:255-260 (CANCELLED branch _wipe_cache), 206 (task_cache path)
- **Problem:** Closing the app while a batch task is rendering deletes every already-rendered frame for that task. closeEvent -> BatchQueue.stop() calls driver.cancel(), the stage loop turns that into StageStatus.CANCELLED, and _run_inner shutil.rmtree's the whole per-task cache dir and resets resume markers (last_completed_frame=-1, completed_stages=0). The task can no longer resume and must restart from frame 0 — genuine data loss with wrong cancel-vs-shutdown semantics (the stop() docstring says it only exists to avoid leaking the runner thread; the wipe is an unintended side effect).
- **Fix:** Make shutdown pause-not-cancel: in stop() call self._driver.pause() (the PAUSED path leaves on-disk frames intact and the task resumable) instead of cancel(). Reserve cancel()/_wipe_cache for explicit user cancel_task(). If a hard non-resumable stop is ever needed, add a distinct driver signal that stops submitting and returns a non-wiping terminal status.

### 2. [P1-high] PNG output crashes every session/task: JPEG-scale quality (1-100) passed to PNGImageWriter which requires 0-9

- **Category:** api-misuse &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/image_writer.py:104-115 (forward), 52-53 (PNG raise); /mnt/c/life/sinner/sinner2/src/sinner2/gui/player_controller.py:400-403; /mnt/c/life/sinner/sinner2/src/sinner2/batch/driver.py:193
- **Problem:** build_image_writer forwards the same integer to both writers despite incompatible scales: JPEG wants 1-100, PNG wants 0-9 and RAISES otherwise. The GUI quality spinboxes are hardcoded 1-100 (default 95) and only disabled for PNG (value 95 retained). Selecting PNG and starting yields PNGImageWriter(compression=95) -> ValueError, which kills session setup in realtime (refuses to start) and marks the batch task FAILED. PNG output is unusable in both paths. The factory docstring claiming clamping is the writers' job is false.
- **Fix:** In build_image_writer, do not forward JPEG-scale quality to PNG: `if image_format is ImageFormat.PNG: return PNGImageWriter()` (use default compression and ignore the JPEG-scale value), or clamp/map the value into 0-9. Alternatively make PNGImageWriter clamp rather than raise. Correct the misleading docstring.

### 3. [P1-high] FaceSwapper.process() does not snapshot backend handles, violating the documented release()/process() race contract (whole-executor stop)

- **Category:** concurrency-race &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/processors/face_swapper.py:229-258, 316-328 (release); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/realtime/executor.py:670-677 (_wait_for_inflight), 923-926 (fatal worker except); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/processors/processor.py:19-28 (contract); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/processors/face_enhancer.py:168-173 (correct pattern)
- **Problem:** FaceSwapper.process() re-reads self._swapper, self._source_face, self._analyser, self._masker off the instance AFTER its line-230 None-check instead of snapshotting them into locals as the Processor contract requires and FaceEnhancer.process does. Because _wait_for_inflight has a bounded 5s timeout, release() (from set_chain/reconfigure) can null those handles while a worker is mid-process() during a live chain swap. The worker then calls None.get(...) -> AttributeError, which the worker loop treats as fatal: it sets _stop_event and breaks, stopping the entire executor and hanging playback. Fires on a >5s in-flight swap concurrent with a chain swap (plausible under rotation-compensation re-detect or GPU contention).
- **Fix:** Snapshot the backend handles into locals at the top of process(): `analyser = self._analyser; swapper = self._swapper; source_face = self._source_face; masker = self._masker`, None-check those locals, and thread them through _swap_one / _resolved_target_sex / the occlusion branch instead of re-reading self.* after the check, mirroring FaceEnhancer.process. A live local ref keeps the backend alive for the call even if release() concurrently nulls the attributes.

### 4. [P1-high] set_chain publishes the new chain to workers before setup() completes (workers run process() on un-set-up processors)

- **Category:** concurrency-race &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/realtime/executor.py:655-660 (_handle_set_chain), 871-908 (worker read/apply, no lock), 687-709 (_handle_reconfigure correct ordering); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/processors/face_swapper.py:230-231 (raise before setup)
- **Problem:** _handle_set_chain assigns self._chain = chain (657) BEFORE running p.setup() on the new processors (658-660). Workers read self._chain (907) and call _apply_chain/process() (908) WITHOUT taking _state_lock. A worker that popped a WorkItem before the line-654 drain and is awaiting its source future can read the new, not-yet-set-up chain and call process() on a processor whose ORT session/model is still None. FaceSwapper.process() raises RuntimeError when called before setup(); that exception in the worker loop (923) sets the global _stop_event (925) and tears the whole executor down. Race window is the full duration of the new processors' setup() (seconds for inswapper/GFPGAN).
- **Fix:** Set up the new processors fully before exposing the chain: compute to_setup = [p for p in chain if p not in old_chain], run p.setup() for each BEFORE the swap, then assign self._chain = chain under _state_lock. Mirror the ordering _handle_reconfigure already uses (setup at 687-692, swap at 709).

### 5. [P1-high] set_roots()/set_paths() replace the library model without cancelling in-flight folder scans; stale folder contents repopulate the new grid

- **Category:** concurrency-race &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/widgets/library_view.py:405-411 (set_paths), 413-439 (set_roots), 466/562 (clear()->_cancel_active_scans bumps epoch), 519/548 (epoch staleness check)
- **Problem:** clear() correctly calls _cancel_active_scans() (bumps _scan_cancel_epoch + flags workers) before wiping the model, but set_roots() and set_paths() wipe the model (clear_paths) without doing so and may start a new scan with the same epoch as a still-running scan. The old scan's already-queued/late batches then pass the staleness check (start_epoch < epoch is False) and reinsert the previous folder's files into the freshly-cleared grid meant to show only the new folder — the exact 'wiped library refills itself' failure the epoch guard exists to prevent. Reachable on startup restore / folder switch when a prior scan is still in flight.
- **Fix:** Call self._cancel_active_scans() at the top of both set_roots() and set_paths() (before clear_paths()), mirroring clear(). That bumps the epoch so old workers' late batches are discarded and only the newly-started scan (carrying the bumped epoch) survives.

### 6. [P1-high] Enhancer/upscaler params missing from processed-frame cache key -> stale cached frames served

- **Category:** correctness-bug &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/player_controller.py:139-144 (_cache_key), 822-841 (_build_chain wraps in PerWorkerProcessor), 404-406 (cache_dir from key); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/processors/per_worker.py:39-46 (no _params)
- **Problem:** _cache_key pulls `_params` off each chain processor, but the enhancer and upscaler are wrapped in PerWorkerProcessor, which exposes no `_params` (only name/_factory) — their params are captured in the factory lambda's closure and never reach the hash. With the default WRITE_READ cache mode, two sessions differing only in enhancer/upscaler knobs (e.g. CodeFormer fidelity, enabled/device) hash to the same cache directory and the second serves the first's already-processed frames from disk. The user changes a setting, reloads, and sees the OLD output. Only FaceSwapper (bare processor with _params) is keyed correctly.
- **Fix:** Make per-worker-wrapped processors contribute their params to the cache key: have PerWorkerProcessor expose a stable cache_key/_params derived from its bound factory params and prefer that in _cache_key; OR compute the key in _build_session from the controller's actual param objects (enhancer_params/upscaler_params model_dump_json(), enabled flags, device strings) rather than reflecting over the built chain.

### 7. [P1-high] CodeFormer shared ONNX session evicted/leaked: N per-worker FaceEnhancer wrappers share one cached session with no refcount

- **Category:** resource-leak &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/processors/codeformer.py:76-79 (get), 102-108 (release); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/model_cache.py:301/543-554 (cache by path), 572 (release pop); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/processors/face_enhancer.py:90 (thread_safe=False); /mnt/c/life/sinner/sinner2/src/sinner2/gui/player_controller.py:99/803/786-789 (worker cap/resize); /mnt/c/life/sinner/sinner2/src/sinner2/batch/stage.py:250-255 (pool builds N)
- **Problem:** FaceEnhancer.thread_safe=False, so each realtime/batch worker builds its own CodeFormerBackend, and all call get_onnx_session(codeformer.onnx) which returns ONE process-wide cached InferenceSession by reference with no refcount. (1) Live realtime worker-count DECREASE: the exiting worker's release() pops the single cache entry while remaining workers still use that session; a later INCREASE misses the cache and builds a SECOND ~377 MB CodeFormer session, so VRAM grows on every shrink+grow cycle (reachable via worker slider or the CodeFormer worker-cap toggle, cap=2). (2) Full teardown: only the first instance pops + gc.collect()s; later instances still hold self._session so the ORT destructor never runs and VRAM is not promptly freed. Batch with workers>1 hits the same hazard.
- **Fix:** Add reference counting to the shared ONNX session cache: store (session, refcount); get_onnx_session increments on hit, release_onnx_session decrements and only pops + del + gc.collect() at 0. Alternatively, since the CodeFormer session is genuinely shared/thread-safe, evict it from a single owning layer (PerWorkerProcessor chain teardown / _ProcessorPool) that runs exactly once, not from each per-worker CodeFormerBackend.release().

### 8. [P1-high] Double-click opens the edit dialog on the RUNNING batch task, racing the queue's store writer and corrupting resume state

- **Category:** concurrency-race &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/widgets/batch_view.py:213-215, 485-488 (_emit_edit_for_row, unguarded); /mnt/c/life/sinner/sinner2/src/sinner2/gui/main_window.py:1462-1472 (_on_edit_batch_task, no guard); /mnt/c/life/sinner/sinner2/src/sinner2/batch/queue.py:245-264/270-290 (_on_progress/_on_completed store.save); /mnt/c/life/sinner/sinner2/src/sinner2/gui/widgets/batch_task_dialog.py:481-546 (to_task preserves snapshot)
- **Problem:** The context menu deliberately hides Edit for the running task (is_running guard) and blocks Delete, but the double-click handler emits editRequested for ANY row with no running check, and _on_edit_batch_task also has no guard. It opens a modal dialog whose nested event loop still pumps the queue's queued cross-thread progress/completed slots. to_task() preserves the dialog-open snapshot of all runtime/resume fields (status, last_completed_frame, total_frames, completed_stages, started_at/finished_at, error_message) because they are omitted from its model_copy update dict. The queue writes fresher markers to the same <id>.json via whole-file os.replace; the two writers clobber each other, corrupting resume state and silently dropping either the edits or live progress.
- **Fix:** Guard the double-click path like the context menu: in _emit_edit_for_row (or before emitting editRequested) bail when task_id == self._queue.current_task_id (optionally show a 'cannot edit a running task' notice), and/or add the same current_task_id guard at the top of _on_edit_batch_task.

### 9. [P1-high] EOF-shrunk frame total not restored on resume -> multi-stage AUTO batch tasks hard-fail with misleading 'frames missing'

- **Category:** correctness-bug &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/batch/driver.py:212-214 (re-read inflated total), 222-235 (trusted-skip), 272-284 (EOF shrink persisted but discarded on resume)
- **Problem:** On resume, _run_inner always re-reads total from reader.frame_count (inflated container nb_frames) and overwrites task.total_frames, discarding the real EOF-resolved length persisted on the prior run. For AUTO-cleanup multi-stage tasks, stage 0 is trusted/skipped on resume so total never re-shrinks; stage 1 then reads frames [real..inflated) that don't exist, the integrity pass reports them missing, and the stage FAILS — converting a resumable task into a hard failure with a misleading 'N frames missing' message. (KEEP-mode impact is milder: stage 0 self-corrects because the reader returns EOF immediately.)
- **Fix:** Persist and prefer the discovered real length on resume: if task.total_frames is a valid positive value from a prior run, seed `total` from it (or use it as an authoritative cap) instead of blindly taking reader.frame_count. Then _stage_complete, the trusted-skip path, and FramesDirInput all operate over the correct [0..R) range.

### 10. [P1-high] cuda118 installer silently broken: unpinned onnxruntime-gpu reinstall overrides CUDA-11.8 pin with a CUDA-12 build (silent CPU fallback)

- **Category:** correctness-bug &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/installer/steps.py:68-71 (ort_gpu_reinstall_command), 28-29 (is_gpu_variant); /mnt/c/life/sinner/sinner2/installer/wizard.py:89-92 (build_install_plan); pyproject.toml:21-26 (cuda118 pin); /mnt/c/life/sinner/sinner2/installer/doctor.py:80-93 (weak provider check)
- **Problem:** For the cuda118 variant the installer first installs the CUDA-11.8-pinned onnxruntime-gpu (>=1.18,<1.19) but then unconditionally runs an unconstrained `uv pip install --reinstall --no-deps onnxruntime-gpu`, which upgrades to the latest (1.20+) built against CUDA 12. On a CUDA-11.8 system CUDAExecutionProvider can't load its CUDA-12 libs and ORT silently falls back to CPU. The doctor only checks get_available_providers() (compiled-in providers, not load-tested) so it reports OK, leaving the breakage invisible while real inference runs on CPU.
- **Fix:** Make ort_gpu_reinstall_command variant-aware: append the matching pin (onnxruntime-gpu>=1.18,<1.19 for cuda118, >=1.20 for cuda), sourced from the same constraint pyproject uses, or pass an explicit constraint to the reinstall so --reinstall can't drop the bound. Defense-in-depth: strengthen the doctor to actually instantiate a tiny ORT session on CUDAExecutionProvider (or check the wheel's CUDA tag).

### 11. [P2-medium] Worker error sets the global stop_event, silently killing the whole executor on a single transient bad frame and leaking GPU state

- **Category:** error-handling &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/realtime/executor.py:923-926 (fatal except), 376-378 (stop()/state/release), 887-894 (non-fatal read-error handling), 935 (per-thread release only)
- **Problem:** A worker catches any exception from _apply_chain or buffer.put (923) and responds by setting the global _stop_event (925), tearing down the dispatcher, all workers, and playback. stop() is never invoked, so _state is left at PLAYING/PAUSED, the shared chain processors are never release()d (the exiting worker only releases its own thread-local instances), and thread handles are never cleared. A single transient per-frame error (disk-full OSError on buffer.put, a momentary swapper failure) silently kills the entire realtime pipeline and leaves the executor inconsistent with GPU resources still held — inconsistent with the non-fatal handling of reader errors 30 lines above. This same fatal-except path is the amplifier for the set_chain and FaceSwapper-snapshot races (ranks 3-4).
- **Fix:** Distinguish recoverable from fatal errors: treat a chain error like a read error — log via status.set and continue to the next item (the finally block already restores _inflight_count) — or, if it must stop, invoke the real stop()/teardown path so the executor reaches a consistent STOPPED state with processors release()d, rather than only flipping _stop_event.

### 12. [P2-medium] BoundedWriteExecutor swallows real write failures and counts them as completed -> silent frame loss, corrupted metrics

- **Category:** error-handling &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/buffer/bounded_write_executor.py:66-78; /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/image_writer.py:64-68/92-96 (writers raise OSError); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/buffer/buffer.py:197 (write_completed metric)
- **Problem:** wrapped() runs FrameStore.write inside try/finally with no except. A raised OSError from the image writer still triggers the finally, which bumps _completed and records latency, then propagates onto a discarded Future (never .result()/.exception()-checked). A persistent disk write failure (full/permission/bad path) is invisible: write_completed increments as success, write_dropped does not, and no error reaches the user. The canonical store silently loses the frame once LRU evicts the cached copy, while BufferMetrics reports healthy.
- **Fix:** Wrap fn(*args, **kwargs) in its own try/except: on success increment _completed; on exception increment a new _failed counter (and optionally invoke an error callback / set a status flag) instead of counting it as completed. Surface _failed in WriteExecutorMetrics and BufferMetrics alongside dropped/completed so disk failures are observable.

### 13. [P2-medium] Detection QThread can be destroyed while still running during close (wait(2000) times out on first model load)

- **Category:** concurrency-race &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/main_window.py:1146-1147 (quit/wait, return ignored), 170-178 (thread parented, queued connection); /mnt/c/life/sinner/sinner2/src/sinner2/gui/face_detection_probe.py:101-121 (synchronous analyze, lazy load); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/processors/face_analyser.py:13-59 (buffalo_l build)
- **Problem:** closeEvent does quit() then wait(2000) and discards the bool. The probe's analyze() slot runs detection synchronously and, on its first call, lazily builds the insightface buffalo_l pack (5 ONNX models + prepare(), can exceed 2s and may download weights). The probe path only runs when overlay is on and swapper is off, so the swapper has not pre-loaded the model. If the user closes while that first detection/load is in flight, quit() cannot stop the running slot, wait(2000) times out and returns False (ignored), and the window-parented QThread is destroyed while still running -> 'QThread: Destroyed while thread is still running' and a potential abort/crash on exit.
- **Fix:** Check the wait() return value; if it times out, retry/loop with a longer wait until the thread finishes (or expose a cooperative cancel flag the probe checks around the heavy detect/model-load) before tearing down. At minimum increase the timeout to cover first-model-load and log/handle the False return rather than silently continuing into super().closeEvent().

### 14. [P2-medium] Switching audio backend mid-session silently kills audio (no media reload, no state restore)

- **Category:** correctness-bug &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/player_controller.py:1040-1049 (set_audio_backend), 1025-1038 (audio_backend factory, no load), 484/626 (only load() sites), 643 (_restore_audio_state early-return)
- **Problem:** set_audio_backend() shuts down the old backend and reconstructs a fresh one that only re-applies cached volume; it never calls backend.load() and never restores seek/play state. With no media loaded, is_loaded() is False, so _on_play/_on_pause/_on_seek all no-op and _restore_audio_state early-returns. If the user switches the audio backend during a live session, audio stays silent for the rest of the session, only returning on the next session start or async swap. self._current_target_path is available, so the reload is possible but not done.
- **Fix:** After reconstructing the backend in set_audio_backend(), if self._current_target_path is not None call backend.load(self._current_target_path), then restore position+play from live executor state (seek to self._executor.current_frame and play/pause to mirror it).

### 15. [P2-medium] Failed async source/target swap leaves audio paused while video keeps playing (A/V desync)

- **Category:** correctness-bug &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/player_controller.py:508-519 (_begin_swap pauses audio), 586-601 (error branch only emits errorOccurred); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/realtime/executor.py:686-696 (reconfigure early-return keeps old state)
- **Problem:** _begin_swap unconditionally pauses the audio backend if loaded. On the failure path (reconfigure_from returns None, or _build_session raises) _on_session_swap_ready's error branch only emits errorOccurred and never restores audio; _restore_audio_state runs only on the success branch. The live executor keeps its old state on failure (_handle_reconfigure returns early without changing _state), so a PLAYING session keeps producing frames. Result: video keeps playing, audio stays paused — the user gets a silent but still-advancing video until they manually toggle play/pause or seek.
- **Fix:** In _on_session_swap_ready's error branch, restore audio to match the still-live old session: derive play/seek from the live self._executor (is_playing/current_frame) and apply to the backend (re-seek + play if playing, else pause), instead of leaving it paused.

### 16. [P2-medium] Shutdown racing a completing swap leaks the new write executor + frame store

- **Category:** resource-leak &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/player_controller.py:844-854 (shutdown), 561-566 (old teardown in worker), 603-619 (_adopt_swapped_bundle), 976-999 (_teardown_session); /mnt/c/life/sinner/sinner2/src/sinner2/gui/main_window.py:1137-1152 (closeEvent)
- **Problem:** On a successful swap the worker thread shuts down the old write_executor/store and emits the new ones in outcome.bundle, expecting the queued _adopt_swapped_bundle to adopt them into self._write_executor/_session_store. shutdown() joins the swap thread and runs _teardown_session before that queued slot fires, so it tears down the OLD (already-shut-down) refs and never sees the new bundle. After closeEvent the app exits without spinning the loop, so _on_session_swap_ready never runs and the new write_executor's threads + PersistentFrameStore's file handles leak (OS reclaims on exit). Narrow but real window.
- **Fix:** In shutdown(), after joining the swap thread, drain the pending swap result before teardown — have _run_swap_job stash the bundle on a controller field and, if the GUI slot hasn't consumed it, explicitly shut down bundle.write_executor and bundle.session_store; or call _on_session_swap_ready directly with the last outcome.

### 17. [P2-medium] OcclusionMasker torch (BiSeNet) model VRAM not freed on FaceSwapper.release()

- **Category:** resource-leak &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/processors/face_swapper.py:316-328 (release nulls masker only); occlusion.py:100-118 (setup loads BiSeNet onto CUDA, no release()); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/processors/face_enhancer.py:205-220 (precedent fix)
- **Problem:** When occlusion masking is enabled, FaceSwapper.setup builds an OcclusionMasker that loads a facexlib torch parser onto CUDA. OcclusionMasker has no release() and never calls torch.cuda.empty_cache(); FaceSwapper.release() only nulls self._masker, so torch's caching allocator keeps the parser's GPU blocks reserved after the ref is dropped. Each chain rebuild/reconfigure with occlusion on re-loads a fresh masker while the old one's VRAM stays reserved until process exit — the exact problem FaceEnhancer.release already documents and fixes for its torch model. Only fires once a user enables occlusion (defaults off).
- **Fix:** Add a release() to OcclusionMasker that drops self._model/self._device and, when the device was CUDA, calls torch.cuda.empty_cache() (track a _device_is_cuda flag in setup like FaceEnhancer). Have FaceSwapper.release() call self._masker.release() (guarded for None) before nulling the reference.

### 18. [P2-medium] Per-frame worker exceptions silently swallowed in batch stage; deterministic failures misreported as 'frames missing'

- **Category:** error-handling &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/batch/stage.py:342-348 (_drain_one drops fut.exception()), 351-360 (_process_write); /mnt/c/life/sinner/sinner2/src/sinner2/batch/driver.py:444-452 (_stage_failed_message)
- **Problem:** _drain_one drops future exceptions on the floor with no logging or capture (stage.py/driver.py have no logging at all). For a DETERMINISTIC failure that recurs every frame (read-only/full output dir making ImageWriter.write raise OSError, recurring CUDA OOM, corrupt model, processor bug), the same exception fires for all frames and is converted into a generic 'K frames missing or empty ... refusing to encode a truncated video' FAILED message. The actual exception type and traceback — the thing needed to diagnose — are lost, and a disk/permissions error is misreported as a corrupt source.
- **Fix:** In _drain_one, when fut.exception() is not None, log it via logging.getLogger(__name__).exception(...) with the frame index, and retain the first exception so run_stage can surface its message in StageResult/error_message. At minimum emit a warning per failed frame so the root cause is recoverable from logs.

### 19. [P2-medium] settings.save() is non-atomic and load() silently swallows corruption -> crash/power-loss truncates settings.json, then defaults overwrite it (preference loss)

- **Category:** correctness-bug &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/config/settings.py:131-134 (save write_text), 121-128 (load swallows errors -> all-None); /mnt/c/life/sinner/sinner2/src/sinner2/gui/main_window.py:1487-1493 (_update_settings model_copy+save)
- **Problem:** save() writes directly to settings.json with no atomic temp-file+rename and no backup, so a crash or power loss mid-write truncates the file. load() then silently swallows the parse/IO error and returns an all-None Settings(); the next _update_settings (model_copy + save) overwrites the corrupt file with defaults, permanently destroying every persisted preference (paths, recents, geometry, library). Affects only user preferences, not media/output. (No concurrent writer — save() is GUI-thread-only — so the vector is crash/power-loss truncation plus the silent load fallback.)
- **Fix:** Write to a sibling temp file in the same directory and os.replace() onto settings.json (atomic on the same filesystem on POSIX and Windows). On load(), if parsing fails, rename the bad file to settings.json.bak and surface a warning, or refuse to save over a file that failed to load, so a transient read error cannot wipe real data.

### 20. [P2-medium] FFmpeg reader forks a fresh ffmpeg process per trailing frame when nb_frames over-counts

- **Category:** performance &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/io/video_target_reader.py:65-73 (read, no _next_index advance on None), 117-122 (nb_frames/estimate), 127-151 (_start_decoder_at Popen); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/realtime/executor.py:824/847-848 (EOS guard / _last_submitted)
- **Problem:** When ffprobe nb_frames (or the duration*fps fallback) over-counts decodable frames, the executor submits phantom trailing indices that pass the frame_index>=frame_count guard. Each phantom read returns None at EOF without advancing _next_index, so the next phantom index mismatches _next_index and triggers _start_decoder_at -> release()+Popen() of a fresh ffmpeg (~100-200ms) that immediately re-hits EOF. Result: a bounded fork+init storm of K short-lived ffmpeg subprocesses (K=overcount) plus K status emits across the Qt bridge at end-of-stream. CV2 backend is unaffected.
- **Fix:** After a None read at an in-range index, treat it as real EOF: set a sticky exhausted flag (or clamp self._frame_count down to the highest successfully-read index) so subsequent reads short-circuit to None without restarting the decoder. Alternatively, in read(), do not let a sequential index one past a known EOF silently spawn a new decoder.

### 21. [P2-medium] smoke.py setup-failure handling is dead code; a model-load failure hangs for the full 300s timeout

- **Category:** error-handling &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/cli/smoke.py:87-104; /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/realtime/executor.py:229-260 (start async), 274-298 (_setup_chain_async swallows), 262-272 (wait_until_ready)
- **Problem:** smoke.py wraps executor.start() expecting setup/model-load errors to raise (returns 3), but start() runs chain.setup() asynchronously; a failure there sets status to 'chain setup failed: ...' and _stop_event without propagating, so the except/return-3 path is dead for the common case. The wait loop only breaks on status == 'end of target', which never occurs after a setup failure, so the run spins the full 300s before printing a misleading 'timeout'. Developer/CI smoke tool — impact is a slow, misleading failure rather than a crash.
- **Fix:** Poll the status observable and/or call executor.wait_until_ready(timeout=...) before/inside the wait loop: after start(), wait_until_ready then `st = executor.status.get(); if st.startswith('chain setup failed') or executor._stop_event.is_set(): return 3`. Also break the playback wait loop on a setup-failed/stopped status. Drop or rework the now-misleading try/except around start().

### 22. [P2-medium] Rotated display re-allocates a full-resolution rotated pixmap on every paint

- **Category:** performance &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/widgets/frame_display.py:147-153 (paintEvent rotated branch), 70-75 (current_pixmap), 92-93 (_on_frame_ready->update)
- **Problem:** In paintEvent the _rotation != 0 branch calls self._pixmap.transformed(QTransform().rotate(...), SmoothTransformation) on every paint, allocating a full-resolution smoothly-rotated QPixmap, then scaling it. paintEvent runs once per displayed frame, so at 30-60 fps with a quarter-turn active each frame incurs a full-res rotation allocation plus the scaled copy, recomputed identically every frame even though only the source pixmap changed. The non-rotated path (scale only) is the minimum; the rotated path doubles per-frame allocation work with no caching.
- **Fix:** Cache the rotated pixmap and invalidate it only when the source frame or rotation changes. Simplest: in _on_frame_ready (and set_rotation), compute the rotated pixmap once and store it as the render source, so paintEvent only scales. Alternatively keep a (_pixmap.cacheKey(), _rotation) -> rotated-pixmap cache reused in paintEvent.

### 23. [P3-low] Shared thumbnail generator's in-flight dedup permanently blanks duplicate tiles across the two libraries

- **Category:** correctness-bug &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/library/thumbnail_generator.py:111-123 (submit dedup returns None); /mnt/c/life/sinner/sinner2/src/sinner2/library/library_model.py:229 (add_path ignores None); side_panel.py:51/60-66 (one generator, two views)
- **Problem:** A single ThumbnailGenerator is shared by both QLibraryView instances. submit() dedups by Path via a process-wide _inflight set and returns None for the second caller, dropping its callback; add_path ignores the None and never requeues. When the same file is present in both libraries (used as both source and target) and submitted while the first job is in flight (e.g. startup restore runs both libraries in close succession), the second model's _on_thumb_outcome is never called and its tile keeps the grey placeholder forever even though the cached JPEG exists. Cosmetic model/view inconsistency, not wrong pipeline output.
- **Fix:** Coalesce callbacks instead of dropping them: make _inflight a dict[Path, list[OnReady]]; on an already-in-flight path, append the new callback and return None without re-submitting, and in _run's completion invoke every registered callback with the same outcome. Alternatively key in-flight tracking by (model-id, path).

### 24. [P3-low] Editing a batch task on a machine missing a requested ONNX provider silently drops that provider

- **Category:** correctness-bug &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/widgets/batch_task_dialog.py:215-225 (checkboxes only for available), 550-556 (_selected_providers), 290-296/331-335 (device-token preservation precedent); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/model_cache.py:324-333 (available_onnx_providers)
- **Problem:** Provider checkboxes are created only for available_onnx_providers() (ort.get_available_providers() for the local install). _selected_providers() returns only checked boxes, so any requested provider not available on this machine has no checkbox and is silently dropped from swapper_execution.providers on accept (e.g. a CUDA+CPU task edited on a CPU-only box loses CUDA permanently). The dialog already preserves unavailable torch-device tokens for the enhancer and upscaler for this exact reason; the providers list was missed. Distinct from the intentional empty-selection->CPU floor.
- **Fix:** Mirror the device-preservation logic: after building checkboxes for available, add a checked checkbox for any provider in wanted (task.swapper_execution.providers) not present in available, so an unavailable-but-requested EP round-trips. Alternatively, in to_task/_selected_providers, union the checkbox selection with originally-requested providers that had no checkbox. Keep the intentional empty-selection->CPU floor.

### 25. [P3-low] Detection probe uses providers fixed at construction; a live providers change can rebuild the shared detector on the stale EP list

- **Category:** concurrency-race &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/face_detection_probe.py:114-121; /mnt/c/life/sinner/sinner2/src/sinner2/gui/player_controller.py:762-765 (reset_shared); /mnt/c/life/sinner/sinner2/src/sinner2/gui/main_window.py:170-172 (probe built once); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/processors/face_analyser.py:13-66
- **Problem:** FaceDetectionProbe caches a FaceAnalyser bound to its construction-time providers and is never told about a later swapper-providers change. That change resets the process-wide insightface singleton; the next caller rebuilds it with its own providers. If the probe (overlay re-detecting) wins the race it rebuilds buffalo_l on the stale startup EP list, and that detector is then reused process-wide (including by the swapper) on an EP list that doesn't match the user's new selection. Narrow (only with overlay active during a providers change) and the detector EP rarely changes detection output.
- **Fix:** Add FaceDetectionProbe.set_providers(...) that resets self._analyser, and call it from main_window when apply_session_config changes providers. Separately, have the providers-change path explicitly rebuild the shared app with the new providers rather than relying on whichever thread races first (and re-document/rename reset_shared_face_analysis to reflect its production use).

### 26. [P3-low] Non-atomic per-frame image writes leave truncated non-zero files that frame_ok accepts and the encoder ships

- **Category:** correctness-bug &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/io/cv2_unicode.py:65-66 (Path.write_bytes); /mnt/c/life/sinner/sinner2/src/sinner2/batch/stage.py:50-54 (frame_ok st_size>0), 158-168 (missing/contiguous); /mnt/c/life/sinner/sinner2/src/sinner2/batch/driver.py:410-414 (_stage_complete); task_store.py:57-76 (atomic precedent)
- **Problem:** imwrite_unicode uses Path.write_bytes (non-atomic). The batch resume/skip logic treats only zero-byte files as incomplete (frame_ok: st_size>0); missing()/contiguous() and _stage_complete all rely on frame_ok. A process kill or full disk partway through a frame's write leaves a truncated non-zero file that resume accepts as done and the encoder ships into the final mp4 — a silent corrupt frame rather than a reprocess. Narrow (one frame's write during a kill).
- **Fix:** Make image writes atomic: cv2.imencode to bytes, write to a temp file in the same directory, then os.replace() onto the final path — mirroring BatchTaskStore.save. A frame file is then either absent or complete, so frame_ok/_stage_complete never accept a half-written frame.

### 27. [P3-low] ReaderPool.read_async/shutdown TOCTOU can enqueue a future that is never resolved or cancelled (latent)

- **Category:** concurrency-race &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/io/reader_pool.py:165-175 (read_async stop-check+put), 211-245 (shutdown drain/cancel)
- **Problem:** read_async's stop-check and queue.put are not atomic with shutdown's drain. A future enqueued after shutdown has drained and reader threads have exited is never resolved or cancelled, breaking the 'callers always get a defined future' contract; a consumer awaiting future.result() would block. All current callers run on the dispatcher thread, which stop()/reconfigure quiesce before shutting the pool down, so the live window does not exist today — latent contract violation, not a firing bug.
- **Fix:** Make the enqueue atomic with the stop check: hold self._lock across both the is_set() check and the queue.put in read_async, and have shutdown() take the same lock while setting _stop_event, then re-drain/cancel under the lock. Alternatively, after the put, re-check _stop_event and cancel the just-enqueued future if shutdown has begun.

### 28. [P3-low] latest_index_at_or_below can return an invalidated index, stalling the playback fallback for one tick

- **Category:** correctness-bug &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/buffer/buffer.py:142-156 (latest_index_at_or_below), 88-102 (invalidate), 158-163 (invalidate_from); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/realtime/executor.py:1053-1057 (fallback)
- **Problem:** invalidate()/invalidate_from() leave stale indices in _recent_indices, so latest_index_at_or_below() can hand the playback fallback an index that get() returns None for (tombstoned/cleared). When the max candidate <= target is invalidated, the fallback yields no frame even though an older valid frame may exist below it, so the display stalls on the prior frame for one tick after a seek/invalidate. Self-correcting on the next tick once the reprocessed put() lands.
- **Fix:** Make latest_index_at_or_below() skip tombstoned indices (filter out i in self._invalidated) so it returns the highest GETTABLE candidate, and/or discard affected indices from _recent_indices in invalidate()/invalidate_from().

### 29. [P3-low] Session-start enforce_size_cap can evict the cache dir the session is about to reuse (no protect passed)

- **Category:** performance &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/player_controller.py:423-425 (enforce_size_cap no protect), 404-406 (cache_dir), 427-430 (meta written after); /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/cache_manager.py:327-340 (LRU age key)
- **Problem:** At session start, enforce_size_cap is called with no protect immediately before PersistentFrameStore(cache_dir) is created, and the active dir's last_used_at metadata is only written afterward. If the cache is over cap and the deterministic cache_dir for this (source,target,chain,writer,scale) is the LRU entry, eviction can delete the directory this session is about to reuse, forcing a full re-render of already-cached frames (cache thrash, not data loss). Requires both over-cap and the active dir being LRU.
- **Fix:** Spare the directory before eviction: touch_last_used(cache_dir) or write its meta first, then call manager.enforce_size_cap(self._cache_size_cap_bytes, protect=[cache_dir]). (Correct behavior of protect depends on fixing the protect-accounting bug, rank 30.)

### 30. [P3-low] enforce_size_cap over-deletes (can wipe the whole cache) when protect is non-empty: protected bytes counted in total but never subtracted (latent)

- **Category:** correctness-bug &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/cache_manager.py:244-262
- **Problem:** In enforce_size_cap, total includes protected entries' bytes (line 245) but the eviction loop only iterates and subtracts non-protected (candidate) sizes. With a non-empty protect set the loop over-evicts by roughly the protected byte count, and if protected alone exceeds max_bytes it deletes every evictable entry without ever satisfying the cap. Latent: the only production caller passes no protect, and the existing protect test protects all entries, so it does not currently fire — but it would fire the moment the rank-29 fix passes a non-empty protect.
- **Fix:** Base the budget only on deletable entries: candidates = [e for e in entries if e.path.resolve() not in protected]; total = sum(e.size_bytes for e in candidates), and do the early-return check against that total. Then the per-delete total -= size accounting stays consistent.

### 31. [P3-low] on_frame callback receives fallback-frame pixels paired with the timeline index, not the frame's own index

- **Category:** correctness-bug &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/realtime/executor.py:1053-1064 (_do_playback_tick fallback); frame_display.py:82 (current consumer ignores index)
- **Problem:** In the fallback branch of _do_playback_tick, frame is set to the latest-at-or-below frame and shown_index to fallback_index, but the on_frame callback passes the timeline target index (self._on_frame(frame, index)) rather than shown_index. The callback receives the fallback frame's pixels labelled with a different frame index. The sole current consumer ignores the index, so this is cosmetic today but a latent contract bug for any future consumer that trusts the index (tagging a saved frame, driving overlay/detection at the correct frame).
- **Fix:** Pass the index that corresponds to the displayed frame: self._on_frame(frame, shown_index). shown_index is guaranteed non-None whenever frame is non-None. Keep current_frame.set(index) using the timeline index for the transport slider.

### 32. [P3-low] Model-download teardown blocks the GUI thread on an unbounded thread.wait()

- **Category:** error-handling &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/model_download.py:144-147; /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/model_cache.py:496/502-513 (urlopen timeout, chunked cancel)
- **Problem:** After the modal dialog closes, ensure_models calls thread.quit() then an unbounded thread.wait() on the GUI thread. The worker's run() is a blocking download loop (not an event loop), so quit() has no effect until run() returns; on cancel the worker is checked only between 256KB chunks and a stalled socket blocks each read up to the 30s urlopen timeout, so wait() can freeze the dialog-less UI for ~30s with no feedback. Startup-only, recoverable.
- **Fix:** Give thread.wait() a bounded timeout and handle the timeout (e.g. wait(35000), then terminate/detach + log if still running), and/or make the download abortable promptly by closing the response/socket on cancel so the blocking read() unwinds. At minimum keep a 'finishing…' indicator visible during the bounded wait.

### 33. [P3-low] thread.wait(5000) on the GUI thread can stall close for up to 5s/job when a scan worker is stuck in os.scandir

- **Category:** performance &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/widgets/library_view.py:610 (shutdown wait 5000), 584 (finished-handler wait 1000, effectively harmless)
- **Problem:** In shutdown() (GUI thread), thread.wait(5000) per job can block the UI for up to 5s each when a worker's run() is stuck inside a slow os.scandir/rglob (e.g. a network share), since quit() doesn't interrupt a running run() and the cooperative cancel flag is only checked between iterations. The finished-handler wait(1000) is effectively harmless because finished is emitted as the last line of run(), so run() has already returned when the queued slot executes.
- **Fix:** For shutdown, use a short bounded wait and let daemon-promoted threads die with the process, or move the join off the GUI thread, so a single stuck network scandir can't freeze close for seconds per job. The finished-handler wait can be dropped in favour of QThread.finished -> deleteLater.

### 34. [P3-low] _update_settings swallows all exceptions, leaving in-memory settings stale and hiding persistence failures

- **Category:** error-handling &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/main_window.py:1487-1493; /mnt/c/life/sinner/sinner2/src/sinner2/config/settings.py:131-134 (save)
- **Problem:** _update_settings routes every settings write through model_copy(update=fields) -> save(updated) -> self._settings = updated inside a bare try/except Exception: pass. save() does Path.write_text, which can raise OSError (disk full / permission / read-only); that failure is silently swallowed with no log and no user feedback. Because the assignment runs only after a successful save(), a save failure leaves self._settings un-updated while the UI shows the new value — an inconsistency between the UI and the in-memory/persisted settings. (Related to the non-atomic save in rank 19.)
- **Fix:** At minimum log via logging.exception instead of pass so persistence failures are observable; ideally narrow the except to OSError/ValueError, surface a status-bar notice on persist failure, and update self._settings before save so the in-memory copy reflects the intended state even if the disk write fails.

### 35. [P3-low] get_onnx_session caches by path only while get_insightface_swap_model caches by (path, providers): wrong-EP session for a future second consumer (latent)

- **Category:** api-misuse &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/pipeline/model_cache.py:301/525-554 (_session_cache by path), 581-614 (_insightface_cache by path+providers)
- **Problem:** _session_cache is keyed by Path alone while _insightface_cache is keyed by (path, providers). On a get_onnx_session cache hit the providers argument is ignored, so a future second consumer of an already-cached ONNX file with a different provider profile would silently receive a session built on the first caller's execution provider, with no error. Does not fire today (each ONNX file has a single provider profile; provider changes flush the whole cache via clear_session_cache), so a latent divergence rather than an active bug.
- **Fix:** Key _session_cache by (path, providers) the same way _insightface_cache is keyed (normalize providers to a tuple), so a differing-providers request builds/returns the correctly-configured session. release_onnx_session and clear_session_cache then need to match on the path component (evict all provider variants for a name).

### 36. [P3-low] Native-size probe opens the target media file synchronously on every keystroke in the target path field

- **Category:** performance &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/widgets/batch_task_dialog.py:461 (textChanged->_refresh_scale_dims), 573-598 (_probe_native_size); cv2_video_target_reader.py:36-58 (VideoCapture probe)
- **Problem:** target_edit.textChanged (fires per character) is connected to _refresh_scale_dims -> _probe_native_size, which for videos builds CV2VideoTargetReader(target), opening cv2.VideoCapture and probing fps/frame_count/dims synchronously on the GUI thread. Partial/invalid paths fail fast, but a complete valid path to a large video (common on paste) blocks the GUI thread opening+probing the file on that keystroke, stalling the dialog.
- **Fix:** Debounce the probe with a short single-shot QTimer restarted on each textChanged (probe ~300ms after typing settles), and/or probe on editingFinished instead of textChanged. Optionally move the probe off the GUI thread.

### 37. [P3-low] _wait_for_tensorrt_build has no re-entrancy guard: a swap completing during a build stacks a second dialog + polling timer

- **Category:** concurrency-race &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/main_window.py:668-728; player_controller.py:586-617 (_on_session_swap_ready->_adopt_swapped_bundle emits sessionScratchDirChanged)
- **Problem:** _wait_for_tensorrt_build shows a non-blocking QProgressDialog plus a 400ms polling QTimer and returns immediately, so the GUI loop keeps running during the ~25-30s dispatcher-thread TRT build. It is reachable from both _on_processor_config_changed and _on_session_scratch_dir. While a config-change build is in progress, an in-flight async swap completing on the GUI thread emits sessionScratchDirChanged, re-entering the function; during the build TRT isn't yet in get_actual_providers() and no .engine file exists, so the guards don't return and a second dialog + timer stack over the first. Both timers self-stop on the same poll condition, so no leak — just two briefly-stacked modals.
- **Fix:** Track an instance-level _trt_wait_active flag (or hold a reference to the active dialog/timer) and return early when a wait is already in progress; clear it when the poll stops the timer and closes the dialog.

### 38. [P3-low] Metrics write_fps/drop_fps rate trackers are not reset when the overlay is re-shown (first reading is a long-window smear)

- **Category:** correctness-bug &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/widgets/metrics_overlay.py:110-117 (_reset_rate_state clears dead fields); main_window.py:949-960 (_set_stats_visible / _restore_metrics_overlay_state), 986-987 (real trackers)
- **Problem:** setVisible(True) -> _reset_rate_state() only clears the dead _prev_* fields (never read in this module). The real rate trackers self._write_rate/self._drop_rate (main_window) are NOT reset on re-show, and the overlay timer stops while hidden, freezing their (prev_count, prev_ts). When re-enabled the first update() computes a delta over the full hidden interval, so the first displayed write_fps/drop_fps is a long-window average (can reflect activity that happened while hidden) instead of a fresh 0.0 baseline. Self-corrects after one tick.
- **Fix:** Reset the real trackers when the overlay becomes visible: have _set_stats_visible / _restore_metrics_overlay_state call self._write_rate.reset() and self._drop_rate.reset() before setVisible(True). Remove the dead _prev_* fields from the overlay, or wire them to the real computation.

### 39. [P3-low] Persisted last_completed_frame uses out-of-order done count; GUI can show progress past unwritten gaps

- **Category:** correctness-bug &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/batch/queue.py:259 (_on_progress); stage.py:171/175-179/341-348 (done count, FIRST_COMPLETED); batch_view.py:303 (display consumer)
- **Problem:** _on_progress persists last_completed_frame = progress.stage_completed - 1, but stage_completed is the out-of-order done counter (a count of finished frames, not a contiguous prefix). With multiple workers it can momentarily exceed the true contiguous-from-0 progress. Impact is limited to the GUI's persisted-progress display; resume is disk-truth (re-scans disk) and the driver's terminal write uses the contiguous count, so it is an inaccurate-on-restart display value, not data loss.
- **Fix:** If the persisted marker is meant to reflect resumable progress, derive it from a contiguous count (expose run_stage's contiguous progress for the throttled save) rather than the out-of-order done counter, or document last_completed_frame as an approximate display-only counter.

### 40. [P3-low] Per-progress-tick allocation of a throwaway _StepTracker via setdefault

- **Category:** performance &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/widgets/batch_view.py:347 (setdefault), 66-69 (_StepTracker __init__ deque), 336 (_on_task_started inserts)
- **Problem:** self._throughput.setdefault(task_id, _StepTracker()) eagerly constructs a _StepTracker (allocating a deque) on every progress tick, discarding it whenever the key already exists. Since _on_task_started already inserts the tracker, the key is present on essentially every _on_task_progress call, so one tracker+deque is allocated and thrown away per progress frame on the GUI thread during a render.
- **Fix:** Avoid eager construction: tracker = self._throughput.get(task_id) then if tracker is None: tracker = self._throughput[task_id] = _StepTracker(). Functionally identical without allocating on the hot path.

### 41. [P3-low] ffmpeg -shortest can truncate rendered video to source-audio length, dropping trailing frames

- **Category:** correctness-bug &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/io/video_encoder.py:108-123 (specifically 122 -shortest, 102 -framerate); driver.py:213/487 (fps, audio source = original target)
- **Problem:** With audio muxing the encoder uses -shortest, which ends output when the first input stream ends. The video is built from a CFR image sequence at the probed fps while audio is copied from the original source. For VFR sources or fps-rounding cases where the reconstructed video runs longer than the source audio, -shortest truncates the video to the audio length, silently dropping trailing processed frames from the final mp4. Narrow (CFR sources with matching audio length won't trigger it).
- **Fix:** Drop -shortest so the video stream defines duration (excess audio is harmless / can be cut without losing frames). If you must bound runaway audio, derive the limit from the known frame count (-frames:v / -t = total/fps) rather than -shortest.

### 42. [P3-low] TensorRT-specific provider tooltip is permanently clobbered by mark_providers_failed() on every launch

- **Category:** maintainability &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/gui/widgets/processor_controls.py:1148-1156 (else-branch overwrite), 280-288 (TRT tooltip set), 291-294 (contradictory copy); main_window.py:365 (_highlight_failed_providers); config/execution.py:30 (DEFAULT_ONNX_PROVIDERS)
- **Problem:** The TensorRT checkbox gets a detailed TRT-specific tooltip at construction, but mark_providers_failed()'s else branch overwrites it with generic text for any provider not in the failed set. _highlight_failed_providers() runs at startup with a non-empty requested set (CUDA+CPU pre-checked from DEFAULT_ONNX_PROVIDERS), so the TRT tooltip is clobbered on every normal launch before the user can read it. The two generic tooltip variants also contradict each other on what unchecking everything does. Cosmetic/UX.
- **Fix:** Build a {provider_name: tooltip} map once at construction and read it from both the constructor and the else branch, so the TensorRT row keeps its specific text and the copy can't drift. Minimally, special-case 'TensorrtExecutionProvider' in the else branch to restore its construction-time tooltip.

### 43. [P3-low] _driver_gate loops on unrecognized input with no guidance and crashes on EOF (no abort branch)

- **Category:** error-handling &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/installer/wizard.py:205-217 (_driver_gate), 190-196 (_select_variant has guidance)
- **Problem:** _driver_gate's prompt loop returns only when the answer starts with r/c/a and has no else branch (unlike the sibling _select_variant, which prints guidance) and no EOFError guard. An interactive typo silently re-prompts with no hint; on non-interactive/exhausted stdin, input() raises EOFError and crashes the wizard mid-flow rather than aborting cleanly. Minor robustness/UX gap.
- **Fix:** Add an else branch that re-prints the valid choices (matching _select_variant), and wrap input() in try/except EOFError that treats EOF as Abort (return None). Apply the same EOF guard to _select_variant for consistency.

### 44. [P3-low] Incorrect rationale for cross-thread read of worker._task in _on_completed (false thread-exit invariant)

- **Category:** maintainability &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/batch/queue.py:270-291 (comment + read), 220-241 (QThread setup), 301 (teardown quit)
- **Problem:** The comment justifying the GUI-thread read of self._worker._task asserts 'the worker thread has exited (completed signal is its last emission before run() returns).' That is false: thread.start() runs QThread's default event loop, so when run() returns the thread idles until _teardown_runner() calls quit(). The read is actually safe because completed.emit is run()'s final action so _task is no longer mutated, but the stated thread-exit invariant is wrong and could mislead a future edit (e.g. adding post-run worker work) into a real cross-thread data race on _task.
- **Fix:** Correct the comment to state the real invariant (the worker has finished run() and emits completed as its final action, so _task is no longer mutated when this slot runs). For a hard guarantee, snapshot the needed state into the completed signal payload, or connect thread.finished, rather than reaching into worker._task.

### 45. [P3-low] ThumbnailGenerator.submit() leaves cancelled-before-start Paths in _inflight at shutdown (bookkeeping-only)

- **Category:** resource-leak &nbsp;·&nbsp; **Confidence:** confirmed
- **Files:** /mnt/c/life/sinner/sinner2/src/sinner2/library/thumbnail_generator.py:111-123 (submit), 147-148 (_run finally discard)
- **Problem:** Cancelled-before-start futures (from shutdown(cancel_futures=True)) never run _run, so their _inflight entries are never discarded. Because the generator is discarded immediately at shutdown (closeEvent -> side_panel.shutdown), this is bookkeeping-only with no live resource impact, no unbounded growth, and no second consumer. The original title's stated trigger (post-shutdown submit leaking the Path) is actually handled — that path rolls back the entry on RuntimeError. Effectively a non-issue.
- **Fix:** If desired for tidiness, set a self._shutdown flag in shutdown() and early-return None from submit() before touching _inflight, and clear _inflight wholesale in shutdown() (or attach a done-callback that discards regardless of whether _run executed).

---

## Systemic notes (worth a single sweep)

- Realtime worker exception handling is fragile and over-broad: the worker loop's `except Exception -> _stop_event.set(); break` (executor.py:923-926) converts ANY per-frame error into a full-executor teardown without calling stop(), leaving _state lying and GPU resources held. This single path is the amplifier for at least three findings (ranks 3, 4, 11) — fixing it to distinguish recoverable-vs-fatal and to route fatal errors through the real stop() should be done first, as it limits the blast radius of the two chain-swap races.
- Non-atomic file writes are a recurring pattern with a known in-repo fix (BatchTaskStore.save uses NamedTemporaryFile + os.replace). Three writers do NOT use it and should be swept together: settings.save() (rank 19), imwrite_unicode/per-frame writes (rank 26), and the general principle. Apply temp-file-in-same-dir + os.replace consistently.
- Broad `except Exception: pass`/silent-swallow with no logging recurs across the codebase and hides real failures: BoundedWriteExecutor.wrapped (rank 12), batch stage _drain_one (rank 18), settings.load() and _update_settings (ranks 19, 34), smoke.py dead except (rank 21). The codebase has essentially no logging in batch/stage.py or driver.py. A single sweep to add logging.getLogger(__name__).exception(...) at these catch sites and to surface failures (metrics counters / status bar) would make most of these diagnosable.
- Per-worker processor lifecycle (thread_safe=False + PerWorkerProcessor/_ProcessorPool) has two systemic gaps: (a) shared cached resources are released per-worker without refcounting, causing the CodeFormer evict/leak (rank 7); (b) per-worker wrappers don't expose _params, breaking the cache key for enhancer/upscaler (rank 6). Both stem from PerWorkerProcessor being a thin opaque wrapper — giving it a proper cache_key/params surface and refcount-aware shared-session release would fix both.
- Async-swap / shutdown / reconfigure ordering on the GUI side has several state-restoration and teardown-race bugs that share a root cause: success-only restoration logic with no symmetric handling on failure/shutdown/re-entry. See ranks 14 (audio backend switch), 15 (failed swap leaves audio paused), 16 (shutdown leaks new bundle), 37 (re-entrant TRT wait). A consistent rule — always restore/adopt state on both success and failure paths, and guard re-entrant GUI handlers — would address the cluster.
- GUI thread does synchronous blocking work in several places that should be debounced or moved off-thread: native-size probe on every keystroke (rank 36), unbounded thread.wait() on model-download teardown (rank 32) and scan-worker shutdown (rank 33), and the first insightface model load inside a 2s close wait (rank 13). Pattern: blocking I/O / model load reached directly from a GUI-thread slot or close path.
- Resume/length correctness in the batch driver depends on container nb_frames being trustworthy, which it isn't (VFR/over-count). This drives the AUTO-task hard-failure (rank 9) and the ffmpeg fork-storm (rank 20) and the -shortest truncation (rank 41). Treating the first real EOF as authoritative length (sticky exhausted flag / clamp + persist) and preferring the persisted real length on resume would resolve all three at the source.
- Cache-key/cache-eviction correctness has a small cluster of latent-or-active bugs that interact: missing enhancer/upscaler params in the key (rank 6, active), session-start eviction of the active dir (rank 29), and the protect-accounting over-delete (rank 30, latent but fires the moment rank 29's protect fix lands). Fix rank 30 before/with rank 29 to avoid introducing the over-delete.
- Double-click / context-menu guard inconsistency: the batch view deliberately guards Edit/Delete on the running task in the context menu but leaves the double-click path unguarded (rank 8). Audit all alternate entry points (double-click, keyboard) to ensure they honor the same running-task guards as the menu.
