"""Preprocessing head-start math.

Preprocessing renders a clip ahead before releasing playback so it then plays
smoothly at the target FPS showing every frame. The minimum head-start (frames
to pre-render) is fixed by throughput:

Play frames [0, N) at F fps while rendering continues at R fps throughout. Start
playback from frame 0 after pre-rendering B frames. At playback time t the
displayed frame is t·F and the rendered count is B + t·R; smoothness needs
``B + t·R >= t·F`` for all t up to the end (t = N/F). That binds at the end:

    B >= (N/F)·(F − R) = N·(1 − R/F)

So when R >= F the pipeline keeps up and no head-start is needed; the slower the
pipeline, the larger the fraction of the clip that must be pre-rendered (e.g.
R=10, F=30 → 2/3 of the clip up front).
"""
from __future__ import annotations

import math


def required_prefill(frame_count: int, process_fps: float, target_fps: float) -> int:
    """Frames to pre-render before releasing playback so it never stalls.

    0 when the pipeline keeps up (R >= F). The whole clip when throughput is
    still unknown (R or F <= 0) — the safe choice (fully render, then play).
    """
    if frame_count <= 0:
        return 0
    if target_fps <= 0 or process_fps <= 0:
        return frame_count
    if process_fps >= target_fps:
        return 0
    # B = N − (frames rendered during playback) = N − floor(N·R/F). Computed
    # this way (not N·(1−R/F)) to stay exact for the common divisible cases and
    # avoid float overshoot; floor UNDER-counts the rendered tail, so B is the
    # conservative (never-under-buffer) head-start.
    rendered_during_playback = math.floor(frame_count * process_fps / target_fps)
    return max(0, min(frame_count, frame_count - rendered_during_playback))
