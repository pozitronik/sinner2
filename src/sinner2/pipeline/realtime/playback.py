"""Pure playback-timing decisions for RealtimeExecutor.

The executor's playback thread owns the loop + I/O (reading the buffer, emitting
on_frame, publishing observables); these helpers hold the two stateless DECISIONS
that loop makes — how long to sleep before the next tick, and which buffered
frame to fall back to when the worker is behind — so each is testable in
isolation without spinning the executor's threads.
"""
from __future__ import annotations

from sinner2.pipeline.playback_mode import PlaybackMode

# Default fixed-rate tick when PlaybackMode.FIXED_30 is selected. Keep at 30 Hz:
# high enough for smooth perceived motion, low enough to be cheap.
_FIXED_PLAYBACK_TICK_S = 1.0 / 30
# Floor for UNLIMITED mode so we still yield to the OS scheduler rather than
# burning a core in a tight loop. The per-tick duplicate-frame guard means we
# don't actually emit more frames than the timeline produces, so this floor
# mostly just bounds wakeup frequency.
_UNLIMITED_PLAYBACK_TICK_S = 0.001


def compute_playback_sleep(
    playing: bool, mode: PlaybackMode, fps: float
) -> float | None:
    """How long to wait before the next playback tick.

    Returns None for "block until woken" — used when nothing is producing frame
    changes (paused or idle). Otherwise the per-mode tick interval. The wake
    event interrupts the wait early on any state change, so this is purely the
    upper bound between ticks during normal playback.
    """
    if not playing:
        return None
    if mode is PlaybackMode.UNLIMITED:
        return _UNLIMITED_PLAYBACK_TICK_S
    if mode is PlaybackMode.SOURCE:
        return 1.0 / fps
    return _FIXED_PLAYBACK_TICK_S


def select_fallback_index(
    fallback_index: int | None, last_shown: int | None
) -> int | None:
    """Which buffered frame to show when the worker is behind (no frame at the
    timeline target) — the newest buffered frame ≤ target, or None to hold the
    current frame.

    Won't repaint an OLDER frame than what's already on screen during forward
    playback — the newest frame ≤ target can drop below the last shown when
    skipped frames complete out of order or an old one is evicted, and repainting
    it is a visible backward stutter. A seek resets ``last_shown`` to None, so
    seeks (incl. backward) repaint freely. (Pause-suppression is the caller's: it
    only consults this when NOT paused.)
    """
    if fallback_index is None:
        return None
    if last_shown is None or fallback_index >= last_shown:
        return fallback_index
    return None
