"""Per-frame pipeline state, for the processing visualiser.

A compact (1 byte/frame) record of where each target frame is in the realtime
pipeline — not reached, queued, processing, ready in memory, evicted to disk,
skipped by the strategy, or invalidated. The executor writes the pre-buffer
transitions (queued / processing / skipped); the FrameBuffer writes the
memory/disk transitions (ready / evicted / invalidated). Both write to the same
map and the GUI polls snapshot() to paint a heatmap aligned to the timeline.

Writes are single-byte assignments / whole-slice assignments, both atomic under
the GIL, so nothing is locked on the hot path; snapshot() copies the array in
one bytes() call. The map is eventually-consistent by design — a column painted
mid-transition just refreshes on the next poll.
"""
from __future__ import annotations

from enum import IntEnum


class FrameState(IntEnum):
    """Where a target frame is in the pipeline. Ordered NOT_REACHED..INVALID;
    the byte value IS the enum value, so the map stores these directly."""

    NOT_REACHED = 0   # not yet submitted (or reset after a chain change)
    SKIPPED = 1       # the strategy jumped past it — never processed
    QUEUED = 2        # submitted, awaiting a source read / a free worker
    PROCESSING = 3    # in flight on a worker
    READY_MEM = 4     # processed, in the memory framebuffer (instant playback)
    READY_DISK = 5    # processed but evicted from memory — on disk only
    INVALID = 6       # stale (chain/param change) — will be reprocessed


class FrameStateMap:
    """A bytearray of FrameState, one entry per target frame. Thread-safe by
    relying on the GIL for the byte/slice writes it makes; no explicit lock."""

    def __init__(self, frame_count: int) -> None:
        self._n = max(0, frame_count)
        self._states = bytearray(self._n)  # zero-filled → all NOT_REACHED

    @property
    def frame_count(self) -> int:
        return self._n

    def set(self, index: int, state: FrameState) -> None:
        """Set one frame's state. Out-of-range indices are ignored (a reconfigure
        can leave a stale writer briefly pointing past the new length)."""
        if 0 <= index < self._n:
            self._states[index] = int(state)  # atomic single-byte write

    def set_range(self, lo: int, hi: int, state: FrameState) -> None:
        """Set [lo, hi) (clamped to bounds) to a state in one slice assignment.
        Used for the strategy's skip gaps and bulk invalidation."""
        lo = max(0, lo)
        hi = min(self._n, hi)
        if hi > lo:
            self._states[lo:hi] = bytes([int(state)]) * (hi - lo)

    def get(self, index: int) -> FrameState:
        if 0 <= index < self._n:
            return FrameState(self._states[index])
        return FrameState.NOT_REACHED

    def reset(self) -> None:
        """Back to all-NOT_REACHED (same length) — e.g. on invalidate_all."""
        self._states = bytearray(self._n)

    def snapshot(self) -> bytes:
        """An immutable copy of the whole array for the GUI to bin + paint."""
        return bytes(self._states)
