"""Timeline section selection — the set of frame ranges to include.

When non-empty, only frames inside these inclusive ``[start, end]`` ranges are
played (live) or processed + written to the output (batch); the gaps are
excluded entirely — a multi-range trim. Empty = no restriction (the whole
timeline, which is the default everywhere).

Ranges are normalized on construction: clamped to ``>= 0``, sorted, and merged
when they touch or overlap (zero excluded frames between two sections means one
continuous region, so they fuse; a real gap keeps them as distinct bands the
user can select and delete). Frozen + value-comparable so a SectionSet can be
diffed for change detection, pushed to the realtime executor, and round-tripped
through a BatchTask / settings as a list of ``[start, end]`` pairs.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def _normalize(pairs: Iterable[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    """Clamp, sort, and merge touching/overlapping ranges into a canonical
    tuple. A range with ``start > end`` is reordered; a range entirely below 0
    is dropped; ``start`` is floored at 0."""
    cleaned: list[tuple[int, int]] = []
    for pair in pairs:
        start, end = int(pair[0]), int(pair[1])
        if start > end:
            start, end = end, start
        if end < 0:
            continue
        cleaned.append((max(0, start), end))
    cleaned.sort()
    merged: list[tuple[int, int]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1] + 1:
            # Touching (gap of 0) or overlapping → one continuous region.
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


@dataclass(frozen=True)
class SectionSet:
    """An ordered, normalized set of inclusive frame ranges. Empty = no
    restriction (play / process the whole timeline)."""

    ranges: tuple[tuple[int, int], ...] = ()

    # ---- Construction ----

    @staticmethod
    def empty() -> "SectionSet":
        return SectionSet(())

    @staticmethod
    def of(pairs: Iterable[Sequence[int]]) -> "SectionSet":
        """Build from raw ``[start, end]`` pairs (normalized)."""
        return SectionSet(_normalize(pairs))

    # ---- Queries ----

    def is_empty(self) -> bool:
        return not self.ranges

    def contains(self, frame: int) -> bool:
        return any(start <= frame <= end for start, end in self.ranges)

    def index_at(self, frame: int) -> int | None:
        """Index of the range containing ``frame``, or None if in a gap."""
        for i, (start, end) in enumerate(self.ranges):
            if start <= frame <= end:
                return i
        return None

    def next_included_frame(self, frame: int) -> int | None:
        """The first included frame ``>= frame``, or None if ``frame`` is past
        the last range. Used to skip a gap during playback / fast-forward the
        playhead onto the next section."""
        for start, end in self.ranges:
            if frame <= end:
                return max(frame, start)
        return None

    def first_frame(self) -> int | None:
        return self.ranges[0][0] if self.ranges else None

    def last_frame(self) -> int | None:
        return self.ranges[-1][1] if self.ranges else None

    def total_frames(self) -> int:
        """Count of included frames across all ranges (the trimmed length)."""
        return sum(end - start + 1 for start, end in self.ranges)

    def count_included_between(self, after: int, before: int) -> int:
        """Number of included frames strictly between ``after`` and ``before``
        (both exclusive). Used for skip accounting so a gap the playhead jumped
        only counts the frames that were actually in a section."""
        n = 0
        for start, end in self.ranges:
            lo = max(start, after + 1)
            hi = min(end, before - 1)
            if lo <= hi:
                n += hi - lo + 1
        return n

    # ---- Edits (return a new normalized SectionSet) ----

    def with_added(self, start: int, end: int) -> "SectionSet":
        return SectionSet.of([*self.ranges, (start, end)])

    def with_range_replaced(self, index: int, start: int, end: int) -> "SectionSet":
        """Replace the range at ``index`` with ``(start, end)`` and re-normalize
        (an edit can merge it into a neighbour). Out-of-range index is a no-op
        add — defensive against a stale selection."""
        kept = [r for j, r in enumerate(self.ranges) if j != index]
        return SectionSet.of([*kept, (start, end)])

    def without_index(self, index: int) -> "SectionSet":
        """Drop the range at ``index`` (no re-merge needed — removal can't fuse
        the survivors)."""
        return SectionSet(
            tuple(r for j, r in enumerate(self.ranges) if j != index)
        )

    def clamp(self, max_frame: int) -> "SectionSet":
        """Trim/drop ranges so none exceeds ``max_frame`` (the last valid frame
        index). Applied when a SectionSet meets a target of known length."""
        out: list[tuple[int, int]] = []
        for start, end in self.ranges:
            s2, e2 = max(0, start), min(end, max_frame)
            if s2 <= e2:
                out.append((s2, e2))
        return SectionSet.of(out)

    # ---- Batch frame plan ----

    def frame_plan(self, total: int) -> list[int]:
        """Ordered list of original frame indices to process, within
        ``[0, total)``. Empty SectionSet → every frame ``0..total-1``."""
        if self.is_empty():
            return list(range(max(0, total)))
        plan: list[int] = []
        for start, end in self.ranges:
            plan.extend(range(max(0, start), min(end, total - 1) + 1))
        return plan

    # ---- Serialization ----

    def to_pairs(self) -> list[list[int]]:
        """As a JSON-friendly list of ``[start, end]`` pairs (for persistence /
        a BatchTask). Empty set → ``[]``."""
        return [[start, end] for start, end in self.ranges]
