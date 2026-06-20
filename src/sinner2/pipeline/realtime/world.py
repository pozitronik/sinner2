"""Immutable snapshot of RealtimeExecutor's source/target + chain state.

The executor's "world" is the set of objects an in-flight frame is processed
against: the reader pool it was read from, the buffer its result is written to,
the timeline / frame-state map it's indexed in, and the processor chain applied
to it — tagged by a monotonic ``generation``. A source/target swap (reconfigure)
or a routing change (face-map / geometry) produces a NEW world via the helpers
below; the executor matches each completed WorkItem to the world it was
submitted under and discards results from a superseded world instead of writing
a stale frame into the new buffer.

Strategy / playback-mode / sections are deliberately NOT part of the world: they
change playback pacing, not the validity of an already-processed frame, so
changing one must not advance the generation and invalidate in-flight results.

Frozen + replace-based so each transition is a single atomic rebind of one
``self._world`` reference — there is no window in which some fields belong to the
new world and others to the old, which is the race the previous piecemeal
field-by-field swap had to guard by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from sinner2.io.reader_pool import ReaderPool
from sinner2.pipeline.buffer.buffer import FrameBuffer
from sinner2.pipeline.buffer.timeline import Timeline
from sinner2.pipeline.processor import Processor
from sinner2.pipeline.realtime.frame_state import FrameStateMap


@dataclass(frozen=True)
class World:
    generation: int
    chain: tuple[Processor, ...]
    reader_pool: ReaderPool
    buffer: FrameBuffer
    timeline: Timeline
    frame_states: FrameStateMap

    def reconfigured(
        self,
        *,
        chain: tuple[Processor, ...],
        reader_pool: ReaderPool,
        buffer: FrameBuffer,
        timeline: Timeline,
        frame_states: FrameStateMap,
    ) -> "World":
        """A new world for a source/target swap: every component replaced and the
        generation advanced, so in-flight results from the old world are
        dropped rather than written into the new buffer."""
        return replace(
            self,
            generation=self.generation + 1,
            chain=chain,
            reader_pool=reader_pool,
            buffer=buffer,
            timeline=timeline,
            frame_states=frame_states,
        )

    def bumped(self) -> "World":
        """A new world for an in-place routing change (face-map / geometry): the
        SAME source/target + chain + buffer objects, the next generation — so a
        worker still processing the old routing has its result discarded rather
        than published as a one-frame stale flash."""
        return replace(self, generation=self.generation + 1)
