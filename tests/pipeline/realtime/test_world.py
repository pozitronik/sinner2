"""Unit tests for the World value object — the executor's immutable
source/target + chain snapshot, tagged by generation.

World is a dumb, frozen container, so the components here are plain sentinels;
what matters is that each transition rebinds ALL the right fields atomically,
advances the generation, and leaves the original instance untouched (so a worker
holding the old World sees a stable snapshot).
"""
from __future__ import annotations

import dataclasses

import pytest

from sinner2.pipeline.realtime.world import World


def _world(generation: int = 0) -> World:
    return World(
        generation=generation,
        chain=("chain0",),
        reader_pool="pool0",
        buffer="buffer0",
        timeline="timeline0",
        frame_states="states0",
    )


class TestConstruction:
    def test_fields_are_preserved(self):
        w = _world(generation=3)
        assert w.generation == 3
        assert w.chain == ("chain0",)
        assert w.reader_pool == "pool0"
        assert w.buffer == "buffer0"
        assert w.timeline == "timeline0"
        assert w.frame_states == "states0"

    def test_is_frozen(self):
        w = _world()
        with pytest.raises(dataclasses.FrozenInstanceError):
            w.generation = 9  # type: ignore[misc]


class TestReconfigured:
    def test_replaces_every_component_and_advances_generation(self):
        w = _world(generation=4)
        nw = w.reconfigured(
            chain=("chain1",),
            reader_pool="pool1",
            buffer="buffer1",
            timeline="timeline1",
            frame_states="states1",
        )
        assert nw.generation == 5
        assert nw.chain == ("chain1",)
        assert nw.reader_pool == "pool1"
        assert nw.buffer == "buffer1"
        assert nw.timeline == "timeline1"
        assert nw.frame_states == "states1"

    def test_leaves_the_original_world_untouched(self):
        w = _world(generation=4)
        w.reconfigured(
            chain=("chain1",), reader_pool="pool1", buffer="buffer1",
            timeline="timeline1", frame_states="states1",
        )
        # The old world a worker may still hold is unchanged.
        assert w.generation == 4
        assert w.buffer == "buffer0"


class TestBumped:
    def test_keeps_components_but_advances_generation(self):
        w = _world(generation=7)
        nw = w.bumped()
        assert nw.generation == 8
        # Same source/target + chain objects (routing change, not a swap).
        assert nw.chain is w.chain
        assert nw.reader_pool is w.reader_pool
        assert nw.buffer is w.buffer
        assert nw.timeline is w.timeline
        assert nw.frame_states is w.frame_states

    def test_leaves_the_original_world_untouched(self):
        w = _world(generation=7)
        w.bumped()
        assert w.generation == 7


class TestWithChain:
    def test_replaces_only_chain_keeping_generation_and_components(self):
        w = _world(generation=4)
        nw = w.with_chain(("chain1",))
        assert nw.generation == 4  # set_chain drained in-flight → no bump
        assert nw.chain == ("chain1",)
        assert nw.reader_pool is w.reader_pool
        assert nw.buffer is w.buffer
        assert nw.timeline is w.timeline
        assert nw.frame_states is w.frame_states

    def test_leaves_the_original_world_untouched(self):
        w = _world(generation=4)
        w.with_chain(("chain1",))
        assert w.chain == ("chain0",)


class TestGenerationMonotonicity:
    def test_advances_one_per_transition_across_a_chain(self):
        w = _world(generation=0)
        w = w.bumped()
        w = w.reconfigured(
            chain=("c",), reader_pool="p", buffer="b",
            timeline="t", frame_states="s",
        )
        w = w.bumped()
        assert w.generation == 3
