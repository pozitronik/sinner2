"""Tests for the preprocessing head-start math."""
from __future__ import annotations

from sinner2.pipeline.realtime.preprocess import required_prefill


class TestRequiredPrefill:
    def test_pipeline_keeps_up_needs_no_head_start(self):
        assert required_prefill(300, process_fps=30.0, target_fps=30.0) == 0
        assert required_prefill(300, process_fps=60.0, target_fps=30.0) == 0

    def test_slow_pipeline_prebuffers_the_deficit_fraction(self):
        # R=10, F=30 → deficit 2/3 → 200 of 300 frames up front.
        assert required_prefill(300, process_fps=10.0, target_fps=30.0) == 200

    def test_half_speed_prebuffers_half(self):
        assert required_prefill(300, process_fps=15.0, target_fps=30.0) == 150

    def test_conservative_rounding(self):
        # N=10, R=11, F=30 → rendered during playback = floor(110/30) = floor(3.6)
        # = 3 → B = 10 - 3 = 7 (round DOWN the rendered tail → never under-buffer).
        assert required_prefill(10, process_fps=11.0, target_fps=30.0) == 7

    def test_unknown_throughput_renders_everything(self):
        assert required_prefill(300, process_fps=0.0, target_fps=30.0) == 300
        assert required_prefill(300, process_fps=10.0, target_fps=0.0) == 300

    def test_capped_at_frame_count(self):
        # Extremely slow → deficit near 1, but never more than the whole clip.
        assert required_prefill(50, process_fps=0.001, target_fps=30.0) == 50

    def test_zero_frames(self):
        assert required_prefill(0, process_fps=10.0, target_fps=30.0) == 0
