import time

import pytest

from sinner2.pipeline.buffer.timeline import Timeline


class TestTimelineInit:
    def test_rejects_zero_fps(self):
        with pytest.raises(ValueError):
            Timeline(fps=0)

    def test_rejects_negative_fps(self):
        with pytest.raises(ValueError):
            Timeline(fps=-30)

    def test_initial_state(self):
        t = Timeline(fps=30)
        assert t.fps == 30
        assert t.is_playing is False
        assert t.current_frame() == 0
        assert t.elapsed_seconds() == 0.0


class TestTimelineStart:
    def test_start_sets_playing(self):
        t = Timeline(fps=30)
        t.start()
        assert t.is_playing is True

    def test_start_from_frame(self):
        t = Timeline(fps=30)
        t.start(from_frame=100)
        assert t.current_frame() >= 100

    def test_start_advances_frame_over_time(self):
        t = Timeline(fps=100)
        t.start()
        time.sleep(0.05)
        assert t.current_frame() >= 4


class TestTimelinePause:
    def test_pause_freezes_frame(self):
        t = Timeline(fps=100)
        t.start()
        time.sleep(0.03)
        t.pause()
        frozen = t.current_frame()
        time.sleep(0.05)
        assert t.current_frame() == frozen

    def test_pause_clears_is_playing(self):
        t = Timeline(fps=30)
        t.start()
        t.pause()
        assert t.is_playing is False

    def test_pause_when_paused_is_noop(self):
        t = Timeline(fps=30)
        t.pause()
        assert t.is_playing is False
        assert t.current_frame() == 0


class TestTimelineSeek:
    def test_seek_jumps_to_frame_while_paused(self):
        t = Timeline(fps=30)
        t.seek(500)
        assert t.current_frame() == 500
        assert t.is_playing is False

    def test_seek_preserves_playing_state(self):
        t = Timeline(fps=30)
        t.start()
        t.seek(100)
        assert t.is_playing is True

    def test_seek_rebases_clock(self):
        t = Timeline(fps=100)
        t.start()
        time.sleep(0.05)
        t.seek(0)
        # right after seek the current frame should be close to 0, not bumped by elapsed
        assert t.current_frame() < 3


class TestTimelineRestart:
    def test_second_start_rebases(self):
        t = Timeline(fps=100)
        t.start(from_frame=0)
        time.sleep(0.03)
        t.start(from_frame=500)
        assert t.current_frame() >= 500
        assert t.current_frame() < 510


class TestTimelineElapsed:
    def test_elapsed_zero_when_paused(self):
        t = Timeline(fps=30)
        assert t.elapsed_seconds() == 0.0

    def test_elapsed_grows_while_playing(self):
        t = Timeline(fps=30)
        t.start()
        time.sleep(0.03)
        assert t.elapsed_seconds() > 0.02
