"""EOF / over-counted-nb_frames handling for the ffmpeg reader.

These are pure-logic tests (the decoder subprocess is stubbed), so unlike the
integration tests in test_video_target_reader.py they don't require ffmpeg.
"""
from __future__ import annotations

import numpy as np

from sinner2.io.video_target_reader import FFmpegVideoTargetReader


def _make_reader(frame_count: int, real_frames: int) -> FFmpegVideoTargetReader:
    """A reader whose probe over-reports frame_count: the metadata claims
    `frame_count` frames but the decoder only ever yields `real_frames`."""
    r = object.__new__(FFmpegVideoTargetReader)
    r._fps = 10.0
    r._frame_count = frame_count
    r._native_width = r._width = 4
    r._native_height = r._height = 4
    r._decoder = None
    r._next_index = 0
    r._eof_index = None
    r._start_calls = 0
    r._real_frames = real_frames

    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def fake_start(index: int) -> None:
        r._start_calls += 1
        r._decoder = object()  # non-None sentinel
        r._next_index = index

    def fake_read_pipe() -> np.ndarray | None:
        # The decoder produces real_frames frames then hits EOF, regardless of
        # how many times it's restarted. _next_index == the index being read.
        return frame if r._next_index < r._real_frames else None

    r._start_decoder_at = fake_start  # type: ignore[method-assign]
    r._read_frame_from_pipe = fake_read_pipe  # type: ignore[method-assign]
    return r


class TestOverCountedFrameCount:
    def test_trailing_phantom_indices_do_not_fork_a_new_decoder(self) -> None:
        # nb_frames says 10 but only 5 frames decode. Reading all 10 indices
        # sequentially must NOT spawn a fresh ffmpeg per trailing phantom frame
        # — exactly one decoder start for the whole sweep.
        r = _make_reader(frame_count=10, real_frames=5)
        results = [r.read(i) for i in range(10)]
        assert [x is not None for x in results] == [True] * 5 + [False] * 5
        assert r._start_calls == 1

    def test_eof_is_sticky_across_reads(self) -> None:
        r = _make_reader(frame_count=10, real_frames=5)
        for i in range(6):  # drive past the real end to discover EOF at 5
            r.read(i)
        before = r._start_calls
        # Any in-range index at or past the discovered EOF short-circuits.
        assert r.read(7) is None
        assert r.read(9) is None
        assert r._start_calls == before  # no new decoders spawned

    def test_backward_seek_below_eof_still_reads(self) -> None:
        # Discovering EOF at 5 must not block legitimate frames before it.
        r = _make_reader(frame_count=10, real_frames=5)
        for i in range(6):
            r.read(i)
        assert r.read(3) is not None
