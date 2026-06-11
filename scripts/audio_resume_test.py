"""Standalone A/V resume diagnostic for the Qt audio backend.

#2 ("audio dies on a source change") can't be pinned from code alone — it's a
QMediaPlayer / Windows Media Foundation runtime behaviour. This script isolates
QtMediaAudioBackend (no GPU, no models, no session) and runs the exact
load -> play -> pause -> RELOAD -> seek -> play cycle a source-only swap does,
for three RELOAD variants, printing whether audio actually resumes. The player's
position advancing after the resume = audio is really rolling.

Run on Windows (the platform that shows the bug), from the project root:

    .venv\\Scripts\\python.exe scripts\\audio_resume_test.py <a_video_with_audio.mp4>

Paste the whole output back. The variant that prints RESUMES is the fix.
"""
from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QApplication

from sinner2.audio.qt_audio_backend import QtMediaAudioBackend


def pump(app: QApplication, seconds: float) -> None:
    """Spin the Qt event loop for `seconds` so async setSource() / playback
    can progress (QMediaPlayer is fully asynchronous)."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def _state(b: QtMediaAudioBackend) -> str:
    p = b._player  # noqa: SLF001
    if p is None:
        return "no-player"
    return f"media={p.mediaStatus()} playback={p.playbackState()} pos={p.position()}ms"


def reload_none(b: QtMediaAudioBackend) -> None:
    """Original behaviour: no reload — just resume on the already-loaded media."""


def reload_same_url(b: QtMediaAudioBackend) -> None:
    """Current shipped fix: setSource() with the UNCHANGED URL."""
    b.reload()


def reload_clear_then_set(b: QtMediaAudioBackend) -> None:
    """Proposed fix: clear the source first so Qt can't short-circuit an
    unchanged URL — guarantees a fresh LoadingMedia -> LoadedMedia transition."""
    p = b._player  # noqa: SLF001
    p.stop()
    p.setSource(QUrl())
    b._media_ready = False         # noqa: SLF001
    b._pending_play = False        # noqa: SLF001
    b._pending_position_ms = None  # noqa: SLF001
    p.setSource(QUrl.fromLocalFile(str(b._loaded_path)))  # noqa: SLF001


def run_variant(
    app: QApplication,
    media: Path,
    label: str,
    do_reload: Callable[[QtMediaAudioBackend], None],
) -> None:
    print(f"\n=== variant: {label} ===")
    b = QtMediaAudioBackend()
    b.set_volume(0.3)
    b.load(media)
    pump(app, 2.5)  # let it load + reach ready
    b.play()
    pump(app, 2.0)  # let playback get going
    pos_initial = b.audio_position_seconds()
    print(f"  after initial play: {_state(b)}  -> pos={pos_initial:.2f}s")
    if pos_initial < 0.2:
        print(
            "  WARNING: initial playback didn't advance — does this file have an "
            "audio track? Try another video."
        )
    # Simulate the swap: pause, then the adopt's reload + seek + resume.
    b.pause()
    pump(app, 0.7)
    print(f"  paused for 'swap':  {_state(b)}")
    do_reload(b)
    b.seek_seconds(pos_initial)
    b.play()
    pump(app, 2.5)
    pos_resumed = b.audio_position_seconds()
    advanced = pos_resumed > pos_initial + 0.3
    print(f"  after resume:       {_state(b)}  -> pos={pos_resumed:.2f}s")
    print(f"  RESULT: {'RESUMES (audio rolling)' if advanced else 'SILENT (position did not advance)'}")
    b.shutdown()
    pump(app, 0.3)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/audio_resume_test.py <video_with_audio>")
        return 2
    media = Path(sys.argv[1])
    if not media.is_file():
        print(f"not a file: {media}")
        return 2
    app = QApplication(sys.argv)
    import PySide6
    print(f"PySide6 {PySide6.__version__}; media={media.name}")
    run_variant(app, media, "1. no reload (original direct play)", reload_none)
    run_variant(app, media, "2. reload same URL (current shipped fix)", reload_same_url)
    run_variant(app, media, "3. clear-then-set (proposed fix)", reload_clear_then_set)
    print(
        "\nDone. The variant printing RESUMES is the fix — paste this whole "
        "output back."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
