import os

# Run Qt tests headlessly. Must be set before Qt is imported anywhere.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_cache_root(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point every GUI test's frame cache at a throwaway tmp dir.

    `default_cache_root()` otherwise resolves to the real ``<repo>/temp``, so a
    window built without an explicit cache root would (a) write session cache
    INTO the working tree and (b) ``stat``-walk the whole thing on the GUI
    thread on every teardown (``_teardown_session`` → ``cacheStorageStatsChanged``
    → ``_refresh_cache_stats``). On a populated temp (86k+ files) that walk never
    finishes within the test timeout — the suite hangs. Reading happens via the
    env var inside ``default_cache_root()`` at construction time, so setting it
    here (autouse, before the ``window`` fixture) isolates each test."""
    monkeypatch.setenv("SINNER2_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))


@pytest.fixture(autouse=True)
def stub_ensure_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never pop the blocking "Download models?" confirm during a window build.

    `_on_processor_config_changed` (fired from the window's `_restore_processor_
    settings`) calls `ensure_models` for any OPTIONAL weight the restored config
    selects (onnx enhancer, alt detector, landmarker, occluder). When that weight
    is absent — which is exactly CI's state, models dir empty — `ensure_models`
    shows a modal `confirm()` dialog and blocks forever: headless there is no one
    to click it, so `SinnerMainWindow()` never returns and the suite hangs in the
    `window` fixture (this was the CI "stuck at 16%"). On a dev box the weights
    exist, so `ensure_models` returns True silently — stub it to do the same so
    tests match that path regardless of what's on disk. Patches the symbol
    main_window imported, NOT `model_download.ensure_models`, so
    `test_model_download.py`'s direct tests of the real function are untouched."""
    from sinner2.gui import main_window

    monkeypatch.setattr(main_window, "ensure_models", lambda *a, **k: True)


@pytest.fixture(autouse=True)
def stub_insightface_model(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Never load the REAL insightface model in GUI tests.

    The detection-overlay probe runs on a background QThread; toggling the
    overlay (swapper off) enqueues a real ``analyse_uncached`` there. On a
    machine without the cached model (CI) that downloads buffalo_l (~300 MB) —
    far longer than closeEvent's join timeout, so the thread outlives the test
    and crashes mid-extract, aborting a later test. Stubbing the shared model
    makes every detection instant + offline, so the probe drains cleanly.

    Autouse + function-scoped: this sets up before the ``window`` fixture and
    tears down AFTER ``window.close()``, so the close()-time probe drain still
    sees the stub.
    """
    from sinner2.pipeline import face_analyser

    app = MagicMock()
    app.get = MagicMock(return_value=[])  # detect nothing, instantly
    monkeypatch.setattr(
        face_analyser, "_get_shared_face_analysis", lambda *a, **k: app
    )
    face_analyser.reset_shared_face_analysis()
    yield app
    face_analyser.reset_shared_face_analysis()
