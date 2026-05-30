"""End-to-end pipeline test on a REAL video with REAL models.

Marked `slow` → deselected from default runs (the `-m 'not slow'` addopts in
pyproject); run explicitly with `pytest -m slow`. Real face detection needs a
real face, so the assets are user-provided via env vars rather than shipped:

    SINNER2_E2E_SOURCE=/path/to/face.jpg \
    SINNER2_E2E_TARGET=/path/to/short.mp4 \
    pytest -m slow

Models must be present in the models dir (or SINNER2_MODELS_DIR). GPU optional
(the CPU execution provider works, just slowly). Skips cleanly when the models
or assets are missing, so it never fails a machine that simply can't run it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from sinner2.batch.driver import BatchDriver
from sinner2.batch.task import (
    BatchOutputFormat,
    BatchTask,
    BatchTaskStatus,
    resolve_output_path,
)
from sinner2.pipeline import model_cache

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _require_models():
    missing = model_cache.missing_models()
    if missing:
        pytest.skip(f"models missing: {', '.join(missing)} (place them or set SINNER2_MODELS_DIR)")


def _asset(env_var: str) -> Path:
    value = os.environ.get(env_var)
    if not value:
        pytest.skip(f"set {env_var} to run the e2e test")
    path = Path(value)
    if not path.is_file():
        pytest.skip(f"{env_var} -> {path} not found")
    return path


def test_batch_swap_and_enhance_to_video(tmp_path):
    source = _asset("SINNER2_E2E_SOURCE")
    target = _asset("SINNER2_E2E_TARGET")
    task = BatchTask(
        source_path=source,
        target_path=target,
        output_path=tmp_path / "e2e_out.mp4",
        output_format=BatchOutputFormat.VIDEO,
        swapper_enabled=True,
        enhancer_enabled=True,
    )
    driver = BatchDriver(cache_root=tmp_path / "cache")
    status = driver.run(task)
    assert status is BatchTaskStatus.COMPLETED, task.error_message
    out = resolve_output_path(task)
    assert out.exists()
    assert out.stat().st_size > 0
