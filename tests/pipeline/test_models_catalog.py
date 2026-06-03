"""Tests for the model catalog (metadata) + the management cache helpers."""
from __future__ import annotations

import pytest

from sinner2.pipeline import model_cache
from sinner2.pipeline.models_catalog import (
    MODEL_CATALOG,
    ModelCategory,
    catalog_entries,
    model_info,
)


@pytest.fixture
def models_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SINNER2_MODELS_DIR", str(tmp_path))
    return tmp_path


class TestCatalog:
    def test_catalog_covers_exactly_model_sources(self):
        # Drift guard: every downloadable model has metadata, and the catalog
        # has no orphan entries.
        assert set(MODEL_CATALOG) == set(model_cache.MODEL_SOURCES)

    def test_entries_well_formed(self):
        for info in MODEL_CATALOG.values():
            assert info.display_name
            assert info.description
            assert isinstance(info.category, ModelCategory)
            assert info.size_mb > 0

    def test_model_info_lookup(self):
        assert model_info("inswapper_128.onnx").category is ModelCategory.SWAPPER
        assert model_info("nope.onnx") is None

    def test_simswap_carries_license_note(self):
        assert "non-commercial" in model_info("simswap_256.onnx").license.lower()

    def test_catalog_entries_ordered_by_category_then_name(self):
        cats = list(ModelCategory)
        entries = catalog_entries()
        keys = [(cats.index(m.category), m.display_name.lower()) for m in entries]
        assert keys == sorted(keys)


class TestCacheHelpers:
    def test_size_on_disk_zero_when_missing(self, models_dir):
        assert model_cache.model_size_on_disk("inswapper_128.onnx") == 0

    def test_size_on_disk_reports_bytes(self, models_dir):
        (models_dir / "span_kendata_x4.onnx").write_bytes(b"x" * 1234)
        assert model_cache.model_size_on_disk("span_kendata_x4.onnx") == 1234

    def test_delete_removes_file(self, models_dir):
        f = models_dir / "span_kendata_x4.onnx"
        f.write_bytes(b"weights")
        assert model_cache.delete_model("span_kendata_x4.onnx") is True
        assert not f.exists()

    def test_delete_missing_returns_false(self, models_dir):
        assert model_cache.delete_model("span_kendata_x4.onnx") is False
