from dataclasses import FrozenInstanceError
from typing import Annotated

import pytest

from sinner2.config.base import ParameterInfo, SinnerBaseModel


class TestParameterInfo:
    def test_defaults(self):
        info = ParameterInfo()
        assert info.cli_names == ()
        assert info.help == ""
        assert info.choices is None
        assert info.required is False

    def test_carries_provided_values(self):
        info = ParameterInfo(
            cli_names=("--quality", "-q"),
            help="Output quality 0-100",
            choices=(50, 75, 90),
            required=True,
        )
        assert info.cli_names == ("--quality", "-q")
        assert info.help == "Output quality 0-100"
        assert info.choices == (50, 75, 90)
        assert info.required is True

    def test_is_frozen(self):
        info = ParameterInfo(help="x")
        with pytest.raises(FrozenInstanceError):
            info.help = "y"  # type: ignore[misc]

    def test_attaches_via_annotated(self):
        class Cfg(SinnerBaseModel):
            quality: Annotated[int, ParameterInfo(help="quality")] = 90

        cfg = Cfg()
        assert cfg.quality == 90


class TestSinnerBaseModel:
    def test_ignores_extra_fields(self):
        class M(SinnerBaseModel):
            name: str

        m = M.model_validate({"name": "a", "unknown_field": "x"})
        assert m.name == "a"
        assert not hasattr(m, "unknown_field")

    def test_validates_defaults(self):
        with pytest.raises(Exception):
            class Broken(SinnerBaseModel):
                count: int = "not an int"  # type: ignore[assignment]

            Broken()
