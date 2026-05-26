from dataclasses import FrozenInstanceError

import pytest

from sinner2.pipeline.messages import (
    PauseMsg,
    PlayMsg,
    SeekMsg,
    SetChainMsg,
    SetParamsMsg,
    SetSkipStrategyMsg,
    StopMsg,
)


class TestMessages:
    def test_play_pause_stop_have_no_payload(self):
        assert PlayMsg() == PlayMsg()
        assert PauseMsg() == PauseMsg()
        assert StopMsg() == StopMsg()

    def test_seek_carries_target_frame(self):
        m = SeekMsg(target_frame=42)
        assert m.target_frame == 42

    def test_set_params_carries_name_and_dict(self):
        m = SetParamsMsg(processor_name="FaceSwapper", params={"many_faces": False})
        assert m.processor_name == "FaceSwapper"
        assert m.params == {"many_faces": False}

    def test_set_chain_carries_factory(self):
        def factory():
            return []

        m = SetChainMsg(chain_factory=factory)
        assert m.chain_factory is factory

    def test_set_skip_strategy_carries_strategy(self):
        from unittest.mock import MagicMock

        s = MagicMock()
        m = SetSkipStrategyMsg(strategy=s)
        assert m.strategy is s

    def test_messages_are_frozen(self):
        with pytest.raises(FrozenInstanceError):
            SeekMsg(target_frame=0).target_frame = 1  # type: ignore[misc]

    def test_messages_are_hashable(self):
        assert hash(PlayMsg())
        assert hash(SeekMsg(target_frame=10))
