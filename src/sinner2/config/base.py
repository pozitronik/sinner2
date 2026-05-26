from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class ParameterInfo:
    """GUI/CLI metadata for Pydantic fields, attached via Annotated.

    Used to keep type hints pure while carrying display info: CLI flag names,
    help text, choice list, required flag. Consumed by the GUI to render
    tooltips and by the CLI to register arguments.
    """

    cli_names: tuple[str, ...] = ()
    help: str = ""
    choices: tuple[Any, ...] | None = None
    required: bool = False


class SinnerBaseModel(BaseModel):
    """Base for every sinner2 Pydantic config.

    Centralizes config policy in one place: unknown fields are silently
    ignored (forward-compatibility for older config files), arbitrary types
    are allowed (numpy arrays, Path), defaults are validated.
    """

    model_config = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,
        validate_default=True,
    )
