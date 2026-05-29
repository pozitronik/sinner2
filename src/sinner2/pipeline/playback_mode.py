from enum import Enum


class PlaybackMode(str, Enum):
    """How fast the display thread polls the buffer.

    FIXED_30:  cap display at 30 Hz regardless of source/processing rate —
               smooth playback with the lowest CPU cost. The default.
    SOURCE:    poll at the source video's fps so every produced frame is
               eligible to be shown at the cadence the content was authored
               for. Useful for high-fps content (60+) when you want to see
               every frame.
    UNLIMITED: poll as fast as possible (1ms floor to yield the GIL). The
               display still won't emit duplicates thanks to the per-tick
               last-shown-index guard, so this mostly buys faster response
               to seeks rather than higher framerate.

    Persisted as the underlying string value via the str mixin.
    """

    FIXED_30 = "fixed_30"
    SOURCE = "source"
    UNLIMITED = "unlimited"
