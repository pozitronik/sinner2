from dataclasses import dataclass


@dataclass(frozen=True)
class BufferMetrics:
    """Observability snapshot for the realtime conveyor.

    All four lag fields are clamped to >= 0. Negative values would mean
    "processing is ahead of playback" which can happen briefly with fast
    workers — clamping keeps the fields useful for "how far behind?" UX.

    cache_hit_ratio is 0.0 until at least one get() call has been made.
    """

    frame_lag: int
    time_lag_s: float
    display_frame_lag: int
    display_time_lag_s: float
    current_frame_miss: int
    memory_used_bytes: int
    cache_hit_ratio: float
