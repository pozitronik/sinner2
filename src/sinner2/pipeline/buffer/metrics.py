from dataclasses import dataclass


@dataclass(frozen=True)
class BufferMetrics:
    """Observability snapshot for the realtime conveyor.

    All four lag fields are clamped to >= 0. Negative values would mean
    "processing is ahead of playback" which can happen briefly with fast
    workers — clamping keeps the fields useful for "how far behind?" UX.

    cache_hit_ratio is 0.0 until at least one get() call has been made.

    The write_* fields surface BoundedWriteExecutor state directly so the
    metrics panel can show one row per concern (cache pressure, write
    queue depth, dropped writes, write latency). They are 0/0.0 when the
    cache_mode is OFF or the executor has not yet run any tasks.
    """

    frame_lag: int
    time_lag_s: float
    display_frame_lag: int
    display_time_lag_s: float
    current_frame_miss: int
    memory_used_bytes: int
    cache_hit_ratio: float
    write_outstanding: int = 0
    write_max_outstanding: int = 0
    write_submitted: int = 0
    write_completed: int = 0
    write_dropped: int = 0
    write_failed: int = 0  # writes that raised (disk full / permission / bad path)
    write_latency_p50_ms: float = 0.0
    write_latency_p95_ms: float = 0.0
