from enum import Enum


class CacheMode(str, Enum):
    """How the FrameBuffer interacts with the persistent disk store.

    WRITE_READ: full behaviour. Every put goes to memory cache AND to the
                bounded write queue. Cache misses fall back to disk reads.
                This is the default for normal use.
    READ_ONLY:  put goes to memory cache only — no new files written.
                Cache misses still fall back to disk reads, so any frames
                from prior sessions (or earlier in this session, before
                the mode change) remain visible. Useful for "view what's
                already processed without polluting the cache with
                experimental parameter changes".
    OFF:        memory only. Both put-to-disk and cache-miss-read-from-disk
                are skipped. No cross-session warmup; backward seeks past
                what's in memory will trigger reprocessing.

    Persisted as the underlying string value via the str mixin.
    """

    WRITE_READ = "write_read"
    READ_ONLY = "read_only"
    OFF = "off"
