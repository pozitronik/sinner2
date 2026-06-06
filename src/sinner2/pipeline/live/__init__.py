"""Live-camera mode: capture a device stream, run it through the processor
chain, and push the result to output sinks (MJPEG today) + an on-screen preview.

Distinct from `pipeline.realtime` (which drives a finite, seekable video file with
a timeline + frame cache): a live stream is infinite, non-seekable, and
latency-first, so this is a lean capture→process→sink loop, not the executor.
"""
