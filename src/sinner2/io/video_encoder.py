"""ffmpeg-backed frame-sequence → mp4 encoder.

Used by the batch driver after every frame in a task has been processed
and written to its cache directory. We always output H.264 / yuv420p
(broadest-compatible mp4 profile) because the batch use-case is "send
this somewhere"; people doing further editing can decode and re-encode.

Two operations:
  - encode_frames_to_mp4(frame_dir, output, fps, audio_source) — runs
    ffmpeg, raising FfmpegMissingError if the binary isn't on PATH so
    the driver can fall back to frames mode.
  - probe_has_audio(media_path) — runs ffprobe to detect whether the
    source has an audio stream we can copy through.

Why not python-ffmpeg / imageio? Subprocess is one dependency we
already implicitly require (FFmpegVideoTargetReader uses it), and the
command line is short enough to read at a glance.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FfmpegMissingError(RuntimeError):
    """ffmpeg / ffprobe not on PATH. Caller should fall back to a
    frames-mode output (just leave the per-frame files in place)."""


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise FfmpegMissingError(
            "ffmpeg not found on PATH — install ffmpeg or switch the task "
            "to FRAMES output."
        )
    return path


def _require_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if path is None:
        raise FfmpegMissingError(
            "ffprobe not found on PATH — install ffmpeg (provides ffprobe) "
            "or switch the task to FRAMES output."
        )
    return path


def probe_has_audio(media_path: Path) -> bool:
    """Returns True when the file has at least one audio stream.

    Used to decide whether to wire the audio-copy arguments. Probing
    avoids ffmpeg failing on a -map 1:a flag when the source is video-
    only.
    """
    try:
        ffprobe = _require_ffprobe()
    except FfmpegMissingError:
        return False
    # ffprobe streams query — print only the codec_type, one stream per line.
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(media_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return "audio" in result.stdout


def encode_frames_to_mp4(
    frame_dir: Path,
    output: Path,
    fps: float,
    frame_ext: str = "jpg",
    audio_source: Path | None = None,
) -> None:
    """Encode the per-frame files at <frame_dir>/00000000.<ext>, ...
    into an H.264/yuv420p mp4 at `output`. Re-muxes audio from
    `audio_source` via stream copy when given AND the source actually
    has an audio stream (silently dropped otherwise — having no audio
    is a valid result, not an error).

    Raises FfmpegMissingError when ffmpeg isn't on PATH. Raises
    subprocess.CalledProcessError on encoder failure (the caller
    surfaces this via task.error_message).
    """
    ffmpeg = _require_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",                          # overwrite if output exists
        "-framerate", f"{fps:.6f}",
        # Image sequence input: %08d matches our zero-padded filenames.
        # start_number 0 because the driver writes 00000000.ext upward.
        "-start_number", "0",
        "-i", str(frame_dir / f"%08d.{frame_ext}"),
    ]
    use_audio = audio_source is not None and probe_has_audio(audio_source)
    if use_audio:
        cmd += ["-i", str(audio_source)]
    cmd += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",                  # visually-lossless default
        "-preset", "medium",
    ]
    if use_audio:
        cmd += [
            "-c:a", "copy",            # re-mux, no re-encode
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",               # don't run past video or audio end
        ]
    else:
        cmd += ["-an"]                 # no audio output
    cmd.append(str(output))
    # capture_output so ffmpeg's stderr doesn't spew through the GUI's
    # parent console; subprocess raises CalledProcessError on non-zero
    # exit and we forward the stderr in the exception message for
    # debuggability.
    subprocess.run(cmd, check=True, capture_output=True, text=True)
