"""ffmpeg-backed frame-sequence → mp4 encoder.

Used by the batch driver after every frame in a task has been processed
and written to its cache directory. We always output H.264 / yuv420p
(broadest-compatible mp4 profile) because the batch use-case is "send
this somewhere"; people doing further editing can decode and re-encode.

Two operations:
  - encode_frames_to_mp4(frame_dir, output, fps, audio_source) — runs
    ffmpeg, raising FfmpegMissingError if the binary isn't on PATH so
    the driver can fall back to frames mode.
  - probe_audio_codec(media_path) — runs ffprobe to read the source's
    audio codec, so we can stream-copy it when MP4 supports it and
    re-encode to AAC when it doesn't (e.g. WMA from a .wmv).

Why not python-ffmpeg / imageio? Subprocess is one dependency we
already implicitly require (FFmpegVideoTargetReader uses it), and the
command line is short enough to read at a glance.
"""
from __future__ import annotations

import shlex
import shutil
import subprocess
import threading
from collections.abc import Callable
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


def probe_audio_codec(media_path: Path) -> str | None:
    """The codec name of the source's first audio stream, or None when there's
    no audio stream (or ffprobe is unavailable).

    Drives two decisions: whether to wire the audio-mux arguments at all (None →
    skip them, avoiding an ffmpeg failure on a -map 1:a flag for a video-only
    source), and whether the codec can be stream-copied into MP4 or must be
    re-encoded (see _MP4_COPYABLE_AUDIO).
    """
    try:
        ffprobe = _require_ffprobe()
    except FfmpegMissingError:
        return None
    # First audio stream's codec name (e.g. "aac", "wmav2"), nothing else.
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name",
        "-of", "csv=p=0",
        str(media_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    codec = result.stdout.strip()
    return codec or None


def probe_has_audio(media_path: Path) -> bool:
    """True when the file has at least one audio stream. Thin wrapper over
    probe_audio_codec for callers that only need presence, not the codec."""
    return probe_audio_codec(media_path) is not None


# libx264/yuv420p requires even width AND height. Force-truncate to even so an
# odd-dimension source (phone / screen capture) doesn't fail the encode with an
# opaque non-zero exit. No-op when dimensions are already even.
_EVEN_SCALE = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
_VIDEO_CODEC = [
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "18",                  # visually-lossless default
    "-preset", "medium",
]
# Audio codecs MP4 can hold via stream copy. Anything else (WMA, Vorbis, Opus,
# FLAC, PCM, ...) is re-encoded to AAC: `-c:a copy` of an unsupported codec into
# MP4 fails at mux time with EINVAL (e.g. WMA from a .wmv source — the failure
# that prompted this). Conservative on purpose — copy only the codecs that mux
# AND play back broadly; re-encode the rest rather than risk an unplayable file.
_MP4_COPYABLE_AUDIO = frozenset({"aac", "mp3", "ac3", "eac3", "alac", "mp2"})


def _av_args(
    use_audio: bool,
    audio_segments: list[tuple[float, float]] | None,
    audio_copy: bool = True,
) -> list[str]:
    """The codec / mapping args after the inputs.

    Three shapes:
      - cut audio (a section trim): re-trim + concatenate the audio segments to
        match the selected frames, re-encoded to AAC (atrim can't stream-copy).
        The even-scale lives in the SAME filter graph because -vf can't coexist
        with -filter_complex.
      - whole audio: stream-copy the source audio when MP4 supports its codec
        (``audio_copy``), else re-encode to AAC; even-scale via -vf.
      - no audio: -an.
    """
    if use_audio and audio_segments:
        trims = [
            f"[1:a]atrim=start={start:.6f}:end={end:.6f},"
            f"asetpts=PTS-STARTPTS[a{i}]"
            for i, (start, end) in enumerate(audio_segments)
        ]
        labels = "".join(f"[a{i}]" for i in range(len(audio_segments)))
        graph = ";".join([
            f"[0:v]{_EVEN_SCALE}[outv]",
            *trims,
            f"{labels}concat=n={len(audio_segments)}:v=0:a=1[outa]",
        ])
        return [
            "-filter_complex", graph,
            "-map", "[outv]", "-map", "[outa]",
            *_VIDEO_CODEC, "-c:a", "aac",
        ]
    args = ["-vf", _EVEN_SCALE, *_VIDEO_CODEC]
    if use_audio:
        # Copy when MP4 supports the source codec; otherwise re-encode to AAC so
        # an incompatible codec (WMA/Vorbis/...) doesn't fail the mux.
        args += ["-c:a", "copy" if audio_copy else "aac"]
        # No -shortest: the video stream (our rendered frame sequence) must
        # define the duration. -shortest would end output at the shorter stream,
        # so for VFR / fps-rounding sources where the video runs slightly longer
        # than the copied audio it silently dropped trailing processed frames.
        args += ["-map", "0:v:0", "-map", "1:a:0"]
    else:
        args += ["-an"]
    return args


def encode_frames_to_mp4(
    frame_dir: Path,
    output: Path,
    fps: float,
    frame_ext: str = "jpg",
    audio_source: Path | None = None,
    audio_segments: list[tuple[float, float]] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    encode_args: str = "",
) -> None:
    """Encode the per-frame files at <frame_dir>/00000000.<ext>, ...
    into an H.264/yuv420p mp4 at `output`. Re-muxes audio from
    `audio_source` when given AND the source actually has an audio stream
    (silently dropped otherwise — having no audio is a valid result, not
    an error): stream-copied when MP4 supports the source codec, else
    re-encoded to AAC (e.g. WMA from a .wmv, which MP4 can't stream-copy).

    `audio_segments`, when given (a section trim), are ``(start_s, end_s)``
    time ranges of `audio_source` to cut out and concatenate so the audio lines
    up with the selected — and likewise concatenated — video frames.

    `progress_callback`, when given, is called with the running count of
    frames encoded (parsed from ffmpeg's `-progress` stream) so the batch
    driver can surface the otherwise-opaque combine/encode step — without
    it the GUI's progress bar froze at the last processor stage's 100%
    while a long video muxed.

    `encode_args`, when given (a power-user string), is shlex-split and appended
    to the output options just before the output file, so it OVERRIDES the
    matching defaults (ffmpeg uses the last value for an option — e.g.
    "-c:v libx265 -crf 24"). The even-scale filter + audio mapping stay intact.

    Raises FfmpegMissingError when ffmpeg isn't on PATH. Raises ValueError when
    `encode_args` can't be parsed (unbalanced quotes). Raises
    subprocess.CalledProcessError on encoder failure (the caller
    surfaces this via task.error_message).
    """
    ffmpeg = _require_ffmpeg()
    try:
        extra_args = shlex.split(encode_args) if encode_args.strip() else []
    except ValueError as exc:
        raise ValueError(f"invalid extra ffmpeg args {encode_args!r}: {exc}") from exc
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
    audio_codec = (
        probe_audio_codec(audio_source) if audio_source is not None else None
    )
    use_audio = audio_codec is not None
    audio_copy = audio_codec in _MP4_COPYABLE_AUDIO
    if use_audio:
        cmd += ["-i", str(audio_source)]
    cmd += _av_args(use_audio, audio_segments, audio_copy)
    # Power-user overrides go last among the output options so ffmpeg's
    # last-value-wins resolves them over the defaults above.
    cmd += extra_args
    if progress_callback is None:
        cmd.append(str(output))
        # capture_output so ffmpeg's stderr doesn't spew through the GUI's
        # parent console; subprocess raises CalledProcessError on non-zero
        # exit and we forward the stderr in the exception message for
        # debuggability.
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return
    # Progress path: ask ffmpeg for a machine-readable progress stream on
    # stdout, parse the `frame=N` lines, and report the count. -nostats
    # silences the human stderr stats; errors still reach stderr (kept for
    # the exception message). pipe:1 = stdout.
    cmd += ["-nostats", "-progress", "pipe:1", str(output)]
    _run_ffmpeg_with_progress(cmd, progress_callback)


def _run_ffmpeg_with_progress(
    cmd: list[str], progress_callback: Callable[[int], None]
) -> None:
    """Run ffmpeg, parsing its `-progress` stdout into frame-count callbacks.

    stderr is drained on a separate thread so a chatty encoder can't deadlock
    against our stdout read; on a non-zero exit we raise CalledProcessError
    carrying that stderr, matching the non-progress path's error surface.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr_chunks: list[str] = []

    def _drain_stderr() -> None:
        if proc.stderr is not None:
            stderr_chunks.append(proc.stderr.read())

    err_thread = threading.Thread(target=_drain_stderr, daemon=True)
    err_thread.start()
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                # ffmpeg -progress emits `key=value` lines; `frame=N` is the
                # encoded-frame counter we surface.
                if line.startswith("frame="):
                    value = line.split("=", 1)[1].strip()
                    if value.isdigit():
                        progress_callback(int(value))
    finally:
        proc.wait()
        err_thread.join(timeout=5)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, stderr="".join(stderr_chunks)
        )
