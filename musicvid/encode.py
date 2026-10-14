"""Frame pipe: stream raw RGB frames to ffmpeg and mux with the audio."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .util import MusicvidError, ffmpeg

# ``ultrafast`` skips x264's lookahead, which also makes it the lightest on
# memory -- useful for long 1080p renders on small machines.
ENCODER_PRESETS = {"ultrafast": "ultrafast", "fast": "veryfast",
                   "medium": "medium", "slow": "slow"}


class FrameWriter:
    """Context manager that writes frames to an H.264 MP4.

    >>> with FrameWriter("out.mp4", (1920, 1080), 30, "song.mp3") as writer:
    ...     for frame in frames:
    ...         writer.write(frame)
    """

    def __init__(self, output: str, size: tuple[int, int], fps: float, audio: str,
                 *, crf: int = 20, preset: str = "medium", audio_bitrate: str = "192k",
                 start: float | None = None, duration: float | None = None,
                 threads: int | None = None):
        self.output = str(output)
        self.width, self.height = int(size[0]), int(size[1])
        self.fps = float(fps)
        self.audio = str(audio)
        self.crf = int(crf)
        self.preset = ENCODER_PRESETS.get(str(preset), str(preset))
        self.audio_bitrate = str(audio_bitrate)
        self.start = float(start) if start else None
        self.duration = float(duration) if duration else None
        self.threads = threads
        self._proc: subprocess.Popen | None = None
        self._stderr = tempfile.NamedTemporaryFile(mode="w+b", suffix=".log", delete=False)
        self._frames = 0

    # -- command ---------------------------------------------------------- #
    def command(self) -> list[str]:
        cmd = [
            ffmpeg(), "-v", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}", "-r", f"{self.fps:.6f}",
            "-i", "pipe:0",
        ]
        # input-side seek: -t then counts from the seek point, which is what the
        # render timeline expects (output-side -ss would make -t absolute)
        if self.start:
            cmd += ["-ss", f"{self.start:.6f}"]
        cmd += ["-i", self.audio, "-map", "0:v:0", "-map", "1:a:0"]
        if self.duration:
            cmd += ["-t", f"{self.duration:.6f}"]
        cmd += [
            "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", self.audio_bitrate, "-ar", "44100", "-ac", "2",
            "-shortest", "-f", "mp4",
        ]
        if self.threads:
            cmd += ["-threads", str(self.threads)]
        cmd.append(self.output)
        return cmd

    # -- lifecycle -------------------------------------------------------- #
    def __enter__(self) -> "FrameWriter":
        Path(self.output).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._proc = subprocess.Popen(
                self.command(), stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=self._stderr,
            )
        except OSError as exc:
            raise MusicvidError(f"Could not start ffmpeg: {exc}") from exc
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise MusicvidError("FrameWriter.write() called outside a context manager.")
        array = np.ascontiguousarray(frame, dtype=np.uint8)
        if array.shape != (self.height, self.width, 3):
            raise MusicvidError(
                f"Frame shape {array.shape} does not match "
                f"({self.height}, {self.width}, 3)."
            )
        try:
            self._proc.stdin.write(array.tobytes())
        except (BrokenPipeError, ValueError) as exc:
            raise MusicvidError(self._failure_message()) from exc
        self._frames += 1

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._proc is None:
            return False
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
        code = self._proc.wait()
        if exc_type is None and code != 0:
            raise MusicvidError(self._failure_message())
        if exc_type is None and self._frames == 0:
            raise MusicvidError("No frames were rendered.")
        return False

    # -- diagnostics ------------------------------------------------------ #
    def _failure_message(self) -> str:
        try:
            self._stderr.flush()
            self._stderr.seek(0)
            text = self._stderr.read().decode("utf8", "replace").strip()
        except Exception:                                  # pragma: no cover
            text = ""
        try:
            if os.path.exists(self._stderr.name):
                os.unlink(self._stderr.name)
        except OSError:                                    # pragma: no cover
            pass
        detail = " / ".join(text.splitlines()[-4:]) or "no output from ffmpeg"
        return f"ffmpeg failed while encoding {self.output}: {detail}"


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Scale a float 0..1 canvas to the uint8 frame ffmpeg expects.

    Scales and clamps in place (the caller has finished with the canvas by the
    time it is handed over), which avoids two full-frame temporaries per frame.
    """
    frame *= 255.0
    np.clip(frame, 0.0, 255.0, out=frame)
    return frame.astype(np.uint8)
