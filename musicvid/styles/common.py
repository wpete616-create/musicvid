"""Shared render context: timeline pre-computation, titles, overlays."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import graphics as gfx
from ..audio import Analysis
from ..graphics import TextRenderer


@dataclass
class RenderContext:
    analysis: Analysis
    cfg: dict
    size: tuple[int, int]
    palette: dict
    text: TextRenderer
    progress_enabled: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    @property
    def cfg_fps(self) -> float:
        return float(self.cfg.get("fps", 30))

    def opt(self, key: str, default=None):
        value = self.cfg.get(key, default)
        return default if value is None else value


def resample_bands(bands: np.ndarray, count: int) -> np.ndarray:
    """Interpolate the band axis so the visual grid can differ from analysis."""
    source = bands.shape[1]
    if count <= 0 or count == source:
        return bands
    if count == 1:
        return bands.mean(axis=1, keepdims=True)
    positions = np.linspace(0.0, source - 1.0, count)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, source - 1)
    weight = (positions - lower)[None, :]
    return (bands[:, lower] * (1.0 - weight) + bands[:, upper] * weight).astype(np.float32)


class Timeline:
    """Per-output-frame arrays: bands, envelopes, beats, waveform windows."""

    def __init__(self, analysis: Analysis, fps: float, duration: float,
                 attack: float = 0.55, decay: float = 0.10,
                 bands: int | None = None):
        self.analysis = analysis
        self.fps = float(fps)
        self.duration = float(duration)
        self.n = max(1, int(round(self.duration * self.fps)))
        self.t = (np.arange(self.n, dtype=np.float64) + 0.5) / self.fps

        index = np.clip(
            np.rint(self.t * analysis.frame_rate).astype(np.int64),
            0, analysis.n_frames - 1,
        )
        self.index = index
        self.bands = analysis.spectrum[index]
        if bands and int(bands) != self.bands.shape[1]:
            self.bands = resample_bands(self.bands, int(bands))
        self.rms = analysis.rms[index]
        self.centroid = analysis.centroid[index]
        self.onset = analysis.onset[index]
        self.bars = _envelope(self.bands, attack=attack, decay=decay)
        self.peaks = _peak_caps(self.bars)
        self.pulse, self.is_beat, self.beat_number = _beat_track(
            analysis.beat_times, self.t, analysis.beat_period,
        )
        self.is_downbeat = _event_flags(analysis.downbeat_times, self.t, self.fps)
        self.sample_rate = analysis.sr

    @property
    def n_bands(self) -> int:
        return int(self.bands.shape[1])

    def waveform(self, frame: int, columns: int, window_seconds: float = 0.15,
                 samples_per_column: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """Min/max sample value per column over a short window of audio.

        The window must hold several samples per column, otherwise the trace
        collapses to a one-pixel hairline instead of a readable envelope.
        """
        samples = self.analysis.samples
        i0 = max(0, int(self.t[frame] * self.sample_rate))
        span = max(int(self.sample_rate * window_seconds),
                   int(self.sample_rate / self.fps),
                   columns * max(1, samples_per_column))
        chunk = samples[i0:i0 + span]
        if chunk.size < columns:
            chunk = np.pad(chunk, (0, columns - chunk.size))
        per_column = max(1, chunk.size // columns)
        usable = per_column * columns
        block = chunk[:usable].reshape(columns, per_column)
        return block.min(axis=1), block.max(axis=1)


def _envelope(bands: np.ndarray, attack: float, decay: float) -> np.ndarray:
    """Per-band attack/decay follower so bars punch and settle like VU meters."""
    out = np.empty_like(bands)
    env = np.zeros(bands.shape[1], np.float32)
    for i in range(bands.shape[0]):
        target = bands[i]
        rising = target > env
        env = np.where(
            rising,
            env + (target - env) * attack,
            env + (target - env) * decay,
        )
        out[i] = env
    return out


def _peak_caps(env: np.ndarray, fall: float = 0.010) -> np.ndarray:
    out = np.empty_like(env)
    current = np.zeros(env.shape[1], np.float32)
    for i in range(env.shape[0]):
        current = np.maximum(current - fall, env[i])
        out[i] = current
    return out


def _beat_track(beat_times: np.ndarray, times: np.ndarray,
                period: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exponentially decaying pulse plus beat flags for every output frame."""
    pulse = np.zeros(times.shape[0], np.float32)
    is_beat = np.zeros(times.shape[0], bool)
    number = np.full(times.shape[0], -1, np.int32)
    if beat_times.size == 0 or times.size == 0:
        return pulse, is_beat, number

    dt = float(times[1] - times[0]) if times.size > 1 else 1.0
    half_frame = 0.5 * dt
    tau = float(np.clip(period * 0.45, 0.08, 0.22))
    for i, beat in enumerate(beat_times):
        if beat < times[0] - half_frame or beat > times[-1] + half_frame:
            continue
        start = min(int(np.searchsorted(times, beat - half_frame)), times.size - 1)
        if abs(float(times[start]) - beat) <= half_frame:
            is_beat[start] = True
            number[start] = i
        stop = int(np.searchsorted(times, beat + tau * 5))
        if stop <= start:
            continue
        window = times[start:stop] - beat
        decayed = np.exp(-window / tau).astype(np.float32)
        np.maximum(pulse[start:stop], decayed, out=pulse[start:stop])
    return pulse, is_beat, number


def _event_flags(event_times: np.ndarray, times: np.ndarray, fps: float) -> np.ndarray:
    flags = np.zeros(times.shape[0], bool)
    if event_times.size == 0 or times.size == 0:
        return flags
    idx = np.rint(event_times * fps).astype(int)
    idx = idx[(idx >= 0) & (idx < times.size)]
    flags[idx] = True
    return flags


# --------------------------------------------------------------------------- #
# overlays shared by the styles
# --------------------------------------------------------------------------- #

def scrim(canvas: np.ndarray, top: float = 0.32, bottom: float = 0.28,
          span: float = 0.42) -> None:
    """Darken the top/bottom of the frame so text stays legible."""
    height = canvas.shape[0]
    n = max(1, int(height * span))
    if top > 0:
        canvas[:n] *= np.linspace(1.0 - top, 1.0, n, dtype=np.float32)[:, None, None]
    if bottom > 0:
        canvas[-n:] *= np.linspace(1.0, 1.0 - bottom, n, dtype=np.float32)[:, None, None]


def title_block(ctx: RenderContext, canvas: np.ndarray, alpha: float = 1.0) -> None:
    """Title + artist in the top-left corner."""
    title = ctx.cfg.get("title")
    artist = ctx.cfg.get("artist")
    if not title and not artist:
        return
    width, height = ctx.size
    margin = int(width * 0.055)
    y = int(height * 0.055)
    shadow = np.zeros(3, np.float32)

    if title:
        size = int(height * (0.052 if height >= width else 0.038))
        title = str(title)
        for offset in (2, 1):
            ctx.text.draw(canvas, title, size,
                          shadow, bold=True, tracking=0.01, alpha=0.22 * alpha,
                          x=margin + offset, y=y + offset)
        _, block_h = ctx.text.draw(canvas, title, size,
                                   ctx.palette["text_rgb"], bold=True, tracking=0.01,
                                   alpha=alpha, x=margin, y=y)
        y += int(block_h * 0.95)

    if artist:
        size = int(height * (0.024 if height >= width else 0.020))
        label = str(artist).upper()
        ctx.text.draw(canvas, label, size, ctx.palette["subtext_rgb"],
                      tracking=0.16, alpha=0.95 * alpha, x=margin, y=y)


@functools.lru_cache(maxsize=8)
def _cover_thumb(path: str, size: int, radius: float) -> np.ndarray:
    with gfx.open_image(path) as image:
        square = gfx.fit_cover(image, size, size)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=int(size * radius), fill=255,
    )
    rgb_arr = np.asarray(square, np.float32) / 255.0
    alpha = np.asarray(mask, np.float32)[..., None] / 255.0
    return np.concatenate([rgb_arr, alpha], axis=2)


def cover_thumb(ctx: RenderContext, canvas: np.ndarray, size: int,
                x: int, y: int, alpha: float = 1.0) -> None:
    cover = ctx.cfg.get("cover")
    if not cover or not Path(cover).exists():
        return
    layer = _cover_thumb(str(cover), int(size), 0.14)
    gfx.paste(canvas, layer, int(x), int(y), alpha)


def progress_bar(ctx: RenderContext, canvas: np.ndarray, t: float,
                 height: int | None = None) -> None:
    if not ctx.cfg.get("progress", True):
        return
    width, frame_h = ctx.size
    bar_h = height or max(3, int(frame_h * 0.0045))
    track = 0.18
    canvas[-bar_h:, :] *= (1.0 - track)
    frac = 0.0 if ctx.analysis.duration <= 0 else min(1.0, max(0.0, t / ctx.analysis.duration))
    end = int(width * frac)
    if end > 0:
        canvas[-bar_h:, :end] = (
            canvas[-bar_h:, :end] * 0.25
            + ctx.palette["accent_rgb"][None, None, :] * 0.75
        )

