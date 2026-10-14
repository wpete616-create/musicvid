"""Audio-reactive spectrum bars with waveform, reflections and peak caps."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .. import graphics as gfx
from ..encode import to_uint8
from .common import RenderContext, Timeline, cover_thumb, progress_bar, scrim, title_block

NAME = "bars"
DESCRIPTION = "Frequency bars that react to the track, with waveform and cover art."
REQUIRES_MEDIA = False


def render(ctx: RenderContext) -> Iterator[np.ndarray]:
    width, height = ctx.size
    palette = ctx.palette
    cfg = ctx.cfg
    gap = float(cfg.get("bar_gap", 0.30))
    n_bars = int(cfg.get("bars") or max(24, int(width / 28)))
    n_bars = max(8, min(320, n_bars))
    timeline = Timeline(ctx.analysis, ctx.cfg_fps, ctx.analysis.duration, bands=n_bars)

    base = gfx.load_background(cfg.get("cover"), width, height, palette)
    if cfg.get("scrim", True):
        scrim(base, top=0.34, bottom=0.22)
    title_block(ctx, base)
    if cfg.get("art") and cfg.get("cover"):
        thumb = int(height * 0.12)
        cover_thumb(ctx, base, thumb, width - thumb - int(width * 0.055), int(height * 0.055))

    portrait = height > width
    baseline = int(height * (0.78 if portrait else 0.80))
    max_bar_h = int(height * (0.34 if portrait else 0.42))
    span = int(width * 0.92)
    margin_x = max(0, (width - span) // 2)
    pitch = span / n_bars
    bar_w = max(2, int(pitch * (1.0 - gap)))
    reflection_h = int(height * 0.15)

    colors = gfx.grad_stops(n_bars, palette["bar_stops_rgb"])
    accent = palette["accent_rgb"]
    glow = palette["glow_rgb"]
    wave_color = palette["wave_rgb"]

    show_wave = bool(cfg.get("waveform", True))
    wave_cols = int(width * 0.94)
    wave_x = max(0, (width - wave_cols) // 2)
    wave_amp = int(height * (0.035 if portrait else 0.05))
    # sit the oscilloscope in the empty upper third so the bars stay readable
    wave_cy = int(height * (0.30 if portrait else 0.34))
    wave_rows = np.arange(wave_cy - wave_amp - 3, wave_cy + wave_amp + 4)[:, None]

    show_reflection = bool(cfg.get("reflection", True))
    show_glow = bool(cfg.get("glow", True))
    do_flash = bool(cfg.get("flash", False))
    tint_strength = np.clip((timeline.centroid - 0.35) * 1.2, 0.0, 1.0) * 0.22
    cap_h = max(1, int(height * 0.0035))
    peak_h = max(2, int(height * 0.0028))
    # precomputed gradients, sliced to length: cheaper than building one per bar
    upper_ramp = np.linspace(0.72, 1.0, max(1, max_bar_h), dtype=np.float32)[:, None, None]
    lower_fade = np.linspace(0.30, 0.0, max(1, reflection_h), dtype=np.float32)[:, None, None]

    for i in range(timeline.n):
        canvas = base.copy()
        pulse = float(timeline.pulse[i])
        heights = np.clip(timeline.bars[i] * max_bar_h, 0.0, max_bar_h)
        caps = np.clip(timeline.peaks[i] * max_bar_h, 0.0, max_bar_h)
        pixel_heights = heights.astype(np.int32)
        mix = float(tint_strength[i])
        frame_colors = colors * (1.0 - mix) + accent[None, :] * mix

        if show_glow:
            gfx.glow_band(canvas, baseline, height * 0.10, glow, 0.05 + 0.22 * pulse)

        for b in range(n_bars):
            bar_h = int(pixel_heights[b])
            x0 = margin_x + int(b * pitch)
            x1 = min(width, x0 + bar_w)
            if x1 <= x0:
                continue
            color = frame_colors[b]

            if bar_h >= 1:
                y0 = baseline - bar_h
                block = canvas[y0:baseline, x0:x1]
                # screen blend, in place: r + t - r*t == r*(1-t) + t
                tint = upper_ramp[max_bar_h - bar_h:] * color
                block *= (1.0 - tint)
                block += tint
                cap = canvas[y0:y0 + cap_h, x0:x1]
                cap *= 0.20
                cap += color * 0.90

            if show_reflection and bar_h >= 1:
                refl_h = min(bar_h, reflection_h)
                y0 = baseline + 1
                block = canvas[y0:y0 + refl_h, x0:x1]
                tint = lower_fade[:refl_h] * color
                block *= (1.0 - tint)
                block += tint

            peak = int(caps[b])
            if peak > 2:
                y0 = max(0, baseline - peak)
                block = canvas[y0:y0 + peak_h, x0:x1]
                block *= 0.30
                block += accent * 0.70

        if show_wave:
            low, high = timeline.waveform(i, wave_cols)
            top = np.rint(wave_cy - high * wave_amp).astype(np.int32)
            bottom = np.rint(wave_cy - low * wave_amp).astype(np.int32)
            mask = (wave_rows >= top[None, :]) & (wave_rows <= bottom[None, :])
            region = canvas[wave_cy - wave_amp - 3:wave_cy + wave_amp + 4, wave_x:wave_x + wave_cols]
            strength = 0.18 + 0.60 * float(timeline.rms[i])
            tint = wave_color * strength
            # note: a boolean mask indexes a copy, so the result must be written
            # back -- in-place ops on `region[mask]` would be discarded
            selected = region[mask]
            region[mask] = selected * (1.0 - tint) + tint

        if do_flash and float(timeline.is_downbeat[i]):
            canvas += accent[None, None, :] * 0.08
        elif pulse > 0.01:
            canvas += accent[None, None, :] * (0.035 * pulse)

        progress_bar(ctx, canvas, float(timeline.t[i]))
        yield to_uint8(canvas)
