"""Timed lyrics as typography, with optional word-level karaoke highlight."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .. import graphics as gfx
from ..lyrics import Line, Lyrics
from ..encode import to_uint8
from .common import RenderContext, Timeline, cover_thumb, progress_bar, scrim, title_block

NAME = "lyrics"
DESCRIPTION = "Timed lyrics over a dimmed cover wash, with optional karaoke highlighting."
REQUIRES_MEDIA = False


def _prepare_lines(ctx: RenderContext, lyrics: Lyrics) -> tuple[list[str], int]:
    """Wrap every line at one shared font size so the layout stays stable."""
    width, height = ctx.size
    portrait = height > width
    max_width = int(width * 0.82)
    size = int(height * (0.072 if portrait else 0.056))
    min_size = int(height * 0.022)
    upper = bool(ctx.cfg.get("uppercase_lyrics"))
    texts = [line.text.upper() if upper else line.text for line in lyrics.lines]

    displays = texts
    while size > min_size:
        displays = [ctx.text.wrap(t, size, max_width, bold=True, max_lines=2) for t in texts]
        widest = 0
        for display in displays:
            for row in display.split("\n"):
                widest = max(widest, ctx.text.measure(row, size, bold=True)[0])
        if widest <= max_width:
            break
        size = int(size * 0.94)
    return displays, size


def _karaoke_layer(ctx: RenderContext, display: str, size: int, words: list[str],
                   active: int) -> np.ndarray:
    """Whole line rendered with past words bright, active word accented."""
    spacing = ctx.text.measure(" ", size, bold=True)[0]
    dim = ctx.palette["subtext_rgb"]
    past_color = ctx.palette["text_rgb"]
    active_color = ctx.palette["accent_rgb"]
    layer = ctx.text.layer(display, size, dim, bold=True).copy()

    def draw_words(indices: list[int], color) -> None:
        if not indices:
            return
        prefix = " ".join(words[:indices[0]])
        x = ctx.text.measure(prefix, size, bold=True)[0] + (spacing if indices[0] else 0)
        for i in indices:
            word_layer = ctx.text.layer(words[i], size, color, bold=True)
            gfx.paste_rgba(layer, word_layer, x, 0)
            x += word_layer.shape[1] + spacing

    draw_words(list(range(active)), past_color)
    draw_words([active], active_color)
    return layer


def render(ctx: RenderContext) -> Iterator[np.ndarray]:
    width, height = ctx.size
    palette = ctx.palette
    cfg = ctx.cfg
    lyrics: Lyrics = ctx.extra.get("lyrics")
    if lyrics is None or not lyrics.lines:
        return

    show_bars = bool(cfg.get("lyrics_bars", True))
    n_bars = max(24, min(320, int(width / 34)))
    timeline = Timeline(ctx.analysis, ctx.cfg_fps, ctx.analysis.duration, bands=n_bars)
    base = gfx.load_background(cfg.get("cover"), width, height, palette)
    scrim(base, top=0.40, bottom=0.45)
    title_block(ctx, base, alpha=0.85)

    portrait = height > width
    displays, size = _prepare_lines(ctx, lyrics)
    line_times = np.array([line.time for line in lyrics.lines], dtype=np.float64)
    center_y = int(height * (0.42 if cfg.get("lyrics_bars", True) else 0.50))
    side_offset = int(height * 0.155)
    small_size = int(size * 0.62)
    karaoke = str(cfg.get("lyrics_style", "line")) == "karaoke" and lyrics.has_word_timings
    show_prev = bool(cfg.get("show_prev", True))
    show_next = bool(cfg.get("show_next", True))
    fade_in = 0.16
    cache: dict[tuple[int, int], np.ndarray] = {}

    span = int(width * 0.92)
    margin_x = max(0, (width - span) // 2)
    pitch = span / n_bars
    bar_w = max(2, int(pitch * 0.68))
    baseline = int(height * (0.94 if portrait else 0.95))
    max_bar_h = int(height * (0.10 if portrait else 0.13))
    colors = gfx.grad_stops(n_bars, palette["bar_stops_rgb"])
    accent = palette["accent_rgb"]

    for i in range(timeline.n):
        t = float(timeline.t[i])
        canvas = base.copy()

        if show_bars:
            pulse = float(timeline.pulse[i])
            gfx.glow_band(canvas, baseline - max_bar_h * 0.5, max_bar_h * 1.3,
                          palette["glow_rgb"], 0.03 + 0.12 * pulse)
            heights = (timeline.bars[i] * max_bar_h).astype(np.int32)
            for b in range(n_bars):
                bar_h = int(heights[b])
                if bar_h < 1:
                    continue
                x0 = margin_x + int(b * pitch)
                x1 = min(width, x0 + bar_w)
                if x1 <= x0:
                    continue
                y0 = baseline - bar_h
                block = canvas[y0:baseline, x0:x1]
                tint = colors[b] * 0.85
                block *= (1.0 - tint)
                block += tint

        index = int(np.searchsorted(line_times, t, side="right")) - 1

        if 0 <= index < len(lyrics.lines):
            current = lyrics.lines[index]
            alpha = min(1.0, max(0.0, (t - current.time) / fade_in))
            rise = int((1.0 - min(1.0, alpha)) * height * 0.012)
            layer = _current_layer(ctx, index, current, displays[index], size, t,
                                   karaoke, cache)
            x = int(width / 2 - layer.shape[1] / 2)
            y = int(center_y - layer.shape[0] / 2) + rise
            gfx.paste(canvas, layer, x, y, alpha)

            if show_prev and index > 0:
                prev = ctx.text.layer(displays[index - 1], small_size,
                                      palette["subtext_rgb"], bold=True)
                gfx.paste(canvas, prev,
                          int(width / 2 - prev.shape[1] / 2),
                          int(center_y - side_offset - prev.shape[0] / 2), 0.42)
            if show_next and index + 1 < len(lyrics.lines):
                nxt = ctx.text.layer(displays[index + 1], small_size,
                                     palette["subtext_rgb"], bold=True)
                gfx.paste(canvas, nxt,
                          int(width / 2 - nxt.shape[1] / 2),
                          int(center_y + side_offset - nxt.shape[0] / 2), 0.42)

        if cfg.get("art") and cfg.get("cover"):
            thumb = int(height * 0.10)
            cover_thumb(ctx, canvas, thumb,
                        int(width * 0.055), height - thumb - int(height * 0.11))

        progress_bar(ctx, canvas, t)
        yield to_uint8(canvas)


def _current_layer(ctx: RenderContext, index: int, line: Line, display: str,
                   size: int, t: float, karaoke: bool,
                   cache: dict) -> np.ndarray:
    """Current line, as plain text or with the active word highlighted."""
    text = display or line.text
    if not karaoke or not line.words or "\n" in text:
        return ctx.text.layer(text, size, ctx.palette["text_rgb"], bold=True)

    words = [word for _, word in line.words]
    active = 0
    for i, (when, _) in enumerate(line.words):
        if t >= when:
            active = i
        else:
            break
    key = (index, active)
    layer = cache.get(key)
    if layer is None:
        layer = _karaoke_layer(ctx, text, size, words, active)
        cache[key] = layer
        if len(cache) > 256:
            cache.clear()
    return layer
