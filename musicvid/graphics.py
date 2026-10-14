"""Drawing primitives: palettes, backgrounds, numpy compositing and text.

Frames are float32 ``(h, w, 3)`` arrays in 0..1. Text layers are float32
``(h, w, 4)`` where RGB is the flat colour and alpha is coverage.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from .util import MusicvidError, font_path

RGB = tuple[float, float, float]


def rgb(*values) -> np.ndarray:
    """Accept ``(1, 0.5, 0)``, ``"#ff8000"`` or ``"ff8000"``."""
    if len(values) == 3:
        return np.asarray(values, np.float32)
    if len(values) == 1:
        value = values[0]
        if isinstance(value, (tuple, list)):
            return np.asarray(value, np.float32)
        text = str(value).strip().lstrip("#")
        if len(text) == 3:
            text = "".join(c * 2 for c in text)
        if len(text) != 6:
            raise MusicvidError(f"Unrecognised colour {value!r}; use #rrggbb.")
        return np.asarray([int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)
    raise MusicvidError(f"Unrecognised colour {values!r}")


PALETTES: dict[str, dict] = {
    "neon": {
        "bg_top": "#0a0618", "bg_bottom": "#02010a",
        "bar_stops": ["#ff2e88", "#ff9a3c", "#2ee6ff"],
        "wave": "#cfe4ff", "text": "#ffffff", "subtext": "#9aa6c4",
        "accent": "#ff3d7f",
        "cover_blur": 46, "cover_mix": 0.55, "darken": 0.55,
        "glow": "#ff2e88",
    },
    "clean": {
        "bg_top": "#101216", "bg_bottom": "#05060a",
        "bar_stops": ["#ffffff", "#e6ecf5", "#bcd2ec"],
        "wave": "#ffffff", "text": "#ffffff", "subtext": "#8f98a8",
        "accent": "#6fa8ff",
        "cover_blur": 60, "cover_mix": 0.32, "darken": 0.68,
        "glow": "#6fa8ff",
    },
    "ember": {
        "bg_top": "#1a0a04", "bg_bottom": "#060201",
        "bar_stops": ["#ff3b00", "#ff9500", "#ffd76e"],
        "wave": "#ffe0b0", "text": "#fff4e6", "subtext": "#c79d78",
        "accent": "#ff7a18",
        "cover_blur": 50, "cover_mix": 0.5, "darken": 0.6,
        "glow": "#ff5a00",
    },
    "mono": {
        "bg_top": "#141414", "bg_bottom": "#000000",
        "bar_stops": ["#f2f2f2", "#cfcfcf", "#8f8f8f"],
        "wave": "#ffffff", "text": "#ffffff", "subtext": "#8c8c8c",
        "accent": "#ffffff",
        "cover_blur": 70, "cover_mix": 0.22, "darken": 0.72,
        "glow": "#ffffff",
    },
    "vapor": {
        "bg_top": "#12003a", "bg_bottom": "#02000f",
        "bar_stops": ["#ff4fd8", "#9d6bff", "#43e8ff"],
        "wave": "#ffd9f6", "text": "#ffffff", "subtext": "#b79ddb",
        "accent": "#ff4fd8",
        "cover_blur": 40, "cover_mix": 0.6, "darken": 0.5,
        "glow": "#ff4fd8",
    },
}


def resolve_palette(name: str, **overrides) -> dict:
    if name not in PALETTES:
        raise MusicvidError(
            f"Unknown palette {name!r}; try one of {', '.join(sorted(PALETTES))}."
        )
    palette = dict(PALETTES[name])
    palette.update({k: v for k, v in overrides.items() if v is not None})
    palette["name"] = name
    palette["bar_stops_rgb"] = [rgb(c) for c in palette["bar_stops"]]
    for key in ("bg_top", "bg_bottom", "wave", "text", "subtext", "accent", "glow"):
        palette[key + "_rgb"] = rgb(palette[key])
    return palette


# --------------------------------------------------------------------------- #
# compositing helpers
# --------------------------------------------------------------------------- #

def paste(dst: np.ndarray, src: np.ndarray, x: int, y: int, alpha: float = 1.0) -> None:
    """Alpha-composite an RGBA layer onto an RGB canvas, clipped to bounds."""
    if src.size == 0 or alpha <= 0:
        return
    sh, sw = src.shape[:2]
    dh, dw = dst.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(dw, int(x) + sw), min(dh, int(y) + sh)
    if x1 <= x0 or y1 <= y0:
        return
    patch = src[y0 - y:y1 - y, x0 - x:x1 - x]
    a = patch[..., 3:4] * float(alpha)
    region = dst[y0:y1, x0:x1]
    dst[y0:y1, x0:x1] = region * (1.0 - a) + patch[..., :3] * a


def paste_rgba(dst: np.ndarray, src: np.ndarray, x: int, y: int, alpha: float = 1.0) -> None:
    """Alpha-composite an RGBA layer onto another RGBA layer."""
    if src.size == 0 or alpha <= 0:
        return
    sh, sw = src.shape[:2]
    dh, dw = dst.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(dw, int(x) + sw), min(dh, int(y) + sh)
    if x1 <= x0 or y1 <= y0:
        return
    patch = src[y0 - y:y1 - y, x0 - x:x1 - x]
    region = dst[y0:y1, x0:x1]
    src_a = patch[..., 3:4] * float(alpha)
    dst_a = region[..., 3:4]
    out_a = src_a + dst_a * (1.0 - src_a)
    safe = np.maximum(out_a, 1e-6)
    rgb = (patch[..., :3] * src_a + region[..., :3] * dst_a * (1.0 - src_a)) / safe
    region[..., :3] = rgb
    region[..., 3:4] = out_a


def glow_band(dst: np.ndarray, center_y: float, radius: float,
              color, alpha: float = 0.1) -> None:
    """Soft horizontal glow across the full width, centred on ``center_y``.

    A hard-edged rectangle reads as a rendering artefact, so the tint falls off
    with a Gaussian profile instead.
    """
    if alpha <= 0 or radius <= 0:
        return
    height, width = dst.shape[:2]
    top = max(0, int(center_y - radius))
    bottom = min(height, int(center_y + radius))
    if bottom <= top:
        return
    rows = np.arange(top, bottom, dtype=np.float32) - float(center_y)
    profile = np.exp(-0.5 * (rows / max(1.0, radius * 0.45)) ** 2)
    tint = profile[:, None, None] * (np.asarray(color, np.float32) * float(alpha))
    region = dst[top:bottom]
    region *= (1.0 - tint)
    region += tint


def grad_stops(count: int, stops: list[np.ndarray]) -> np.ndarray:
    """Interpolate ``stops`` into ``count`` colours."""
    if count <= 0:
        return np.zeros((0, 3), np.float32)
    if len(stops) == 1:
        return np.repeat(stops[0][None, :], count, axis=0)
    positions = np.linspace(0.0, 1.0, count)
    scaled = positions * (len(stops) - 1)
    out = np.empty((count, 3), np.float32)
    for i in range(len(stops) - 1):
        lo, hi = scaled >= i, scaled <= i + 1
        mask = lo & hi
        if not np.any(mask):
            continue
        t = (scaled[mask] - i)[:, None]
        out[mask] = stops[i][None, :] * (1 - t) + stops[i + 1][None, :] * t
    return out


@functools.lru_cache(maxsize=8)
def vertical_gradient(width: int, height: int, top: tuple, bottom: tuple) -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    top_arr = np.asarray(top, np.float32)[None, None, :]
    bottom_arr = np.asarray(bottom, np.float32)[None, None, :]
    grad = top_arr * (1 - ramp) + bottom_arr * ramp
    return np.repeat(grad, width, axis=1).astype(np.float32)


@functools.lru_cache(maxsize=8)
def vignette(width: int, height: int, strength: float = 0.55) -> np.ndarray:
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radius = np.sqrt(xs * xs + ys * ys) / math.sqrt(2.0)
    mask = 1.0 - strength * np.clip(radius - 0.25, 0.0, 1.0) ** 1.6
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def fit_cover(image: Image.Image, width: int, height: int) -> Image.Image:
    """Scale + centre-crop an image so it exactly fills ``width`` x ``height``."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def fit_contain(image: Image.Image, width: int, height: int) -> tuple[Image.Image, Image.Image]:
    """Return ``(background, foreground)`` for a letterboxed fit."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    cover = fit_cover(image, width, height)
    background = cover.filter(ImageFilter.GaussianBlur(max(8, width // 40)))
    scale = min(width / image.width, height / image.height)
    fg_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    foreground = image.resize(fg_size, Image.LANCZOS)
    return background, foreground


def load_background(cover: str | Path | None, width: int, height: int,
                    palette: dict) -> np.ndarray:
    """Base canvas: gradient, plus a blurred cover-art wash when supplied."""
    canvas = vertical_gradient(
        width, height, tuple(palette["bg_top_rgb"]), tuple(palette["bg_bottom_rgb"]),
    ).copy()

    if cover:
        path = Path(cover).expanduser()
        if not path.exists():
            raise MusicvidError(f"Cover image not found: {cover}")
        with Image.open(path) as image:
            wash = fit_cover(image, width, height)
        wash = wash.filter(ImageFilter.GaussianBlur(float(palette["cover_blur"])))
        wash_arr = np.asarray(wash, np.float32) / 255.0
        mix = float(palette["cover_mix"])
        canvas = canvas * (1.0 - mix) + wash_arr * mix

    darken = float(palette["darken"])
    if darken > 0:
        canvas = canvas * (1.0 - darken * 0.55)

    canvas *= vignette(width, height)[..., None]
    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=64)
def _font(size: int, bold: bool, override: str | None) -> ImageFont.FreeTypeFont:
    size = max(6, int(size))
    path = font_path(bold, override)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                    # Pillow < 10.1
        return ImageFont.load_default()


class TextRenderer:
    """Renders text to cached RGBA layers."""

    def __init__(self, font_override: str | None = None):
        self.font_override = font_override
        self._cache: dict[tuple, np.ndarray] = {}

    def layer(self, text: str, size: int, color, bold: bool = False,
              tracking: float = 0.0, stroke: int = 0) -> np.ndarray:
        key = (text, int(size), tuple(np.asarray(color, np.float32).tolist()),
               bold, round(float(tracking), 4), int(stroke))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        font = _font(int(size), bold, self.font_override)
        color_arr = np.asarray(color, np.float32)
        lines = text.split("\n") or [""]
        tracking_px = float(tracking) * size

        widths = []
        for line in lines:
            if tracking_px:
                widths.append(sum(font.getlength(ch) for ch in line)
                              + tracking_px * max(0, len(line) - 1))
            else:
                widths.append(font.getlength(line))
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        pad = max(2, int(stroke) + 2)

        canvas_w = int(math.ceil(max(widths) if widths else 1)) + pad * 2
        canvas_h = int(math.ceil(line_height * len(lines))) + pad * 2
        mask = Image.new("L", (max(1, canvas_w), max(1, canvas_h)), 0)
        draw = ImageDraw.Draw(mask)

        y = float(pad)
        for line, width in zip(lines, widths):
            x = float(pad)
            if tracking_px:
                for ch in line:
                    draw.text((x, y), ch, font=font, fill=255)
                    x += font.getlength(ch) + tracking_px
            else:
                draw.text((x, y), line, font=font, fill=255)
            y += line_height

        alpha = np.asarray(mask, np.float32) / 255.0
        layer = np.empty((mask.height, mask.width, 4), np.float32)
        layer[..., :3] = color_arr
        layer[..., 3] = alpha
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = layer
        return layer

    def draw(self, canvas: np.ndarray, text: str, size: int, color, *,
             bold: bool = False, tracking: float = 0.0, alpha: float = 1.0,
             x: int | None = None, y: int | None = None,
             center: tuple[int, int] | None = None) -> tuple[int, int]:
        layer = self.layer(text, size, color, bold=bold, tracking=tracking)
        if x is None or y is None:
            if center is None:
                raise MusicvidError("TextRenderer.draw needs x/y or center.")
            x = int(center[0] - layer.shape[1] / 2)
            y = int(center[1] - layer.shape[0] / 2)
        paste(canvas, layer, int(x), int(y), alpha)
        return layer.shape[1], layer.shape[0]

    def measure(self, text: str, size: int, bold: bool = False,
                tracking: float = 0.0) -> tuple[int, int]:
        layer = self.layer(text, size, np.zeros(3, np.float32), bold=bold, tracking=tracking)
        return layer.shape[1], layer.shape[0]

    def wrap(self, text: str, size: int, max_width: int, bold: bool = False,
             max_lines: int = 3) -> str:
        words = text.split()
        if not words:
            return ""
        font = _font(int(size), bold, self.font_override)
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.getlength(candidate) <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1].rstrip() + "..."
        return "\n".join(lines)

    def fit_size(self, text: str, max_width: int, start_size: int,
                 bold: bool = False, min_size: int = 18) -> int:
        size = int(start_size)
        while size > min_size:
            font = _font(size, bold, self.font_override)
            widest = max(font.getlength(line) for line in text.split("\n")) if text else 0
            if widest <= max_width:
                return size
            size = int(size * 0.94)
        return min_size
