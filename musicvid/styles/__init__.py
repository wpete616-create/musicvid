"""Render styles and their registry."""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

from ..util import MusicvidError
from .common import RenderContext
from . import bars, lyrics, slideshow

REGISTRY: dict[str, Callable[[RenderContext], Iterator[np.ndarray]]] = {
    bars.NAME: bars.render,
    lyrics.NAME: lyrics.render,
    slideshow.NAME: slideshow.render,
}

DESCRIPTIONS = {
    bars.NAME: bars.DESCRIPTION,
    lyrics.NAME: lyrics.DESCRIPTION,
    slideshow.NAME: slideshow.DESCRIPTION,
}

# Which styles consume --media/--media-dir; the other ones ignore them.
REQUIRES_MEDIA = {
    bars.NAME: bars.REQUIRES_MEDIA,
    lyrics.NAME: lyrics.REQUIRES_MEDIA,
    slideshow.NAME: slideshow.REQUIRES_MEDIA,
}

__all__ = ["REGISTRY", "DESCRIPTIONS", "REQUIRES_MEDIA", "RenderContext",
           "get_style", "style_names", "style_requires_media"]


def style_names() -> list[str]:
    return sorted(REGISTRY)


def style_requires_media(name: str) -> bool:
    return bool(REQUIRES_MEDIA.get(name, False))


def get_style(name: str) -> Callable[[RenderContext], Iterator[np.ndarray]]:
    if name not in REGISTRY:
        raise MusicvidError(
            f"Unknown style {name!r}; available: {', '.join(style_names())}."
        )
    return REGISTRY[name]
