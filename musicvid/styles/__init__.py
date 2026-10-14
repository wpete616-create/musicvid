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

__all__ = ["REGISTRY", "DESCRIPTIONS", "RenderContext", "get_style", "style_names"]


def style_names() -> list[str]:
    return sorted(REGISTRY)


def get_style(name: str) -> Callable[[RenderContext], Iterator[np.ndarray]]:
    if name not in REGISTRY:
        raise MusicvidError(
            f"Unknown style {name!r}; available: {', '.join(style_names())}."
        )
    return REGISTRY[name]
