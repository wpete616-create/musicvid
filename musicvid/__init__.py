"""musicvid: turn any song into a music video, over and over.

Three pipelines share one audio analysis pass:

* ``bars``      audio-reactive spectrum bars
* ``lyrics``    timed lyrics typography (optional word-level karaoke)
* ``slideshow`` photos/clips cut on the beat
"""

from __future__ import annotations

__version__ = "0.1.0"

from .audio import Analysis, analyze
from .config import DEFAULTS, PRESETS, base_config
from .styles import DESCRIPTIONS, get_style, style_names

__all__ = [
    "__version__",
    "Analysis",
    "analyze",
    "DEFAULTS",
    "PRESETS",
    "base_config",
    "DESCRIPTIONS",
    "get_style",
    "style_names",
]
