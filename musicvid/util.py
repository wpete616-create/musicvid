"""Small shared helpers: external binaries, sizing, logging, fonts."""

from __future__ import annotations

import functools
import os
import shutil
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS = PACKAGE_ROOT / "assets"
PROJECT_ROOT = PACKAGE_ROOT.parent


class MusicvidError(RuntimeError):
    """A user-facing failure (bad input, missing tool, unsupported media)."""


# --------------------------------------------------------------------------- #
# external binaries
# --------------------------------------------------------------------------- #

def _candidate_paths(name: str) -> list[Path]:
    return [
        PROJECT_ROOT / ".tools" / "current" / name,
        PROJECT_ROOT / ".tools" / "bin" / name,
        Path.home() / ".local" / "bin" / name,
    ]


@functools.lru_cache(maxsize=None)
def find_binary(name: str) -> str:
    """Locate ``ffmpeg``/``ffprobe``.

    Order: ``$MUSICVID_<NAME>`` override, ``PATH``, then a project-local static
    build under ``.tools/`` (see ``tools/install_ffmpeg.sh``).
    """
    override = os.environ.get(f"MUSICVID_{name.upper()}")
    if override:
        if Path(override).exists():
            return override
        raise MusicvidError(
            f"$MUSICVID_{name.upper()} points at {override!r}, which does not exist."
        )

    found = shutil.which(name)
    if found:
        return found

    for candidate in _candidate_paths(name):
        if candidate.exists():
            return str(candidate)

    raise MusicvidError(
        f"Could not find {name!r}. Install it (e.g. `apt install ffmpeg`, "
        "`brew install ffmpeg`) or run tools/install_ffmpeg.sh, or set "
        f"MUSICVID_{name.upper()} to its full path."
    )


def ffmpeg() -> str:
    return find_binary("ffmpeg")


def ffprobe() -> str:
    return find_binary("ffprobe")


# --------------------------------------------------------------------------- #
# misc helpers
# --------------------------------------------------------------------------- #

SIZE_PRESETS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "4k": (3840, 2160),
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "vertical": (1080, 1920),
    "square": (1080, 1080),
}


def parse_size(value) -> tuple[int, int]:
    """Accept ``1920x1080``, ``1080x1920`` or a preset name."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    text = str(value).strip().lower()
    if text in SIZE_PRESETS:
        return SIZE_PRESETS[text]
    for sep in ("x", "*", ":"):
        if sep in text:
            parts = text.split(sep)
            if len(parts) == 2:
                try:
                    w, h = int(parts[0]), int(parts[1])
                except ValueError:
                    break
                if w > 0 and h > 0:
                    # even numbers keep libx264 happy
                    return w - (w % 2), h - (h % 2)
    raise MusicvidError(
        f"Unrecognised size {value!r}; use WxH (e.g. 1920x1080) or one of "
        f"{', '.join(sorted(SIZE_PRESETS))}."
    )


def parse_timestamp(value) -> float | None:
    """Accept ``12``, ``12.5``, ``1:03`` or ``1:02:03.5`` seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    negative = text.startswith("-")
    text = text.lstrip("-")
    parts = text.split(":")
    if len(parts) > 3:
        raise MusicvidError(f"Unrecognised timestamp {value!r}.")
    try:
        total = 0.0
        for part in parts:
            total = total * 60.0 + float(part)
    except ValueError as exc:
        raise MusicvidError(f"Unrecognised timestamp {value!r}.") from exc
    return -total if negative else total


def format_seconds(value: float) -> str:
    minutes, seconds = divmod(max(0.0, float(value)), 60.0)
    return f"{int(minutes):d}:{seconds:04.1f}"


def log(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


class Progress:
    """Single-line progress meter written to stderr."""

    def __init__(self, total: int, label: str, enabled: bool = True):
        self.total = max(1, int(total))
        self.label = label
        self.enabled = enabled
        self._last = -1

    def update(self, done: int, extra: str = "") -> None:
        if not self.enabled:
            return
        done = max(0, min(self.total, int(done)))
        pct = int(done * 100 / self.total)
        if pct == self._last and done != self.total:
            return
        self._last = pct
        width = 28
        filled = int(width * pct / 100)
        bar = "#" * filled + "-" * (width - filled)
        suffix = f" {extra}" if extra else ""
        print(f"\r{self.label} [{bar}] {pct:3d}%{suffix}", end="", file=sys.stderr, flush=True)

    def done(self, extra: str = "") -> None:
        if not self.enabled:
            return
        self.update(self.total, extra)
        print("", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# fonts
# --------------------------------------------------------------------------- #

_BUNDLED = {
    False: ASSETS / "fonts" / "DejaVuSans.ttf",
    True: ASSETS / "fonts" / "DejaVuSans-Bold.ttf",
}

_SYSTEM = {
    False: [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
    True: [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ],
}


@functools.lru_cache(maxsize=None)
def font_path(bold: bool = False, override: str | None = None) -> str | None:
    """Resolve a font file.

    Preference order: explicit ``override``, bundled DejaVu, common system
    fonts. ``None`` means "let Pillow use its built-in default".
    """
    if override:
        path = Path(override).expanduser()
        if not path.exists():
            raise MusicvidError(f"Font file not found: {override}")
        return str(path)

    bundled = _BUNDLED[bold]
    if bundled.exists():
        return str(bundled)

    for candidate in _SYSTEM[bold]:
        if Path(candidate).exists():
            return candidate

    return None


def have_font(bold: bool = False) -> bool:
    return font_path(bold) is not None
