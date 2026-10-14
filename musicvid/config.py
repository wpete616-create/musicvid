"""Settings: defaults, built-in presets and JSON config files.

Resolution order (later wins): DEFAULTS -> preset -> --config file -> CLI flags.
"""

from __future__ import annotations

import json
from pathlib import Path

from .util import MusicvidError

DEFAULTS: dict = {
    # output
    "style": "bars",
    "palette": "neon",
    "seed": 0,
    "size": "1920x1080",
    "fps": 30,
    "crf": 20,
    "encoder_preset": "medium",
    "audio_bitrate": "192k",
    "threads": None,           # None = let x264 choose; lower it to cut memory use
    # trim
    "start": None,
    "end": None,
    # analysis
    "bands": 48,
    # shared look
    "title": None,
    "artist": None,
    "cover": None,
    "font": None,
    "art": False,
    "progress": True,
    "scrim": True,
    # palette overrides
    "bg_top": None,
    "bg_bottom": None,
    "accent": None,
    "glow_color": None,
    "cover_mix": None,
    "darken": None,
    # bars style
    "bars": None,              # None = auto from frame width
    "bar_gap": 0.30,
    "waveform": True,
    "reflection": True,
    "glow": True,
    # slideshow style
    "media": [],
    "media_dir": None,
    "cut_every": 8,
    "cut_on": "downbeats",
    "transition": 0.4,
    "zoom": 0.07,
    "fit": "cover",
    "flash": False,
    "shuffle": False,
    # lyrics style
    "lyrics": None,
    "lyrics_style": "line",    # line | karaoke
    "show_next": True,
    "show_prev": True,
    "lyrics_bars": True,
    "uppercase_lyrics": False,
}

PRESETS: dict[str, dict] = {
    # audio-reactive bars
    "neon": {"style": "bars", "palette": "neon"},
    "clean": {"style": "bars", "palette": "clean", "waveform": False},
    "ember": {"style": "bars", "palette": "ember"},
    "mono": {"style": "bars", "palette": "mono", "waveform": False},
    "vapor": {"style": "bars", "palette": "vapor"},
    # lyrics
    "lyric": {"style": "lyrics", "palette": "clean", "lyrics_style": "line"},
    "karaoke": {"style": "lyrics", "palette": "neon", "lyrics_style": "karaoke"},
    # beat-cut slideshow
    "cut": {"style": "slideshow", "palette": "mono", "transition": 0.0,
            "cut_every": 4, "zoom": 0.0},
    "cinema": {"style": "slideshow", "palette": "ember", "transition": 0.6,
               "cut_every": 8},
}

VALID_STYLES = ("bars", "slideshow", "lyrics")


def preset_names() -> list[str]:
    return sorted(PRESETS)


def apply_preset(cfg: dict, name: str) -> dict:
    if name not in PRESETS:
        raise MusicvidError(
            f"Unknown preset {name!r}; available: {', '.join(preset_names())}."
        )
    out = dict(cfg)
    out.update(PRESETS[name])
    out["preset"] = name
    return out


def load_config(path: str) -> tuple[str | None, dict]:
    """Read a JSON config file; returns ``(preset, settings)``.

    Files are produced by ``--write-config`` but are plain JSON and can be
    hand-edited; unknown keys are rejected so typos never pass silently.
    """
    file = Path(path).expanduser()
    if not file.exists():
        raise MusicvidError(f"Config file not found: {path}")
    try:
        data = json.loads(file.read_text(encoding="utf8"))
    except json.JSONDecodeError as exc:
        raise MusicvidError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise MusicvidError(f"{path} must contain a JSON object.")

    preset = data.pop("preset", None)
    unknown = [key for key in data if key not in DEFAULTS]
    if unknown:
        raise MusicvidError(
            f"Unknown config key(s) in {path}: {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(DEFAULTS))}."
        )
    return (str(preset) if preset else None), data


def base_config(preset: str | None = None) -> dict:
    cfg = dict(DEFAULTS)
    if preset:
        cfg = apply_preset(cfg, preset)
    return cfg


def save_config(cfg: dict, path: str) -> None:
    """Persist the effective settings, minus per-run inputs."""
    skip = {"config_path", "media_dir"}
    data = {}
    for key, value in sorted(cfg.items()):
        if key in skip or key not in DEFAULTS:
            continue
        if key == "media" and not value:
            continue
        data[key] = value
    preset = cfg.get("preset")
    ordered = {"preset": preset, **data} if preset else data
    file = Path(path).expanduser()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf8")
