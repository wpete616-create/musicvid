"""Command line interface: ``musicvid make|analyze|presets|doctor``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from . import audio as audio_mod
from . import config as config_mod
from . import graphics as gfx
from . import lyrics as lyrics_mod
from .encode import FrameWriter
from .graphics import TextRenderer
from .styles import DESCRIPTIONS, get_style, style_names, style_requires_media
from .styles.common import RenderContext
from .styles.slideshow import collect_media
from .util import (
    MusicvidError, Progress, ffmpeg, font_path, format_seconds, have_font, log,
    parse_size, parse_timestamp,
)


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="musicvid",
        description="Turn any song into a music video: audio-reactive bars, "
                    "lyric typography, or beat-cut slideshows.",
    )
    parser.add_argument("--version", action="version", version=f"musicvid {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    make = subparsers.add_parser("make", help="render a music video from a song")
    make.add_argument("song", help="audio file (mp3, wav, flac, m4a, ...)")
    make.add_argument("-o", "--output", help="output .mp4 path (default: ./<song>.mp4)")

    look = make.add_argument_group("look")
    look.add_argument("--style", choices=style_names(), help="render style")
    look.add_argument("--preset", choices=config_mod.preset_names(),
                      help="built-in look, e.g. neon, clean, lyric, karaoke, cut, cinema")
    look.add_argument("--config", help="JSON config file of saved settings")
    look.add_argument("--write-config", metavar="PATH",
                      help="save the effective settings for reuse")
    look.add_argument("--palette", choices=sorted(gfx.PALETTES), help="colour palette")
    look.add_argument("--bg-top", help="background top colour, e.g. '#0a0618'")
    look.add_argument("--bg-bottom", help="background bottom colour")
    look.add_argument("--accent", help="accent colour used for beats and highlights")
    look.add_argument("--glow-color", help="glow colour")
    look.add_argument("--cover-mix", type=float, help="cover-art wash strength 0..1")
    look.add_argument("--darken", type=float, help="background darkening 0..1")
    look.add_argument("--size", help="frame size, e.g. 1920x1080, 1080x1920, square")
    look.add_argument("--fps", type=int, help="frames per second (default 30)")
    look.add_argument("--crf", type=int, help="x264 quality, lower is better (default 20)")
    look.add_argument("--encoder-preset",
                      choices=("ultrafast", "fast", "medium", "slow"),
                      help="x264 speed preset (ultrafast is also the lightest)")
    look.add_argument("--audio-bitrate", help="AAC bitrate for the output audio")
    look.add_argument("--threads", type=int,
                      help="encoder threads; lower it if the renderer runs out of memory")
    look.add_argument("--seed", type=int, help="random seed for shuffled media")
    look.add_argument("--title", help="title shown in the corner")
    look.add_argument("--artist", help="artist/subtitle shown under the title")
    look.add_argument("--cover", help="cover art image: background wash and thumbnail")
    look.add_argument("--font", help="path to a .ttf/.otf font")
    look.add_argument("--art", action="store_true", default=None,
                      help="show a cover-art thumbnail")
    look.add_argument("--no-progress-bar", dest="no_progress", action="store_true",
                      default=None, help="hide the playback progress bar")
    look.add_argument("--no-scrim", dest="no_scrim", action="store_true", default=None,
                      help="skip the top/bottom darkening")

    trim = make.add_argument_group("trim")
    trim.add_argument("--start", help="start time, e.g. 42 or 1:02")
    trim.add_argument("--end", help="end time")

    bars = make.add_argument_group("bars style")
    bars.add_argument("--bars", type=int, help="number of frequency bars")
    bars.add_argument("--bar-gap", type=float, help="gap between bars 0..0.9")
    bars.add_argument("--no-waveform", dest="no_waveform", action="store_true",
                      default=None, help="hide the oscilloscope waveform")
    bars.add_argument("--no-reflection", dest="no_reflection", action="store_true",
                      default=None, help="hide the mirrored reflection")
    bars.add_argument("--no-glow", dest="no_glow", action="store_true", default=None,
                      help="hide the horizon glow")

    show = make.add_argument_group("slideshow style")
    show.add_argument("--media", action="append",
                      help="image/clip path or folder; repeatable, comma separated")
    show.add_argument("--media-dir", help="folder of images/clips to cycle")
    show.add_argument("--cut-every", type=int, help="beats between cuts (default 8)")
    show.add_argument("--cut-on", choices=("downbeats", "beats"),
                      help="cut on bar starts or every beat")
    show.add_argument("--transition", type=float, help="crossfade seconds (0 = hard cut)")
    show.add_argument("--zoom", type=float, help="Ken Burns zoom amount, e.g. 0.07")
    show.add_argument("--fit", choices=("cover", "contain"), help="how media fills the frame")
    show.add_argument("--flash", action="store_true", default=None,
                      help="flash the accent colour on each cut")
    show.add_argument("--shuffle", action="store_true", default=None,
                      help="shuffle media order (deterministic per --seed)")

    lyric = make.add_argument_group("lyrics style")
    lyric.add_argument("--lyrics", help=".lrc or .srt file (default: song.lrc sidecar)")
    lyric.add_argument("--lyrics-style", choices=("line", "karaoke"),
                       help="line-level or word-level highlighting")
    lyric.add_argument("--no-next-line", dest="no_next", action="store_true",
                       default=None, help="hide the upcoming line")
    lyric.add_argument("--no-prev-line", dest="no_prev", action="store_true",
                       default=None, help="hide the previous line")
    lyric.add_argument("--no-lyrics-bars", dest="no_lyrics_bars", action="store_true",
                       default=None, help="hide the spectrum strip under the lyrics")
    lyric.add_argument("--uppercase-lyrics", action="store_true", default=None,
                       help="render lyrics in uppercase")

    make.add_argument("--quiet", action="store_true", help="suppress progress output")
    make.set_defaults(handler=cmd_make)

    analyze = subparsers.add_parser("analyze", help="print tempo and beat information")
    analyze.add_argument("song")
    analyze.add_argument("--start")
    analyze.add_argument("--end")
    analyze.add_argument("--bands", type=int, default=48, help="spectrum bands to measure")
    analyze.add_argument("--json", action="store_true", help="machine-readable output")
    analyze.set_defaults(handler=cmd_analyze)

    presets = subparsers.add_parser("presets", help="list built-in looks")
    presets.set_defaults(handler=cmd_presets)

    doctor = subparsers.add_parser("doctor", help="check dependencies")
    doctor.set_defaults(handler=cmd_doctor)

    return parser


def collect_overrides(args) -> dict:
    """Map CLI flags onto config keys (only flags the user actually passed)."""
    out: dict = {}

    def take(attr, key=None, transform=None):
        value = getattr(args, attr, None)
        if value is None:
            return
        out[key or attr] = transform(value) if transform else value

    take("style")
    take("palette")
    take("size")
    take("fps")
    take("crf")
    take("encoder_preset")
    take("audio_bitrate")
    take("threads")
    take("seed")
    take("title")
    take("artist")
    take("cover")
    take("font", "font")
    take("bg_top")
    take("bg_bottom")
    take("accent")
    take("glow_color")
    take("cover_mix")
    take("darken")
    take("bars")
    take("bar_gap")
    take("cut_every")
    take("cut_on")
    take("transition")
    take("zoom")
    take("fit")
    take("lyrics")
    take("lyrics_style")
    take("media_dir")

    if args.media:
        media: list[str] = []
        for entry in args.media:
            media.extend(part.strip() for part in str(entry).split(",") if part.strip())
        out["media"] = media

    flags = (
        ("art", "art", True), ("no_progress", "progress", False),
        ("no_scrim", "scrim", False), ("no_waveform", "waveform", False),
        ("no_reflection", "reflection", False), ("no_glow", "glow", False),
        ("flash", "flash", True), ("shuffle", "shuffle", True),
        ("no_next", "show_next", False), ("no_prev", "show_prev", False),
        ("no_lyrics_bars", "lyrics_bars", False),
        ("uppercase_lyrics", "uppercase_lyrics", True),
    )
    for attr, key, value in flags:
        if getattr(args, attr, None):
            out[key] = value

    for attr in ("start", "end"):
        raw = getattr(args, attr, None)
        if raw is not None:
            out[attr] = parse_timestamp(raw)
    return out


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_make(args) -> int:
    song = Path(args.song).expanduser()
    if not song.exists():
        raise MusicvidError(f"Audio file not found: {args.song}")

    preset = args.preset
    file_settings: dict = {}
    if args.config:
        file_preset, file_settings = config_mod.load_config(args.config)
        preset = preset or file_preset

    cfg = config_mod.base_config(preset)
    cfg.update(file_settings)
    cfg.update(collect_overrides(args))

    size = parse_size(cfg["size"])
    fps = float(cfg["fps"])
    style = str(cfg["style"])
    quiet = bool(args.quiet)

    if cfg.get("start") is not None and cfg.get("end") is not None:
        if float(cfg["end"]) <= float(cfg["start"]):
            raise MusicvidError("--end must be later than --start.")

    if (cfg.get("media") or cfg.get("media_dir")) and not style_requires_media(style):
        log(
            f"  note: the {style!r} style ignores --media/--media-dir; only the "
            f"slideshow style shows them (try --preset cut or --style slideshow).",
            quiet=quiet,
        )

    log(f"Analysing {song.name} ...", quiet=quiet)
    started = time.time()
    analysis = audio_mod.analyze(str(song), start=cfg.get("start"), end=cfg.get("end"),
                                bands=int(cfg.get("bands", 48)))
    log(
        f"  {format_seconds(analysis.duration)} · {analysis.bpm:.1f} BPM · "
        f"{len(analysis.beat_times)} beats · {analysis.n_bands} bands "
        f"({time.time() - started:.1f}s)",
        quiet=quiet,
    )

    extra: dict = {}
    if style == "lyrics":
        lyrics_path = cfg.get("lyrics") or lyrics_mod.find_sidecar(str(song))
        if not lyrics_path:
            raise MusicvidError(
                "No lyrics found. Pass --lyrics song.lrc (or .srt), or put a .lrc "
                "file next to the audio. Timestamps are required -- musicvid does "
                "not transcribe audio."
            )
        parsed = lyrics_mod.load(lyrics_path)
        if not parsed.lines:
            raise MusicvidError(f"No timed lines found in {lyrics_path}.")
        if str(cfg.get("lyrics_style", "line")) == "karaoke" and not parsed.has_word_timings:
            log("  note: no word-level timings in the lyrics file; using line mode.",
                quiet=quiet)
        extra["lyrics"] = parsed
        log(f"  lyrics: {len(parsed.lines)} lines from {Path(lyrics_path).name}",
            quiet=quiet)
    elif style == "slideshow":
        extra["media"] = collect_media(cfg, seed=int(cfg.get("seed", 0) or 0))
        names = ", ".join(Path(item.path).name for item in extra["media"][:4])
        log(f"  media: {len(extra['media'])} item(s) ({names}"
            f"{', ...' if len(extra['media']) > 4 else ''})", quiet=quiet)

    palette = gfx.resolve_palette(
        str(cfg["palette"]),
        bg_top=cfg.get("bg_top"), bg_bottom=cfg.get("bg_bottom"),
        accent=cfg.get("accent"), glow=cfg.get("glow_color"),
        cover_mix=cfg.get("cover_mix"), darken=cfg.get("darken"),
    )
    if cfg.get("cover") and not Path(str(cfg["cover"])).exists():
        raise MusicvidError(f"Cover image not found: {cfg['cover']}")

    ctx = RenderContext(
        analysis=analysis,
        cfg=cfg,
        size=size,
        palette=palette,
        text=TextRenderer(cfg.get("font")),
        extra=extra,
    )

    frames = get_style(style)(ctx)
    total = int(round(analysis.duration * fps))
    output = Path(args.output).expanduser() if args.output else Path.cwd() / f"{song.stem}.mp4"

    if args.write_config:
        config_mod.save_config(cfg, args.write_config)
        log(f"Saved settings to {args.write_config}", quiet=quiet)

    log(f"Rendering {total} frames at {size[0]}x{size[1]} {fps:g}fps ...", quiet=quiet)
    progress = Progress(total, "  encoding", enabled=not quiet)
    started = time.time()
    written = 0
    with FrameWriter(
        str(output), size, fps, str(song),
        crf=int(cfg["crf"]), preset=str(cfg["encoder_preset"]),
        audio_bitrate=str(cfg["audio_bitrate"]),
        threads=int(cfg["threads"]) if cfg.get("threads") else None,
        start=cfg.get("start"), duration=analysis.duration,
    ) as writer:
        for written, frame in enumerate(frames, start=1):
            writer.write(frame)
            progress.update(written, extra=f"{written / max(1e-6, time.time() - started):.0f} fps")
    progress.done()

    megabytes = output.stat().st_size / 1e6 if output.exists() else 0.0
    log(f"Wrote {output} ({megabytes:.1f} MB, {written} frames, "
        f"{time.time() - started:.1f}s)", quiet=quiet)
    return 0


def cmd_analyze(args) -> int:
    analysis = audio_mod.analyze(
        str(Path(args.song).expanduser()),
        start=parse_timestamp(args.start), end=parse_timestamp(args.end),
        bands=int(args.bands),
    )
    if args.json:
        payload = {
            "source": analysis.source,
            "duration": round(analysis.duration, 4),
            "bpm": round(analysis.bpm, 3),
            "beat_period": round(analysis.beat_period, 5),
            "beats": [round(float(t), 4) for t in analysis.beat_times],
            "downbeats": [round(float(t), 4) for t in analysis.downbeat_times],
            "rms_mean": round(float(np.mean(analysis.rms)), 5),
            "rms_peak": round(float(np.max(analysis.rms)), 5),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"source        {analysis.source}")
    print(f"duration      {format_seconds(analysis.duration)}")
    print(f"tempo         {analysis.bpm:.1f} BPM ({analysis.beat_period:.3f}s per beat)")
    print(f"beats         {len(analysis.beat_times)}")
    print(f"downbeats     {len(analysis.downbeat_times)}")
    print(f"loudness      mean {float(np.mean(analysis.rms)):.3f}, "
          f"peak {float(np.max(analysis.rms)):.3f}")
    if analysis.beat_times.size:
        head = ", ".join(f"{t:.2f}" for t in analysis.beat_times[:12])
        print(f"first beats   {head} ...")
    return 0


def cmd_presets(args) -> int:
    print("styles")
    for name in style_names():
        print(f"  {name:<10} {DESCRIPTIONS.get(name, '')}")
    print("\npresets")
    for name in config_mod.preset_names():
        settings = ", ".join(f"{k}={v}" for k, v in config_mod.PRESETS[name].items())
        print(f"  {name:<10} {settings}")
    return 0


def cmd_doctor(args) -> int:
    ok = True

    print(f"python        {sys.version.split()[0]}")
    print(f"musicvid      {__version__}")
    for label, module in (("numpy", "numpy"), ("pillow", "PIL")):
        try:
            imported = __import__(module)
            print(f"{label:<13} {getattr(imported, '__version__', 'ok')}")
        except ImportError as exc:
            ok = False
            print(f"{label:<13} MISSING ({exc})")

    try:
        binary = ffmpeg()
        print(f"ffmpeg        {binary}")
    except MusicvidError as exc:
        ok = False
        print(f"ffmpeg        MISSING ({exc})")

    fonts = font_path(False), font_path(True)
    if have_font():
        print(f"font          {fonts[0]}")
    else:
        print("font          none found; falling back to Pillow's built-in font")
    print(f"styles        {', '.join(style_names())}")
    print(f"palettes      {', '.join(sorted(gfx.PALETTES))}")
    print("\n" + ("ready to render" if ok else "missing dependencies -- see above"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 2
    try:
        return int(args.handler(args))
    except MusicvidError as exc:
        print(f"musicvid: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("musicvid: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())
