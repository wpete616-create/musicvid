"""Beat-cut slideshow: photos and video clips cut on the beat."""

from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from .. import graphics as gfx
from ..audio import Analysis
from ..encode import to_uint8
from ..util import MusicvidError, ffmpeg
from .common import RenderContext, Timeline, progress_bar, scrim, title_block

NAME = "slideshow"
DESCRIPTION = "Photos and clips cut on the beat, with Ken Burns motion and transitions."
REQUIRES_MEDIA = True

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
HEIF_EXT = {".heic", ".heif"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg"}

# Pillow alone cannot decode HEIC/HEIF, so those extensions are accepted only
# when the optional pillow-heif plugin is installed -- the tool never advertises
# a format it cannot actually read.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    IMAGE_EXT |= HEIF_EXT
except ImportError:                                # pragma: no cover - optional
    pass


@dataclass
class Media:
    path: str
    kind: str                       # "image" | "video"
    frames: list[str] = field(default_factory=list)


def collect_media(cfg: dict, seed: int = 0) -> list[Media]:
    """Expand ``--media``/``--media-dir`` into validated media items."""
    raw = cfg.get("media") or []
    if isinstance(raw, str):
        raw = [part for part in raw.split(",") if part.strip()]

    paths: list[Path] = []
    for item in raw:
        path = Path(str(item)).expanduser()
        if not path.exists():
            raise MusicvidError(f"Media not found: {item}")
        if path.is_dir():
            found = _dir_media(path)
            if not found:
                raise MusicvidError(_empty_folder_message(path))
            paths.extend(found)
        else:
            paths.append(path)

    if cfg.get("media_dir"):
        directory = Path(str(cfg["media_dir"])).expanduser()
        if not directory.is_dir():
            raise MusicvidError(f"--media-dir is not a directory: {directory}")
        found = _dir_media(directory)
        if not found:
            raise MusicvidError(_empty_folder_message(directory))
        paths.extend(found)

    if not paths and cfg.get("cover"):
        paths.append(Path(str(cfg["cover"])).expanduser())

    if not paths:
        raise MusicvidError(
            "The slideshow style needs media: pass --media FILE[,FILE] (or a folder), "
            "--media-dir DIR, or --cover IMAGE."
        )

    items: list[Media] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXT:
            items.append(Media(str(path), "image"))
        elif suffix in VIDEO_EXT:
            items.append(Media(str(path), "video"))
        else:
            raise MusicvidError(
                f"Unsupported media type: {path.name}. {_supported_hint(suffix)}"
            )

    if cfg.get("shuffle"):
        random.Random(seed).shuffle(items)
    return items


def _dir_media(directory: Path) -> list[Path]:
    allowed = IMAGE_EXT | VIDEO_EXT
    return sorted(
        (path for path in directory.iterdir()
         if path.is_file() and path.suffix.lower() in allowed),
        key=lambda path: path.name.lower(),
    )


def _supported_hint(suffix: str = "") -> str:
    """Describe the accepted formats, calling out HEIC/HEIF when relevant."""
    listed = (f"images: {', '.join(sorted(IMAGE_EXT))}; "
              f"video: {', '.join(sorted(VIDEO_EXT))}")
    if suffix in HEIF_EXT:
        return (f"HEIC/HEIF needs the optional pillow-heif plugin "
                f"(`pip install 'musicvid[heic]'`). Supported {listed}.")
    return f"Supported {listed}."


def _empty_folder_message(directory: Path) -> str:
    message = f"No usable media in {directory}. {_supported_hint()}"
    heif = sorted(entry.name for entry in directory.iterdir()
                  if entry.suffix.lower() in HEIF_EXT)
    if heif:
        message += (f" It holds {len(heif)} HEIC/HEIF file(s) (e.g. {heif[0]}), which "
                    "need the optional pillow-heif plugin: pip install 'musicvid[heic]'.")
    return message


def cut_times(analysis: Analysis, cfg: dict, min_segment: float = 0.7) -> np.ndarray:
    """Segment start times, snapped to the beat grid."""
    cut_every = max(1, int(cfg.get("cut_every", 8)))
    mode = str(cfg.get("cut_on", "downbeats"))
    duration = analysis.duration

    if mode == "beats" or analysis.downbeat_times.size < 4:
        events = analysis.beat_times[::cut_every]
    else:
        every = max(1, int(round(cut_every / 4.0)))
        events = analysis.downbeat_times[::every]

    starts = [0.0]
    for when in events:
        when = float(when)
        if when - starts[-1] >= min_segment and when < duration - min_segment * 0.5:
            starts.append(when)
    return np.asarray(starts, dtype=np.float64)


def _prepare_image(path: str, size: tuple[int, int], fit: str) -> Image.Image:
    with gfx.open_image(path) as image:
        if fit == "contain":
            background, foreground = gfx.fit_contain(image, size[0], size[1])
            x = (size[0] - foreground.width) // 2
            y = (size[1] - foreground.height) // 2
            background.paste(foreground, (x, y))
            return background
        return gfx.fit_cover(image, size[0], size[1])


def _ken_burns(prepared: Image.Image, progress: float, size: tuple[int, int],
               zoom: float, direction: int) -> np.ndarray:
    """Crop a breathing window out of an already-fitted image."""
    width, height = size
    if zoom <= 0.0:
        return np.asarray(prepared, np.float32) / 255.0

    ease = 1.0 - abs(2.0 * min(1.0, max(0.0, progress)) - 1.0)   # 0 -> 1 -> 0
    scale = 1.0 + zoom * ease
    crop_w, crop_h = width / scale, height / scale
    half_w, half_h = crop_w / 2.0, crop_h / 2.0
    pan = width * 0.07 * direction * (2.0 * progress - 1.0)
    cx = float(np.clip(width / 2.0 + pan, half_w, width - half_w))
    cy = float(np.clip(height / 2.0, half_h, height - half_h))
    box = (
        int(round(cx - half_w)), int(round(cy - half_h)),
        int(round(cx + half_w)), int(round(cy + half_h)),
    )
    return np.asarray(prepared.crop(box).resize(size, Image.BILINEAR), np.float32) / 255.0


def _extract_frames(item: Media, size: tuple[int, int], fps: float,
                    workdir: str, index: int, limit: float) -> None:
    """Decode a video clip to JPEG frames matching the output size/fps."""
    out_dir = Path(workdir) / f"clip{index:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = size
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase,"
          f"crop={width}:{height},fps={fps:.6f}")
    cmd = [ffmpeg(), "-v", "error", "-nostdin", "-y", "-i", item.path]
    if limit > 0:
        cmd += ["-t", f"{limit:.3f}"]
    cmd += ["-vf", vf, "-q:v", "3", "-an", str(out_dir / "f%05d.jpg")]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf8", "replace").strip().splitlines()
        raise MusicvidError(
            f"Could not decode clip {Path(item.path).name}: "
            f"{' / '.join(detail[-2:]) if detail else 'ffmpeg error'}"
        )
    item.frames = [str(path) for path in sorted(out_dir.glob("*.jpg"))]
    if not item.frames:
        raise MusicvidError(f"Clip {Path(item.path).name} produced no frames.")


def render(ctx: RenderContext) -> Iterator[np.ndarray]:
    cfg = ctx.cfg
    width, height = ctx.size
    size = (width, height)
    fps = ctx.cfg_fps
    analysis = ctx.analysis
    duration = analysis.duration

    media: list[Media] = ctx.extra.get("media") or collect_media(
        cfg, seed=int(cfg.get("seed", 0) or 0))
    starts = cut_times(analysis, cfg)
    timeline = Timeline(analysis, fps, duration)

    transition = max(0.0, float(cfg.get("transition", 0.4) or 0.0))
    zoom = max(0.0, float(cfg.get("zoom", 0.0) or 0.0))
    fit = str(cfg.get("fit", "cover"))
    flash = bool(cfg.get("flash", False))
    accent = ctx.palette["accent_rgb"]
    # shots are full-bleed, so corner text needs its own legibility scrim
    wants_scrim = bool(cfg.get("scrim", True)) and bool(cfg.get("title") or cfg.get("artist"))

    if fit == "contain":
        zoom = min(zoom, 0.04)      # letterboxed shots look unstable when cropped

    segments = [float(s) for s in starts]
    ends = segments[1:] + [duration]
    max_segment = max((end - start for start, end in zip(segments, ends)), default=0.0)

    workdir = tempfile.mkdtemp(prefix="musicvid-clips-")
    try:
        for index, item in enumerate(media):
            if item.kind == "video":
                _extract_frames(item, size, fps, workdir, index, limit=max_segment)

        prepared: dict[int, Image.Image] = {}

        def shot(segment_index: int, local: float, segment_length: float) -> np.ndarray:
            media_index = segment_index % len(media)
            item = media[media_index]
            progress = 0.0 if segment_length <= 0 else max(0.0, local) / segment_length
            direction = 1 if segment_index % 2 == 0 else -1
            if item.kind == "video":
                frame_index = int(min(max(0.0, local) * fps, len(item.frames) - 1))
                with gfx.open_image(item.frames[frame_index]) as frame:
                    return np.asarray(frame.convert("RGB"), np.float32) / 255.0
            cached = prepared.get(media_index)
            if cached is None:
                cached = _prepare_image(item.path, size, fit)
                prepared[media_index] = cached
            return _ken_burns(cached, progress, size, zoom, direction)

        for i in range(timeline.n):
            t = float(timeline.t[i])
            segment = int(np.searchsorted(starts, t, side="right")) - 1
            segment = max(0, min(segment, len(segments) - 1))
            start, end = segments[segment], ends[segment]
            local = t - start
            canvas = shot(segment, local, end - start)

            if transition > 0 and segment + 1 < len(segments):
                overlap_start = end - transition
                if t > overlap_start:
                    blend = min(1.0, (t - overlap_start) / transition)
                    following = shot(segment + 1, t - end, ends[segment + 1] - end)
                    canvas = canvas * (1.0 - blend) + following * blend

            if flash and local < 0.09:
                canvas = np.clip(canvas + accent[None, None, :] * (0.45 * (1.0 - local / 0.09)),
                                 0.0, 1.0)

            if wants_scrim:
                scrim(canvas, top=0.22, bottom=0.12)
            title_block(ctx, canvas, alpha=0.95)
            progress_bar(ctx, canvas, t)
            yield to_uint8(canvas)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
