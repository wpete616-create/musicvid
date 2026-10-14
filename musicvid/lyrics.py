"""Lyrics parsing: LRC (including enhanced word timings) and SRT/WebVTT."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .util import MusicvidError

_LRC_TIME = re.compile(r"\[(\d{1,3}):(\d{1,2}(?:[.:]\d{1,3})?)\]")
_LRC_WORD = re.compile(r"<(\d{1,3}):(\d{1,2}(?:[.:]\d{1,3})?)>")
_LRC_META = re.compile(r"^\[(ti|ar|al|by|offset|re|ve|length):(.*)\]$", re.IGNORECASE)
_SRT_INDEX = re.compile(r"^\d+$")
_SRT_TIME = re.compile(
    r"(\d{1,3}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->\s*(\d{1,3}):(\d{2}):(\d{2})[.,](\d{1,3})"
)
_TAG = re.compile(r"<[^>]+>")


@dataclass
class Line:
    time: float
    text: str
    words: list[tuple[float, str]] = field(default_factory=list)


@dataclass
class Lyrics:
    lines: list[Line]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def has_word_timings(self) -> bool:
        return bool(self.lines) and all(line.words for line in self.lines)


def _seconds(minutes: str, seconds: str) -> float:
    return float(minutes) * 60.0 + float(str(seconds).replace(":", "."))


def parse(text: str) -> Lyrics:
    """Parse LRC or SRT text. Auto-detects the format."""
    lines = text.replace("\r\n", "\n").split("\n")
    if any(_SRT_TIME.search(line) for line in lines):
        return _parse_srt(lines)
    if any(_LRC_TIME.search(line) for line in lines):
        return _parse_lrc(lines)
    raise MusicvidError(
        "No timestamps found. Provide a .lrc file (`[mm:ss.xx] line`) or a "
        ".srt file, or pass timings with --lyrics-format."
    )


def _parse_lrc(raw_lines: list[str]) -> Lyrics:
    metadata: dict[str, str] = {}
    entries: list[Line] = []
    offset = 0.0

    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        meta = _LRC_META.match(line)
        if meta:
            key, value = meta.group(1).lower(), meta.group(2).strip()
            metadata[key] = value
            if key == "offset":
                try:
                    offset = -float(value) / 1000.0
                except ValueError:
                    offset = 0.0
            continue

        stamps = list(_LRC_TIME.finditer(line))
        if not stamps:
            continue
        body = line[stamps[-1].end():].strip()

        words: list[tuple[float, str]] = []
        if _LRC_WORD.search(body):
            stripped = _LRC_TIME.sub("", body)
            marks = list(_LRC_WORD.finditer(stripped))
            for i, mark in enumerate(marks):
                start = mark.end()
                stop = marks[i + 1].start() if i + 1 < len(marks) else len(stripped)
                chunk = stripped[start:stop].strip()
                if chunk:
                    words.append((_seconds(mark.group(1), mark.group(2)) + offset, chunk))
            body = _TAG.sub(" ", stripped)
            body = " ".join(body.split())

        if not body and not words:
            # a timestamp with no text carries nothing; LRC exports usually end
            # with one and it would blank the final line
            continue

        for stamp in stamps:
            when = _seconds(stamp.group(1), stamp.group(2)) + offset
            entries.append(Line(time=max(0.0, when), text=body,
                                words=[(t, w) for t, w in words] if words else []))

    entries.sort(key=lambda item: item.time)
    return Lyrics(lines=_dedupe(entries), metadata=metadata)


def _parse_srt(raw_lines: list[str]) -> Lyrics:
    entries: list[Line] = []
    pending: list[str] = []
    when: float | None = None

    def flush() -> None:
        nonlocal pending, when
        if when is not None:
            text = " ".join(" ".join(pending).split())
            if text:
                entries.append(Line(time=max(0.0, when), text=_TAG.sub("", text)))
        pending, when = [], None

    for raw in raw_lines:
        line = raw.strip()
        match = _SRT_TIME.search(line)
        if match:
            flush()
            hours, minutes, seconds, millis = match.group(1, 2, 3, 4)
            when = (float(hours) * 3600 + float(minutes) * 60 + float(seconds)
                    + float(millis.ljust(3, "0")) / 1000.0)
            continue
        if not line or _SRT_INDEX.match(line):
            continue
        if when is not None:
            pending.append(line)
    flush()

    entries.sort(key=lambda item: item.time)
    return Lyrics(lines=_dedupe(entries))


def _dedupe(lines: list[Line]) -> list[Line]:
    out: list[Line] = []
    for line in lines:
        if out and abs(out[-1].time - line.time) < 1e-3 and out[-1].text == line.text:
            continue
        out.append(line)
    return out


def load(path: str) -> Lyrics:
    file = Path(path).expanduser()
    if not file.exists():
        raise MusicvidError(f"Lyrics file not found: {path}")
    return parse(file.read_text(encoding="utf8", errors="replace"))


def find_sidecar(audio_path: str) -> str | None:
    """Look for ``song.lrc`` / ``song.srt`` next to the audio file."""
    audio = Path(audio_path)
    for suffix in (".lrc", ".srt", ".vtt", ".txt"):
        candidate = audio.with_suffix(suffix)
        if candidate.exists():
            try:
                if parse(candidate.read_text(encoding="utf8", errors="replace")).lines:
                    return str(candidate)
            except MusicvidError:
                continue
    return None
