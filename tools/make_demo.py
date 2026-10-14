#!/usr/bin/env python3
"""Generate royalty-free demo material: a 24s beat, an .lrc and some shots.

    python tools/make_demo.py [outdir]

Writes ``demo.wav``, ``demo.lrc`` and ``shots/shotN.jpg`` so the three styles can
be tried without any third-party assets.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SR = 44100
BPM = 120.0
BEAT = 60.0 / BPM
BARS = 12
DURATION = BEAT * 4 * BARS


def _kick(duration: float = 0.40) -> np.ndarray:
    time = np.arange(int(duration * SR)) / SR
    envelope = np.exp(-time * 20.0)
    freq = 48.0 + 110.0 * np.exp(-time * 32.0)
    return (np.sin(2 * np.pi * np.cumsum(freq) / SR) * envelope).astype(np.float32)


def _clap(duration: float = 0.22, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(int(duration * SR)) / SR
    noise = rng.standard_normal(len(time)).astype(np.float32)
    noise = np.convolve(noise, np.array([1.0, -0.7, 0.3], np.float32), mode="same")
    return (noise * np.exp(-time * 34.0)).astype(np.float32)


def _hat(duration: float = 0.06, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(int(duration * SR)) / SR
    noise = np.diff(rng.standard_normal(len(time)), prepend=0.0)
    return (noise * np.exp(-time * 90.0)).astype(np.float32) * 0.6


def _bass(freq: float, duration: float) -> np.ndarray:
    time = np.arange(int(duration * SR)) / SR
    phase = 2 * np.pi * freq * time
    wave_ = np.sin(phase) + 0.28 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)
    envelope = np.exp(-time * 6.0) * (1.0 - np.exp(-time * 400.0))
    return (wave_ * envelope).astype(np.float32)


def _pad(freqs: list[float], duration: float) -> np.ndarray:
    time = np.arange(int(duration * SR)) / SR
    signal = np.zeros_like(time, dtype=np.float32)
    for index, freq in enumerate(freqs):
        for detune in (-0.4, 0.4):
            signal += np.sin(2 * np.pi * (freq + detune) * time).astype(np.float32)
        signal += 0.3 * np.sin(4 * np.pi * freq * time).astype(np.float32)
    attack = 1.0 - np.exp(-time * 4.0)
    release = np.clip((duration - time) * 3.0, 0.0, 1.0)
    return (signal * attack * release / max(1, len(freqs) * 3.0)).astype(np.float32)


def build_track() -> np.ndarray:
    total = int(DURATION * SR)
    mix = np.zeros(total, dtype=np.float32)

    def place(signal: np.ndarray, start: float, gain: float) -> None:
        i0 = int(start * SR)
        i1 = min(total, i0 + signal.size)
        if i1 > i0:
            mix[i0:i1] += signal[:i1 - i0] * gain

    kick, clap, hat = _kick(), _clap(), _hat()
    progression = [
        (110.00, [220.00, 261.63, 329.63]),   # Am
        (87.31, [174.61, 220.00, 261.63]),    # F
        (130.81, [196.00, 261.63, 329.63]),   # C
        (98.00, [196.00, 246.94, 294.00]),    # G
    ]

    for bar in range(BARS):
        bar_start = bar * 4 * BEAT
        root, chord = progression[bar % len(progression)]
        intensity = 0.55 + 0.45 * (bar / max(1, BARS - 1))

        for beat in range(4):
            time = bar_start + beat * BEAT
            if beat in (0, 2):
                place(kick, time, 1.0 * intensity)
            if beat in (1, 3):
                place(clap, time, 0.55 * intensity)
            for half in range(2):
                place(hat, time + half * BEAT / 2.0, 0.35 * intensity)

        # bass: driving eighths with a syncopated skip
        for step in range(8):
            if step == 6:
                continue
            place(_bass(root, BEAT * 0.5 * 0.95),
                  bar_start + step * BEAT / 2.0, 0.5 * intensity)
        place(_pad(chord, BEAT * 4), bar_start, 0.16 + 0.10 * intensity)

    # simple limiter so the encoder never clips
    peak = float(np.abs(mix).max())
    if peak > 0:
        mix *= 0.88 / peak
    return mix


def write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((pcm * 32767.0).astype("<i2").tobytes())


LYRICS = [
    (0.0, "musicvid demo"),
    (BEAT * 4, "no film crew needed"),
    (BEAT * 8, "just your track and a folder of shots"),
    (BEAT * 16, "bars react to the mix"),
    (BEAT * 24, "lyrics ride the beat"),
    (BEAT * 32, "cuts land on the downbeat"),
    (BEAT * 40, "render it again for the next song"),
]


def write_lrc(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["[ar:musicvid]", "[ti:Demo Loop]", "[offset:0]"]
    for when, text in LYRICS:
        minutes = int(when // 60)
        seconds = when - minutes * 60
        lines.append(f"[{minutes:02d}:{seconds:05.2f}]{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def write_shots(directory: Path, count: int = 6) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    palettes = [
        ("#12002f", "#ff3d7f"), ("#02010a", "#2ee6ff"), ("#2b0a00", "#ffb03a"),
        ("#001616", "#38ffd0"), ("#1a0018", "#b46bff"), ("#0b0b0b", "#ffffff"),
    ]
    width, height = 1600, 1200
    for index in range(count):
        top, bottom = palettes[index % len(palettes)]
        ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        top_rgb = np.array([int(top[1:][i:i + 2], 16) for i in (0, 2, 4)], np.float32)
        bottom_rgb = np.array([int(bottom[1:][i:i + 2], 16) for i in (0, 2, 4)], np.float32)
        gradient = top_rgb[None, :] * (1 - ramp) + bottom_rgb[None, :] * ramp
        canvas = np.repeat(gradient[:, None, :], width, axis=1)
        # diagonal light streak so static shots still have structure
        ys = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        xs = np.linspace(0, 1, width, dtype=np.float32)[None, :]
        streak = np.exp(-((xs - ys * 0.6 - 0.2) ** 2) / 0.02)
        canvas = np.clip(canvas + streak[..., None] * 30.0, 0, 255)
        image = Image.fromarray(canvas.astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((80, height - 260), f"SHOT {index + 1}", fill=(255, 255, 255))
        draw.text((84, height - 200), "musicvid demo asset", fill=(220, 220, 220))
        image.save(directory / f"shot{index + 1}.jpg", quality=90)


def main(argv: list[str]) -> int:
    outdir = Path(argv[1] if len(argv) > 1 else "out").expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"synthesising {DURATION:.1f}s at {BPM:g} BPM ...")
    write_wav(outdir / "demo.wav", build_track())
    write_lrc(outdir / "demo.lrc")
    write_shots(outdir / "shots")
    print(f"wrote {outdir/'demo.wav'}, {outdir/'demo.lrc'}, {outdir/'shots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
