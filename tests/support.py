"""Shared helpers for the test suite."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 22050


def tone(freq: float, duration: float, sr: int = SR, amp: float = 0.6) -> np.ndarray:
    time = np.arange(int(duration * sr)) / sr
    return (np.sin(2 * np.pi * freq * time) * amp).astype(np.float32)


def click_track(bpm: float = 120.0, bars: int = 8, sr: int = SR,
                with_offbeat: bool = True) -> np.ndarray:
    """A synthesised drum loop: strong downbeats, weaker offbeats."""
    beat = 60.0 / bpm
    total = int((beat * 4 * bars + 0.5) * sr)
    buffer = np.zeros(total, np.float32)

    def add(signal: np.ndarray, start: float, gain: float) -> None:
        i0 = int(start * sr)
        i1 = min(total, i0 + signal.size)
        if i1 > i0:
            buffer[i0:i1] += signal[:i1 - i0] * gain

    burst = np.arange(int(0.09 * sr)) / sr
    for bar in range(bars):
        for beat_index in range(4):
            when = bar * 4 * beat + beat_index * beat
            if beat_index == 0:
                envelope = np.exp(-burst * 26.0)
                add((np.sin(2 * np.pi * 70.0 * burst) * envelope).astype(np.float32), when, 1.0)
            elif beat_index == 2:
                envelope = np.exp(-burst * 30.0)
                add((np.sin(2 * np.pi * 85.0 * burst) * envelope).astype(np.float32), when, 0.9)
            else:
                envelope = np.exp(-burst * 40.0)
                add((np.sin(2 * np.pi * 200.0 * burst) * envelope).astype(np.float32), when, 0.4)
            if with_offbeat:
                envelope = np.exp(-burst * 120.0)
                add((np.sin(2 * np.pi * 900.0 * burst) * envelope).astype(np.float32),
                    when + beat / 2.0, 0.18)
    return buffer


def write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes((pcm * 32767.0).astype("<i2").tobytes())
    return path


def solid_image(path: Path, color: tuple[int, int, int], size=(640, 480)) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path
