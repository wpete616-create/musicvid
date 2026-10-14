"""Analysis accuracy: tempo, beat placement, bands and trimming."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from musicvid import audio
from tests.support import SR, click_track, tone, write_wav


class TempoTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_detects_tempo_within_four_percent(self) -> None:
        for bpm in (100.0, 120.0, 140.0):
            with self.subTest(bpm=bpm):
                path = write_wav(self.dir / f"click{int(bpm)}.wav",
                                 click_track(bpm=bpm, bars=8))
                analysis = audio.analyze(str(path))
                self.assertAlmostEqual(analysis.bpm, bpm, delta=bpm * 0.04)

    def test_beats_land_on_the_grid(self) -> None:
        bpm = 120.0
        period = 60.0 / bpm
        path = write_wav(self.dir / "grid.wav", click_track(bpm=bpm, bars=8))
        analysis = audio.analyze(str(path))

        self.assertGreaterEqual(len(analysis.beat_times), 28)   # 8 bars * 4 beats
        self.assertLessEqual(len(analysis.beat_times), 34)

        gaps = np.diff(analysis.beat_times)
        self.assertLess(float(np.median(np.abs(gaps - period))), period * 0.12)

        # beats must sit on the true grid, not merely on a consistent offset
        grid = np.arange(0.0, 16.5, period)
        distance = np.abs(analysis.beat_times[:, None] - grid[None, :]).min(axis=1)
        self.assertLess(float(np.percentile(distance, 90)), 0.06)

    def test_downbeats_track_the_bar_line(self) -> None:
        path = write_wav(self.dir / "bars.wav", click_track(bpm=120.0, bars=8))
        analysis = audio.analyze(str(path))
        self.assertGreater(len(analysis.downbeat_times), 3)
        gaps = np.diff(analysis.downbeat_times)
        self.assertLess(float(np.median(np.abs(gaps - 2.0))), 0.06)   # one 4/4 bar
        self.assertLess(abs(float(analysis.downbeat_times[0])), 0.06)

    def test_downbeats_are_a_subset_of_beats(self) -> None:
        path = write_wav(self.dir / "down.wav", click_track(bpm=125.0, bars=8))
        analysis = audio.analyze(str(path))
        self.assertGreater(len(analysis.downbeat_times), 0)
        for when in analysis.downbeat_times:
            self.assertLess(float(np.min(np.abs(analysis.beat_times - when))), 1e-6)


class SpectrumTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_low_tone_energises_low_bands(self) -> None:
        # raw band energy keeps spectral placement; `spectrum` is normalised per
        # band on purpose, so a steady tone saturates every band there
        path = write_wav(self.dir / "low.wav", tone(60.0, 3.0))
        analysis = audio.analyze(str(path))
        profile = analysis.raw_energy.mean(axis=0)
        self.assertLessEqual(int(np.argmax(profile)), 2)
        self.assertGreater(float(profile[0]), float(profile[-1]) * 100.0)

    def test_high_tone_energises_high_bands(self) -> None:
        path = write_wav(self.dir / "high.wav", tone(7000.0, 3.0))
        analysis = audio.analyze(str(path))
        profile = analysis.raw_energy.mean(axis=0)
        self.assertGreaterEqual(int(np.argmax(profile)), int(analysis.n_bands * 0.75))
        self.assertGreater(float(profile.max()), float(profile[0]) * 100.0)

    def test_spectrum_is_normalised_per_band(self) -> None:
        path = write_wav(self.dir / "tones.wav", click_track(bars=4))
        analysis = audio.analyze(str(path))
        self.assertEqual(analysis.spectrum.shape, analysis.raw_energy.shape)
        self.assertGreaterEqual(float(analysis.spectrum.min()), 0.0)
        self.assertLessEqual(float(analysis.spectrum.max()), 1.0)
        self.assertGreater(float(analysis.raw_energy.max()), 0.0)

    def test_rms_and_centroid_shapes(self) -> None:
        path = write_wav(self.dir / "shapes.wav", click_track(bars=2))
        analysis = audio.analyze(str(path))
        self.assertEqual(analysis.spectrum.shape[0], analysis.rms.shape[0])
        self.assertEqual(analysis.spectrum.shape[0], analysis.centroid.shape[0])
        self.assertGreaterEqual(float(analysis.rms.min()), 0.0)
        self.assertLessEqual(float(analysis.rms.max()), 1.0)
        self.assertGreaterEqual(float(analysis.centroid.min()), 0.0)
        self.assertLessEqual(float(analysis.centroid.max()), 1.0)

    def test_trimming_shifts_times_to_zero(self) -> None:
        path = write_wav(self.dir / "trim.wav", click_track(bpm=120.0, bars=8))
        full = audio.analyze(str(path))
        clip = audio.analyze(str(path), start=2.0, end=6.0)

        self.assertAlmostEqual(full.duration, 16.5, delta=0.3)
        self.assertAlmostEqual(clip.duration, 4.0, delta=0.15)
        expected = full.beat_times[(full.beat_times >= 2.0) & (full.beat_times < 6.0)] - 2.0
        self.assertAlmostEqual(len(clip.beat_times), len(expected), delta=1)
        if len(clip.beat_times):
            self.assertLess(float(clip.beat_times.min()), 1.0)

    def test_windowed_is_noop_without_bounds(self) -> None:
        path = write_wav(self.dir / "noop.wav", click_track(bars=2))
        analysis = audio.analyze(str(path))
        self.assertIs(analysis.windowed(None, None), analysis)


class DecodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_file_raises(self) -> None:
        from musicvid.util import MusicvidError

        with self.assertRaises(MusicvidError):
            audio.analyze(str(self.dir / "nope.wav"))

    def test_very_short_audio_raises(self) -> None:
        from musicvid.util import MusicvidError

        path = write_wav(self.dir / "tiny.wav", tone(440.0, 0.2))
        with self.assertRaises(MusicvidError):
            audio.analyze(str(path))

    def test_decodes_at_requested_rate(self) -> None:
        path = write_wav(self.dir / "rate.wav", tone(440.0, 1.0, sr=48000), sr=48000)
        samples = audio.decode(str(path), sr=16000)
        self.assertAlmostEqual(len(samples) / 16000, 1.0, delta=0.05)
        self.assertEqual(samples.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
