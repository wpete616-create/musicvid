"""End-to-end render checks, driven through the real CLI."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from musicvid import audio
from musicvid.cli import main
from musicvid.styles.slideshow import cut_times
from musicvid.util import ffprobe
from tests.support import click_track, solid_image, write_wav

LRC = "[ar:Test]\n[00:00.50]first line\n[00:02.00]second line\n[00:03.50]third line\n"


def frame_at(path: Path, seconds: float, size=(320, 180)) -> np.ndarray:
    """Decode one grayscale frame so renders can be compared pixel by pixel."""
    from musicvid.util import ffmpeg

    result = subprocess.run(
        [ffmpeg(), "-v", "error", "-ss", f"{seconds:.3f}", "-i", str(path),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE, check=True,
    )
    return np.frombuffer(result.stdout, np.uint8).reshape(size[1], size[0])


def probe(path: Path) -> dict:
    result = subprocess.run(
        [ffprobe(), "-v", "error", "-show_streams", "-show_format",
         "-count_frames", "-of", "json", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return json.loads(result.stdout)


class CutTimesTest(unittest.TestCase):
    def test_cuts_land_on_bar_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            song = write_wav(Path(tmp) / "cut.wav", click_track(bpm=120.0, bars=8))
            analysis = audio.analyze(str(song))
            starts = cut_times(analysis, {"cut_every": 8, "cut_on": "downbeats"})

            self.assertGreaterEqual(len(starts), 4)
            self.assertEqual(starts[0], 0.0)
            for start in starts[1:]:
                self.assertLess(abs(start / 4.0 - round(start / 4.0)) * 4.0, 0.2)

    def test_short_tail_does_not_create_a_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            song = write_wav(Path(tmp) / "tail.wav", click_track(bpm=120.0, bars=4))
            analysis = audio.analyze(str(song))
            starts = cut_times(analysis, {"cut_every": 2, "cut_on": "beats"})
            self.assertTrue(np.all(np.diff(starts) >= 0.7))


class RenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.root = root
        cls.song = write_wav(root / "song.wav", click_track(bpm=120.0, bars=2))
        cls.lyrics_file = root / "song.lrc"
        cls.lyrics_file.write_text(LRC, encoding="utf8")
        cls.shots = [
            solid_image(root / f"shot{index}.jpg", (30 + index * 60, 20, 90))
            for index in range(3)
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def render(self, name: str, *extra: str) -> Path:
        output = self.root / f"{name}.mp4"
        code = main([
            "make", str(self.song), "-o", str(output),
            "--size", "320x180", "--fps", "12", "--quiet", *extra,
        ])
        self.assertEqual(code, 0, f"CLI failed for {name}")
        self.assertTrue(output.exists())
        return output

    def test_bars_style_renders_video_and_audio(self) -> None:
        output = self.render("bars", "--style", "bars", "--preset", "neon",
                             "--title", "Test Song", "--artist", "Test Artist")
        info = probe(output)
        kinds = {stream["codec_type"]: stream for stream in info["streams"]}
        self.assertIn("video", kinds)
        self.assertIn("audio", kinds)
        self.assertEqual(int(kinds["video"]["width"]), 320)
        self.assertEqual(int(kinds["video"]["height"]), 180)
        frames = int(kinds["video"]["nb_read_frames"])
        self.assertAlmostEqual(frames, 54, delta=3)
        self.assertAlmostEqual(float(info["format"]["duration"]), 4.5, delta=0.3)

    def test_lyrics_style_uses_the_sidecar(self) -> None:
        output = self.render("lyrics", "--style", "lyrics", "--palette", "clean")
        info = probe(output)
        frames = int(next(s for s in info["streams"] if s["codec_type"] == "video")["nb_read_frames"])
        self.assertAlmostEqual(frames, 54, delta=3)

    def test_slideshow_style_renders_the_media(self) -> None:
        output = self.render(
            "slideshow", "--style", "slideshow", "--preset", "cinema",
            "--media", ",".join(str(path) for path in self.shots),
            "--cut-every", "4", "--zoom", "0.06", "--transition", "0.3",
        )
        info = probe(output)
        self.assertGreater(int(next(
            s for s in info["streams"] if s["codec_type"] == "video"
        )["nb_read_frames"]), 40)

    def test_frames_actually_differ_over_time(self) -> None:
        """A static render would mean the analysis never reached the pixels."""
        output = self.render("motion", "--style", "bars", "--no-reflection")
        from musicvid.util import ffmpeg

        result = subprocess.run(
            [ffmpeg(), "-v", "error", "-i", str(output), "-vf", "select='eq(n\\,2)+eq(n\\,20)'",
             "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            stdout=subprocess.PIPE, check=True,
        )
        raw = np.frombuffer(result.stdout, np.uint8)
        self.assertEqual(raw.size, 320 * 180 * 2)
        early, later = raw[:320 * 180].astype(float), raw[320 * 180:].astype(float)
        self.assertGreater(float(np.abs(early - later).mean()), 1.0)

    def test_waveform_can_be_toggled(self) -> None:
        """Guards a bug class where an overlay silently draws nothing."""
        with_wave = frame_at(self.render(
            "wave_on", "--style", "bars", "--palette", "mono",
            "--title", "T", "--end", "3"), 2.0)
        without = frame_at(self.render(
            "wave_off", "--style", "bars", "--palette", "mono",
            "--title", "T", "--no-waveform", "--end", "3"), 2.0)
        differing = int((np.abs(with_wave.astype(int) - without.astype(int)) > 6).sum())
        self.assertGreater(differing, 200, "waveform overlay is not drawn")

    def test_bars_react_to_the_audio(self) -> None:
        output = self.render("react", "--style", "bars", "--palette", "neon",
                             "--end", "3", "--no-waveform")
        quiet = frame_at(output, 0.05)
        loud = frame_at(output, 1.0)
        self.assertGreater(float(np.abs(quiet.astype(int) - loud.astype(int)).mean()), 2.0)

    def test_saved_config_is_reusable(self) -> None:
        config = self.root / "look.json"
        self.render("configured", "--style", "bars", "--no-waveform",
                    "--preset", "mono", "--write-config", str(config))
        payload = json.loads(config.read_text(encoding="utf8"))
        self.assertEqual(payload["preset"], "mono")
        self.assertEqual(payload["size"], "320x180")
        self.assertFalse(payload["waveform"])

        output = self.root / "reused.mp4"
        code = main(["make", str(self.song), "-o", str(output),
                     "--config", str(config), "--quiet"])
        self.assertEqual(code, 0)
        info = probe(output)
        video = next(s for s in info["streams"] if s["codec_type"] == "video")
        self.assertEqual(int(video["width"]), 320)

    def test_trim_only_renders_the_window(self) -> None:
        output = self.root / "trim.mp4"
        code = main([
            "make", str(self.song), "-o", str(output), "--size", "320x180",
            "--fps", "12", "--quiet", "--preset", "clean",
            "--start", "1", "--end", "2.5",
        ])
        self.assertEqual(code, 0)
        info = probe(output)
        self.assertAlmostEqual(float(info["format"]["duration"]), 1.5, delta=0.25)

    def test_missing_lyrics_is_a_clean_error(self) -> None:
        bare = write_wav(self.root / "bare.wav", click_track(bpm=120.0, bars=1))
        code = main(["make", str(bare), "-o", str(self.root / "nope.mp4"),
                     "--style", "lyrics", "--size", "320x180", "--fps", "12", "--quiet"])
        self.assertEqual(code, 2)

    def test_slideshow_without_media_is_a_clean_error(self) -> None:
        bare = write_wav(self.root / "bare2.wav", click_track(bpm=120.0, bars=1))
        code = main(["make", str(bare), "-o", str(self.root / "nope2.mp4"),
                     "--style", "slideshow", "--size", "320x180", "--fps", "12", "--quiet"])
        self.assertEqual(code, 2)

    def test_unknown_config_key_is_rejected(self) -> None:
        bad = self.root / "bad.json"
        bad.write_text(json.dumps({"pallate": "neon"}), encoding="utf8")
        code = main(["make", str(self.song), "-o", str(self.root / "bad.mp4"),
                     "--config", str(bad), "--quiet"])
        self.assertEqual(code, 2)

    def test_analyze_command_reports_tempo(self) -> None:
        self.assertEqual(main(["analyze", str(self.song), "--json"]), 0)
        self.assertEqual(main(["presets"]), 0)
        self.assertEqual(main(["doctor"]), 0)


if __name__ == "__main__":
    unittest.main()
