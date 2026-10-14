"""Lyrics parsing: LRC, enhanced LRC, SRT and sidecar discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from musicvid import lyrics
from musicvid.util import MusicvidError

LRC = """[ar:Test Artist]
[ti:Test Song]
[offset:-500]
[00:01.00]first line
[00:05.50]second line
[00:10.25][00:12.25]repeated chorus
[00:15.00]
"""

SRT = """1
00:00:01,000 --> 00:00:03,000
hello there

2
00:00:03,500 --> 00:00:06,000
<i>second</i> cue
spanning lines
"""


class LrcTest(unittest.TestCase):
    def test_metadata_and_offset(self) -> None:
        parsed = lyrics.parse(LRC)
        self.assertEqual(parsed.metadata["ar"], "Test Artist")
        self.assertEqual(parsed.metadata["ti"], "Test Song")
        # offset -500ms means lines appear 0.5s later
        self.assertAlmostEqual(parsed.lines[0].time, 1.5, delta=1e-6)

    def test_repeated_timestamp_expands(self) -> None:
        parsed = lyrics.parse(LRC)
        repeated = [line for line in parsed.lines if line.text == "repeated chorus"]
        self.assertEqual(len(repeated), 2)
        self.assertAlmostEqual(repeated[0].time, 10.75, delta=1e-6)
        self.assertAlmostEqual(repeated[1].time, 12.75, delta=1e-6)

    def test_lines_are_sorted_and_blank_dropped(self) -> None:
        parsed = lyrics.parse(LRC)
        times = [line.time for line in parsed.lines]
        self.assertEqual(times, sorted(times))
        self.assertNotIn("", [line.text for line in parsed.lines])

    def test_enhanced_word_timings(self) -> None:
        parsed = lyrics.parse("[00:01.00]<00:01.00>Hello <00:01.60>world\n")
        line = parsed.lines[0]
        self.assertEqual(line.text, "Hello world")
        self.assertEqual([word for _, word in line.words], ["Hello", "world"])
        self.assertAlmostEqual(line.words[1][0], 1.6, delta=1e-6)
        self.assertTrue(parsed.has_word_timings)
        self.assertFalse(lyrics.parse(LRC).has_word_timings)


class SrtTest(unittest.TestCase):
    def test_parses_cues_and_strips_tags(self) -> None:
        parsed = lyrics.parse(SRT)
        self.assertEqual(len(parsed.lines), 2)
        self.assertAlmostEqual(parsed.lines[0].time, 1.0, delta=1e-6)
        self.assertEqual(parsed.lines[1].text, "second cue spanning lines")


class LoadingTest(unittest.TestCase):
    def test_untimed_text_is_rejected(self) -> None:
        with self.assertRaises(MusicvidError):
            lyrics.parse("just some words\nand more words\n")

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(MusicvidError):
            lyrics.load("/nonexistent/song.lrc")

    def test_sidecar_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "song.wav").write_bytes(b"")
            self.assertIsNone(lyrics.find_sidecar(str(root / "song.wav")))
            (root / "song.lrc").write_text(LRC, encoding="utf8")
            found = lyrics.find_sidecar(str(root / "song.wav"))
            self.assertIsNotNone(found)
            self.assertTrue(str(found).endswith("song.lrc"))


if __name__ == "__main__":
    unittest.main()
