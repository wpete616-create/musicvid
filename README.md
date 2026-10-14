# musicvid

Turn any song into a music video — over and over. One command in, one MP4 out:

```bash
musicvid make song.mp3                                  # audio-reactive bars
musicvid make song.mp3 --style lyrics --title "My Song"  # timed lyric video
musicvid make song.mp3 --style slideshow --media shots/  # beat-cut slideshow
```

Every style shares one audio analysis pass: it detects tempo, beat grid, downbeats
and per-band energy, so cuts, pulses and highlights land on the music.

| style | what it renders |
| --- | --- |
| `bars` | frequency bars that react to the track, plus oscilloscope, reflections and peak caps |
| `lyrics` | timed lyrics typography with optional word-level karaoke highlighting |
| `slideshow` | photos and video clips cut on the beat with Ken Burns motion and transitions |

## Install

```bash
pip install -e .          # needs Python 3.9+, installs numpy + pillow
```

You also need **ffmpeg** on your `PATH` (`apt install ffmpeg`, `brew install ffmpeg`, …).
No root? Grab a static build into `.tools/` instead, which musicvid picks up
automatically:

```bash
bash tools/install_ffmpeg.sh
musicvid doctor           # verifies python, numpy, pillow, ffmpeg and fonts
```

## Quick start

```bash
# 24 seconds of royalty-free demo audio, lyrics and shots
python tools/make_demo.py out

# 1. spectrum bars over a blurred cover wash
musicvid make out/demo.wav -o bars.mp4 --preset neon \
  --title "Demo Loop" --artist "you" --cover out/shots/shot1.jpg --art

# 2. lyric video (finds out/demo.lrc automatically next to the audio)
musicvid make out/demo.wav -o lyrics.mp4 --style lyrics --palette clean

# 3. slideshow cut every 8 beats with crossfades
musicvid make out/demo.wav -o cut.mp4 --preset cinema --media out/shots

# vertical for Shorts/TikTok
musicvid make song.mp3 -o short.mp4 --preset vapor --size 1080x1920

# a 30-second teaser
musicvid make song.mp3 --start 0:52 --end 1:22 --preset clean
```

## Reusing a look

Save the settings you like and replay them on the next song:

```bash
musicvid make song.mp3 -o one.mp4 --preset karaoke --palette vapor \
  --accent "#ff2d55" --bars 72 --write-config mylook.json

musicvid make another-song.mp3 -o two.mp4 --config mylook.json
```

The config is plain JSON, validated against known keys (typos are errors, not
silent no-ops), and CLI flags still override it. `musicvid presets` lists the
built-in looks: `neon`, `clean`, `ember`, `mono`, `vapor`, `lyric`, `karaoke`,
`cut`, `cinema`.

## Options worth knowing

**Shared** `--size 1920x1080|1080x1920|square|4:5`, `--fps`, `--crf`,
`--encoder-preset ultrafast|fast|medium|slow`, `--threads N`, `--palette`,
`--bg-top`/`--bg-bottom`/
`--accent`/`--glow-color`, `--title`, `--artist`, `--cover`, `--font`, `--start`,
`--end`, `--art`, `--no-progress-bar`, `--no-scrim`, `--seed`.

**Bars** `--bars N`, `--bar-gap`, `--no-waveform`, `--no-reflection`, `--no-glow`.

**Lyrics** `--lyrics file.lrc`, `--lyrics-style line|karaoke`, `--no-next-line`,
`--no-prev-line`, `--no-lyrics-bars`, `--uppercase-lyrics`.

**Slideshow** `--media FILE[,FILE]` or `--media-dir DIR`, `--cut-every N` (beats),
`--cut-on downbeats|beats`, `--transition` seconds (`0` = hard cut), `--zoom`
(Ken Burns amount), `--fit cover|contain`, `--flash`, `--shuffle`.

Lyrics come from a `.lrc` or `.srt` file — normally `<song>.lrc` next to the
audio. Word-level karaoke needs enhanced LRC timings
(`[00:12.00]<00:12.00>Hello <00:12.50>world`). **musicvid does not transcribe
audio**; you need timestamped lyrics.

`musicvid analyze song.mp3 --json` prints the detected tempo, beats and downbeats
if you want to sanity-check a track or feed the numbers elsewhere.

## How it works

```
decode (ffmpeg → mono float32)
  └─ STFT → log-magnitude flux → onset envelope (kick/bass weighted)
       ├─ autocorrelation + harmonic summation → tempo, folded onto the tactus (80–160 BPM)
       ├─ phase search + onset snapping → beat times (quiet correction for flux lead)
       ├─ low-end level per beat phase → downbeats
       └─ per-band energy → visual spectrum (each band auto-levelled)
render (numpy frames) → raw RGB pipe → ffmpeg (libx264 + AAC) → MP4
```

Rendering is done in numpy at float32 with in-place compositing; text and cover
art are cached RGBA layers. On a 2-core machine the frame generator runs at
about **20–25 fps of frames at 1080p** and the whole pipeline (including x264)
lands around **9–14 fps of output**, so a three-minute song takes roughly 3–6
minutes; smaller sizes and `--encoder-preset ultrafast` are quicker. Memory is
dominated by the encoder — x264 peaks near 600 MB at 1080p (about 490 MB with
`--threads 1`) — so on a small or shared box, `--threads 1` or `ultrafast` is the
thing to reach for.

## Tests

```bash
python -m unittest discover -t . -s tests       # or: pytest tests -q
```

The suite synthesises its own audio (click tracks at known tempos, tones for the
band placement) and checks tempo accuracy, beat-to-grid alignment, downbeat
spacing, LRC/SRT parsing, trimming, config round-trips, and renders every style
end to end — including A/B checks that overlays are actually drawn and that
frames change with the music.

## Limitations

- No music or footage generation: it renders visuals from audio you already have.
- Timestamps are required for lyrics; there is no speech-to-text.
- Tempo is octave-ambiguous by nature (a beat and its double both fit). The
  analyser folds onto 80–160 BPM and prefers the earliest phase, which matches
  what a listener taps — but very sparse or drumless material can still land a
  beat off. `musicvid analyze` shows what it decided.
- Bundled DejaVu fonts ship for consistent text rendering; see
  `musicvid/assets/fonts/LICENSE.txt`.
