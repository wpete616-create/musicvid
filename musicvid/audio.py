"""Audio analysis: decode, spectrum bands, onsets, tempo and beat tracking.

Everything here is plain numpy so the only external dependency is ffmpeg.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import numpy as np

from .util import MusicvidError, ffmpeg

DEFAULT_SR = 22050
N_FFT = 2048
HOP = 512
# spectral flux peaks just before the transient it detects; this offset (about
# one analysis hop at the default settings) realigns beats with the audio
ONSET_LATENCY = HOP / DEFAULT_SR


def decode(path: str, sr: int = DEFAULT_SR) -> np.ndarray:
    """Decode any audio/video file to mono float32 PCM at ``sr``."""
    cmd = [
        ffmpeg(), "-v", "error", "-nostdin", "-i", str(path),
        "-vn", "-f", "f32le", "-ac", "1", "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf8", "replace").strip().splitlines()
        tail = " / ".join(detail[-3:]) if detail else "unknown ffmpeg error"
        raise MusicvidError(f"Could not decode audio from {path!r}: {tail}")
    samples = np.frombuffer(proc.stdout, dtype="<f4")
    if samples.size == 0:
        raise MusicvidError(f"No audio found in {path!r}.")
    return np.ascontiguousarray(samples, dtype=np.float32)


def _stft_magnitude(samples: np.ndarray, n_fft: int = N_FFT, hop: int = HOP,
                    chunk: int = 1024) -> np.ndarray:
    """Spectrogram magnitude, shape ``(frames, n_fft // 2 + 1)``."""
    window = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    padded = np.concatenate([
        np.zeros(pad, np.float32), samples, np.zeros(pad + n_fft, np.float32),
    ])
    n_frames = 1 + (len(padded) - n_fft) // hop
    out = np.empty((n_frames, n_fft // 2 + 1), np.float32)
    offsets = np.arange(n_fft)
    for start in range(0, n_frames, chunk):
        stop = min(n_frames, start + chunk)
        idx = (np.arange(start, stop)[:, None] * hop) + offsets[None, :]
        out[start:stop] = np.abs(np.fft.rfft(padded[idx] * window, axis=1))
    return out


def _band_slices(n_bands: int, sr: int, n_fft: int, fmin: float = 35.0,
                 fmax: float | None = None) -> list[tuple[int, int]]:
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fmax = fmax if fmax is not None else sr * 0.46
    edges = np.geomspace(fmin, min(fmax, freqs[-1]), n_bands + 1)
    bins = np.searchsorted(freqs, edges)
    bins = np.clip(bins, 0, len(freqs))
    for i in range(1, len(bins)):           # keep bands non-empty & ordered
        bins[i] = max(bins[i], bins[i - 1] + 1)
    bins = np.clip(bins, 0, len(freqs))
    return [(int(bins[i]), int(bins[i + 1])) for i in range(n_bands)]


def _band_energy(mag: np.ndarray, slices: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((mag.shape[0], len(slices)), np.float32)
    for i, (lo, hi) in enumerate(slices):
        if hi <= lo:
            continue
        seg = mag[:, lo:hi]
        out[:, i] = np.sqrt(np.mean(seg * seg, axis=1))
    return out


def _normalise_bands(energy: np.ndarray, gamma: float = 0.65) -> np.ndarray:
    """Scale each band by its own loud percentile so quiet treble still shows."""
    ref = np.percentile(energy, 97, axis=0)
    ref = np.maximum(ref, 1e-6)
    return np.clip(energy / ref, 0.0, 1.0) ** gamma


def _spectral_flux(mag: np.ndarray, sr: int, emphasis_hz: float = 400.0) -> np.ndarray:
    """Spectral flux with a low-frequency emphasis.

    A flat broadband sum lets hi-hats and cymbals dominate the envelope and pull
    the beat grid onto the offbeats; weighting toward the kick/bass/snare body
    keeps the tactus where a listener would tap it.
    """
    logmag = np.log1p(mag * 25.0)
    # compare against silence rather than against frame 0 repeated, otherwise an
    # onset on the very first frame produces no flux and the first downbeat is lost
    previous = np.concatenate(
        [np.zeros((1, logmag.shape[1]), logmag.dtype), logmag[:-1]], axis=0,
    )
    flux = np.maximum(logmag - previous, 0.0)
    freqs = np.fft.rfftfreq((mag.shape[1] - 1) * 2, 1.0 / sr)
    weight = (1.0 / (1.0 + (freqs / emphasis_hz) ** 2)).astype(np.float32)
    return (flux * weight[None, :]).sum(axis=1).astype(np.float32)


def _onset_envelope(mag: np.ndarray, sr: int, hop: int,
                    emphasis_hz: float = 400.0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(onset, tempo_env)``.

    ``onset`` is the peaked flux used for phase and beat snapping (its maxima sit
    on the transients), ``tempo_env`` has the slow trend removed, which is what
    autocorrelation wants.
    """
    raw = _spectral_flux(mag, sr, emphasis_hz)
    peak = float(raw.max())
    onset = raw / peak if peak > 0 else raw

    win = max(3, int(round(0.35 * sr / hop)) | 1)
    trend = np.convolve(raw, np.ones(win, np.float32) / win, mode="same")
    highpassed = np.maximum(raw - trend, 0.0)
    peak = float(highpassed.max())
    tempo_env = highpassed / peak if peak > 0 else highpassed
    return onset.astype(np.float32), tempo_env.astype(np.float32)


def _rms_envelope(samples: np.ndarray, hop: int, n_frames: int) -> np.ndarray:
    idx = np.arange(n_frames)[:, None] * hop + np.arange(hop)[None, :]
    np.clip(idx, 0, len(samples) - 1, out=idx)
    frames = samples[idx]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    ref = float(np.percentile(rms, 97)) or 1.0
    return np.clip(rms / ref, 0.0, 1.0).astype(np.float32)


def _centroid(mag: np.ndarray, sr: int, n_fft: int) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    total = mag.sum(axis=1)
    total = np.where(total <= 0, 1e-6, total)
    c = (mag * freqs[None, :]).sum(axis=1) / total
    return np.clip(c / (sr * 0.5), 0.0, 1.0).astype(np.float32)


def _peak_score(scores: np.ndarray, index: int) -> float:
    """Parabolic sub-sample refinement of a peak in ``scores``."""
    if not (0 < index < len(scores) - 1):
        return 0.0
    y0, y1, y2 = scores[index - 1], scores[index], scores[index + 1]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))


def _estimate_tempo(env: np.ndarray, fps: float, min_bpm: float = 60.0,
                    max_bpm: float = 190.0, harmonics: int = 3,
                    tactus_low: float = 80.0,
                    tactus_high: float = 160.0) -> tuple[float, float]:
    """Autocorrelation tempo estimate folded onto the tactus range.

    Autocorrelation cannot tell the beat from its half or double on its own, so
    the strongest periodicity is folded into ``[tactus_low, tactus_high]``: a
    60 BPM peak becomes 120, a 70 BPM peak becomes 140, and a 150 BPM peak stays
    where the listener would tap. Summing harmonics of the autocorrelation keeps
    the winner at a real periodicity rather than at a sub-multiple.
    """
    centred = env.astype(np.float64) - float(env.mean())
    n = len(centred)
    if n < 8:
        return 120.0, 0.5
    acf = np.correlate(centred, centred, mode="full")[n - 1:]
    if acf[0] <= 0:
        return 120.0, 0.5
    acf = acf / acf[0]

    lag_min = max(1, int(np.floor(fps * 60.0 / max_bpm)))
    lag_max = min(n - 2, int(np.ceil(fps * 60.0 / min_bpm)))
    if lag_max <= lag_min:
        return 120.0, 0.5

    def comb(lag: float) -> float:
        total = 0.0
        for k in range(1, harmonics + 1):
            index = int(round(lag * k))
            if index >= len(acf):
                break
            total += float(acf[index]) / k
        return total

    lags = np.arange(lag_min, lag_max + 1)
    scores = np.nan_to_num(np.array([comb(float(lag)) for lag in lags]))
    if scores.size == 0 or scores.max() <= 0:
        return 120.0, 0.5

    best = int(np.argmax(scores))
    period_frames = float(lags[best]) + _peak_score(scores, best)
    bpm = 60.0 * fps / max(1.0, period_frames)

    # fold onto the tactus octave a listener would tap
    while bpm < tactus_low:
        bpm *= 2.0
    while bpm > tactus_high:
        bpm /= 2.0
    period_frames = 60.0 * fps / bpm

    # re-centre on the best real periodicity close to the folded lag
    window = max(2, int(round(period_frames * 0.08)))
    lo = max(lag_min, int(round(period_frames)) - window)
    hi = min(lag_max, int(round(period_frames)) + window)
    if hi > lo:
        local = np.array([comb(float(lag)) for lag in range(lo, hi + 1)])
        if local.size and local.max() > 0:
            offset = int(np.argmax(local))
            period_frames = float(lo + offset) + _peak_score(local, offset)

    period = period_frames / fps
    if not np.isfinite(period) or period <= 0:
        return 120.0, 0.5
    return 60.0 / period, period


def _estimate_phase(env: np.ndarray, period_frames: float, resolutions: int = 96) -> float:
    """Offset of the beat grid. Ties go to the earliest offset.

    Music almost always opens on a downbeat, and starting the grid late is how a
    beat gets paired with the wrong side of the bar.
    """
    n = len(env)
    times = np.arange(n)
    offsets = np.linspace(0.0, period_frames, resolutions, endpoint=False)
    scores = np.full(offsets.shape, -np.inf)
    for i, offset in enumerate(offsets):
        grid = np.arange(offset, n - 1, period_frames)
        if grid.size == 0:
            continue
        scores[i] = float(np.interp(grid, times, env).mean())
    if not np.any(np.isfinite(scores)) or scores.max() <= 0:
        return 0.0
    near_best = offsets[scores >= scores.max() * 0.98]
    return float(near_best.min()) if near_best.size else float(offsets[int(np.argmax(scores))])


def _snap_beats(env: np.ndarray, grid: np.ndarray, period_frames: float) -> np.ndarray:
    """Pull each grid beat onto the nearest local onset peak for tighter sync."""
    n = len(env)
    window = max(1, int(round(period_frames * 0.11)))
    peaks: list[float] = []
    for g in grid:
        lo = int(max(0, np.floor(g) - window))
        hi = int(min(n, np.ceil(g) + window + 1))
        if hi <= lo:
            peaks.append(float(g))
            continue
        local = env[lo:hi]
        if local.max() <= 0.02:
            peaks.append(float(g))
            continue
        peaks.append(float(lo + int(np.argmax(local))))

    # keep the beat sequence monotonic and evenly-ish spaced
    snapped: list[float] = []
    for i, p in enumerate(peaks):
        if not snapped:
            snapped.append(p)
            continue
        previous = snapped[-1]
        low = previous + 0.62 * period_frames
        high = previous + 1.45 * period_frames
        p = min(max(p, low), high)
        if i + 1 < len(peaks) and p >= peaks[i + 1]:
            p = previous + period_frames
        snapped.append(p)
    return np.asarray(snapped, dtype=np.float64)


def _downbeats(low_energy: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
    """Pick the 4/4 phase whose beats carry the most low-end weight.

    ``low_energy`` is the bass/kick level rather than flux: the bar start is
    marked by the low end, not by whichever transient is sharpest. Which phase
    is "one" stays ambiguous for some material, so when two phases score within
    5% the earliest wins -- tracks almost always open on a downbeat.
    """
    if len(beat_frames) < 8:
        return beat_frames
    scores = []
    for phase in range(4):
        group = beat_frames[phase::4]
        scores.append(float(low_energy[group].mean()) if group.size else -np.inf)
    best = max(scores)
    if not np.isfinite(best) or best <= 0:
        return beat_frames[0::4]
    candidates = [p for p in range(4) if np.isfinite(scores[p]) and scores[p] >= best * 0.95]
    phase = min(candidates) if candidates else int(np.argmax(scores))
    return beat_frames[phase::4]


@dataclass
class Analysis:
    """Everything the renderers need, all indexed in analysis frames."""

    samples: np.ndarray
    sr: int
    frame_rate: float
    spectrum: np.ndarray            # (frames, bands) normalised 0..1
    raw_energy: np.ndarray          # (frames, bands) unnormalised band RMS
    rms: np.ndarray                 # (frames,) 0..1
    centroid: np.ndarray            # (frames,) 0..1
    onset: np.ndarray               # (frames,) 0..1
    bpm: float
    beat_period: float
    beat_times: np.ndarray          # seconds
    beat_frames: np.ndarray         # analysis frame indices
    downbeat_times: np.ndarray
    duration: float
    n_bands: int = 0
    source: str = ""
    frame_times: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # -- convenience ------------------------------------------------------- #
    @property
    def n_frames(self) -> int:
        return int(self.spectrum.shape[0])

    @property
    def peak(self) -> float:
        return float(np.abs(self.samples).max()) if self.samples.size else 1.0

    def frame_index(self, seconds: float) -> int:
        return int(np.clip(round(seconds * self.frame_rate), 0, self.n_frames - 1))

    def windowed(self, start: float | None = None, end: float | None = None) -> "Analysis":
        """Trim to ``[start, end)`` seconds; returned times start at zero."""
        start = max(0.0, float(start or 0.0))
        end = float(end) if end is not None else self.duration
        end = min(max(end, start), self.duration)
        if start <= 0.0 and end >= self.duration - 1e-9:
            return self

        s0 = int(start * self.sr)
        s1 = int(end * self.sr)
        f0 = int(start * self.frame_rate)
        f1 = min(self.n_frames, int(np.ceil(end * self.frame_rate)))

        keep = (self.beat_times >= start) & (self.beat_times < end)
        beat_times = np.asarray(self.beat_times[keep]) - start
        # frames follow from the latency-corrected times so both stay consistent
        beat_frames = np.rint(
            (self.beat_times[keep] - ONSET_LATENCY) * self.frame_rate
        ).astype(int) - f0
        keep = (self.downbeat_times >= start) & (self.downbeat_times < end)
        downbeats = np.asarray(self.downbeat_times[keep]) - start

        return Analysis(
            samples=self.samples[s0:s1].copy(),
            sr=self.sr,
            frame_rate=self.frame_rate,
            spectrum=self.spectrum[f0:f1].copy(),
            raw_energy=self.raw_energy[f0:f1].copy(),
            rms=self.rms[f0:f1].copy(),
            centroid=self.centroid[f0:f1].copy(),
            onset=self.onset[f0:f1].copy(),
            bpm=self.bpm,
            beat_period=self.beat_period,
            beat_times=np.asarray(beat_times),
            beat_frames=np.asarray(beat_frames, dtype=int),
            downbeat_times=np.asarray(downbeats),
            duration=end - start,
            n_bands=self.n_bands,
            source=self.source,
            frame_times=np.arange(f0, f1, dtype=np.float64) / self.frame_rate - start,
        )


def analyze(path: str, sr: int = DEFAULT_SR, n_fft: int = N_FFT, hop: int = HOP,
            bands: int = 48, start: float | None = None,
            end: float | None = None) -> Analysis:
    """Decode ``path`` and extract spectrum, onsets, tempo and beats."""
    samples = decode(path, sr=sr)
    duration = len(samples) / sr
    if duration < 0.5:
        raise MusicvidError("Audio is shorter than half a second; nothing to render.")

    mag = _stft_magnitude(samples, n_fft=n_fft, hop=hop)
    frame_rate = sr / hop
    raw_energy = _band_energy(mag, _band_slices(bands, sr, n_fft))
    spectrum = _normalise_bands(raw_energy)
    onset, tempo_env = _onset_envelope(mag, sr, hop)
    rms = _rms_envelope(samples, hop, mag.shape[0])
    centroid = _centroid(mag, sr, n_fft)

    bpm, period = _estimate_tempo(tempo_env, frame_rate)
    period_frames = period * frame_rate
    phase = _estimate_phase(onset, period_frames)
    # one period of slack: the phase lands on the envelope peaks, which sit just
    # before the transients, so without this the opening downbeat is dropped
    grid = np.arange(phase - period_frames, mag.shape[0] - 1, period_frames)
    grid = grid[grid >= -period_frames * 0.99]
    if grid.size == 0:
        grid = np.array([float(phase)])
    snapped = _snap_beats(onset, grid, period_frames)
    beat_frames = np.clip(snapped.round().astype(int), 0, mag.shape[0] - 1)
    # the flux peaks about one hop before the transient it belongs to; nudge the
    # grid forward so cuts and lyrics land on the sound, not just ahead of it
    beat_times = beat_frames / frame_rate + ONSET_LATENCY
    low_band = max(2, bands // 6)
    low_energy = raw_energy[:, :low_band].sum(axis=1)
    downbeats = _downbeats(low_energy, beat_frames) / frame_rate + ONSET_LATENCY

    analysis = Analysis(
        samples=samples,
        sr=sr,
        frame_rate=frame_rate,
        spectrum=spectrum,
        raw_energy=raw_energy,
        rms=rms,
        centroid=centroid,
        onset=onset,
        bpm=bpm,
        beat_period=period,
        beat_times=beat_times,
        beat_frames=beat_frames,
        downbeat_times=np.asarray(downbeats),
        duration=duration,
        n_bands=bands,
        source=str(path),
        frame_times=np.arange(mag.shape[0], dtype=np.float64) / frame_rate,
    )
    return analysis.windowed(start, end)
