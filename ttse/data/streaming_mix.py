"""Streaming sequence construction from LibriSpeech/LibriMix sources.

Each training/eval example is a scripted scenario built on the fly:

  Phase I   target + masker           len_a  seconds, TMR in [tmr_lo, tmr_hi] dB
  Phase II  masker only (absence)     D      seconds
  Phase III target + masker           len_c  seconds

Utterances per speaker are concatenated (crossfaded) to reach phase lengths.
Returns mix, clean target (zeros during absence), per-chunk oracle activity,
and an enrollment utterance from a *different* recording of the target.

Train curriculum: D <= 4 s, TMR >= -5 dB.  Eval: D up to 30 s, TMR to -15 dB.
"""

import random
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


SR = 16000


def _load(path, sr=SR):
    x, fs = sf.read(path, dtype="float32", always_2d=False)
    assert fs == sr, f"{path}: {fs} != {sr}"
    if x.ndim > 1:
        x = x[:, 0]
    return x


def _cat_to_len(paths, n_samples, rng, xfade=160):
    """Concatenate utterances (with short crossfade) until n_samples, then crop
    at a random offset."""
    out = np.zeros(0, dtype=np.float32)
    paths = list(paths)
    while len(out) < n_samples + SR:
        p = rng.choice(paths)
        x = _load(p)
        if len(out) == 0:
            out = x
        else:
            a, b = out[:-xfade], out[-xfade:]
            ramp = np.linspace(1, 0, xfade, dtype=np.float32)
            out = np.concatenate([a, b * ramp + x[:xfade] * (1 - ramp), x[xfade:]])
    off = rng.randint(0, max(1, len(out) - n_samples))
    return out[off:off + n_samples]


def _rms(x, eps=1e-8):
    return float(np.sqrt((x ** 2).mean()) + eps)


def _random_channel(x, nprng, severity=2):
    """Random spectral shaping: first-order tilt + peaking biquads; severity 1-3
    scales tilt/peak count/gain, severity 3 adds telephone-style bandpass.
    Simulates enrollment/mixture channel mismatch. Pure numpy (no scipy)."""
    n_peaks = severity + 1
    tilt = 0.3 * severity
    peak_db = 4.0 + 4.0 * severity
    # tilt: y[n] = x[n] + g*x[n-1]
    g = nprng.uniform(-tilt, tilt)
    y = x.copy()
    y[1:] += g * x[:-1]
    # peaking biquads via direct form I
    for _ in range(n_peaks):
        f0 = nprng.uniform(300, 6000) / SR
        q = nprng.uniform(0.7, 2.0)
        gain_db = nprng.uniform(-peak_db, peak_db)
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * f0
        alpha = np.sin(w0) / (2 * q)
        b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
        a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
        b, a = b / a[0], a / a[0]
        out = np.zeros_like(y)
        for n in range(len(y)):
            out[n] = b[0] * y[n]
            if n >= 1:
                out[n] += b[1] * y[n - 1] - a[1] * out[n - 1]
            if n >= 2:
                out[n] += b[2] * y[n - 2] - a[2] * out[n - 2]
        y = out
    if severity >= 3:
        # telephone band: brutal high/low cut via FFT mask (zero-phase, cheap)
        Y = np.fft.rfft(y)
        f = np.fft.rfftfreq(len(y), 1 / SR)
        Y[(f < 300) | (f > 3400)] = 0
        y = np.fft.irfft(Y, n=len(y)).astype(np.float32)
    return (y / (np.abs(y).max() + 1e-8) * (np.abs(x).max() + 1e-8)).astype(np.float32)


def _scale_to_tmr(tgt, msk, tmr_db):
    """Scale masker so that 10log10(P_tgt/P_msk) = tmr_db (target kept at unit-ish)."""
    g = _rms(tgt) / (_rms(msk) * 10 ** (tmr_db / 20))
    return tgt, msk * g


class StreamingScenario(Dataset):
    """spk2utts: dict speaker_id -> list of wav paths (LibriSpeech layout)."""

    def __init__(self, spk2utts, n_examples=10000, seed=0,
                 len_a=5.0, len_c=8.0,
                 absence_choices=(0.0, 1.0, 2.0, 4.0),
                 tmr_range=(-5.0, 5.0),
                 channel_shift=0,
                 chunk_ms=250, enroll_sec=5.0):
        self.spk2utts = {k: v for k, v in spk2utts.items() if len(v) >= 3}
        self.spks = sorted(self.spk2utts)
        if len(self.spks) < 2:
            raise ValueError("Need at least two speakers, each with three FLAC recordings")
        self.n = n_examples
        self.seed = seed
        self.len_a, self.len_c = len_a, len_c
        self.absence_choices = absence_choices
        self.tmr_range = tmr_range
        self.channel_shift = channel_shift
        self.chunk = int(SR * chunk_ms / 1000)
        self.enroll = int(SR * enroll_sec)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        nprng = np.random.RandomState(self.seed * 7 + i)
        spk_a, spk_b = rng.sample(self.spks, 2)
        D = rng.choice(self.absence_choices)
        tmr = rng.uniform(*self.tmr_range)

        na = int(self.len_a * SR)
        nd = int(D * SR)
        nc = int(self.len_c * SR)
        T = na + nd + nc

        utts_a = self.spk2utts[spk_a]
        enroll_path = rng.choice(utts_a)
        speech_paths = [p for p in utts_a if p != enroll_path]

        tgt_speech = _cat_to_len(speech_paths, na + nc, RngShim(rng))
        if self.channel_shift:
            # channel_shift=-1: aug severity 0/0/1/2; -2: aug severity 0/1/2/3
            if self.channel_shift == -1:
                sev = rng.choice([0, 0, 1, 2])
            elif self.channel_shift == -2:
                sev = rng.choice([0, 1, 2, 3])
            else:
                sev = int(self.channel_shift)
            if sev > 0:
                tgt_speech = _random_channel(tgt_speech, nprng, severity=sev)
        msk = _cat_to_len(self.spk2utts[spk_b], T, RngShim(rng))

        tgt = np.zeros(T, dtype=np.float32)
        tgt[:na] = tgt_speech[:na]
        tgt[na + nd:] = tgt_speech[na:]

        tgt, msk = _scale_to_tmr(tgt, msk, tmr)
        mix = tgt + msk
        peak = max(np.abs(mix).max(), 1e-4)
        if peak > 0.99:
            mix, tgt, msk = mix / peak, tgt / peak, msk / peak

        enroll = _cat_to_len([enroll_path], self.enroll, RngShim(rng))

        n_chunks = (T + self.chunk - 1) // self.chunk
        active = np.zeros(n_chunks, dtype=np.float32)
        for t in range(n_chunks):
            seg = tgt[t * self.chunk:(t + 1) * self.chunk]
            active[t] = float((seg ** 2).mean() > 1e-6)

        return {
            "mix": torch.from_numpy(mix),
            "target": torch.from_numpy(tgt),
            "masker": torch.from_numpy(msk),
            "enroll": torch.from_numpy(enroll),
            "active": torch.from_numpy(active),
            "absence_sec": float(D),
            "tmr_db": float(tmr),
            "spk_a": spk_a, "spk_b": spk_b,
            "bounds": (na, na + nd, T),  # phase boundaries in samples
        }


class RngShim:
    """random.Random -> numpy-ish shim for the two methods _cat_to_len uses."""

    def __init__(self, rng):
        self.rng = rng

    def choice(self, seq):
        return self.rng.choice(seq)

    def randint(self, lo, hi):
        return self.rng.randint(lo, max(lo, hi - 1))


def librispeech_index(root):
    """root: LibriSpeech split dir (e.g. .../train-clean-360). Returns spk2utts."""
    import pathlib
    spk2utts = {}
    for f in sorted(pathlib.Path(root).rglob("*.flac")):
        spk = f.parts[-3]
        spk2utts.setdefault(spk, []).append(str(f))
    return spk2utts
