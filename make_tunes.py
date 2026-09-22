"""Three 'locked in' focus tunes, synthesized from scratch with numpy (no samples).

    python make_tunes.py            -> renders all three .wav files next to this script
    python make_tunes.py 2          -> renders only tune 2

1. Deep Focus   - 78 BPM lo-fi study beat (F minor, jazzy EP chords, vinyl crackle)
2. Flow State   - 124 BPM hypnotic minimal techno (A minor, rolling bass, phasing 7-note arp)
3. Night Drive  - 100 BPM synthwave (D minor, octave bass, big pads, a lead hook)
"""
import functools
import os
import sys
import wave

import numpy as np

SR = 44100
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

_PC = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6,
       'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}


def m(name):
    """'A3' -> MIDI note number."""
    return 12 * (int(name[-1]) + 1) + _PC[name[:-1]]


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def tvec(n):
    return np.arange(n) / SR


def ns(sec):
    return max(1, int(round(sec * SR)))


def gate(t, dur, rel):
    """1 while the note is held, then a linear release to 0."""
    return np.clip(1 - (t - dur) / max(rel, 1e-4), 0, 1)


def edge(n, fin=0.0005, fout=0.005):
    e = np.ones(n)
    a, b = min(n, ns(fin)), min(n, ns(fout))
    e[:a] = np.linspace(0, 1, a)
    e[n - b:] *= np.linspace(1, 0, b)
    return e


def fft_filter(x, lo=None, hi=None, order=2):
    """Zero-phase Butterworth-magnitude low/high-pass done in the frequency domain."""
    n = x.shape[0]
    f = np.fft.rfftfreq(n, 1 / SR)
    H = np.ones_like(f)
    if hi:
        H /= np.sqrt(1 + (f / hi) ** (2 * order))
    if lo:
        H /= np.sqrt(1 + (lo / np.maximum(f, 1e-6)) ** (2 * order))
    if x.ndim == 2:
        H = H[:, None]
    return np.fft.irfft(np.fft.rfft(x, axis=0) * H, n, axis=0)


# ----------------------------------------------------------------------------- drums

@functools.lru_cache(maxsize=None)
def kick(f0=160.0, f1=48.0, pdec=0.035, adec=6.0, click=0.3, dur=0.5, drive=2.0):
    n = ns(dur)
    t = tvec(n)
    f = f1 + (f0 - f1) * np.exp(-t / pdec)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * adec)
    cl = fft_filter(np.random.default_rng(1).standard_normal(n), lo=1500, hi=9000)
    cl = cl / np.abs(cl).max() * np.exp(-t * 300) * click
    x = np.tanh(drive * (body + cl)) / np.tanh(drive)
    return x * edge(n)


@functools.lru_cache(maxsize=None)
def snare(tone=190.0, tdec=22.0, ndec=16.0, lo=1200.0, hi=7000.0, dur=0.4, mix=0.65, var=0):
    n = ns(dur)
    t = tvec(n)
    body = (np.sin(2 * np.pi * tone * t) + 0.5 * np.sin(2 * np.pi * tone * 1.62 * t)) * np.exp(-t * tdec)
    nz = fft_filter(np.random.default_rng(100 + var).standard_normal(n), lo=lo, hi=hi)
    nz = nz / np.abs(nz).max() * np.exp(-t * ndec)
    x = (1 - mix) * body + mix * nz * 1.4
    return x / np.abs(x).max() * edge(n)


@functools.lru_cache(maxsize=None)
def hat(decay=40.0, dur=0.15, lo=7000.0, var=0):
    n = ns(dur)
    t = tvec(n)
    nz = fft_filter(np.random.default_rng(200 + var).standard_normal(n), lo=lo, hi=16000)
    x = nz * np.exp(-t * decay)
    return x / np.abs(x).max() * edge(n)


@functools.lru_cache(maxsize=None)
def clap(var=0):
    n = ns(0.45)
    t = tvec(n)
    nz = fft_filter(np.random.default_rng(300 + var).standard_normal(n), lo=900, hi=5000)
    env = np.zeros(n)
    for d in (0.0, 0.009, 0.019):
        on = t >= d
        env[on] = np.maximum(env[on], np.exp(-(t[on] - d) * 220))
    tail = np.where(t >= 0.028, np.exp(-np.maximum(t - 0.028, 0) * 18), 0) * 0.7
    x = nz * np.maximum(env, tail)
    return x / np.abs(x).max() * edge(n)


@functools.lru_cache(maxsize=None)
def rim():
    n = ns(0.12)
    t = tvec(n)
    nz = fft_filter(np.random.default_rng(400).standard_normal(n), lo=2000, hi=8000)
    x = (np.sin(2 * np.pi * 1650 * t) * np.exp(-t * 60)
         + 0.6 * np.sin(2 * np.pi * 820 * t) * np.exp(-t * 45)
         + 0.4 * nz / np.abs(nz).max() * np.exp(-t * 120))
    return x / np.abs(x).max() * edge(n)


@functools.lru_cache(maxsize=None)
def tom(f0, dur=0.5):
    n = ns(dur)
    t = tvec(n)
    f = f0 * (1 + 0.5 * np.exp(-t / 0.04))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 6)
    nz = fft_filter(np.random.default_rng(500).standard_normal(n), lo=300, hi=3000)
    x = body + 0.15 * nz / np.abs(nz).max() * np.exp(-t * 30)
    return np.tanh(1.5 * x) * edge(n)


@functools.lru_cache(maxsize=None)
def crash(dur=2.8):
    n = ns(dur)
    t = tvec(n)
    nz = fft_filter(np.random.default_rng(600).standard_normal(n), lo=3500, hi=15000)
    x = nz / np.abs(nz).max() * np.exp(-t * 1.6) * (1 - np.exp(-t * 800))
    return x * edge(n)


# ----------------------------------------------------------------------------- tonal

def saw(freq, n, cutoff, res=0.0, phase=0.0, vib=None):
    """Band-limited sawtooth built from harmonics, each weighted by a 12 dB/oct
    low-pass response at `cutoff` (scalar or per-sample array) with an optional resonant bump."""
    t = tvec(n)
    w = 2 * np.pi * (freq * t if vib is None else np.cumsum(freq * vib) / SR) + phase
    cutoff = np.asarray(cutoff, float)
    kmax = int(max(1, min(18000 / freq, 4 * float(cutoff.max()) / freq + 2, 160)))
    out = np.zeros(n)
    for k in range(1, kmax + 1):
        r = k * freq / cutoff
        g = 1 / np.sqrt(1 + r ** 4)
        if res:
            g = g * (1 + res * np.exp(-((r - 1) / 0.3) ** 2))
        out += g * np.sin(k * w) / k
    return out


@functools.lru_cache(maxsize=None)
def pluck(midi, dur, base, amt=2.5, fdec=0.06, res=0.4, adec=10.0, rel=0.05):
    n = ns(dur + rel)
    t = tvec(n)
    x = saw(hz(midi), n, base * (1 + amt * np.exp(-t / fdec)), res)
    return x * np.exp(-t * adec) * (1 - np.exp(-t * 800)) * gate(t, dur, rel)


@functools.lru_cache(maxsize=None)
def epiano(midi, dur, vel):
    """Two-operator FM electric piano with a short bell 'tine'."""
    f = hz(midi)
    rel = 0.35
    n = ns(dur + rel)
    t = tvec(n)
    idx = (0.6 + 1.4 * vel) * np.exp(-t * 3.0) + 0.12
    body = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * t))
    tine = 0.22 * vel * np.sin(2 * np.pi * f * 7 * t) * np.exp(-t * 20)
    env = np.exp(-t * (0.7 + f / 1000)) * (1 - np.exp(-t * 350)) * gate(t, dur, rel)
    return (body + tine) * env * (0.4 + 0.6 * vel)


@functools.lru_cache(maxsize=None)
def vibes(midi, dur, vel):
    f = hz(midi)
    rel = 0.6
    n = ns(dur + rel)
    t = tvec(n)
    x = (np.sin(2 * np.pi * f * t)
         + 0.25 * np.sin(2 * np.pi * 4 * f * t) * np.exp(-t * 7)
         + 0.08 * np.sin(2 * np.pi * 10 * f * t) * np.exp(-t * 25))
    env = np.exp(-t * 1.6) * (1 - np.exp(-t * 600)) * gate(t, dur, rel)
    trem = 1 - 0.22 * (0.5 + 0.5 * np.sin(2 * np.pi * 4.8 * t))
    return x * env * trem * vel


@functools.lru_cache(maxsize=None)
def sub_bass(midi, dur, vel=1.0):
    f = hz(midi)
    rel = 0.08
    n = ns(dur + rel)
    t = tvec(n)
    x = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t)
         + 0.12 * np.sin(2 * np.pi * 3 * f * t))
    env = (1 - np.exp(-t * 250)) * np.exp(-t * 0.8) * gate(t, dur, rel)
    return np.tanh(1.3 * x * env) * vel


@functools.lru_cache(maxsize=None)
def pad(midis, dur, cutoff=1500.0, att=0.8, rel=1.5, detune=9.0):
    """Three detuned saws per note, spread left/centre/right."""
    n = ns(dur + rel)
    t = tvec(n)
    r = np.random.default_rng(sum(midis))
    L = np.zeros(n)
    R = np.zeros(n)
    for mm in midis:
        f = hz(mm)
        for j, c in enumerate((-detune, 0.0, detune)):
            x = saw(f * 2 ** (c / 1200), n, cutoff, phase=r.uniform(0, 2 * np.pi))
            if j == 0:
                L += x
            elif j == 2:
                R += x
            else:
                L += 0.7 * x
                R += 0.7 * x
    env = (0.5 - 0.5 * np.cos(np.pi * np.clip(t / att, 0, 1))) * gate(t, dur, rel)
    return np.stack([L, R], axis=1) * env[:, None] / len(midis)


@functools.lru_cache(maxsize=None)
def lead(midi, dur):
    f = hz(midi)
    rel = 0.18
    n = ns(dur + rel)
    t = tvec(n)
    vib = 2 ** (18 * np.clip((t - 0.22) / 0.35, 0, 1) * np.sin(2 * np.pi * 5.3 * t) / 1200)
    cut = 2600 + 2400 * np.exp(-t / 0.12)
    a = saw(f * 2 ** (-6 / 1200), n, cut, 0.25, vib=vib)
    b = saw(f * 2 ** (6 / 1200), n, cut, 0.25, phase=1.3, vib=vib)
    env = np.minimum(1, t / 0.008) * (0.75 + 0.25 * np.exp(-t * 5)) * gate(t, dur, rel)
    return np.stack([a + 0.6 * b, b + 0.6 * a], axis=1) * env[:, None]


def vinyl(seconds, seed=3):
    n = ns(seconds)
    r = np.random.default_rng(seed)
    out = np.zeros((n, 2))
    for c in range(2):
        x = np.zeros(n)
        k = r.poisson(10 * seconds)
        np.add.at(x, r.integers(0, n, k), r.exponential(1, k) * r.choice([-1, 1], k))
        x = np.convolve(x, np.exp(-np.arange(40) / 6), mode='same')
        x = fft_filter(x, lo=800, hi=9000)
        hiss = fft_filter(r.standard_normal(n), lo=300, hi=5000) * 0.08
        out[:, c] = x + hiss
    return out


# ----------------------------------------------------------------------------- mixing

class Bus:
    def __init__(self, seconds):
        self.x = np.zeros((ns(seconds), 2))

    def add(self, sig, t0, gain=1.0, pan=0.0):
        i = int(round(t0 * SR))
        if i < 0:
            sig, i = sig[-i:], 0
        e = min(i + len(sig), len(self.x))
        if e <= i:
            return
        seg = sig[:e - i] * gain
        if seg.ndim == 1:
            a = (pan + 1) * np.pi / 4
            self.x[i:e, 0] += seg * np.cos(a) * np.sqrt(2)
            self.x[i:e, 1] += seg * np.sin(a) * np.sqrt(2)
        else:
            self.x[i:e] += seg


def make_ir(rt=2.0, pre=0.02, hi=6000.0, lo=200.0, gated=None, seed=0):
    r = np.random.default_rng(seed)
    length = gated if gated else rt * 1.3
    n = ns(length)
    t = tvec(n)
    env = np.exp(-t * 1.5) * np.clip((length - t) / 0.02, 0, 1) if gated else np.exp(-t * 6.9 / rt)
    ir = fft_filter(r.standard_normal((n, 2)) * env[:, None], lo=lo, hi=hi)
    ir *= np.clip(t / 0.004, 0, 1)[:, None]
    ir = np.vstack([np.zeros((ns(pre), 2)), ir])
    return ir / np.sqrt((ir ** 2).sum(axis=0))


def reverb(x, ir):
    mono = x.mean(axis=1)
    nfft = 1 << (len(mono) + len(ir) - 2).bit_length()
    M = np.fft.rfft(mono, nfft)
    out = np.empty((len(mono), 2))
    for c in range(2):
        out[:, c] = np.fft.irfft(M * np.fft.rfft(ir[:, c], nfft), nfft)[:len(mono)]
    return out


def pingpong(x, dt, fb=0.45, reps=6, lo=250.0, hi=4500.0):
    d = ns(dt)
    src = fft_filter(x.mean(axis=1), lo=lo, hi=hi)
    out = np.zeros_like(x)
    for i in range(1, reps + 1):
        sh = d * i
        if sh >= len(src):
            break
        out[sh:, (i - 1) % 2] += src[:len(src) - sh] * fb ** (i - 1)
    return out


def wow(x, depth=0.0011, rate=0.4, flut=0.00005, frate=5.5):
    """Tape wow & flutter: a slowly modulated delay line."""
    n = len(x)
    t = tvec(n)
    d = 0.003 + depth * np.sin(2 * np.pi * rate * t) + flut * np.sin(2 * np.pi * frate * t)
    idx = np.arange(n)
    return np.stack([np.interp(idx - d * SR, idx, x[:, c]) for c in range(2)], axis=1)


def pump(n, kicks, depth=0.6, rel=0.1):
    """Sidechain-style gain curve that ducks after every kick."""
    c = np.ones(n)
    L = ns(0.45)
    shape = 1 - depth * np.exp(-tvec(L) / rel)
    a = ns(0.004)
    shape[:a] = np.linspace(1, shape[a], a)
    for k in kicks:
        i = int(k * SR)
        if i >= n:
            continue
        e = min(i + L, n)
        c[i:e] = np.minimum(c[i:e], shape[:e - i])
    return c[:, None]


def norm_rms(x, target_db):
    """Scale so the RMS of the bus's active (non-silent) stretches hits target_db."""
    fl = 2205
    nfr = len(x) // fl
    e = np.sqrt((x[:nfr * fl].reshape(nfr, fl, 2) ** 2).mean(axis=(1, 2)))
    if e.max() == 0:
        return x
    rms = np.sqrt((e[e > 0.1 * e.max()] ** 2).mean())
    return x * (10 ** (target_db / 20) / rms)


def finish(mix, filename, total, bar, fade=6.0):
    mix = fft_filter(mix[:ns(total + 1.5)], lo=30.0)
    p = np.percentile(np.abs(mix), 99.95)
    y = np.tanh(1.3 * mix / p) * 0.97
    t = tvec(len(y))
    y *= (np.clip((t[-1] - t) / fade, 0, 1) ** 1.5 * np.clip(t / 0.02, 0, 1))[:, None]
    path = os.path.join(OUT_DIR, filename)
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype('<i2').tobytes())
    blk = ns(8 * bar)
    per8 = [20 * np.log10(np.sqrt((y[i:i + blk] ** 2).mean()) + 1e-9) for i in range(0, len(y), blk)]
    print(f'{filename}: {len(y) / SR:.1f}s  peak {np.abs(y).max():.3f}  '
          f'rms {20 * np.log10(np.sqrt((y ** 2).mean())):.1f} dBFS  nan={np.isnan(y).any()}')
    print('   rms per 8 bars:', ' '.join(f'{v:.0f}' for v in per8))


# ============================================================================ 1. Deep Focus

def deep_focus():
    bpm = 78
    beat = 60 / bpm
    s16 = beat / 4
    bar = 4 * beat
    nbars = 48
    total = nbars * bar
    swing = 0.22
    rng = np.random.default_rng(11)

    def T(b, step, jit=0.004):
        t = b * bar + step * s16
        if int(round(step)) % 2 == 1:
            t += swing * s16
        return t + rng.normal(0, jit)

    prog = [('F2', ['Ab3', 'C4', 'Eb4', 'G4']),    # Fm9
            ('Bb1', ['Ab3', 'C4', 'Db4', 'F4']),   # Bbm9
            ('Eb2', ['G3', 'C4', 'Db4', 'F4']),    # Eb13
            ('Ab1', ['G3', 'Bb3', 'C4', 'Eb4']),   # Abmaj9
            ('Db2', ['F3', 'Ab3', 'C4', 'Eb4']),   # Dbmaj9
            ('G1', ['F3', 'Bb3', 'Db4', 'F4']),    # Gm7b5
            ('C2', ['E3', 'Bb3', 'Db4', 'G4']),    # C7b9
            ('C2', ['E3', 'G3', 'Bb3', 'Db4'])]    # C7b9
    melody = [[(0, 'C5', 4), (6, 'Eb5', 2), (8, 'F5', 6)],
              [(2, 'Db5', 2), (4, 'C5', 4), (10, 'Ab4', 4)],
              [(0, 'Bb4', 6), (8, 'C5', 2), (10, 'Bb4', 2), (12, 'G4', 4)],
              [(0, 'Ab4', 8), (12, 'Eb5', 4)],
              [(0, 'F5', 4), (4, 'Eb5', 2), (6, 'C5', 6), (14, 'Ab4', 2)],
              [(0, 'Bb4', 6), (8, 'Db5', 4), (12, 'C5', 4)],
              [(0, 'Bb4', 4), (4, 'G4', 4), (8, 'E4', 6)],
              [(0, 'Db5', 4), (4, 'C5', 10)]]

    L = total + 6
    mix = np.zeros((ns(L), 2))
    send = np.zeros_like(mix)
    t_all = tvec(len(mix))

    # drums (bars 8-43, with a two-beat dropout closing bars 23 and 39)
    dr = Bus(L)
    kicks = []
    for b in range(8, 44):
        brk = b in (23, 39)
        ks = [(0, 1.0), (10, 0.8)] if b % 2 == 0 else [(0, 1.0), (7, 0.55), (10, 0.8)]
        if b % 8 == 7:
            ks.append((14, 0.6))
        for st, v in ks:
            if brk and st >= 8:
                continue
            t0 = T(b, st)
            dr.add(kick(150.0, 48.0, 0.04, 5.0, 0.15, 0.6, 1.6), t0, v)
            kicks.append(t0)
        for st in (4, 12):
            if brk and st >= 8:
                continue
            dr.add(snare(205.0, 25.0, 20.0, 1000.0, 5500.0, 0.35, 0.6, b % 3), T(b, st), 0.75 + rng.normal(0, 0.04))
        if not brk and rng.random() < 0.35:
            dr.add(snare(205.0, 25.0, 20.0, 1000.0, 5500.0, 0.35, 0.6, 0), T(b, 15), 0.18)
        for st in range(0, 16, 2):
            v = 0.42 if st % 4 == 0 else 0.3
            dr.add(hat(55.0, 0.12, 6500.0, st % 4), T(b, st), v * rng.uniform(0.85, 1.1), pan=0.2)
        for st in (3, 11):
            if rng.random() < 0.5:
                dr.add(hat(70.0, 0.1, 7000.0, 5), T(b, st), 0.14, pan=0.25)
    x = fft_filter(dr.x, hi=8500)
    x = np.tanh(2 * x / np.abs(x).max())
    x = norm_rms(x, -14)
    mix += x
    send += x * 0.12
    del dr, x

    # electric piano: filter opens over bars 6-8, closes again in the outro
    ep = Bus(L)
    for b in range(nbars):
        voic = prog[b % 8][1]
        patt = [(0, 7, 0.75), (10, 5, 0.55)] if b % 4 != 3 else [(0, 5, 0.75), (6, 3, 0.45), (12, 4, 0.6)]
        for st, dsteps, vel in patt:
            t0 = T(b, st)
            for i, v in enumerate(voic):
                vv = round(float(np.clip(vel + rng.normal(0, 0.05), 0.2, 1)) * 20) / 20
                ep.add(epiano(m(v), round(dsteps * s16, 3), vv), t0 + i * 0.011, 0.9, pan=-0.3 + 0.2 * i)
    c = np.minimum(np.clip((t_all - 6 * bar) / (2 * bar), 0, 1),
                   np.clip(1 - (t_all - 44 * bar) / (3.5 * bar), 0, 1))[:, None]
    x = fft_filter(ep.x, hi=700) * (1 - c) + fft_filter(ep.x, hi=5500) * c
    x = wow(x) * pump(len(x), kicks, 0.15, 0.15)
    x = norm_rms(x, -17)
    mix += x
    send += x * 0.35
    del ep, x

    # bass
    bs = Bus(L)
    for b in range(8, 44):
        root = m(prog[b % 8][0])
        notes = [(0, root, 7)] if b in (23, 39) else [(0, root, 9), (10, root, 3), (14, root + 12, 2)]
        for st, nn, d in notes:
            bs.add(sub_bass(nn, round(d * s16, 3), 0.6 if st == 14 else 1.0), T(b, st, 0.003))
    x = bs.x * pump(len(bs.x), kicks, 0.35, 0.12)
    x = norm_rms(x, -17)
    mix += x
    del bs, x

    # melody (vibes), bars 16-23 and 32-39
    mel = Bus(L)
    for b in list(range(16, 24)) + list(range(32, 40)):
        for st, nm, d in melody[b % 8]:
            mel.add(vibes(m(nm), round(d * s16, 3), 0.8), T(b, st, 0.006), 1.0, pan=0.1)
    x = wow(mel.x)
    x = x + pingpong(x, 3 * beat / 4, 0.4) * 0.35
    x = norm_rms(x, -20)
    mix += x
    send += x * 0.4
    del mel, x

    # counter-arpeggio, bars 24-31
    cnt = Bus(L)
    for b in range(24, 32):
        tones = [m(v) + 12 for v in prog[b % 8][1]]
        for j, st in enumerate([0, 3, 6, 8, 11, 14]):
            cnt.add(vibes(tones[j % len(tones)], round(2 * s16, 3), 0.5), T(b, st), 1.0, pan=0.35 if j % 2 else -0.35)
    x = cnt.x + pingpong(cnt.x, 3 * beat / 4, 0.4) * 0.4
    x = norm_rms(x, -24)
    mix += x
    send += x * 0.5
    del cnt, x

    mix += norm_rms(vinyl(L), -36)
    mix += reverb(send, make_ir(1.8, 0.02, 5000.0))
    finish(mix, '1 - Deep Focus (lo-fi, 78 bpm).wav', total, bar)


# ============================================================================ 2. Flow State

def flow_state():
    bpm = 124
    beat = 60 / bpm
    s16 = beat / 4
    bar = 4 * beat
    nbars = 80
    total = nbars * bar
    rng = np.random.default_rng(22)

    def T(b, st):
        return b * bar + st * s16

    def brk(b):
        return 32 <= b < 40

    L = total + 6
    mix = np.zeros((ns(L), 2))
    send = np.zeros_like(mix)

    # kick: four on the floor everywhere except the breakdown
    kb = Bus(L)
    kicks = []
    for b in range(nbars):
        if brk(b):
            continue
        for st in (0, 4, 8, 12):
            t0 = T(b, st)
            kb.add(kick(170.0, 50.0, 0.03, 7.0, 0.35, 0.45, 2.5), t0)
            kicks.append(t0)
    x = norm_rms(kb.x, -13)
    mix += x
    send += x * 0.03
    del kb, x

    # percussion
    pc = Bus(L)
    for b in range(nbars):
        if brk(b):
            continue
        for st in range(16):
            if b < 72 or st % 2 == 0:
                v = 0.55 if st % 4 == 2 else 0.3
                pc.add(hat(60.0, 0.1, 8000.0, st % 4), T(b, st), v * rng.uniform(0.9, 1.1), pan=0.15)
        if 8 <= b < 72:
            for st in (2, 6, 10, 14):
                pc.add(hat(12.0, 0.3, 6000.0, 7), T(b, st), 0.45, pan=-0.1)
        if 16 <= b < 32 or 40 <= b < 64:
            for st in (4, 12):
                pc.add(clap(b % 2), T(b, st), 0.7)
        if 40 <= b < 64:
            for st in (3, 10, 13):
                pc.add(rim(), T(b, st), 0.4, pan=0.3)
    # snare roll into the drop
    for i, st in enumerate(range(0, 16, 2)):
        pc.add(snare(230.0, 30.0, 22.0, 1500.0, 8000.0, 0.25, 0.7, i % 3), T(38, st), 0.15 + 0.03 * i)
    for i in range(8):
        pc.add(snare(230.0, 30.0, 22.0, 1500.0, 8000.0, 0.25, 0.7, i % 3), T(39, i), 0.4 + 0.03 * i)
    for i in range(16):
        pc.add(snare(230.0, 30.0, 22.0, 1500.0, 8000.0, 0.25, 0.7, i % 3), T(39, 8 + i / 2), 0.65 + 0.022 * i)
    riser_n = ns(4 * bar)
    rt = tvec(riser_n)
    riser = fft_filter(np.random.default_rng(7).standard_normal(riser_n), lo=3000, hi=12000)
    riser = riser / np.abs(riser).max() * (rt / rt[-1]) ** 3
    pc.add(riser, T(36, 0), 0.5)
    for b in (16, 40):
        pc.add(crash(), T(b, 0), 0.5)
    x = norm_rms(fft_filter(pc.x, hi=11000.0), -21)
    mix += x
    send += x * 0.12
    del pc, x

    # rolling bass
    bs = Bus(L)
    for b in range(8, 72):
        if brk(b):
            continue
        roots = ['A1', 'A1', 'A1', 'A1'] if b < 16 else ['A1', 'A1', 'C2', 'G1']
        root = m(roots[b % 4])
        for st in (2, 3, 6, 7, 10, 11, 14, 15):
            note = root + (12 if (st == 15 and b % 2 == 1) else 0)
            x = pluck(note, round(s16 * 0.9, 4), 180.0, 5.0, 0.045, 0.6, 9.0, 0.02)
            bs.add(np.tanh(1.8 * x), T(b, st), 1.0 if st % 2 == 0 else 0.7)
    x = fft_filter(bs.x, lo=35.0) * pump(len(bs.x), kicks, 0.7, 0.09)
    x = norm_rms(x, -16)
    mix += x
    send += x * 0.02
    del bs, x

    # hypnotic arp: a 7-note cycle phasing against the 16-step grid
    seq = [m(v) for v in ['A3', 'E4', 'C4', 'A4', 'G4', 'E4', 'B3']]

    def arp_cut(pos):
        if pos < 32:
            return 350 + (pos - 16) / 16 * 1800
        if pos < 40:
            return 900
        if pos < 64:
            return 2200
        return 2200 - (pos - 64) / 8 * 1700

    ar = Bus(L)
    g = 0
    for b in range(16, 72):
        for st in range(16):
            t0 = T(b, st)
            base = arp_cut(b + st / 16) * (1 + 0.45 * np.sin(2 * np.pi * t0 / (8 * bar)))
            base = float(max(200, round(base / 40) * 40))
            acc = 1.0 if st % 4 == 2 else (0.8 if st % 2 == 0 else 0.65)
            ar.add(pluck(seq[g % 7], round(s16 * 1.2, 4), base, 3.0, 0.05, 0.5, 11.0, 0.03),
                   t0, acc, pan=0.25 if g % 2 else -0.25)
            g += 1
    x = ar.x + pingpong(ar.x, 3 * s16, 0.45) * 0.4
    x = x * pump(len(x), kicks, 0.45, 0.1)
    x = norm_rms(x, -19)
    mix += x
    send += x * 0.3
    del ar, x

    # pad: full in the breakdown, quiet bed under the drop
    am9 = tuple(m(v) for v in ['A2', 'E3', 'G3', 'B3', 'C4'])
    fmaj9 = tuple(m(v) for v in ['F2', 'C3', 'E3', 'G3', 'A3'])
    pd = Bus(L)
    pd.add(pad(am9, 4 * bar, 1600.0, 1.8, 2.5), T(32, 0))
    pd.add(pad(fmaj9, 4 * bar, 1600.0, 1.8, 2.5), T(36, 0))
    for b in range(40, 64, 4):
        pd.add(pad(am9, 4 * bar, 1600.0, 1.8, 2.5), T(b, 0), 0.4)
    x = pd.x * pump(len(pd.x), kicks, 0.6, 0.12)
    x = norm_rms(x, -20)
    mix += x
    send += x * 0.5
    del pd, x

    # dub stabs in the drop
    sb = Bus(L)
    for b in range(40, 64):
        hits = [14] if b % 2 == 1 else ([6] if b % 4 == 0 else [])
        for st in hits:
            for i, v in enumerate(['A3', 'C4', 'E4', 'G4']):
                sb.add(pluck(m(v), 0.15, 900.0, 2.0, 0.05, 0.3, 7.0, 0.05), T(b, st), 1.0, pan=-0.2 + 0.13 * i)
    x = sb.x + pingpong(sb.x, 3 * s16, 0.55, 8) * 0.6
    x = x * pump(len(x), kicks, 0.5, 0.1)
    x = norm_rms(x, -22)
    mix += x
    send += x * 0.6
    del sb, x

    mix += reverb(send, make_ir(2.6, 0.03, 6000.0, seed=2))
    finish(mix, '2 - Flow State (minimal techno, 124 bpm).wav', total, bar)


# ============================================================================ 3. Night Drive

def night_drive():
    bpm = 100
    beat = 60 / bpm
    s16 = beat / 4
    bar = 4 * beat
    nbars = 64
    total = nbars * bar
    rng = np.random.default_rng(33)

    def T(b, st):
        return b * bar + st * s16

    prog = [('D2', ['D3', 'A3', 'D4', 'F4']),     # Dm
            ('Bb1', ['Bb2', 'Bb3', 'D4', 'F4']),  # Bb
            ('F2', ['F3', 'A3', 'C4', 'F4']),     # F
            ('C2', ['C3', 'G3', 'C4', 'E4'])]     # C
    hook = [[(0, 'A4', 4), (4, 'D5', 4), (8, 'F5', 6), (14, 'E5', 2)],
            [(0, 'D5', 8), (8, 'C5', 4), (12, 'Bb4', 4)],
            [(0, 'A4', 4), (4, 'C5', 4), (8, 'F5', 4), (12, 'A5', 4)],
            [(0, 'G5', 6), (6, 'F5', 2), (8, 'E5', 4), (12, 'C5', 4)],
            [(0, 'F5', 4), (4, 'E5', 2), (6, 'D5', 6), (12, 'A4', 4)],
            [(0, 'Bb4', 4), (4, 'D5', 4), (8, 'F5', 8)],
            [(0, 'A5', 6), (6, 'G5', 2), (8, 'F5', 4), (12, 'C5', 4)],
            [(0, 'E5', 8), (8, 'C5', 4), (12, 'E5', 4)]]

    def drums_on(b):
        return 8 <= b < 32 or 40 <= b < 56

    def chorus(b):
        return 16 <= b < 32 or 40 <= b < 56

    L = total + 6
    mix = np.zeros((ns(L), 2))
    send = np.zeros_like(mix)
    t_all = tvec(len(mix))

    # drums + gated-reverb snare
    dr = Bus(L)
    sn = Bus(L)
    hh = Bus(L)
    kicks = []
    for b in range(nbars):
        if b in (8, 16, 40, 48, 56):
            hh.add(crash(), T(b, 0), 0.6)
        if not drums_on(b):
            continue
        fill = b % 8 == 7
        for st in [0, 8] + ([10] if b % 2 == 1 else []):
            t0 = T(b, st)
            dr.add(kick(155.0, 45.0, 0.04, 5.5, 0.25, 0.55, 2.0), t0)
            kicks.append(t0)
        for st in (4, 12):
            if fill and st == 12:
                continue
            s = snare(180.0, 18.0, 12.0, 900.0, 7000.0, 0.4, 0.6, b % 3)
            dr.add(s, T(b, st), 0.8)
            sn.add(s, T(b, st), 0.8)
        if fill:
            for st, f0 in zip((8, 10, 12, 14), (210.0, 170.0, 140.0, 110.0)):
                dr.add(tom(f0), T(b, st), 0.8, pan=0.4 - 0.25 * (st - 8) / 2)
        for st in range(0, 16, 1 if chorus(b) else 2):
            if fill and st >= 8:
                break
            v = 0.5 if st % 4 == 2 else (0.35 if st % 2 == 0 else 0.22)
            hh.add(hat(45.0, 0.12, 7500.0, st % 4), T(b, st), v * rng.uniform(0.9, 1.1), pan=-0.2)
    x = dr.x + reverb(sn.x, make_ir(0.9, 0.005, 7000.0, 300.0, gated=0.26, seed=5)) * 0.8
    x = norm_rms(x, -14)
    mix += x
    send += x * 0.08
    x = norm_rms(hh.x, -22)
    mix += x
    send += x * 0.1
    del dr, sn, hh, x

    # pads: filter opens over bars 4-8, closes in the outro
    pd = Bus(L)
    for b in range(nbars):
        pd.add(pad(tuple(m(v) for v in prog[b % 4][1]), bar, 1500.0, 0.25, 0.7), T(b, 0))
    c = np.minimum(np.clip((t_all - 4 * bar) / (4 * bar), 0, 1),
                   np.clip(1 - (t_all - 57 * bar) / (6 * bar), 0, 1))[:, None]
    x = fft_filter(pd.x, hi=500) * (1 - c) + pd.x * c
    x = x * pump(len(x), kicks, 0.45, 0.12)
    x = norm_rms(x, -19)
    mix += x
    send += x * 0.4
    del pd, x

    # arp
    ar = Bus(L)
    for b in range(nbars):
        v = [m(n_) for n_ in prog[b % 4][1][1:]]
        seq = [v[0], v[1], v[2], v[0] + 12, v[1] + 12, v[0] + 12, v[2], v[1]]
        if b < 8:
            base = 600 + 60 * b
        elif b < 16:
            base = 1000.0
        elif b < 32:
            base = 1500.0
        elif b < 40:
            base = 900.0
        elif b < 56:
            base = 1800.0
        else:
            base = 1800 - 150 * (b - 56)
        for st in range(16):
            ar.add(pluck(seq[st % 8], round(s16 * 1.3, 4), float(base), 2.5, 0.05, 0.4, 9.0, 0.04),
                   T(b, st), 1.0 if st % 4 == 0 else 0.75, pan=0.3 if st % 2 else -0.3)
    x = ar.x + pingpong(ar.x, 3 * beat / 4, 0.4) * 0.4
    x = x * pump(len(x), kicks, 0.35, 0.1)
    x = norm_rms(x, -20)
    mix += x
    send += x * 0.3
    del ar, x

    # bass: octave-bouncing eighths, long sub notes in the breakdown
    bs = Bus(L)
    for b in range(8, 56):
        root = m(prog[b % 4][0])
        if 32 <= b < 40:
            bs.add(sub_bass(root, round(bar * 0.95, 3), 0.8), T(b, 0))
            continue
        for i, st in enumerate(range(0, 16, 2)):
            x = pluck(root + (12 if i % 2 else 0), round(s16 * 1.7, 4), 260.0, 4.0, 0.07, 0.5, 5.0, 0.03)
            bs.add(np.tanh(1.5 * x), T(b, st))
    x = fft_filter(bs.x, lo=35.0) * pump(len(bs.x), kicks, 0.5, 0.1)
    x = norm_rms(x, -16)
    mix += x
    del bs, x

    # lead hook, bars 16-31 and 40-55 (with an octave-up double on the last pass)
    ld = Bus(L)
    for b in list(range(16, 32)) + list(range(40, 56)):
        for st, nm, d in hook[b % 8]:
            ld.add(lead(m(nm), round(d * s16 * 0.92, 3)), T(b, st))
            if b >= 48:
                ld.add(lead(m(nm) + 12, round(d * s16 * 0.92, 3)), T(b, st), 0.3)
    x = ld.x + pingpong(ld.x, 3 * beat / 4, 0.35) * 0.3
    x = norm_rms(x, -16)
    mix += x
    send += x * 0.35
    del ld, x

    mix += reverb(send, make_ir(2.4, 0.025, 6500.0, seed=3))
    finish(mix, '3 - Night Drive (synthwave, 100 bpm).wav', total, bar)


if __name__ == '__main__':
    songs = {'1': deep_focus, '2': flow_state, '3': night_drive}
    for k in (sys.argv[1:] or ['1', '2', '3']):
        songs[k]()
        for fn in (kick, snare, hat, clap, rim, tom, crash, pluck, epiano, vibes, sub_bass, pad, lead):
            fn.cache_clear()
