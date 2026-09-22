# locked-in-tunes

Three focus tracks synthesized entirely from code — no samples, no DAW, no AI music model.
Every kick, snare, synth and chord is built from raw math with numpy and written straight to a `.wav`.

| # | Track | Style | Tempo / Key | What's going on |
|---|-------|-------|-------------|-----------------|
| 1 | **Deep Focus** | lo-fi study beat | 78 BPM, F minor | Jazzy FM electric piano with tape wobble, swung boom-bap drums, vibraphone melody, vinyl crackle |
| 2 | **Flow State** | hypnotic minimal techno | 124 BPM, A minor | Four-on-the-floor, sidechained rolling bass, a 7-note arp phasing against the 16-step grid, dub stabs |
| 3 | **Night Drive** | synthwave | 100 BPM, D minor | Dm–B♭–F–C, octave bass, gated-reverb snare, supersaw pads, lead hook |

**Listen:** grab the WAVs from the [Releases page](../../releases/latest).

## Render them yourself

Needs Python 3 and numpy.

```bash
pip install numpy
python make_tunes.py        # renders all three (~1 minute)
python make_tunes.py 2      # renders just one
```

## How it works

Everything lives in [`make_tunes.py`](make_tunes.py):

- **Drums** — kick is a sine with a fast pitch drop plus a noise click; snares, hats and claps are filtered noise with shaped envelopes.
- **Synths** — saws are built additively, harmonic by harmonic, with each harmonic weighted by a low-pass filter curve (so filter sweeps and resonance are cheap and alias-free). The electric piano and vibes are small FM synths.
- **Mixing** — a `Bus` per instrument group, sidechain "pump" curves keyed off the kick times, ping-pong delay, convolution reverb (with a gated variant for the synthwave snare), tape wow/flutter for the lo-fi track, and a soft-clip master.
- **Arrangement** — each song is plain Python: chord progressions, melodies and drum patterns as lists of `(step, note, length)`, laid out bar by bar into intro / build / breakdown / drop / outro sections.

Tweak a chord list, a tempo or a drum pattern and re-run to make your own version.
