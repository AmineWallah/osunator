"""key_diagnostics.py — does the key head PRESS, and on time?"""
import numpy as np
from tensorflow import keras
from config import load_norm_stats
from src.osunator.parsing import beatmap_replay_pairs, build_training_example
from predict_replay import generate_replay

REPLAY_PATH = '/home/amine/PycharmProjects/osunator/replays/suitable/INFERNOBESTMAP - GALNERYUS - RAISE MY SWORD [A THOUSAND FLAMES] (2023-10-08) Osu.osr'
MODEL_PATH = '../best_model.keras'
THRESHOLD = 0.4

stats = load_norm_stats()
model = keras.models.load_model(MODEL_PATH, compile=False)
beatmap, _, replay, path = next(beatmap_replay_pairs([REPLAY_PATH]))
example = build_training_example(beatmap, replay)
gen = generate_replay(model, beatmap, stats, temperature=0.0)

n = min(len(example['cursor_x']), len(gen['pred_cursor_x']))
h_on = np.asarray(example['key_onset'], bool)[:, :n]          # (2, n)
g_prob = gen['pred_key_onset'][:n]                            # (n, 2)
g_on = (g_prob > THRESHOLD).T                                 # (2, n)

print(f"map: {path}\n")
print("[1] raw probability stats (is the head even trying?)")
for s, name in [(0, 'K1'), (1, 'K2')]:
    p = g_prob[:, s]
    print(f"    onset[{name}]: mean {p.mean():.3f}  p95 {np.percentile(p,95):.3f}  "
          f"p99 {np.percentile(p,99):.3f}  max {p.max():.3f}  >thr: {(p>THRESHOLD).sum()}")

print(f"\n[2] press counts (onsets), threshold {THRESHOLD}")
for s, name in [(0, 'K1'), (1, 'K2')]:
    print(f"    {name}: human {int(h_on[s].sum())}   generated {int(g_on[s].sum())}")
print(f"    total: human {int(h_on.sum())}   generated {int(g_on.sum())}")

print("\n[3] onset timing: each generated onset -> nearest human onset (either slot)")
h_ticks = np.flatnonzero(h_on.any(axis=0))
g_ticks = np.flatnonzero(g_on.any(axis=0))
if len(g_ticks) and len(h_ticks):
    d = np.array([np.abs(h_ticks - t).min() for t in g_ticks])
    print(f"    generated onsets: {len(g_ticks)}")
    print(f"    |error| ticks: median {np.median(d):.1f}  p90 {np.percentile(d,90):.1f}  "
          f"within ±2: {(d<=2).mean():.0%}  within ±5: {(d<=5).mean():.0%}")
    hd = np.array([np.abs(g_ticks - t).min() for t in h_ticks])
    print(f"    human onsets with a generated onset within ±3: {(hd<=3).mean():.0%}  (recall proxy)")
else:
    print("    no generated onsets above threshold — see [1] for how close it gets")

# ---------------------------------------------------------------------------
# [4] offset head audit (never measured before)
# ---------------------------------------------------------------------------
h_off = np.asarray(example['key_offset'], bool)[:, :n]        # (2, n)
g_off_prob = gen['pred_key_offset'][:n]                       # (n, 2)
g_off = (g_off_prob > THRESHOLD).T                            # (2, n)

print(f"\n[4] offset (release) head, threshold {THRESHOLD}")
for s, name in [(0, 'K1'), (1, 'K2')]:
    p = g_off_prob[:, s]
    print(f"    offset[{name}]: p95 {np.percentile(p,95):.3f}  p99 {np.percentile(p,99):.3f}  "
          f"max {p.max():.3f}  releases: human {int(h_off[s].sum())}  gen {int(g_off[s].sum())}")

# ---------------------------------------------------------------------------
# [5] press durations: reconstruct held state EXACTLY like result_to_replay,
#     measure onset->release gaps. This is what the client actually plays.
# ---------------------------------------------------------------------------
def press_durations(onsets, offsets):
    """onsets/offsets: (2, n) bool. Returns list of press lengths in ticks,
    walking the same held-state logic as result_to_replay."""
    durs = []
    for s in (0, 1):
        held_since = None
        for t in range(onsets.shape[1]):
            if onsets[s, t] and held_since is None:
                held_since = t
            if offsets[s, t] and held_since is not None:
                durs.append(t - held_since)
                held_since = None
    return np.array(durs)

d_h = press_durations(h_on, h_off)
d_g = press_durations(g_on, g_off)

print(f"\n[5] press durations (ticks; 1 tick = 16.7ms)")
print(f"    human: n={len(d_h)}  median {np.median(d_h):.0f}  p10 {np.percentile(d_h,10):.0f}  "
      f"p90 {np.percentile(d_h,90):.0f}")
if len(d_g):
    print(f"    gen  : n={len(d_g)}  median {np.median(d_g):.0f}  p10 {np.percentile(d_g,10):.0f}  "
          f"p90 {np.percentile(d_g,90):.0f}")
    print(f"    gen presses lasting <=2 ticks (~33ms, overlay-invisible territory): "
          f"{(d_g <= 2).mean():.0%}   human: {(d_h <= 2).mean():.0%}")
else:
    print("    gen: NO complete presses reconstructed — onsets never met a release,")
    print("    or releases fire same-tick (check [4] counts vs onset counts)")

# ---------------------------------------------------------------------------
# [6] double-tap check: gaps BETWEEN consecutive generated onsets
# ---------------------------------------------------------------------------
g_ticks = np.flatnonzero(g_on.any(axis=0))
h_ticks = np.flatnonzero(h_on.any(axis=0))
if len(g_ticks) > 1:
    gaps_g = np.diff(g_ticks)
    gaps_h = np.diff(h_ticks)
    print(f"\n[6] inter-onset gaps (ticks)")
    print(f"    human: median {np.median(gaps_h):.0f}  gaps<=3: {(gaps_h<=3).mean():.0%}")
    print(f"    gen  : median {np.median(gaps_g):.0f}  gaps<=3: {(gaps_g<=3).mean():.0%}")
    print(f"    -> gen gaps<=3 >> human = plateau double-fire confirmed (the de-bounce fix)")