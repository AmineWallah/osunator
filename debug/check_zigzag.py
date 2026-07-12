"""Visual check for the overshoot-oscillation theory of the temporal drift.

Plots a short window (~200 ticks, ~3.3s) of the generated trajectory against
the human's path over the same map-time, twice:

1. XY OVERLAY — the actual paths on the playfield, with per-tick dots. If
   the model oscillates (overshoot-correct cycles), its path will visibly
   zigzag around the human's smooth arc — unmistakable at dot level. If
   instead it's smooth but takes longer arcs, that's a different story
   (path-shape inefficiency, not oscillation).

2. HEADING CHANGE PER TICK — turn angle between consecutive movement
   vectors, generated vs human. Oscillation = frequent near-180° reversals
   (the signature of overshoot-correct); a human path turns smoothly, so
   its heading changes stay small except at deliberate direction changes.
   This makes the zigzag quantitative: % of ticks with >90° reversals.

Uses a training-set map (same one as debug_speed) so model quality on the
map isn't the confound. Change WINDOW_START_TICK to inspect different parts
of the map — early (before much lateness accumulates) vs late.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed; writes a PNG
import matplotlib.pyplot as plt
from tensorflow import keras

from config import load_norm_stats
from src.osunator.parsing import beatmap_replay_pairs, build_training_example
from predict_replay import generate_replay

REPLAY_PATH = '../replays/suitable/INFERNOBESTMAP - GALNERYUS - RAISE MY SWORD [A THOUSAND FLAMES] (2023-10-08) Osu.osr'
MODEL_PATH = '../best_model.keras'
OUT_PNG = 'zigzag_check.png'

WINDOW_START_TICK = 14000    # where the inspected window begins (600 = 10s in)
WINDOW_TICKS = 200         # ~3.3 seconds at 60Hz


def heading_changes(xs, ys):
    """Angle (degrees) between consecutive movement vectors. 0 = straight,
    180 = full reversal. Ticks where either vector is near-zero are masked
    out (heading of a stationary cursor is noise)."""
    dx, dy = np.diff(xs), np.diff(ys)
    v1x, v1y = dx[:-1], dy[:-1]
    v2x, v2y = dx[1:], dy[1:]
    n1 = np.sqrt(v1x**2 + v1y**2)
    n2 = np.sqrt(v2x**2 + v2y**2)
    ok = (n1 > 0.5) & (n2 > 0.5)
    cos = np.full(v1x.shape, np.nan)
    cos[ok] = (v1x[ok]*v2x[ok] + v1y[ok]*v2y[ok]) / (n1[ok] * n2[ok])
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def main():
    stats = load_norm_stats()
    model = keras.models.load_model(MODEL_PATH, compile=False)
    beatmap, beatmap_id, replay, path = next(beatmap_replay_pairs([REPLAY_PATH]))
    print(f"map/replay: {path}")

    example = build_training_example(beatmap, replay)   # human path on the same grid
    gen = generate_replay(model, beatmap, stats, temperature=0.0)

    lo = WINDOW_START_TICK
    hi = lo + WINDOW_TICKS
    n = min(len(gen['pred_cursor_x']), len(example['cursor_x']))
    hi = min(hi, n)

    gx, gy = gen['pred_cursor_x'][lo:hi], gen['pred_cursor_y'][lo:hi]
    hx, hy = example['cursor_x'][lo:hi], example['cursor_y'][lo:hi]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- panel 1: XY overlay ---
    ax1.plot(hx, hy, '-o', color='tab:blue', markersize=2, linewidth=1, label='human')
    ax1.plot(gx, gy, '-o', color='tab:red', markersize=2, linewidth=1, label='generated')
    ax1.plot(gx[0], gy[0], 'k^', markersize=10, label='window start')
    ax1.set_xlim(-20, 532)
    ax1.set_ylim(404, -20)   # osu y grows downward
    ax1.set_title(f'trajectories, ticks {lo}-{hi} (~{WINDOW_TICKS/60:.1f}s)')
    ax1.legend()
    ax1.set_aspect('equal')

    # --- panel 2: heading change per tick ---
    turn_g = heading_changes(gx, gy)
    turn_h = heading_changes(hx, hy)
    ax2.plot(turn_h, color='tab:blue', alpha=0.7, label='human')
    ax2.plot(turn_g, color='tab:red', alpha=0.7, label='generated')
    ax2.axhline(90, color='gray', linestyle='--', linewidth=1)
    ax2.set_title('heading change per tick (spikes to ~180° = overshoot-correct reversals)')
    ax2.set_ylabel('degrees')
    ax2.set_xlabel('tick in window')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    print(f"wrote {OUT_PNG}")

    # --- the number that settles it ---
    rev_g = np.nanmean(turn_g > 90) * 100
    rev_h = np.nanmean(turn_h > 90) * 100
    print(f"\n% of moving ticks with >90° heading reversal:")
    print(f"  generated: {rev_g:.1f}%")
    print(f"  human:     {rev_h:.1f}%")
    print("\noscillation confirmed if generated >> human (e.g. 20%+ vs a few %);")
    print("if both are low and similar, the inefficiency is smooth path-shape,")
    print("not zigzag — different problem, different fix.")


if __name__ == "__main__":
    main()