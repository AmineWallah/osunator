"""Speed/lateness diagnostic for the accumulating temporal drift.

History of what this file has ruled out (kept so the numbers mean something):
  - UNDERSHOOT theory: dead. Teacher-forced speed ratio measured 1.027 —
    per-tick delta magnitudes are fine, slightly above human.
  - ZIGZAG theory: fixed by the MDN head at temperature=0 (31.8% -> 2.5%
    heading reversals), but lateness persists.

Current question this run answers: with the zigzag gone, is the closed-loop
speed ratio now ~1.0? That discriminates the two surviving explanations:
  ratio ~1.0 while still late  -> path length matches the human's; lateness
      is a recoverable schedule displacement -> lag-DART is the right tool.
  ratio still >1 (e.g. ~1.1+)  -> excess motion survives that the reversal
      metric can't see (wide arcs / soft corners) -> path-shape problem,
      lag-DART only partially helps.

Run on a TRAINING-set map so model quality isn't the confound.
"""
import numpy as np
from tensorflow import keras

from osunator.config import load_norm_stats
from osunator.parsing import beatmap_replay_pairs
from osunator.generate import predict_replay, generate_replay

REPLAY_PATH = '/replays/suitable/INFERNOBESTMAP - GALNERYUS - RAISE MY SWORD [A THOUSAND FLAMES] (2023-10-08) Osu.osr'
MODEL_PATH = '../best_model.keras'
TEMPERATURE = 0.0   # greedy — the current operating point; the ratio question is asked at T=0


def speed(xs, ys):
    return np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)


def main():
    stats = load_norm_stats()
    model = keras.models.load_model(MODEL_PATH, compile=False)   # inference only; skips mdn_nll deserialization
    beatmap, beatmap_id, replay, path = beatmap_replay_pairs([REPLAY_PATH])[0]
    print(f"map/replay under test: {path}")
    print(f"temperature: {TEMPERATURE}\n")

    # ---------- 1 & 2: teacher-forced ----------
    r = predict_replay(model, beatmap, replay, stats)
    pred_speed = speed(r['pred_cursor_x'], r['pred_cursor_y'])
    true_speed = speed(r['true_cursor_x'], r['true_cursor_y'])

    ratio = pred_speed.mean() / true_speed.mean()
    print(f"[1] teacher-forced speed ratio (pred/true): {ratio:.3f}")
    print(f"    pred mean speed: {pred_speed.mean():.2f} px/tick   "
          f"true: {true_speed.mean():.2f} px/tick\n")

    print("[2] ratio per map segment:")
    n_seg = 10
    seg_len = len(pred_speed) // n_seg
    for s in range(n_seg):
        lo, hi = s * seg_len, (s + 1) * seg_len
        denom = true_speed[lo:hi].mean()
        if denom < 1e-6:
            print(f"    segment {s:2d}: (human stationary — skipped)")
            continue
        seg_ratio = pred_speed[lo:hi].mean() / denom
        bar = '#' * min(int(seg_ratio * 40), 60)
        print(f"    segment {s:2d}: {seg_ratio:.3f}  {bar}")
    print()

    # ---------- 3: closed-loop — the deciding number ----------
    g = generate_replay(model, beatmap, stats, temperature=TEMPERATURE)
    gen_speed = speed(g['pred_cursor_x'], g['pred_cursor_y'])
    cl_ratio = gen_speed.mean() / true_speed.mean()
    print(f"[3] closed-loop mean speed: {gen_speed.mean():.2f} px/tick   "
          f"(human: {true_speed.mean():.2f})")
    print(f"    closed-loop/human ratio: {cl_ratio:.3f}")
    print(f"    (pre-MDN baseline was 1.148 — ~1.0 now means path length matches")
    print(f"     and the lateness is schedule displacement -> lag-DART justified;")
    print(f"     still >1.1 means wide-arc excess motion -> path-shape problem)")


if __name__ == "__main__":
    main()