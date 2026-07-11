"""
diagnose_lateness.py — full trajectory diagnostic, one map, one checkpoint.

  [A] per-target ARRIVAL lateness (first entry into hit radius vs schedule)
  [B] excess motion split: human-moving vs human-holding ticks
  [C] missed targets characterized (near-miss vs skip, slider/spinner, deciles)
  [D] cursor->target distance AT the scheduled tick — what the eye sees.
      [A] can be on-time while [D] is bad: cursor reaches the ring on
      schedule but settles into the circle late (undershoot-then-settle).

Set REPLAY_PATH and run. No retraining; generation takes ~1-2s per map-minute.
"""

import numpy as np
from tensorflow import keras

from config import load_norm_stats
from parsing import beatmap_replay_pairs, build_training_example
from predict_replay import generate_replay

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
REPLAY_PATH = '/replays/suitable/222mm - MAKOOTO - Tanukichi no Bouken [usagi] (2026-02-09) Osu.osr'
MODEL_PATH = '../best_model.keras'
TEMPERATURE = 0.0

WINDOW_TICKS = None       # e.g. 2700 = first 45s only (dodge late-map dead zone);
                          # None = whole map. Long maps: use 2700.
HIT_RADIUS_PX = 50.0      # arrival radius (~CS4 circle + slack)
MOVE_THRESH = 1.0         # px/tick below which the human counts as "holding"
SEARCH_AHEAD = 240        # ticks (~4s) to look for an arrival before "never arrived"
TICK_MS = 1000.0 / 60.0

# ---------------------------------------------------------------------------
# LOAD + GENERATE
# ---------------------------------------------------------------------------
stats = load_norm_stats()
model = keras.models.load_model(MODEL_PATH, compile=False)
beatmap, beatmap_id, replay, path = beatmap_replay_pairs([REPLAY_PATH])[0]
print(f"map/replay: {path}")
print(f"temperature: {TEMPERATURE}   window: "
      f"{'whole map' if WINDOW_TICKS is None else f'first {WINDOW_TICKS} ticks (~{WINDOW_TICKS/60:.0f}s)'}\n")

example = build_training_example(beatmap, replay)   # human, raw, pre-perturb
hx = example['cursor_x'].astype(float)
hy = example['cursor_y'].astype(float)

gen = generate_replay(model, beatmap, stats, temperature=TEMPERATURE)
gx = gen['pred_cursor_x'].astype(float)
gy = gen['pred_cursor_y'].astype(float)

n = min(len(hx), len(gx))
if WINDOW_TICKS is not None:
    n = min(n, WINDOW_TICKS)
hx, hy, gx, gy = hx[:n], hy[:n], gx[:n], gy[:n]

# absolute target positions: target = human cursor + raw target_dx
tx = hx + example['target_dx'][:n].astype(float)
ty = hy + example['target_dy'][:n].astype(float)
is_active = example['is_active'][:n].astype(bool)
is_slider = example['is_slider'][:n].astype(bool)
is_spinner = example['is_spinner'][:n].astype(bool)

print(f"ticks analyzed: {n}   gen mean speed "
      f"{np.hypot(np.diff(gx), np.diff(gy)).mean():.2f} px/tick   "
      f"human {np.hypot(np.diff(hx), np.diff(hy)).mean():.2f} px/tick")

# ---------------------------------------------------------------------------
# target events: ticks where the active target position jumps (new object)
# ---------------------------------------------------------------------------
jump = np.hypot(np.diff(tx), np.diff(ty)) > 5.0
event_ticks = np.where(jump & is_active[1:])[0] + 1
if len(event_ticks):
    keep = np.concatenate([[True], np.diff(event_ticks) > 3])   # de-dup slider wiggle
    event_ticks = event_ticks[keep]

if len(event_ticks) == 0:
    raise SystemExit("no target events found in window — widen WINDOW_TICKS?")


def first_arrival(px, py, t0, target_x, target_y):
    """first tick in [t0-12, t0+SEARCH_AHEAD) where cursor is within radius"""
    lo = max(t0 - 12, 0)
    hi = min(t0 + SEARCH_AHEAD, n)
    d = np.hypot(px[lo:hi] - target_x, py[lo:hi] - target_y)
    idx = np.where(d < HIT_RADIUS_PX)[0]
    return lo + idx[0] if len(idx) else None


# ---------------------------------------------------------------------------
# [A] per-target arrival lateness
# ---------------------------------------------------------------------------
rows = []             # (event_tick, human_lateness, gen_lateness)
missed_t, nearest = [], []
for t0 in event_ticks:
    ha = first_arrival(hx, hy, t0, tx[t0], ty[t0])
    if ha is None:
        continue                                   # degenerate event; skip
    ga = first_arrival(gx, gy, t0, tx[t0], ty[t0])
    if ga is None:
        lo, hi_ = max(t0 - 12, 0), min(t0 + SEARCH_AHEAD, n)
        d = np.hypot(gx[lo:hi_] - tx[t0], gy[lo:hi_] - ty[t0])
        missed_t.append(t0)
        nearest.append(d.min())
        continue
    rows.append((t0, ha - t0, ga - t0))

print(f"\n[A] targets evaluated: {len(rows)}   (gen never arrived: {len(missed_t)})")
if rows:
    rows = np.array(rows, dtype=float)
    t_ev, late_h, late_g = rows[:, 0], rows[:, 1], rows[:, 2]
    rel = late_g - late_h
    print(f"    lateness vs human: mean {rel.mean():+.2f} ticks ({rel.mean()*TICK_MS:+.1f}ms)   "
          f"median {np.median(rel):+.2f}   p90 {np.percentile(rel, 90):+.2f} ({np.percentile(rel,90)*TICK_MS:+.1f}ms)")
    if len(rows) >= 10:
        A = np.vstack([t_ev, np.ones_like(t_ev)]).T
        slope, intercept = np.linalg.lstsq(A, rel, rcond=None)[0]
        print(f"    trend: {intercept:+.2f} ticks at start, {slope * 3600:+.2f} ticks/min drift")
        print("    lateness by map decile (ticks):")
        for i, idx in enumerate(np.array_split(np.argsort(t_ev), 10)):
            if len(idx) == 0:
                continue
            m = rel[np.sort(idx)].mean()
            print(f"      {i}: {m:+6.2f}  " + "#" * max(int(m * 4) + 8, 0))
else:
    print("    (no targets reached — nothing to score)")

# ---------------------------------------------------------------------------
# [B] excess motion: human-moving vs human-holding ticks
# ---------------------------------------------------------------------------
hspd = np.hypot(np.diff(hx), np.diff(hy))
gspd = np.hypot(np.diff(gx), np.diff(gy))
moving = hspd > MOVE_THRESH
holding = ~moving

print(f"\n[B] human moving {moving.mean():.0%} / holding {holding.mean():.0%} of ticks")
if moving.any():
    print(f"    while human MOVING : human {hspd[moving].mean():6.2f} px/tick   "
          f"gen {gspd[moving].mean():6.2f}   ratio {gspd[moving].mean()/hspd[moving].mean():.3f}")
if holding.any():
    print(f"    while human HOLDING: human {hspd[holding].mean():6.2f} px/tick   "
          f"gen {gspd[holding].mean():6.2f}")
total_excess = gspd.sum() - hspd.sum()
hold_excess = gspd[holding].sum() - hspd[holding].sum() if holding.any() else 0.0
move_excess = gspd[moving].sum() - hspd[moving].sum() if moving.any() else 0.0
print(f"    excess path: total {total_excess:+.0f}px = {hold_excess:+.0f}px in holds "
      f"{move_excess:+.0f}px in movement")

# ---------------------------------------------------------------------------
# [C] missed targets characterized
# ---------------------------------------------------------------------------
print(f"\n[C] missed targets: {len(missed_t)}")
if missed_t:
    missed_t = np.array(missed_t)
    nearest = np.array(nearest)
    print(f"    nearest-approach: median {np.median(nearest):.0f}px   p90 {np.percentile(nearest, 90):.0f}px")
    print(f"    near-misses (<80px): {(nearest < 80).mean():.0%}   true skips (>150px): {(nearest > 150).mean():.0%}")
    print(f"    on slider ticks: {is_slider[missed_t].mean():.0%}   on spinner ticks: {is_spinner[missed_t].mean():.0%}")
    hist, _ = np.histogram(missed_t, bins=10, range=(0, n))
    print(f"    misses by map decile: {hist}")

# ---------------------------------------------------------------------------
# [D] distance to target AT the scheduled tick — what the eye sees
# ---------------------------------------------------------------------------
et = event_ticks
d_gen = np.hypot(gx[et] - tx[et], gy[et] - ty[et])
d_hum = np.hypot(hx[et] - tx[et], hy[et] - ty[et])
lagging = d_gen - d_hum
print(f"\n[D] cursor->target distance at the scheduled tick (px), {len(et)} events:")
print(f"    human: median {np.median(d_hum):5.1f}   p90 {np.percentile(d_hum, 90):5.1f}")
print(f"    gen  : median {np.median(d_gen):5.1f}   p90 {np.percentile(d_gen, 90):5.1f}")
print(f"    gen minus human: median {np.median(lagging):+5.1f}   p90 {np.percentile(lagging, 90):+5.1f}")
print(f"    share of events where gen is >30px farther than human: {(lagging > 30).mean():.0%}")
print("    -> [D] bad while [A] on-time = reaches the ring on schedule but")
print("       settles into the circle late (undershoot-then-settle approach);")
print("    -> [D] ~ human = trajectory is genuinely clean; perceived lag is")
print("       rhythm-tail (p90 in [A]) and/or the silent key head.")