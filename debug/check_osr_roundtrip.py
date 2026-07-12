"""
check_osr_roundtrip.py — is the lag in the .osr file rather than the model?

Regenerates the trajectory in-process (same grid arrays every diagnostic has
been scoring), writes generated.osr exactly like predict_replay.__main__ does,
then parses that file back with osrparse + convert_to_absolute and compares:

  [1] frame count in vs out
  [2] recovered absolute times vs the grid  (a constant offset here = the lag)
  [3] recovered positions vs the generated arrays (should be ~exact)
  [4] the lead-in structure a REAL replay has vs what ours has

If [2] shows a constant time offset, every object in the client looks
uniformly late while all grid-based metrics read perfect — the model was
innocent and the bug lives in result_to_replay.
"""

import numpy as np
import csv
from tensorflow import keras
from osrparse import Replay
from config import load_norm_stats, FEATURES_DIR
from parsing import beatmap_replay_pairs, convert_to_absolute
from predict_replay import generate_replay, result_to_replay

# ---------------------------------------------------------------------------
rows = list(csv.DictReader(open(FEATURES_DIR / 'manifest.csv')))
row = next(r for r in rows if r['split'] == 'train')   # or pick a specific map by name
REPLAY_PATH = row['replay_path']
MODEL_PATH = '/home/amine/PycharmProjects/osunator/best_model.keras'
OSR_OUT = 'roundtrip_test.osr'
# ---------------------------------------------------------------------------

stats = load_norm_stats()
model = keras.models.load_model(MODEL_PATH, compile=False)
beatmap, beatmap_id, human_replay, path = next(beatmap_replay_pairs([REPLAY_PATH]))
print(f"map/replay: {path}\n")

# --- generate + write, exactly like predict_replay.__main__ ---
result = generate_replay(model, beatmap, stats, temperature=0.0)
grid = np.asarray(result['grid'], dtype=float)
gx = result['pred_cursor_x']
gy = result['pred_cursor_y']

written = result_to_replay(result, human_replay.beatmap_hash)
written.write_path(OSR_OUT)
print(f"wrote {OSR_OUT}")

# --- parse it back ---
back = Replay.from_path(OSR_OUT)
events = back.replay_data[1:]
print(f"\n[1] frames: written {len(grid)}   parsed back {len(events)}")

# recover absolute times by cumulative-summing the deltas ourselves
# (transparent — no dependence on convert_to_absolute conventions)
deltas = np.array([e.time_delta for e in events], dtype=float)
abs_times = np.cumsum(deltas)
rx = np.array([e.x for e in events], dtype=float)
ry = np.array([e.y for e in events], dtype=float)

m = min(len(grid), len(abs_times))
dt = abs_times[:m] - grid[:m]
print(f"\n[2] recovered absolute time minus grid time (ms):")
print(f"    first frame: {dt[0]:+.1f}   median {np.median(dt):+.1f}   "
      f"min {dt.min():+.1f}   max {dt.max():+.1f}")
print(f"    drift across file: {dt[-1] - dt[0]:+.1f}ms")
if abs(np.median(dt)) > 5:
    print(f"    *** CONSTANT OFFSET ~{np.median(dt):+.0f}ms — this is the visible lag. ***")
elif abs(dt[-1] - dt[0]) > 5:
    print(f"    *** times DRIFT through the file — rounding accumulation in time_deltas. ***")
else:
    print("    times align — the .osr timing is innocent.")

dpos = np.hypot(rx[:m] - gx[:m], ry[:m] - gy[:m])
print(f"\n[3] recovered position vs generated arrays (px):")
print(f"    median {np.median(dpos):.3f}   max {dpos.max():.3f}   (should be ~0)")

# --- also run YOUR canonical decoder on it, same as training data goes through ---
try:
    convert_to_absolute(back)
    print("\n    convert_to_absolute() accepted the file without error.")
except ValueError as e:
    print(f"\n    *** convert_to_absolute() REJECTED our own file: {e} ***")

# --- [4] structure comparison against the real human replay ---
h_deltas = [e.time_delta for e in human_replay.replay_data[:8]]
w_deltas = [e.time_delta for e in events[:8]]
print(f"\n[4] first 8 frame deltas —")
print(f"    real human replay : {h_deltas}")
print(f"    our written replay: {w_deltas}")
print(f"    real replay negative-delta lead-in frames: "
      f"{sum(1 for e in human_replay.replay_data if e.time_delta < 0)}")
print(f"    our replay negative-delta frames         : "
      f"{int((deltas < 0).sum())}")
print("\n    if the real replay opens with lead-in frames ours lacks, the client")
print("    may anchor our first frame differently relative to audio start —")
print("    that anchoring difference is invisible to every grid-based metric.")