"""
threshold_sweep.py — calibrate ONSET_THR / RELEASE_THR on the RECONSTRUCTED keys.

Measures the shipped artifact stage (full_alternate output), not the raw head:
for each (onset_thr, release_thr) cell, reconstruct keys_per_tick and score it
against the human replay on the same map. Pick the cell where onset count and
median duration land nearest the human row printed at the bottom.

Run from anywhere (paths anchored via config). First run generates the gen dict
once (~30s) and caches it as rms_gen_T0.npz next to the project root; later
runs sweep instantly with no model in the loop.
"""

import csv
import numpy as np
from osrparse import Replay
from osrparse.utils import Key
import slider

from config import ROOT, load_norm_stats, MANIFEST_PATH
from src.osunator.parsing import build_training_example
from predict_replay import full_alternate, generate_replay

# ---------------------------------------------------------------------------
MAP_NAME_SUBSTR = "a thousand flames"          # matched case-insensitively in manifest beatmap_name
GEN_CACHE = ROOT / "rms_gen_T0.npz"
MODEL_PATH = ROOT / "best_model.keras"
ONSET_THRS = (0.35, 0.40, 0.45)
RELEASE_THRS = (0.40, 0.45, 0.50, 0.55)
COOLDOWN = 3
RECALL_TOL = 3                              # ticks; "gen onset within ±3 of human onset"
# ---------------------------------------------------------------------------

K1 = int(Key.K1)
K2 = int(Key.K2)


def find_manifest_row():
    with open(MANIFEST_PATH) as f:
        rows = [r for r in csv.DictReader(f)
                if MAP_NAME_SUBSTR in r["beatmap_name"].lower()]
    if not rows:
        raise SystemExit(f"no manifest row matching {MAP_NAME_SUBSTR!r}")
    # several human replays exist for the map; take the highest-accuracy one
    # as the reference and SAY which, so the numbers are attributable
    row = max(rows, key=lambda r: float(r["accuracy"]))
    print(f"reference: {row['beatmap_name']}")
    print(f"human replay: {row['replay_path']}  (acc {float(row['accuracy']):.2f}, "
          f"{len(rows)} candidates)\n")
    return row


def onsets_durations_gaps(keys_per_tick):
    """From a reconstructed Key list: merged onset ticks, press durations, inter-onset gaps."""
    arr = np.array([int(k) for k in keys_per_tick])
    onset_ticks, durations = [], []
    for bit in (K1, K2):
        held = (arr & bit) != 0
        prev = np.concatenate([[False], held[:-1]])
        rises = np.flatnonzero(held & ~prev)
        falls = np.flatnonzero(~held & prev)          # tick where key is first UP again
        for r in rises:
            onset_ticks.append(r)
            later = falls[falls > r]
            durations.append((later[0] - r) if len(later) else (len(arr) - r))
    order = np.argsort(onset_ticks)
    onset_ticks = np.array(onset_ticks)[order]
    durations = np.array(durations)[order]
    gaps = np.diff(onset_ticks) if len(onset_ticks) > 1 else np.array([])
    return onset_ticks, durations, gaps


def human_reference(beatmap, replay):
    ex = build_training_example(beatmap, replay)
    onset = np.asarray(ex["key_onset"], dtype=bool)    # (2, n)
    held = np.asarray(ex["key_held"], dtype=bool)
    onset_ticks, durations = [], []
    for s in (0, 1):
        rises = np.flatnonzero(onset[s])
        h = held[s]
        prev = np.concatenate([[False], h[:-1]])
        falls = np.flatnonzero(~h & prev)
        for r in rises:
            onset_ticks.append(r)
            later = falls[falls > r]
            durations.append((later[0] - r) if len(later) else (h.shape[0] - r))
    order = np.argsort(onset_ticks)
    onset_ticks = np.array(onset_ticks)[order]
    durations = np.array(durations)[order]
    gaps = np.diff(onset_ticks) if len(onset_ticks) > 1 else np.array([])
    return onset_ticks, durations, gaps


def recall(human_ticks, gen_ticks, tol=RECALL_TOL):
    if len(human_ticks) == 0 or len(gen_ticks) == 0:
        return 0.0
    idx = np.searchsorted(gen_ticks, human_ticks)
    lo = np.clip(idx - 1, 0, len(gen_ticks) - 1)
    hi = np.clip(idx, 0, len(gen_ticks) - 1)
    nearest = np.minimum(np.abs(gen_ticks[lo] - human_ticks),
                         np.abs(gen_ticks[hi] - human_ticks))
    return float((nearest <= tol).mean())


def main():
    row = find_manifest_row()
    beatmap = slider.beatmap.Beatmap.from_path(row["beatmap_path"])
    human_replay = Replay.from_path(row["replay_path"])

    # --- gen dict: cached, or generate once ---
    if False:
        result = dict(np.load(GEN_CACHE))
        print(f"loaded cached gen dict: {GEN_CACHE.name} ({len(result['grid'])} ticks)\n")
    else:
        from tensorflow import keras
        stats = load_norm_stats()
        model = keras.models.load_model(MODEL_PATH, compile=False)
        result = generate_replay(model, beatmap, stats, temperature=0.0)
        np.savez(GEN_CACHE, **result)
        print(f"generated and cached: {GEN_CACHE.name}\n")

    n = len(result["grid"])
    onset_prob = result["pred_key_onset"]
    offset_prob = result["pred_key_offset"]

    h_ticks, h_dur, h_gaps = human_reference(beatmap, human_replay)
    h_gaps3 = float((h_gaps <= 3).mean()) if len(h_gaps) else 0.0

    print(f"{'onset':>6} {'release':>8} | {'onsets':>7} {'recall±3':>9} "
          f"{'gaps<=3':>8} {'med dur':>8}")
    print("-" * 56)

    tables = {}
    for o_thr in ONSET_THRS:
        for r_thr in RELEASE_THRS:
            keys = full_alternate(n, onset_prob, offset_prob, o_thr, r_thr, COOLDOWN)
            g_ticks, g_dur, g_gaps = onsets_durations_gaps(keys)
            rec = recall(h_ticks, g_ticks)
            gaps3 = float((g_gaps <= 3).mean()) if len(g_gaps) else 0.0
            med = float(np.median(g_dur)) if len(g_dur) else 0.0
            tables[(o_thr, r_thr)] = (len(g_ticks), rec, gaps3, med)
            print(f"{o_thr:>6.2f} {r_thr:>8.2f} | {len(g_ticks):>7d} {rec:>8.0%} "
                  f"{gaps3:>7.0%} {med:>8.1f}")

    print("-" * 56)
    print(f"{'HUMAN':>15} | {len(h_ticks):>7d} {'—':>9} "
          f"{h_gaps3:>7.0%} {float(np.median(h_dur)):>8.1f}")

    # self-check: if full_alternate ignores its threshold params (the shadowing
    # bug), every cell is identical — catch that loudly instead of tabling lies
    if len(set(tables.values())) == 1:
        print("\n*** WARNING: all 12 cells identical — full_alternate is almost "
              "certainly still reading the module constants instead of its "
              "parameters. Fix the comparisons inside it and rerun. ***")


if __name__ == "__main__":
    main()