"""
battery.py — the full diagnostic battery in one run, manifest-picked.

Runs, in order, on ONE map/replay pair chosen from the manifest:
  [R] .osr roundtrip      — writer fidelity (frames, timing, positions)
  [K] key reconstruction  — onset count / recall / gaps / durations vs human,
                            at the module constants in predict_replay
  [L] lateness            — arrival lateness vs human, miss count, distance
                            at scheduled tick, split by map third
  [Z] zigzag              — heading-reversal %% across several windows, with
                            median |move| on reversal ticks (jitter-vs-zigzag
                            discriminator)

NOTE on [L]: this is a self-contained reimplementation, definitions below.
Numbers are comparable across runs of THIS script; do not compare
decimal-for-decimal against diagnose_lateness.py's output.

Usage:
  python battery.py --split test                 # seeded random test-split map
  python battery.py --name "raise my sword"      # name match (any split)
  python battery.py --split test --seed 7        # different pick
  python battery.py --name "..." --temp 0.3      # nonzero temperature

Generation is cached per-map (gen_cache/gen_<beatmap_id>_T<temp>.npz) and the
cache stores the beatmap hash — a stale/mismatched cache is refused, not used.
"""

import argparse
import csv
import hashlib
import numpy as np
import slider
from osrparse import Replay
from osrparse.utils import Key

from config import ROOT, load_norm_stats, MANIFEST_PATH
from src.osunator.parsing import build_training_example, convert_to_absolute, to_ms, TICK_MS
from slider.beatmap import Circle, Slider

MODEL_PATH = ROOT / "best_model.keras"
GEN_CACHE_DIR = ROOT / "gen_cache"

K1, K2 = int(Key.K1), int(Key.K2)
ARRIVAL_WINDOW_TICKS = 60          # search ±1s around the scheduled tick
RECALL_TOL = 3                     # ticks
MOVE_THRESH = 2.0                  # px/tick: below this the human is "holding"
REV_MIN_MOVE = 0.5                 # px: mask near-stationary vectors in heading calc


# ---------------------------------------------------------------- manifest --

def pick_from_manifest(name_substr=None, split=None, seed=0):
    """Pick ONE (map, replay) from the manifest. Filters by split and/or a
    case-insensitive name substring; picks the map deterministically (seeded)
    from the candidates, then the highest-accuracy replay of that map.
    Returns the manifest row (dict). Prints what it picked and why."""
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [r for r in rows if r["split"] == split]
    if name_substr:
        rows = [r for r in rows if name_substr.lower() in r["beatmap_name"].lower()]
    if not rows:
        raise SystemExit(f"no manifest rows match split={split!r} name={name_substr!r}")

    by_map = {}
    for r in rows:
        by_map.setdefault(r["beatmap_id"], []).append(r)
    map_ids = sorted(by_map)
    rng = np.random.default_rng(seed)
    chosen_map = map_ids[int(rng.integers(len(map_ids)))] if name_substr is None \
        else map_ids[0] if len(map_ids) == 1 else None
    if chosen_map is None:
        print(f"name matched {len(map_ids)} maps:")
        for m in map_ids:
            print(f"  {m}: {by_map[m][0]['beatmap_name']}")
        raise SystemExit("narrow the --name filter to one map")

    row = max(by_map[chosen_map], key=lambda r: float(r["accuracy"]))
    print(f"picked: {row['beatmap_name']}  [split={row['split']}, "
          f"{len(by_map)} candidate maps, seed={seed}]")
    print(f"human:  {row['replay_path']}  (acc {float(row['accuracy']):.2f}, "
          f"{len(by_map[chosen_map])} replays for this map)\n")
    return row


# -------------------------------------------------------------- generation --

def load_or_generate(row, temperature):
    beatmap = slider.beatmap.Beatmap.from_path(row["beatmap_path"])
    with open(row["beatmap_path"], "rb") as f:
        map_hash = hashlib.md5(f.read()).hexdigest()

    GEN_CACHE_DIR.mkdir(exist_ok=True)
    cache = GEN_CACHE_DIR / f"gen_{row['beatmap_id']}_T{temperature}.npz"

    if cache.exists():
        d = dict(np.load(cache, allow_pickle=False))
        if str(d.get("beatmap_hash")) == map_hash:
            print(f"loaded cached generation: {cache.name} "
                  f"({len(d['grid'])} ticks)\n")
            return beatmap, map_hash, d
        print(f"cache {cache.name} is for a DIFFERENT map version — regenerating\n")

    from tensorflow import keras
    from predict_replay import generate_replay
    stats = load_norm_stats()
    model = keras.models.load_model(MODEL_PATH, compile=False)
    result = generate_replay(model, beatmap, stats, temperature=temperature)
    result["beatmap_hash"] = np.array(map_hash)
    np.savez(cache, **result)
    print(f"generated and cached: {cache.name}\n")
    return beatmap, map_hash, result


# --------------------------------------------------------------- roundtrip --

def run_roundtrip(result, map_hash):
    from predict_replay import result_to_replay
    grid = np.asarray(result["grid"], dtype=float)
    out = GEN_CACHE_DIR / "battery_roundtrip.osr"
    result_to_replay(result, map_hash).write_path(str(out))

    back = Replay.from_path(str(out))
    events = back.replay_data[1:]                     # skip anchor frame
    deltas = np.array([e.time_delta for e in events], dtype=float)
    abs_times = np.cumsum(deltas)
    rx = np.array([e.x for e in events])
    ry = np.array([e.y for e in events])
    m = min(len(grid), len(abs_times))
    dt = abs_times[:m] - grid[:m]
    dpos = np.hypot(rx[:m] - result["pred_cursor_x"][:m],
                    ry[:m] - result["pred_cursor_y"][:m])
    try:
        convert_to_absolute(back)
        decoder = "accepted"
    except ValueError as e:
        decoder = f"REJECTED: {e}"

    ok = (len(events) == len(grid) and abs(np.median(dt)) < 1
          and abs(dt[-1] - dt[0]) < 1 and dpos.max() < 0.51
          and decoder == "accepted")
    print(f"[R] roundtrip: frames {len(grid)}/{len(events)}  "
          f"dt median {np.median(dt):+.1f}ms drift {dt[-1]-dt[0]:+.1f}ms  "
          f"pos max {dpos.max():.3f}px  decoder {decoder}"
          f"   {'PASS' if ok else '*** FAIL ***'}")
    return ok


# --------------------------------------------------------------- key stats --

def _onsets_durations(arr_int):
    onset_ticks, durations = [], []
    for bit in (K1, K2):
        held = (arr_int & bit) != 0
        prev = np.concatenate([[False], held[:-1]])
        rises = np.flatnonzero(held & ~prev)
        falls = np.flatnonzero(~held & prev)
        for r in rises:
            onset_ticks.append(r)
            later = falls[falls > r]
            durations.append((later[0] - r) if len(later) else (len(held) - r))
    order = np.argsort(onset_ticks)
    return np.array(onset_ticks)[order], np.array(durations)[order]


def _recall(human_ticks, gen_ticks, tol=RECALL_TOL):
    if len(human_ticks) == 0 or len(gen_ticks) == 0:
        return 0.0
    idx = np.searchsorted(gen_ticks, human_ticks)
    lo = np.clip(idx - 1, 0, len(gen_ticks) - 1)
    hi = np.clip(idx, 0, len(gen_ticks) - 1)
    nearest = np.minimum(np.abs(gen_ticks[lo] - human_ticks),
                         np.abs(gen_ticks[hi] - human_ticks))
    return float((nearest <= tol).mean())


def run_keys(result, example):
    from predict_replay import full_alternate, ONSET_THR, RELEASE_THR, COOLDOWN_TICKS
    n = len(result["grid"])
    keys = full_alternate(n, result["pred_key_onset"], result["pred_key_offset"],
                          ONSET_THR, RELEASE_THR, COOLDOWN_TICKS)
    g_ticks, g_dur = _onsets_durations(np.array([int(k) for k in keys]))

    onset = np.asarray(example["key_onset"], dtype=bool)
    held = np.asarray(example["key_held"], dtype=bool)
    h_ticks, h_dur = [], []
    for s in (0, 1):
        rises = np.flatnonzero(onset[s])
        prev = np.concatenate([[False], held[s][:-1]])
        falls = np.flatnonzero(~held[s] & prev)
        for r in rises:
            h_ticks.append(r)
            later = falls[falls > r]
            h_dur.append((later[0] - r) if len(later) else (held.shape[1] - r))
    order = np.argsort(h_ticks)
    h_ticks = np.array(h_ticks)[order]
    h_dur = np.array(h_dur)[order]

    g_gaps = np.diff(g_ticks) if len(g_ticks) > 1 else np.array([np.inf])
    h_gaps = np.diff(h_ticks) if len(h_ticks) > 1 else np.array([np.inf])
    print(f"[K] keys (thr {ONSET_THR}/{RELEASE_THR}): "
          f"onsets {len(g_ticks)} vs human {len(h_ticks)}   "
          f"recall±{RECALL_TOL} {_recall(h_ticks, g_ticks):.0%}   "
          f"gaps<=3 {float((g_gaps <= 3).mean()):.0%} vs {float((h_gaps <= 3).mean()):.0%}   "
          f"med dur {np.median(g_dur):.0f} vs {np.median(h_dur):.0f} ticks")


# ---------------------------------------------------------------- lateness --

def run_lateness(beatmap, result, example):
    """Targets = circles + slider heads. For each, 'arrival' = first tick
    within the circle radius, searched ±ARRIVAL_WINDOW_TICKS around the
    scheduled tick. Lateness = gen arrival tick − human arrival tick,
    over targets where BOTH arrive. Miss = gen never enters the radius."""
    radius = 54.4 - 4.48 * beatmap.circle_size
    grid = np.asarray(result["grid"], dtype=float)
    gx, gy = result["pred_cursor_x"], result["pred_cursor_y"]
    hx, hy = example["cursor_x"], example["cursor_y"]
    n = len(grid)

    targets = [(to_ms(o.time), o.position.x, o.position.y)
               for o in beatmap._hit_objects if isinstance(o, (Circle, Slider))]

    def arrival(xs, ys, sched_tick, tx, ty):
        lo = max(0, sched_tick - ARRIVAL_WINDOW_TICKS)
        hi = min(n, sched_tick + ARRIVAL_WINDOW_TICKS)
        d = np.hypot(xs[lo:hi] - tx, ys[lo:hi] - ty)
        inside = np.flatnonzero(d <= radius)
        return (lo + inside[0]) if len(inside) else None

    lat, dist_g, dist_h, thirds = [], [], [], []
    misses = evaluated = 0
    for t_ms, tx, ty in targets:
        st = int(round((t_ms - grid[0]) / TICK_MS))
        if not (0 <= st < n):
            continue
        evaluated += 1
        a_g = arrival(gx, gy, st, tx, ty)
        a_h = arrival(hx, hy, st, tx, ty)
        if a_g is None:
            misses += 1
        if a_g is not None and a_h is not None:
            lat.append(a_g - a_h)
            thirds.append(min(2, 3 * st // n))
        dist_g.append(np.hypot(gx[st] - tx, gy[st] - ty))
        dist_h.append(np.hypot(hx[st] - tx, hy[st] - ty))

    lat = np.array(lat); dist_g = np.array(dist_g); dist_h = np.array(dist_h)
    thirds = np.array(thirds)
    by_third = [f"{lat[thirds == i].mean():+.2f}" if (thirds == i).any() else "—"
                for i in range(3)]

    # moving/holding split (human-defined)
    hspd = np.hypot(np.diff(hx), np.diff(hy))
    gspd = np.hypot(np.diff(gx), np.diff(gy))
    moving = hspd > MOVE_THRESH
    ratio = gspd[moving].mean() / hspd[moving].mean() if moving.any() else np.nan

    print(f"[L] lateness ({evaluated} targets, radius {radius:.0f}px): "
          f"mean {lat.mean():+.2f}t median {np.median(lat):+.1f}t "
          f"p90 {np.percentile(lat, 90):+.1f}t   misses {misses}")
    print(f"    by map third: early {by_third[0]}  mid {by_third[1]}  "
          f"late {by_third[2]}   (front-loaded = warm-up signature)")
    print(f"    dist@sched: gen med {np.median(dist_g):.1f}px vs human "
          f"{np.median(dist_h):.1f}px   moving-speed ratio {ratio:.3f}")


# ------------------------------------------------------------------ zigzag --

def run_zigzag(result, example):
    gx, gy = result["pred_cursor_x"], result["pred_cursor_y"]
    hx, hy = example["cursor_x"], example["cursor_y"]
    n = min(len(gx), len(hx))
    win = 600  # ~10s

    def rev_stats(xs, ys, lo, hi):
        dx, dy = np.diff(xs[lo:hi]), np.diff(ys[lo:hi])
        v1x, v1y, v2x, v2y = dx[:-1], dy[:-1], dx[1:], dy[1:]
        n1, n2 = np.hypot(v1x, v1y), np.hypot(v2x, v2y)
        ok = (n1 > REV_MIN_MOVE) & (n2 > REV_MIN_MOVE)
        if not ok.any():
            return np.nan, np.nan
        cos = (v1x[ok]*v2x[ok] + v1y[ok]*v2y[ok]) / (n1[ok]*n2[ok])
        rev = np.degrees(np.arccos(np.clip(cos, -1, 1))) > 90
        med_move = np.median(n2[ok][rev]) if rev.any() else np.nan
        return 100 * rev.mean(), med_move

    print(f"[Z] heading reversals >90° (window {win} ticks; "
          f"'med px' = median move size ON reversal ticks — small = parked jitter)")
    fracs = (0.05, 0.25, 0.50, 0.75, 0.90)
    for f in fracs:
        lo = int(f * (n - win)) if n > win else 0
        hi = min(lo + win, n)
        rg, mg = rev_stats(gx, gy, lo, hi)
        rh, mh = rev_stats(hx, hy, lo, hi)
        print(f"    @{f:>4.0%} (tick {lo:>6d}): gen {rg:5.1f}% (med {mg:4.1f}px)"
              f"   human {rh:4.1f}% (med {mh:4.1f}px)")


# -------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="beatmap name substring")
    ap.add_argument("--split", default=None, choices=("train", "test"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.0)
    args = ap.parse_args()

    row = pick_from_manifest(args.name, args.split, args.seed)
    beatmap, map_hash, result = load_or_generate(row, args.temp)
    human_replay = Replay.from_path(row["replay_path"])
    example = build_training_example(beatmap, human_replay)

    ng, nh = len(result["grid"]), len(example["grid"])
    if ng != nh:
        print(f"*** grid length mismatch gen {ng} vs example {nh} — "
              f"stale cache or map version drift; aborting ***")
        return

    print(f"=== battery: {row['beatmap_name']} | split {row['split']} | "
          f"T={args.temp} | {ng} ticks ===\n")
    run_roundtrip(result, map_hash)
    run_keys(result, example)
    run_lateness(beatmap, result, example)
    run_zigzag(result, example)
    print("\n(paste the lines above into training_log.txt with date + checkpoint)")


if __name__ == "__main__":
    main()