"""Compare a human .osr against a generated one, field by field.

Purpose: isolate why the osu! client (a) doesn't display held keys on the
key overlay for generated replays, and (b) shows a "replay corrupted"
warning with D-rank/100%/zeroed stats.

Hypotheses this instrument discriminates:
  H1 (overlay): real client writes K1|M1 (=5) / K2|M2 (=10) for keyboard
      presses; we write bare K1 (4) / K2 (8). -> visible in the key-value
      census: human frames show 5/10, generated show 4/8.
  H2 (warning): placeholder metadata (empty replay_hash, zeroed counts,
      impossible game_version) trips a client sanity check. -> visible in
      the metadata table.

Usage:
    python diagnostics/compare_replays.py <human.osr> <generated.osr>
"""

import sys
from collections import Counter

from osrparse import Replay


META_FIELDS = [
    'mode', 'game_version', 'beatmap_hash', 'username', 'replay_hash',
    'count_300', 'count_100', 'count_50', 'count_geki', 'count_katu',
    'count_miss', 'score', 'max_combo', 'perfect', 'mods',
    'replay_id', 'rng_seed',
]

KEY_BITS = {1: 'M1', 2: 'M2', 4: 'K1', 8: 'K2', 16: 'SMOKE'}


def decode_keys(v: int) -> str:
    names = [n for bit, n in KEY_BITS.items() if v & bit]
    return '+'.join(names) if names else 'none'


def key_census(replay):
    """Counter of raw key-field integer values across all frames."""
    return Counter(int(ev.keys) for ev in replay.replay_data)


def hold_runs(replay):
    """Lengths (in frames) of consecutive runs where any key bit is down.
    If the generated replay has sane runs here, the holds EXIST in the data
    and the overlay problem is representational (H1), not structural."""
    runs, cur = [], 0
    for ev in replay.replay_data:
        if int(ev.keys) != 0:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def first_frames(replay, n=5):
    return [(ev.time_delta, round(ev.x, 1), round(ev.y, 1), int(ev.keys))
            for ev in replay.replay_data[:n]]


def report(label, replay):
    print(f"\n=== {label} ===")
    print("-- metadata --")
    for f in META_FIELDS:
        print(f"  {f:14s} = {getattr(replay, f, '<absent>')!r}")

    print("-- key-value census (raw int: frames) --")
    for v, c in sorted(key_census(replay).items()):
        print(f"  {v:3d} ({decode_keys(v):9s}): {c} frames")

    runs = hold_runs(replay)
    if runs:
        import numpy as np
        arr = np.array(runs)
        print(f"-- hold runs: n={len(arr)} median={np.median(arr):.0f} "
              f"min={arr.min()} max={arr.max()} frames --")
    else:
        print("-- hold runs: NONE (no frame has any key bit set) --")

    print(f"-- first {5} frames (dt, x, y, keys) --")
    for row in first_frames(replay):
        print(f"  {row}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    human = Replay.from_path(sys.argv[1])
    gen = Replay.from_path(sys.argv[2])
    report("HUMAN", human)
    report("GENERATED", gen)

    print("\n=== verdict hints ===")
    hk, gk = set(key_census(human)), set(key_census(gen))
    if (5 in hk or 10 in hk) and (4 in gk or 8 in gk) and not (5 in gk or 10 in gk):
        print("H1 SUPPORTED: human uses K|M combos (5/10), generated uses bare K bits (4/8).")
    else:
        print("H1 not clearly supported by key census — read the tables.")
    if not gen.replay_hash:
        print("H2 candidate: generated replay_hash is empty.")


if __name__ == "__main__":
    main()