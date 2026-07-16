"""Compare a human .osr against a generated one, field by field.

v2: adds hold-duration percentile comparison (p50/p90/p99/max) and a
tail-vs-shift verdict, to discriminate:
  TAIL-ONLY   p50/p90 at parity, p99/max explode -> thresholds are fine;
              a few holds never release (suspect: quiet map stretches /
              dead-zone neighborhood). Fix candidate: max-hold cap in
              full_alternate, NOT a threshold change.
  SHIFT       p50 already high -> live-head regime diverges from the sweep's
              reconstructed-key regime; probe that, don't nudge thresholds.

Also reports START positions of the longest generated holds, so they can be
cross-referenced against map object density (dead-zone probe).

Usage:
    python diagnostics/compare_replays.py <human.osr> <generated.osr>
"""

import sys
from collections import Counter

import numpy as np
from osrparse import Replay


META_FIELDS = [
    'mode', 'game_version', 'beatmap_hash', 'username', 'replay_hash',
    'count_300', 'count_100', 'count_50', 'count_geki', 'count_katu',
    'count_miss', 'score', 'max_combo', 'perfect', 'mods',
    'replay_id', 'rng_seed',
]

KEY_BITS = {1: 'M1', 2: 'M2', 4: 'K1', 8: 'K2', 16: 'SMOKE'}
CLICK_MASK = 1 | 2 | 4 | 8   # any click-ish bit


def decode_keys(v: int) -> str:
    names = [n for bit, n in KEY_BITS.items() if v & bit]
    return '+'.join(names) if names else 'none'


def key_census(replay):
    return Counter(int(ev.keys) for ev in replay.replay_data)


def hold_runs_with_pos(replay, mask=CLICK_MASK):
    """(length, start_frame_index, start_abs_ms) for every consecutive run
    of frames with any bit in `mask` set. Trailer frames (dt=-12345) excluded.

    CAVEAT (measured, this cycle): with mask=CLICK_MASK this measures
    any-key-down runs, which CONFLATES gapless alternation with stuck holds
    — full_alternate releases the old key on the same tick the new one
    presses, so a whole dense section reads as one giant "run". Per-key
    masks (K1: 4|1, K2: 8|2) measure actual per-key hold durations."""
    runs = []
    cur_len, cur_start_idx, cur_start_ms = 0, None, None
    abs_ms = 0
    for i, ev in enumerate(replay.replay_data):
        if ev.time_delta == -12345:     # rng-seed trailer, not gameplay
            continue
        abs_ms += ev.time_delta
        if int(ev.keys) & mask:
            if cur_len == 0:
                cur_start_idx, cur_start_ms = i, abs_ms
            cur_len += 1
        elif cur_len:
            runs.append((cur_len, cur_start_idx, cur_start_ms))
            cur_len = 0
    if cur_len:
        runs.append((cur_len, cur_start_idx, cur_start_ms))
    return runs


def pct_table(label, lengths):
    a = np.asarray(lengths)
    if a.size == 0:
        print(f"  {label:9s}: no holds")
        return None
    row = {p: np.percentile(a, p) for p in (50, 90, 99)}
    print(f"  {label:9s}: n={a.size:5d}  p50={row[50]:5.1f}  p90={row[90]:5.1f}  "
          f"p99={row[99]:6.1f}  max={a.max():5d}  (frames)")
    return row, int(a.max())


def report_meta_and_census(label, replay):
    print(f"\n=== {label} ===")
    print("-- metadata --")
    for f in META_FIELDS:
        print(f"  {f:14s} = {getattr(replay, f, '<absent>')!r}")
    print("-- key-value census --")
    for v, c in sorted(key_census(replay).items()):
        print(f"  {v:3d} ({decode_keys(v):12s}): {c} frames")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    human = Replay.from_path(sys.argv[1])
    gen = Replay.from_path(sys.argv[2])

    report_meta_and_census("HUMAN", human)
    report_meta_and_census("GENERATED", gen)

    K1_MASK, K2_MASK = 4 | 1, 8 | 2   # key + its mouse-alias bit

    print("\n=== PER-KEY hold-duration distributions (the honest metric) ===")
    per_key = {}
    for label, rep in (("HUMAN", human), ("GENERATED", gen)):
        k1 = hold_runs_with_pos(rep, K1_MASK)
        k2 = hold_runs_with_pos(rep, K2_MASK)
        both = [r[0] for r in k1 + k2]
        per_key[label] = (pct_table(label, both), k1 + k2)

    print("\n-- longest 10 generated PER-KEY holds (len_frames, start_frame, start_ms) --")
    for run in sorted(per_key["GENERATED"][1], reverse=True)[:10]:
        print(f"  {run}")

    print("\n=== any-key-down runs (alternation continuity, NOT holds — see caveat) ===")
    for label, rep in (("HUMAN", human), ("GENERATED", gen)):
        pct_table(label, [r[0] for r in hold_runs_with_pos(rep)])

    h, g = per_key["HUMAN"][0], per_key["GENERATED"][0]
    if h and g:
        (hp, hmax), (gp, gmax) = h, g
        print("\n=== verdict (per-key) ===")
        p50_ratio = gp[50] / max(hp[50], 1e-9)
        tail_blown = gmax > 3 * hmax or gp[99] > 3 * hp[99]
        if p50_ratio < 1.5 and tail_blown:
            print("RELEASE STARVATION IN SPARSE PLAY: per-key body at parity but")
            print("tail blown — releases are press-driven (swap on next note),")
            print("exposed wherever notes are sparse. Cross-check offset_prob in")
            print("gen_cache over the long-hold windows; fix candidate: scaffold")
            print("release on N consecutive inactive ticks, thresholds untouched.")
        elif p50_ratio >= 1.5:
            print(f"SHIFT: generated per-key p50 is {p50_ratio:.1f}x human — the")
            print("live head diverges from the sweep's reconstructed-key regime")
            print("across the board. Probe the regime; don't nudge thresholds blind.")
        else:
            print("MIXED/UNCLEAR — read the tables.")




if __name__ == "__main__":
    main()