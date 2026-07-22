import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

DB = sys.argv[1] if len(sys.argv) > 1 else "study.db"
PAIRS_PATH = sys.argv[2] if len(sys.argv) > 2 else "pairs.json"


def binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test (sum of outcomes with prob <= P(k))."""
    def pmf(i: int) -> float:
        return math.comb(n, i) * p**i * (1 - p) ** (n - i)
    pk = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= pk + 1e-12))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return ((centre - margin) / denom, (centre + margin) / denom)


def line(label: str, k: int, n: int) -> str:
    if n == 0:
        return f"{label:<28} n=0"
    lo, hi = wilson_ci(k, n)
    return (f"{label:<28} {k:>4}/{n:<4} = {k/n:6.1%}"
            f"   95% CI [{lo:5.1%}, {hi:5.1%}]"
            f"   p(two-sided vs 50%) = {binom_two_sided_p(k, n):.4f}")


def main() -> None:
    pairs = json.loads(Path(PAIRS_PATH).read_text())
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    submitted = {r["id"]: r for r in conn.execute(
        "SELECT * FROM raters WHERE submitted_at IS NOT NULL")}
    unsubmitted = conn.execute(
        "SELECT COUNT(*) c FROM raters WHERE submitted_at IS NULL").fetchone()["c"]
    print(f"raters: {len(submitted)} submitted, {unsubmitted} excluded (never submitted)\n")

    rows = [r for r in conn.execute("SELECT * FROM responses")
            if r["rater_id"] in submitted]

    def correct(r) -> bool:
        return (r["chosen_slot"] == "A") == bool(r["slot_a_is_human"])

    k = sum(correct(r) for r in rows)
    n = len(rows)
    print(line("POOLED", k, n))

    for field in ("cohort", "experience"):
        print(f"\nby {field}:")
        groups: dict = defaultdict(lambda: [0, 0])
        for r in rows:
            g = submitted[r["rater_id"]][field]
            groups[g][0] += correct(r)
            groups[g][1] += 1
        for g in sorted(groups):
            print("  " + line(g, *groups[g]))

    print("\nby map:")
    groups = defaultdict(lambda: [0, 0])
    for r in rows:
        m = pairs.get(r["pair_id"], {}).get("map", "?")
        groups[m][0] += correct(r)
        groups[m][1] += 1
    for m in sorted(groups):
        print("  " + line(m, *groups[m]))

    print("\nrevision effect:")
    for flag, label in ((0, "final = first answer"), (1, "revised answers")):
        sub = [r for r in rows if r["revised"] == flag]
        print("  " + line(label, sum(correct(r) for r in sub), len(sub)))

    print("\nper-rater scores (for the writeup's distribution figure):")
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["rater_id"]][0] += correct(r)
        per[r["rater_id"]][1] += 1
    scores = sorted(f"{k}/{n}" for k, n in per.values())
    print("  " + ", ".join(scores))


if __name__ == "__main__":
    main()