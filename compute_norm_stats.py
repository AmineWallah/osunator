import csv
import json
import numpy as np

from config import FEATURES_DIR

MANIFEST_PATH = FEATURES_DIR / 'manifest.csv'
STATS_PATH = FEATURES_DIR / 'norm_stats.json'


def summarize(name, arr):
    print(f"{name}: n={len(arr)}  mean={arr.mean():.3f}  std={arr.std():.3f}  "
          f"min={arr.min():.3f}  max={arr.max():.3f}  "
          f"p50={np.percentile(arr, 50):.3f}  p99={np.percentile(arr, 99):.3f}")


def compute_stats():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    train_rows = [r for r in rows if r['split'] == 'train']
    print(f"pooling stats from {len(train_rows)} train examples...")

    cursor_dx_all, cursor_dy_all, time_to_next_all = [], [], []
    for row in train_rows:
        data = np.load(FEATURES_DIR / f"{row['example_id']}.npz")
        cursor_dx_all.append(data['cursor_dx'])
        cursor_dy_all.append(data['cursor_dy'])
        time_to_next_all.append(data['time_to_next'])

    cursor_dx_all = np.concatenate(cursor_dx_all)
    cursor_dy_all = np.concatenate(cursor_dy_all)
    time_to_next_all = np.concatenate(time_to_next_all)
    log_time_to_next_all = np.log1p(time_to_next_all)   # log1p handles the 0-ms case cleanly (log1p(0)=0)

    print()
    summarize('cursor_dx', cursor_dx_all)
    summarize('cursor_dy', cursor_dy_all)
    summarize('time_to_next (raw)', time_to_next_all)
    summarize('time_to_next (log1p)', log_time_to_next_all)

    stats = {
        'target_dx': {'transform': 'divide', 'scale': 512.0},   # not data-driven — fixed playfield width
        'target_dy': {'transform': 'divide', 'scale': 384.0},   # fixed playfield height
        'cursor_dx': {'transform': 'zscore',
                      'mean': float(cursor_dx_all.mean()), 'std': float(cursor_dx_all.std())},
        'cursor_dy': {'transform': 'zscore',
                      'mean': float(cursor_dy_all.mean()), 'std': float(cursor_dy_all.std())},
        'time_to_next': {'transform': 'log1p_zscore',
                          'mean': float(log_time_to_next_all.mean()),
                          'std': float(log_time_to_next_all.std())},
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nsaved stats to {STATS_PATH}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    compute_stats()