import csv
import hashlib
from tqdm import tqdm
import numpy as np
from collections import Counter
from config import SUITABLE_DIR, FEATURES_DIR, OSU_DIR, MANIFEST_PATH, REPLAY_CENSUS_PATH, HASH_FREQ_PATH, \
    SELECTED_REPLAYS_PATH
from src.osunator.parsing import beatmap_replay_pairs, build_training_example
from dataset_cleanup import get_map_accuracy
from osrparse import Replay
from pathlib import Path

MANIFEST_FIELDS = [
    'example_id', 'beatmap_id', 'beatmap_hash', 'beatmap_path', 'beatmap_name',
    'replay_path','npz_path', 'accuracy', 'split',
]

REPLAY_CENSUS_FIELDS = [
    'replay_path', 'replay_id', 'accuracy', 'beatmap_hash', 'mods', 'mode'
]

TEST_FRACTION = 0.15   # ~15% of MAPS (not replays) go to test


def assign_split(beatmap_id, test_fraction=TEST_FRACTION):
    """
    Deterministically assign a beatmap_id to 'train' or 'test' via a stable
    hash of the id itself — not a random shuffle.
    """
    digest = hashlib.md5(str(beatmap_id).encode()).hexdigest()
    bucket = int(digest, 16) % 100
    return 'test' if bucket < test_fraction * 100 else 'train'


def build_dataset():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # paths = list(SUITABLE_DIR.rglob('*.osr'))
    sel = csv.DictReader(open(SELECTED_REPLAYS_PATH))
    paths = [Path(r['replay_path']) for r in sel]

    pairs = beatmap_replay_pairs(paths)   # (beatmap, beatmap_id, replay, path) 4-tuples

    manifest_rows = []
    built = skipped_existing = failed = 0
    with open(MANIFEST_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()

        for beatmap, beatmap_id, replay, path in tqdm(pairs, total=len(paths), desc="building examples"):
            example_id = path.stem  # replay filename without .osr — already unique
            out_path = FEATURES_DIR / f"{example_id}.npz"

            if out_path.exists():
                skipped_existing += 1
            else:
                try:
                    example = build_training_example(beatmap, replay)
                except ValueError as e:
                    tqdm.write(f"skip {path.name}: {e}")
                    failed += 1
                    continue

                np.savez(out_path, **example)  # uncompressed — savez, not savez_compressed
                built += 1

            row = {
                'example_id': example_id,
                'beatmap_id': beatmap_id,
                'beatmap_hash': replay.beatmap_hash,  # hash this replay was recorded against
                'beatmap_path': str(OSU_DIR / f"{beatmap_id}.osu"),
                'beatmap_name': beatmap.display_name,  # "Artist - Title [Difficulty]"
                'replay_path': str(path),
                'npz_path': str(out_path),  # Added to help lazy loading
                'accuracy': get_map_accuracy(replay),
                'split': assign_split(beatmap_id),
            }
            manifest_rows.append(row)
            writer.writerow(row)



    n_train_maps = len({r['beatmap_id'] for r in manifest_rows if r['split'] == 'train'})
    n_test_maps = len({r['beatmap_id'] for r in manifest_rows if r['split'] == 'test'})
    n_train_rows = sum(1 for r in manifest_rows if r['split'] == 'train')
    n_test_rows = sum(1 for r in manifest_rows if r['split'] == 'test')

    print(f"build: {built} written, {skipped_existing} already existed, {failed} failed, "
          f"{len(manifest_rows)} total in manifest")
    print(f"split: {n_train_maps} train maps ({n_train_rows} replays), "
          f"{n_test_maps} test maps ({n_test_rows} replays)")

def build_replay_census():
    replay_paths = list(SUITABLE_DIR.rglob('*.osr'))
    failed = 0
    hash_counts = Counter()

    with open(REPLAY_CENSUS_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=REPLAY_CENSUS_FIELDS)
        writer.writeheader()
        for path in tqdm(replay_paths):
            try:
                replay = Replay.from_path(str(path))
            except Exception:
                failed += 1
                continue
            total = replay.count_300 + replay.count_100 + replay.count_50 + replay.count_miss
            if total == 0:
                failed += 1
                continue
            writer.writerow({
                'replay_path': str(path),
                'replay_id': replay.replay_id,
                'accuracy': get_map_accuracy(replay),
                'beatmap_hash': replay.beatmap_hash,
                'mods': int(replay.mods),
                'mode': replay.mode.value,
            })
            if replay.mode.value == 0:                 # std only feeds the cut decision
                hash_counts[replay.beatmap_hash] += 1

    print(f"census: {len(replay_paths) - failed} rows, {failed} skipped")

    # frequency table + threshold curve (std-only)
    with open(HASH_FREQ_PATH, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['beatmap_hash', 'n_replays'])
        w.writerows(hash_counts.most_common())

    for N in (1, 2, 3, 5, 8, 10, 15):
        maps = sum(1 for c in hash_counts.values() if c >= N)
        reps = sum(c for c in hash_counts.values() if c >= N)
        print(f"maps with >={N:2d} replays: {maps:6d}   replays they hold: {reps:6d}")


if __name__ == "__main__":
    build_dataset()