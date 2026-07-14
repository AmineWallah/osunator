from osunator.config import HASH_FREQ_PATH, REPLAY_CENSUS_PATH, SUITABLE_DIR
from build import REPLAY_CENSUS_FIELDS
from osunator.data.cleanup import get_map_accuracy
from tqdm import tqdm
from osrparse import Replay
from collections import Counter
import csv

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