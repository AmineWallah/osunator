import pandas as pd
from pathlib import Path
from config import FEATURES_DIR, OSU_DIR, load_cache, SELECTED_REPLAYS_PATH, REPLAY_CENSUS_PATH, HASH_FREQ_PATH

MIN_REPLAYS_PER_MAP = 10
TOP_PER_MAP = 5

census = REPLAY_CENSUS_PATH
freq = HASH_FREQ_PATH
cache = load_cache()

# maps that made the cut AND resolved AND have their .osu on disk
hash_index = cache['hash_index']                        # hash -> beatmap_id
cut_hashes = set(freq[freq.n_replays >= MIN_REPLAYS_PER_MAP].beatmap_hash)
usable = {h for h in cut_hashes
          if h in hash_index and (OSU_DIR / f"{hash_index[h]}.osu").exists()}
print(f"maps: {len(cut_hashes)} in cut, {len(usable)} resolved+downloaded")

# top-N by accuracy within each usable map
rows = census[census.beatmap_hash.isin(usable)]
rows = (rows.sort_values('accuracy', ascending=False)
            .groupby('beatmap_hash')
            .head(TOP_PER_MAP))

# stale-path guard: prune moved sub-cut files, and surprises happen
exists = rows.replay_path.map(lambda p: Path(p).exists())
missing = int((~exists).sum())
if missing:
    print(f"WARNING: {missing} selected paths no longer exist — skipped")
rows = rows[exists]

out = SELECTED_REPLAYS_PATH
rows[['replay_path', 'beatmap_hash', 'accuracy']].to_csv(out, index=False)
print(f"selected {len(rows)} replays over {rows.beatmap_hash.nunique()} maps -> {out}")
print(rows.accuracy.describe())