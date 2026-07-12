import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_META = ROOT / 'data_meta'

REPLAYS = ROOT / 'replays'
PLAYERS_DIR = REPLAYS / 'players'
SUITABLE_DIR = REPLAYS / 'suitable'
CORRUPT_DIR = REPLAYS / 'corrupt'
UNRESOLVED_DIR = REPLAYS / 'unresolved'
OSU_DIR = ROOT / 'maps' / 'downloaded'
FEATURES_DIR = ROOT / 'features'  
STATS_PATH = DATA_META / 'norm_stats.json'
HASH_FREQ_PATH = DATA_META / 'hash_freq.csv'
CACHE_PATH = DATA_META / 'cache.json'
MANIFEST_PATH = DATA_META / 'manifest.csv'


def load_cache():
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"maps": {}, "hash_index": {}}


def save_cache(cache):
    with open('data_meta/cache.json', 'w') as f:
        json.dump(cache, f, indent=2)


def load_norm_stats(path=STATS_PATH):
    with open(path) as f:
        return json.load(f)

def load_hash_freq(path=HASH_FREQ_PATH):
    with open(path, 'r', encoding='utf-8') as file:
        return {
            row['beatmap_hash']: int(row['n_replays'])
            for row in csv.DictReader(file)
        }