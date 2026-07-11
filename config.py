import csv
import json
from pathlib import Path

ROOT = Path('/home/amine/PycharmProjects/osunator')
REPLAYS = ROOT / 'replays'
PLAYERS_DIR = REPLAYS / 'players'
SUITABLE_DIR = REPLAYS / 'suitable'
CORRUPT_DIR = REPLAYS / 'corrupt'
UNRESOLVED_DIR = REPLAYS / 'unresolved'
OSU_DIR = ROOT / 'maps' / 'downloaded'
FEATURES_DIR = ROOT / 'features'  
STATS_PATH = FEATURES_DIR / 'norm_stats.json'
HASH_FREQ_PATH = FEATURES_DIR / 'hash_freq.csv'


def load_cache():
    try:
        with open('cache.json') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"maps": {}, "hash_index": {}}


def save_cache(cache):
    with open('cache.json', 'w') as f:
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