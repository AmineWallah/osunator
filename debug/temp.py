import csv
from osrparse import Replay
from config import load_cache, FEATURES_DIR, MANIFEST_PATH

cache = load_cache()
replay = Replay.from_path('/home/amine/PycharmProjects/osunator/replays/suitable/9852d676135cf535b6d28d2861aa1730.osr')
h = replay.beatmap_hash
print("replay hash :", repr(h))

# is it there under different case/whitespace?
keys = list(cache['hash_index'].keys())
print("sample key  :", repr(keys[0]))
print("lower match :", h.lower() in {k.lower() for k in keys})

# and what did the manifest record for this replay?
rows = list(csv.DictReader(open(MANIFEST_PATH)))
mine = [r for r in rows if '9852d676135cf535b6d28d2861aa1730' in r['replay_path']]
print("manifest row hash:", repr(mine[0]['beatmap_hash']) if mine else "ROW NOT FOUND")