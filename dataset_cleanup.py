import asyncio
import os
from dotenv import load_dotenv
from ossapi import OssapiAsync
from osrparse import Replay, Mod
from tqdm.asyncio import tqdm
import hashlib
import aiohttp
import shutil
from src.osunator.parsing import convert_to_absolute
from config import (PLAYERS_DIR, SUITABLE_DIR, CORRUPT_DIR, UNRESOLVED_DIR,
                    OSU_DIR, load_cache, save_cache, load_hash_freq)

load_dotenv()

client_secret = os.getenv("CLIENT_SECRET")
client_id = os.getenv("CLIENT_ID")

api = OssapiAsync(client_id, client_secret)

# --- Constants ---

OSU_ENDPOINT = "https://osu.ppy.sh/osu/{}"          # per-difficulty .osu download (no audio, MD5-verifiable)
USER_AGENT = "osunator-dataset-builder/0.1 (personal research project)"

SEM = asyncio.Semaphore(5)      # bounds concurrent replay-resolve API calls
DL_SEM = asyncio.Semaphore(5)   # bounds concurrent .osu downloads
PACE = 0.25                     # seconds; polite delay before each download while holding DL_SEM

BOM = b'\xef\xbb\xbf'           # UTF-8 byte-order-mark some .osu files start with
MAGIC = b'osu file format v'    # every valid .osu file starts with this (BOM-stripped) header

THRESHOLD = 98                                  # min in-game accuracy % for a replay to be "suitable"
ALLOWED_MODS = [Mod.NoMod, Mod.Hidden]          # exact-match list — only pure NoMod/HD, not combos (e.g. HD+HR)
REPLAYS_PER_MAP = 10

def mods_ok(replay):
    """True if the replay's mod combo is in our allowed list. Exact match only —
    correct for single mods, would need revisiting if DT/HR combos are ever wanted."""
    return replay.mods in ALLOWED_MODS


def get_map_accuracy(replay):
    """osu! standard accuracy formula from hit counts. Caller should guard against
    a replay with zero total hits (would ZeroDivisionError) before calling this."""
    return (replay.count_300 * 100 + replay.count_100 * 100/3 + replay.count_50 * 100/6) / (replay.count_300 + replay.count_100 + replay.count_50 + replay.count_miss)


def is_suitable(replay):
    """A replay is usable training data if it's accurate enough and played with
    an allowed mod combo."""
    acc = get_map_accuracy(replay)
    return acc >= THRESHOLD and mods_ok(replay)


def filter_replays():
    SUITABLE_DIR.mkdir(parents=True, exist_ok=True)
    paths = list(PLAYERS_DIR.rglob('*.osr'))

    hashes = set()
    moved = skipped = failed = collided = 0
    for path in tqdm(paths):                     # progress bar — you already import tqdm
        try:
            replay = Replay.from_path(str(path))
        except Exception:
            failed += 1
            continue                              # dropped the per-file print — 300k lines of spam otherwise

        if not is_suitable(replay):
            skipped += 1
            continue

        hashes.add(replay.beatmap_hash)          # free API-budget count

        dest = SUITABLE_DIR / path.name
        if dest.exists():
            collided += 1
            continue
        shutil.copy(str(path), str(dest))
        moved += 1

    print(f"filter: {moved} copied, {skipped} rejected, {collided} skipped (exists), {failed} parse-fail")
    print(f"unique beatmap hashes among survivors: {len(hashes)}")


def _move_to(path, dest_dir, dry_run):
    """Move a single file into dest_dir, creating it if needed. Skips (with a
    warning) if a same-named file already exists there, rather than overwriting.
    When dry_run=True, reports what it *would* do without touching the filesystem —
    always run once with dry_run=True before trusting a prune."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    if dest.exists():
        tqdm.write(f"collision, leaving in place: {path.name}")
        return
    if not dry_run:
        path.rename(dest)   # same filesystem — atomic, cheap


def prune_suitable(cache, dry_run=True):
    """Post-resolve cleanup pass over suitable/. Must run AFTER resolve + download,
    since it needs the full cache to know which maps exist.

    Sorts every replay in suitable/ into one of:
    - parse_fail / corrupt -> moved to CORRUPT_DIR (unparseable, or convert_to_absolute
      raised a genuine corruption error — see CORRUPT_CAP in parsing.py)
    - unresolved -> moved to UNRESOLVED_DIR (replay's map hash never resolved to a
      beatmap id; almost always a custom/unranked map that will never have a .osu)
    - missing_osu -> left in place in suitable/, NOT moved. This case means the map
      resolved but its .osu file isn't on disk yet — that's likely just "downloads
      haven't run/finished," not "this map doesn't exist." Deleting/moving these
      would silently destroy recoverable replays; re-run the download phase instead.
    - kept -> genuinely fine, stays in suitable/.

    Always moves (never deletes) so a bad verdict is recoverable by hand.
    """
    from collections import Counter
    stats = Counter()
    for path in SUITABLE_DIR.rglob('*.osr'):
        try:
            replay = Replay.from_path(str(path))
        except Exception:
            stats['parse_fail'] += 1
            _move_to(path, CORRUPT_DIR, dry_run); continue

        if replay.beatmap_hash not in cache['hash_index']:
            stats['unresolved'] += 1
            _move_to(path, UNRESOLVED_DIR, dry_run); continue

        try:
            convert_to_absolute(replay)
        except ValueError:
            stats['corrupt'] += 1
            _move_to(path, CORRUPT_DIR, dry_run); continue

        beatmap_id = cache['hash_index'][replay.beatmap_hash]
        if not (OSU_DIR / f"{beatmap_id}.osu").exists():
            stats['missing_osu (left in suitable)'] += 1
            continue

        stats['kept'] += 1
    print(f"{'DRY RUN — ' if dry_run else ''}{dict(stats)}")

async def get_map_from_hash(h, cache):
    if h in cache['hash_index']:
        return cache['hash_index'][h]

    try:
        beatmap = await api.beatmap(checksum=h)
    except ValueError as e:
        if "api returned an error" in str(e):
            return None
        raise

    beatmap_id = str(beatmap.id)
    cache['maps'][beatmap_id] = {
        'hash': beatmap.checksum,
        'beatmapset_id': beatmap.beatmapset_id,
        'resolved': True,
        'downloaded': False,
        'verified': False,
    }
    cache['hash_index'][h] = beatmap_id
    return beatmap_id

async def get_map_from_replay(replay, cache):
    """Resolve one replay's beatmap_hash to a beatmap_id, using cache['hash_index']
    as a dedup layer so the same map is never looked up twice via the API.
    On a cache miss, calls the osu! API by checksum. A checksum the API doesn't
    recognize (custom/deleted map) raises ValueError with "api returned an error"
    in the message — that specific case returns None (unresolvable, skip); any
    other ValueError is a real problem and is re-raised.
    cache is passed in and mutated in place, not loaded/saved here (load-once/
    save-once happens at the edges, in main())."""
    h = replay.beatmap_hash
    if h in cache['hash_index']:
        return cache['hash_index'][h]

    try:
        beatmap = await api.beatmap(checksum=h)
    except ValueError as e:
        if "api returned an error" in str(e):
            return None
        raise

    beatmap_id = str(beatmap.id)
    cache['maps'][beatmap_id] = {
        'hash': beatmap.checksum,
        'beatmapset_id': beatmap.beatmapset_id,
        'resolved': True,
        'downloaded': False,
        'verified': False,
    }
    cache['hash_index'][h] = beatmap_id
    return beatmap_id

async def resolve_one(path, cache):
    """Resolve a single replay's map under SEM (bounds how many resolutions run
    concurrently). Skips (with a message) any replay that fails to parse at all."""
    async with SEM:
        try:
            replay = Replay.from_path(str(path))
        except Exception as e:
            tqdm.write(f"skip (parse fail) {path.name}: {e}")
            return
        await get_map_from_replay(replay, cache)

async def resolve_one_hash_version(h, cache):
    """Resolve a single replay's map under SEM (bounds how many resolutions run
    concurrently). Skips (with a message) any replay that fails to parse at all."""
    async with SEM:
        await get_map_from_hash(h, cache)

async def download_osu(session, beatmap_id, record):
    """Download one map's .osu file, verify it, and write it to disk atomically.

    - Skips entirely if already downloaded and the file exists (dedup).
    - Paced/bounded by DL_SEM + PACE, since this endpoint's rate limit is undocumented.
    - Validates the response actually looks like a .osu file (checks the header
      magic bytes, BOM-stripped) — deleted/unavailable maps return HTML or an
      empty body instead of a 404, so a non-200 status isn't the only failure mode.
    - MD5s the RAW bytes (not decoded text — BOM changes the hash) and compares
      against the replay's beatmap_hash to catch hash drift (map edited since the
      replay was recorded). A mismatch is saved anyway but flagged verified=False.
    - Writes to a .part temp file and renames atomically, so a crash mid-download
      never leaves a corrupt half-file sitting at the real path.
    """
    dest = OSU_DIR / f"{beatmap_id}.osu"

    # file-level dedup: already have it -> skip (resolved-but-not-downloaded only get here anyway)
    if record.get('downloaded') and dest.exists():
        return

    url = OSU_ENDPOINT.format(beatmap_id)
    async with DL_SEM:
        await asyncio.sleep(PACE)              # gentle pacing while holding a slot
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    tqdm.write(f"skip {beatmap_id}: HTTP {resp.status}")
                    return
                content = await resp.read()    # RAW bytes — required for a correct MD5
        except Exception as e:
            tqdm.write(f"fail {beatmap_id}: {e}")
            return

    # sanity: unavailable/deleted maps return HTML or an empty body, not a .osu
    body = content[3:] if content.startswith(BOM) else content
    if not body.startswith(MAGIC):
        tqdm.write(f"skip {beatmap_id}: not a valid .osu (unavailable?)")
        return

    # verify: MD5 of the exact bytes must equal the replay's beatmap hash
    actual = hashlib.md5(content).hexdigest()
    expected = record['hash'].lower()
    verified = (actual == expected)
    if not verified:
        tqdm.write(f"warn {beatmap_id}: md5 mismatch (hash drift) — saved but unverified")

    # write to temp then atomically rename, so a crash never leaves a half file at the real path
    tmp = dest.parent / (dest.name + '.part')
    with open(tmp, 'wb') as f:                 # binary — preserve exact bytes
        f.write(content)
    tmp.rename(dest)

    record['downloaded'] = True
    record['verified'] = verified


async def main():
    cache = load_cache()
    hash_freq = load_hash_freq()

    # resolve from hashes (census regime) — the ONLY correct phase-1 now
    await tqdm.gather(*[resolve_one_hash_version(h, cache)
                        for h in hash_freq.keys() if hash_freq[h] >= REPLAYS_PER_MAP])

    # download: no-ops for files on disk IF we pre-mark them (fresh cache says downloaded=False)
    for bid, rec in cache['maps'].items():
        if (OSU_DIR / f"{bid}.osu").exists():
            rec['downloaded'] = True  # file's already here from the original run

    targets = [(bid, rec) for bid, rec in cache['maps'].items()
               if rec.get('resolved') and not rec.get('downloaded')]
    print(f"downloads needed: {len(targets)}")  # expect ~0-60 (the originally-failed ones)
    async with aiohttp.ClientSession(headers={"User-Agent": USER_AGENT}) as session:
        await tqdm.gather(*[download_osu(session, bid, rec) for bid, rec in targets])

    save_cache(cache)  # THE POINT OF THE RUN


if __name__ == "__main__":
    asyncio.run(main())
    # filter_replays()