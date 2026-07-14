"""Manifest access + replay picking for diagnostics.

One authoritative way to answer "which replay am I looking at?" so that
every instrument in diagnostics/ stops hardcoding paths (commit 4). All
lookups go through data_meta/manifest.csv via config.MANIFEST_PATH — no
bare relative strings anywhere in this module (cwd-dependent path incident,
this cycle).

Contracts callers rely on:
- Rows are plain dicts with the manifest's string fields:
  example_id, beatmap_id, beatmap_hash, beatmap_path, beatmap_name,
  replay_path, npz_path, accuracy, split. Nothing is coerced — `accuracy`
  stays a string; cast at the point of use.
- pick_from_manifest REFUSES ambiguity instead of guessing: a name query
  matching >1 distinct replay raises ValueError listing the candidates.
  Instruments must fail loudly, never silently diagnose the wrong replay
  (map-identity incidents, twice).
- Seeded random picks are reproducible: same (split, seed, manifest) ->
  same row, independent of global RNG state.
"""

import csv

from osunator.config import MANIFEST_PATH

import numpy as np


def load_manifest(split=None):
    """All manifest rows, optionally filtered to one split.

    :param split: None (everything), 'train', or 'test'
    :return: list of row dicts (cheap — strings only, arrays stay on disk)
    :raises FileNotFoundError: manifest.csv missing (build_dataset not run,
        or running against a tree where data_meta/ wasn't restored)
    :raises ValueError: split given but matches zero rows — catches typos
        ('Test', 'val') instead of returning [] and letting the caller
        crash later with an opaque empty-sequence error
    """
    with open(MANIFEST_PATH, newline='') as f:
        rows = list(csv.DictReader(f))

    if split is not None:
        rows = [r for r in rows if r['split'] == split]
        if not rows:
            raise ValueError(f"split={split!r} matches no manifest rows "
                             f"(valid: 'train', 'test')")
    return rows


def pick_from_manifest(name=None, split=None, seed=None):
    """Pick exactly one manifest row, by name query or seeded random draw.

    Selection modes:
    - name given: case-insensitive substring match against beatmap_name
      AND example_id (so both "raise my sword" and a replay filename stem
      work). Exactly one match required.
    - name None: uniform random row from the (split-filtered) manifest,
      seeded if seed is given.

    :param name: substring to match (case-insensitive), or None for random
    :param split: restrict pool to 'train'/'test' before matching/drawing
    :param seed: int for reproducible random picks; ignored when name is given
    :return: single manifest row dict
    :raises ValueError: zero matches, or ambiguous name (>1 match — error
        message lists up to 10 candidates so the caller can narrow the query)
    """
    rows = load_manifest(split=split)

    if name is not None:
        q = name.lower()
        matches = [r for r in rows
                   if q in r['beatmap_name'].lower()
                   or q in r['example_id'].lower()]
        if not matches:
            raise ValueError(f"no manifest row matches {name!r}"
                             f"{f' in split={split!r}' if split else ''}")
        if len(matches) > 1:
            preview = "\n  ".join(
                f"{r['example_id']}  ({r['beatmap_name']}, {r['split']})"
                for r in matches[:10])
            more = f"\n  ... and {len(matches) - 10} more" if len(matches) > 10 else ""
            raise ValueError(
                f"ambiguous: {name!r} matches {len(matches)} replays — "
                f"narrow the query:\n  {preview}{more}")
        return matches[0]

    rng = np.random.default_rng(seed)
    return rows[int(rng.integers(len(rows)))]