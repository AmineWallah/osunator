import hashlib
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import slider
from config import load_norm_stats
from parsing import build_grid, resample_map_features, build_training_example
from training_data import normalize_example, chunk_example, assemble_xy
from mdn import CorrelatedSampler
from osrparse import Replay
from osrparse.utils import ReplayEventOsu, Key, Mod, GameMode
from datetime import datetime, timezone
BEATMAP_PATH = "/home/amine/.local/share/osu-wine/osu!/Songs/781509 Vickeblanka - Black Rover (TV Size) [no video]/Vickeblanka - Black Rover (TV Size) (Sotarks) [Extreme].osu"   # set this to a real .osu file on your machine
MODEL_PATH = 'best_model.keras'
OUTPUT_REPLAY_PATH = 'generated.osr'
ONSET_THR = 0.35
RELEASE_THR = 0.35
COOLDOWN_TICKS = 3


def reset_all_states(model):
    for layer in model.layers:
        if hasattr(layer, 'reset_states'):
            layer.reset_states()


def generate_replay(model, beatmap, stats, start_x=256.0, start_y=192.0, temperature=0.0):
    grid = build_grid(beatmap)
    map_feats = resample_map_features(beatmap, grid)
    sampler = CorrelatedSampler(temperature=temperature)

    reset_all_states(model)
    sampler.reset()

    @tf.function
    def predict_step(x):
        return model(x, training=False)

    cursor_x, cursor_y = start_x, start_y
    pred_cursor_x, pred_cursor_y = [], []
    pred_key_onset, pred_key_offset = [], []

    for t in tqdm(range(len(grid)), desc="generating"):
        target_dx = (map_feats['target_x'][t] - cursor_x) / stats['target_dx']['scale']
        target_dy = (map_feats['target_y'][t] - cursor_y) / stats['target_dy']['scale']
        approach_progress = map_feats['approach_progress'][t]
        log_ttn = np.log1p(map_feats['time_to_next'][t])
        time_to_next_norm = (log_ttn - stats['time_to_next']['mean']) / stats['time_to_next']['std']

        X = np.array([[[
            target_dx, target_dy, time_to_next_norm,
            float(map_feats['is_active'][t]),
            float(map_feats['is_slider'][t]),
            float(map_feats['is_spinner'][t]),
            cursor_x / 512.0,
            cursor_y / 384.0,
            float(approach_progress),
        ]]], dtype=np.float32)

        cursor_pred, key_pred = predict_step(X)

        sampled = sampler.sample(cursor_pred[0, 0].numpy()[None, None, :])
        dx_norm, dy_norm = sampled[0, 0, 0], sampled[0, 0, 1]
        dx_px = dx_norm * stats['cursor_dx']['std'] + stats['cursor_dx']['mean']
        dy_px = dy_norm * stats['cursor_dy']['std'] + stats['cursor_dy']['mean']

        cursor_x += dx_px   # THIS tick's own prediction becomes NEXT tick's position — the closed loop
        cursor_y += dy_px

        pred_cursor_x.append(cursor_x)
        pred_cursor_y.append(cursor_y)
        pred_key_onset.append(key_pred[0, 0, 0:2].numpy())
        pred_key_offset.append(key_pred[0, 0, 2:4].numpy())

    return {
        'grid': grid,
        'pred_cursor_x': np.array(pred_cursor_x),
        'pred_cursor_y': np.array(pred_cursor_y),
        'pred_key_onset': np.array(pred_key_onset),
        'pred_key_offset': np.array(pred_key_offset),
    }


def result_to_replay(result, beatmap_hash, username="osunator-bot"):
    grid = result['grid']
    pred_x = result['pred_cursor_x']
    pred_y = result['pred_cursor_y']
    onset_prob = result['pred_key_onset']  # (n, 2) raw probabilities
    offset_prob = result['pred_key_offset']
    n = len(grid)
    press_now = onset_prob.max(axis=1) > ONSET_THR  # slot-agnostic: model says WHEN
    release_now = offset_prob.max(axis=1) > RELEASE_THR

    held_slot = None
    last_press_tick = -10 ** 9
    last_slot = 1
    keys_per_tick = []
    for t in range(n):
        if press_now[t] and (t - last_press_tick) >= COOLDOWN_TICKS:
            slot = 1 - last_slot  # full alternation (your style)
            held_slot = slot
            last_press_tick, last_slot = t, slot
        elif held_slot is not None and release_now[t] and (t - last_press_tick) >= 2:
            held_slot = None  # >=2: never release same/next tick
        k = Key(0)
        if held_slot == 0: k = k | Key.K1
        if held_slot == 1: k = k | Key.K2
        keys_per_tick.append(k)

    # absolute ms -> per-frame deltas. ROUND THE ABSOLUTE TIMES FIRST, then
    # diff — rounding each delta independently leaks +0.33ms/frame on a
    # 16.67ms grid and accumulates to seconds over a map (the visible-lag bug).
    abs_ms = np.round(np.asarray(grid, dtype=float)).astype(int)
    time_deltas = np.diff(abs_ms, prepend=0)

    # anchor frame first (real client convention: 0-delta frame parked
    # off-screen, then the lead-in delta on the first real frame)
    replay_data = [ReplayEventOsu(0, 256.0, -500.0, Key(0))]
    replay_data += [
        ReplayEventOsu(int(time_deltas[t]), float(pred_x[t]), float(pred_y[t]), keys_per_tick[t])
        for t in tqdm(range(n), desc="building replay events")
    ]

    return Replay(
        mode=GameMode.STD,
        game_version=20231234,          # arbitrary but plausible osu client version number
        beatmap_hash=beatmap_hash,      # REAL hash — associates this replay with the correct map
        username=username,
        replay_hash="",                 # not validated locally; left empty
        count_300=0, count_100=0, count_50=0, count_geki=0, count_katu=0, count_miss=0,
        score=0, max_combo=0, perfect=False,   # placeholders — no hit-judgement simulation exists
        mods=Mod.NoFail,
        life_bar_graph=None,
        timestamp=datetime.now(timezone.utc),
        replay_data=replay_data,
        replay_id=0,
        rng_seed=None,                  # osrparse's packer omits
    )

def predict_replay(model, beatmap, replay, stats, temperature=0.0):
    """TEACHER-FORCED diagnostic: at every tick the model sees the REAL human
    cursor position (target_dx/dy computed from ground truth, exactly like
    training), but the returned trajectory is built from the MODEL's
    predicted deltas. Isolates "is the per-tick prediction good" from the
    closed-loop drift problem — if THIS looks good but generate_replay looks
    bad, the model is fine and the failure lives in the closed loop; if THIS
    is also bad, the model (or its loading) is the problem.
    """
    example = build_training_example(beatmap, replay)
    normalized = normalize_example(example, stats)
    chunks = chunk_example(normalized)
    sampler = CorrelatedSampler(temperature=temperature)
    reset_all_states(model)


    pred_dx_chunks, pred_dy_chunks, mask_chunks = [], [], []
    for chunk in chunks:
        X, _, _, mask = assemble_xy(chunk)
        cursor_pred, key_pred = model.predict(X[None, ...], verbose=0)
        sampled = sampler.sample(cursor_pred[0])  # (600, 20) -> (600, 2), greedy
        pred_dx_chunks.append(sampled[:, 0])
        pred_dy_chunks.append(sampled[:, 1])
        mask_chunks.append(mask)

    pred_dx = np.concatenate(pred_dx_chunks)
    pred_dy = np.concatenate(pred_dy_chunks)
    full_mask = np.concatenate(mask_chunks)

    n_real = int(full_mask.sum())   # only the last chunk has padding
    pred_dx = pred_dx[:n_real]
    pred_dy = pred_dy[:n_real]

    # un-normalize predicted deltas back to pixel units
    pred_dx_px = pred_dx * stats['cursor_dx']['std'] + stats['cursor_dx']['mean']
    pred_dy_px = pred_dy * stats['cursor_dy']['std'] + stats['cursor_dy']['mean']

    true_x = example['cursor_x'][:n_real]
    true_y = example['cursor_y'][:n_real]

    # integrate predicted deltas from the human's true start — exact inverse
    # of np.diff(..., prepend=x[0]) used to build cursor_dx in training
    pred_x = true_x[0] + np.cumsum(pred_dx_px)
    pred_y = true_y[0] + np.cumsum(pred_dy_px)

    return {
        'true_cursor_x': true_x, 'true_cursor_y': true_y,
        'pred_cursor_x': pred_x, 'pred_cursor_y': pred_y,
    }

if __name__ == "__main__":
    stats = load_norm_stats()
    model = keras.models.load_model(MODEL_PATH, compile=False)
    beatmap = slider.beatmap.Beatmap.from_path(BEATMAP_PATH)
    with open(BEATMAP_PATH, 'rb') as f:  # raw bytes — BOM matters, same as download_osu
        beatmap_hash = hashlib.md5(f.read()).hexdigest()

    print("generating...")
    result = generate_replay(model, beatmap, stats, temperature=0.0) # try with other temp value
    print(f"generated {len(result['grid'])} ticks")
    print(f"pred_cursor_x range: {result['pred_cursor_x'].min():.1f} to {result['pred_cursor_x'].max():.1f}  (playfield: 0-512)")
    print(f"pred_cursor_y range: {result['pred_cursor_y'].min():.1f} to {result['pred_cursor_y'].max():.1f}  (playfield: 0-384)")

    replay = result_to_replay(result, beatmap_hash)
    replay.write_path(OUTPUT_REPLAY_PATH)
    print(f"wrote {OUTPUT_REPLAY_PATH}")