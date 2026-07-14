"""Raw example -> training tensors: perturb, normalize, chunk, assemble.

Order is a contract: perturb needs RAW pixels, normalize needs the
perturbed values, chunk goes last (never perturb padded chunks).
Full perturbation rationale: docs/perturbation.md.
"""

import numpy as np
from osunator.parsing import CHUNK_TICKS


def perturb_example(example: dict[str, np.ndarray], noise_std_px=10.0,
                    offset_std_px=25.0, offset_decay=0.95,
                    rng=None) -> dict[str, np.ndarray]:
    """DART-style perturbation with a proportional-controller expert.

    Displaces the training path (smooth OU schedule offset, stationary std
    offset_std_px, mean-reversion 1-offset_decay per tick + white noise
    noise_std_px) and relabels every tick with the corrective action from
    the displaced position. Why OU: earlier constant-lag and trapezoid
    variants made teleport labels at segment boundaries — measured, reverted.
    Details + label algebra: docs/perturbation.md.

    :param example: RAW example from build_training_example() — pixel units,
        unchunked, unnormalized.
    :param rng: np.random.Generator; fresh unseeded if None.
    :return: NEW dict. Rewritten: target_dx/dy (inputs from perturbed pos),
        cursor_dx/dy (corrective labels), cursor_x/y (PERTURBED positions —
        no longer ground truth). Other keys alias the input.
    """
    if rng is None:
        rng = np.random.default_rng()
    out = dict(example)

    x_true = example['cursor_x']
    y_true = example['cursor_y']
    n = len(x_true)

    # -- smooth OU offset track --
    off_x = np.zeros(n)
    off_y = np.zeros(n)
    if offset_std_px > 0 and 0 <= offset_decay < 1:
        # stationary-variance identity: var(o) = sigma_w^2 / (1 - decay^2)
        sigma_w = offset_std_px * np.sqrt(1.0 - offset_decay ** 2)
        wx = rng.normal(0.0, sigma_w, size=n)
        wy = rng.normal(0.0, sigma_w, size=n)
        # start at the stationary distribution so early ticks aren't special
        ox = rng.normal(0.0, offset_std_px)
        oy = rng.normal(0.0, offset_std_px)
        for t in range(n):
            off_x[t] = ox
            off_y[t] = oy
            ox = offset_decay * ox + wx[t]
            oy = offset_decay * oy + wy[t]

    # base = true path displaced by the wandering offset
    x_base = x_true + off_x
    y_base = y_true + off_y

    # --- gaussian noise on top of the base ---
    x_pert = x_base + rng.normal(0.0, noise_std_px, size=n)
    y_pert = y_base + rng.normal(0.0, noise_std_px, size=n)

    # inputs: target vectors now point FROM the perturbed position
    out['target_dx'] = example['target_dx'] + (x_true - x_pert)
    out['target_dy'] = example['target_dy'] + (y_true - y_pert)

    # labels: one tick ahead along the base path, from the perturbed position
    # = true motion + proportional offset correction + noise correction
    label_dx = np.empty_like(x_true)
    label_dx[:-1] = x_base[1:] - x_pert[:-1]
    label_dx[-1] = x_true[-1] - x_pert[-1]  # last tick: pure correction to true
    label_dy = np.empty_like(y_true)
    label_dy[:-1] = y_base[1:] - y_pert[:-1]
    label_dy[-1] = y_true[-1] - y_pert[-1]
    out['cursor_dx'] = label_dx
    out['cursor_dy'] = label_dy

    # keep perturbed positions visible for debugging/plotting
    out['cursor_x'] = x_pert
    out['cursor_y'] = y_pert
    out['offset_mag'] = np.hypot(off_x, off_y)  # debug
    return out


def normalize_example(example, stats):
    out = dict(example)   # shallow copy — only transformed keys replaced

    out['target_dx'] = example['target_dx'] / stats['target_dx']['scale']
    out['target_dy'] = example['target_dy'] / stats['target_dy']['scale']

    out['cursor_dx'] = (example['cursor_dx'] - stats['cursor_dx']['mean']) / stats['cursor_dx']['std']
    out['cursor_dy'] = (example['cursor_dy'] - stats['cursor_dy']['mean']) / stats['cursor_dy']['std']

    out['cursor_x_norm'] = example['cursor_x'] / 512.0
    out['cursor_y_norm'] = example['cursor_y'] / 384.0

    log_ttn = np.log1p(example['time_to_next'])   # log1p: 0 ms -> 0, no -inf
    out['time_to_next'] = (log_ttn - stats['time_to_next']['mean']) / stats['time_to_next']['std']

    return out


def chunk_example(example, chunk_ticks=CHUNK_TICKS):
    n = len(example['grid'])
    n_chunks = int(np.ceil(n / chunk_ticks)) if n > 0 else 0

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_ticks
        end = min(start + chunk_ticks, n)
        n_valid = end - start

        chunk = {}
        for key, arr in example.items():
            piece = arr[..., start:end]
            if piece.shape[-1] < chunk_ticks:
                pad_width = [(0, 0)] * (piece.ndim - 1) + [(0, chunk_ticks - piece.shape[-1])]
                piece = np.pad(piece, pad_width, mode='constant')
            chunk[key] = piece

        mask = np.zeros(chunk_ticks, dtype=bool)
        mask[:n_valid] = True
        chunk['mask'] = mask
        chunk['chunk_index'] = i
        chunk['is_last_chunk'] = (i == n_chunks - 1)
        chunks.append(chunk)

    return chunks


# ORDER IS THE CONTRACT: generate.py builds X by hand and must mirror this
# list exactly — same features, same order, same normalization. Any edit
# here requires the matching edit there; nothing checks agreement at runtime.
INPUT_KEYS = ['target_dx', 'target_dy', 'time_to_next', 'is_active',
              'is_slider', 'is_spinner', 'cursor_x_norm', 'cursor_y_norm',
              'approach_progress']
Y_CURSOR_COLUMNS = ['cursor_dx', 'cursor_dy']
Y_KEY_COLUMNS = ['key_onset[A]', 'key_onset[B]', 'key_offset[A]', 'key_offset[B]']


def assemble_xy(chunk):
    X = np.stack([chunk[k].astype(np.float32) for k in INPUT_KEYS], axis=-1)

    y_cursor = np.stack([
        chunk['cursor_dx'].astype(np.float32),
        chunk['cursor_dy'].astype(np.float32),
    ], axis=-1)

    y_keys = np.concatenate([
        chunk['key_onset'].astype(np.float32).T,     # (2, T) -> (T, 2)
        chunk['key_offset'].astype(np.float32).T,    # (2, T) -> (T, 2)
    ], axis=-1)

    mask = chunk['mask'].astype(np.float32)

    return X, y_cursor, y_keys, mask


def build_key_weight(chunk, pos_weight=30.0, tol_ticks=2, tol_weight=0.0):
    onset = np.asarray(chunk['key_onset'], dtype=bool)  # (2, T)
    offset = np.asarray(chunk['key_offset'], dtype=bool)  # (2, T)
    event = onset.any(axis=0) | offset.any(axis=0)  # (T,)
    T = event.shape[0]

    w = np.ones(T, dtype=np.float32)

    if tol_ticks > 0 and event.any():
        near = np.zeros(T, dtype=bool)
        idx = np.flatnonzero(event)
        for k in range(1, tol_ticks + 1):
            lo = np.clip(idx - k, 0, T - 1)
            hi = np.clip(idx + k, 0, T - 1)
            near[lo] = True
            near[hi] = True
        near &= ~event  # events keep their own weight
        w[near] = tol_weight

    w[event] = pos_weight
    return w * chunk['mask'].astype(np.float32)


def measure_key_positive_rate(raw_examples):
    """Fraction of ticks carrying any key event, over an iterable of raw
    examples (generators fine — train.py streams one at a time).
    Feeds train()'s data-derived pos_weight."""
    events = total = 0
    for ex in raw_examples:
        onset = np.asarray(ex['key_onset'], dtype=bool)
        offset = np.asarray(ex['key_offset'], dtype=bool)
        ev = onset.any(axis=0) | offset.any(axis=0)
        events += int(ev.sum())
        total += ev.shape[0]
    return events / max(total, 1)