import numpy as np
from parsing import CHUNK_TICKS


def perturb_example(example, noise_std_px=10.0, offset_std_px=25.0, offset_decay=0.95,
                    rng=None):
    """DART-style perturbation with a proportional-controller expert.

    Two displacement sources on top of the true path:
      1. per-tick gaussian noise (as always),
      2. a SMOOTH schedule offset following an Ornstein-Uhlenbeck process:
             o[t+1] = offset_decay * o[t] + w[t],   w ~ N(0, sigma_w)
         with sigma_w chosen so the stationary std of o is offset_std_px.
         The offset wanders continuously (no segments, no jumps) and always
         reverts toward the true path at rate (1 - offset_decay) per tick.

    The perturbed base path is x_true + o. Labels point one tick ahead ALONG
    THE BASE PATH: label[t] = base[t+1] - pert[t]
                            = true_delta + (decay-1)*o[t] + w[t] - noise[t].
    The corrective policy taught is one coherent rule: every tick, close a
    fixed fraction (1-decay) of your current schedule error, plus normal
    motion. 50px behind at decay=0.95 -> +2.5px/tick of catch-up. The same
    observation (behind-ness) always maps to the same answer (proportional
    correction) — no phases, no hidden state, and because the offset is
    continuous there are no teleport labels anywhere, boundaries included.
    """
    if rng is None:
        rng = np.random.default_rng()

    out = dict(example)

    x_true = example['cursor_x']
    y_true = example['cursor_y']
    n = len(x_true)

    # --- smooth OU offset track ---
    off_x = np.zeros(n)
    off_y = np.zeros(n)
    if offset_std_px > 0 and 0 <= offset_decay < 1:
        sigma_w = offset_std_px * np.sqrt(1.0 - offset_decay ** 2)
        wx = rng.normal(0.0, sigma_w, size=n)
        wy = rng.normal(0.0, sigma_w, size=n)
        # start at stationary distribution so early ticks aren't special
        ox = rng.normal(0.0, offset_std_px)
        oy = rng.normal(0.0, offset_std_px)
        for t in range(n):
            off_x[t] = ox
            off_y[t] = oy
            ox = offset_decay * ox + wx[t]
            oy = offset_decay * oy + wy[t]

    # base = true path displaced by the smoothly wandering offset
    x_base = x_true + off_x
    y_base = y_true + off_y

    # --- gaussian noise on top of the base ---
    x_pert = x_base + rng.normal(0.0, noise_std_px, size=n)
    y_pert = y_base + rng.normal(0.0, noise_std_px, size=n)

    # inputs: shift target vectors so they now point FROM the perturbed position
    out['target_dx'] = example['target_dx'] + (x_true - x_pert)
    out['target_dy'] = example['target_dy'] + (y_true - y_pert)

    # labels: one tick ahead along the base path, from the perturbed position.
    # = true motion + proportional offset correction + noise correction.
    label_dx = np.empty_like(x_true)
    label_dx[:-1] = x_base[1:] - x_pert[:-1]
    label_dx[-1] = x_true[-1] - x_pert[-1]  # last tick: pure correction to true
    label_dy = np.empty_like(y_true)
    label_dy[:-1] = y_base[1:] - y_pert[:-1]
    label_dy[-1] = y_true[-1] - y_pert[-1]
    out['cursor_dx'] = label_dx
    out['cursor_dy'] = label_dy

    # keep the perturbed positions visible for debugging/plotting
    out['cursor_x'] = x_pert
    out['cursor_y'] = y_pert
    out['offset_mag'] = np.hypot(off_x, off_y)  # debug

    return out



def normalize_example(example, stats):
    out = dict(example)   # shallow copy — only the transformed keys get replaced

    out['target_dx'] = example['target_dx'] / stats['target_dx']['scale']
    out['target_dy'] = example['target_dy'] / stats['target_dy']['scale']

    out['cursor_dx'] = (example['cursor_dx'] - stats['cursor_dx']['mean']) / stats['cursor_dx']['std']
    out['cursor_dy'] = (example['cursor_dy'] - stats['cursor_dy']['mean']) / stats['cursor_dy']['std']

    out['cursor_x_norm'] = example['cursor_x'] / 512.0
    out['cursor_y_norm'] = example['cursor_y'] / 384.0

    log_ttn = np.log1p(example['time_to_next'])
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


INPUT_KEYS = ['target_dx', 'target_dy', 'time_to_next', 'is_active', 'is_slider', 'is_spinner', 'cursor_x_norm', 'cursor_y_norm', 'approach_progress']
Y_CURSOR_COLUMNS = ['cursor_dx', 'cursor_dy']
Y_KEY_COLUMNS = ['key_onset[A]', 'key_onset[B]', 'key_offset[A]', 'key_offset[B]']


def assemble_xy(chunk):
    X = np.stack([chunk[k].astype(np.float32) for k in INPUT_KEYS], axis=-1)

    y_cursor = np.stack([
        chunk['cursor_dx'].astype(np.float32),
        chunk['cursor_dy'].astype(np.float32),
    ], axis=-1)

    y_keys = np.concatenate([
        chunk['key_onset'].astype(np.float32).T,     # (2, n) -> (n, 2)
        chunk['key_offset'].astype(np.float32).T,    # (2, n) -> (n, 2)
    ], axis=-1)

    mask = chunk['mask'].astype(np.float32)

    return X, y_cursor, y_keys, mask


def build_key_weight(chunk, pos_weight=30.0, tol_ticks=2, tol_weight=0.0):
    """Per-tick sample_weight for the key head: weighted BCE + tolerance window.

    Three tick classes (priority order — event beats tolerance beats background):
      EVENT ticks      (any of the 4 onset/offset labels is 1): weight = pos_weight.
                       Missing a real press is now pos_weight times worse than a
                       background mistake — this is what breaks the all-zeros optimum.
      TOLERANCE ticks  (within +-tol_ticks of an event, label 0): weight = tol_weight.
                       A prediction 1-2 ticks off the human's exact tick is human-level
                       timing jitter, not an error; tol_weight=0 means "not graded",
                       so the model isn't punished twice (miss at t + false-pos at t+-1)
                       for a press that's merely slightly early/late.
      BACKGROUND ticks: weight = 1 (false presses in empty space stay fully punished).

    Multiplied by the padding mask, so it REPLACES the mask in the key slot of
    sample_weight (do NOT also multiply the mask in train_model).
    Keras note: sample_weight is per-(batch, time) — one weight shared across the
    4 output columns; per-column weighting isn't expressible this way. Good enough:
    key events on different columns rarely coincide within a tolerance radius.
    """
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
    """Fraction of ticks carrying any key event, over a list of raw examples.
    Used by train() to derive pos_weight from data instead of a guess."""
    events = total = 0
    for ex in raw_examples:
        onset = np.asarray(ex['key_onset'], dtype=bool)
        offset = np.asarray(ex['key_offset'], dtype=bool)
        ev = onset.any(axis=0) | offset.any(axis=0)
        events += int(ev.sum())
        total += ev.shape[0]
    return events / max(total, 1)
