import csv
import numpy as np
from tqdm import tqdm
from config import FEATURES_DIR, load_norm_stats, MANIFEST_PATH
from training_data import (perturb_example, normalize_example, chunk_example,
                           assemble_xy, measure_key_positive_rate, build_key_weight)
from build_model import build_model, compile_model


EPOCHS = 25
NOISE_STD_PX = 10.0
EPOCH_REPLAYS = 400
VAL_REPLAYS = 100
POS_RATE_SAMPLE = 200

def reset_all_states(model):
    for layer in model.layers:
        if hasattr(layer, 'reset_states'):
            layer.reset_states()


def load_rows_for_split(split):
    """All manifest rows (as dicts) for a split. Rows are cheap (strings);
    the ARRAYS stay on disk until a row is actually visited."""
    with open(MANIFEST_PATH) as f:
        return [row for row in csv.DictReader(f) if row['split'] == split]


def load_example(row):
    """The lazy load: one replay's precomputed example off disk. ~1-2ms on NVMe.
    dict() materializes the NpzFile into real arrays so downstream copying
    (perturb_example's dict(example)) behaves normally."""
    return dict(np.load(row['npz_path']))


def replay_to_chunks(row, stats, noise_std_px=0.0, rng=None):
    """Load -> (perturb) -> normalize -> chunk for ONE replay.
    Replaces prepare_epoch_chunks: the unit of work is now a single replay,
    materialized only when the training loop reaches it."""
    example = load_example(row)
    if noise_std_px > 0:
        example = perturb_example(example, noise_std_px=noise_std_px, rng=rng)
    normalized = normalize_example(example, stats)
    return chunk_example(normalized)


def train_one_replay(model, chunks, pos_weight):
    replay_losses = []
    for chunk in chunks:
        X, y_cursor, y_keys, mask = assemble_xy(chunk)
        X = X[None, ...]
        y_cursor = y_cursor[None, ...]
        y_keys = y_keys[None, ...]
        mask = mask[None, ...]

        key_w = build_key_weight(chunk, pos_weight=pos_weight)[None, ...]
        losses = model.train_on_batch(X, [y_cursor, y_keys], sample_weight=[mask, key_w])
        replay_losses.append(losses)
    return replay_losses   # list of [total, cursor_loss, key_loss] per chunk


def evaluate_one_replay(model, chunks, pos_weight):
    replay_losses = []
    for chunk in chunks:
        X, y_cursor, y_keys, mask = assemble_xy(chunk)
        X = X[None, ...]
        y_cursor = y_cursor[None, ...]
        y_keys = y_keys[None, ...]
        mask = mask[None, ...]

        key_w = build_key_weight(chunk, pos_weight=pos_weight)[None, ...]
        losses = model.test_on_batch(X, [y_cursor, y_keys], sample_weight=[mask, key_w])
        replay_losses.append(losses)
    return replay_losses


def evaluate(model, all_replay_chunks, pos_weight):
    all_losses = []
    for chunks in all_replay_chunks:
        reset_all_states(model)
        all_losses.extend(evaluate_one_replay(model, chunks, pos_weight))
    all_losses = np.array(all_losses)
    return all_losses.mean(axis=0)   # [mean_total, mean_cursor, mean_key]


def estimate_pos_rate(train_rows, rng, n_sample=POS_RATE_SAMPLE):
    """Streaming estimate of the key-event positive rate: load a sample of
    train examples ONE AT A TIME (generator feeds measure_key_positive_rate,
    which only ever iterates — nothing is held in memory simultaneously)."""
    idx = rng.choice(len(train_rows), size=min(n_sample, len(train_rows)), replace=False)
    return measure_key_positive_rate(load_example(train_rows[i]) for i in idx)


def train(epochs=EPOCHS, patience=5, checkpoint_path='best_model.keras',
          noise_std_px=NOISE_STD_PX, epoch_replays=EPOCH_REPLAYS,
          log_path='training_log.txt'):
    stats = load_norm_stats()
    model = compile_model(build_model())
    rng = np.random.default_rng()          # perturbation + epoch sampling: fresh every run
    val_rng = np.random.default_rng(1337)  # val subset: SEEDED, same 100 replays every run

    train_rows = load_rows_for_split('train')
    test_rows = load_rows_for_split('test')
    print(f"manifest: {len(train_rows)} train / {len(test_rows)} test replays")

    pos_rate = estimate_pos_rate(train_rows, rng)
    pos_weight = min(0.5 / pos_rate, 60.0)   # half of full rebalance, capped
    print(f"key positive rate (sampled): {pos_rate:.4f} -> pos_weight {pos_weight:.1f}")

    # fixed val subset, prepared eagerly ONCE (clean: no perturbation, so chunks
    # are identical every epoch — same rationale as the old design, now sampled)
    val_idx = val_rng.choice(len(test_rows), size=min(VAL_REPLAYS, len(test_rows)), replace=False)
    test_chunks = [replay_to_chunks(test_rows[i], stats, noise_std_px=0.0) for i in val_idx]
    print(f"val subset: {len(test_chunks)} replays (fixed, seed 1337)\n")

    best_val_loss = np.inf
    best_weights = None
    epochs_without_improvement = 0

    log = open(log_path, 'a')
    log.write(f"\n--- new run: epochs={epochs} noise_std_px={noise_std_px} "
              f"pos_rate={pos_rate:.4f} pos_weight={pos_weight:.1f} "
              f"epoch_replays={epoch_replays} val_replays={len(test_chunks)} "
              f"train_pool={len(train_rows)} "
              f"offset_std_px=25.0 offset_decay=0.95 tol_ticks=2 lstm=(256,128,64) ---\n")

    for epoch in range(epochs):
        # sample WHICH replays this epoch sees (without replacement within the
        # epoch; fresh sample every epoch, so the model tours the whole pool
        # across epochs). Order is already random -> no separate shuffle needed.
        epoch_idx = rng.choice(len(train_rows), size=min(epoch_replays, len(train_rows)),
                               replace=False)

        epoch_losses = []
        bar = tqdm(epoch_idx, desc=f"epoch {epoch + 1}/{epochs}", unit="replay")
        for i in bar:
            chunks = replay_to_chunks(train_rows[i], stats, noise_std_px=noise_std_px, rng=rng)
            reset_all_states(model)
            epoch_losses.extend(train_one_replay(model, chunks, pos_weight))
            m = np.array(epoch_losses).mean(axis=0)
            bar.set_postfix(loss=f"{m[0]:.3f}", cursor=f"{m[1]:.3f}", key=f"{m[2]:.3f}")

        train_loss, train_cursor, train_key = np.array(epoch_losses).mean(axis=0)
        val_loss, val_cursor, val_key = evaluate(model, test_chunks, pos_weight)

        line = (f"epoch {epoch+1}/{epochs}  "
                f"loss={train_loss:.4f} (cursor={train_cursor:.4f} key={train_key:.4f})  "
                f"val_loss={val_loss:.4f} (cursor={val_cursor:.4f} key={val_key:.4f})")
        print(line)
        log.write(line + "\n")
        log.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            model.save(checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                line = f"early stopping: no val_loss improvement in {patience} epochs"
                print("\n" + line)
                log.write(line + "\n")
                break

    log.close()
    if best_weights is not None:
        model.set_weights(best_weights)
    return model


if __name__ == "__main__":
    train()