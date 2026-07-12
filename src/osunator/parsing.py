import datetime
from pathlib import Path

import osrparse
import slider
import bisect
import numpy as np
from slider import Position
from tqdm import tqdm
from osrparse import Replay
from collections.abc import Iterator
from config import load_cache, OSU_DIR, SUITABLE_DIR
from slider.beatmap import Circle, Slider, Spinner

CORRUPT_CAP = 500 # Tolerance gap for negative time-deltas, used to evaluate whether a replay is corrupt (went back in time) or not
TICK_MS = 1000 / 60 # We are going for a 60 ticks per second approach, so we divide 1000ms (1 second) by the number of ticks (60)
CHUNK_SECONDS = 10
CHUNK_TICKS = int(CHUNK_SECONDS * 1000 / TICK_MS) # for claude: should we move this variable somewhere else? it's not used in parsing.py
LEAD_IN_POS = (256.0, -500.0) # osu replays usually carry this frame as a placeholder of some sort, we store it in a constant for future checks


def to_ms(td: datetime.timedelta) -> float:
    """
    Converts a given time-delta to milliseconds

    :param td: time-delta
    :return: same duration in milliseconds
    """
    return td.total_seconds() * 1000


def slider_position(slider_obj: Slider, timestamp: float) -> tuple[float, Position]:
    """
    Calculates the position of a slider ball at a given timestamp, makes it easy for the model to process sliders

    :param slider_obj: the slider hit-object in question
    :param timestamp: when the ball is supposed to be at this position (milliseconds), must lie within [slider_start, slider_end]
    :return: timestamp and position of the ball
    """
    # Start and end times in milliseconds for the slider object
    start = to_ms(slider_obj.time)
    end = to_ms(slider_obj.end_time)

    total = end - start

    global_progress = (timestamp - start) / total # Fraction of the slider's elapsed time span (0 at start, 1 at end including repeats)
    repeat = slider_obj.repeat # single pass = 1, multiple passes >= 2

    scaled = global_progress * repeat
    pass_index = int(scaled) # Which pass is the slider ball currently at
    local_progress = scaled - pass_index # Fraction of the current pass (0 at start, 1 at end)

    if pass_index >= repeat: # Handles global_progress == 1.0, it would index a pass that doesn't exist
        pass_index = repeat - 1
        local_progress = 1.0

    if pass_index % 2 == 1: # Handles back-and-forth passes
        local_progress = 1.0 - local_progress

    pos = slider_obj.curve(local_progress) # position of the slider ball accounting for its curve
    return timestamp, pos


def convert_to_absolute(replay: osrparse.Replay) -> list[tuple[float, float, float, int]]:
    """
    Parses osu! replay events and returns a list of (timestamp, x, y, keys) tuples (with the timestamp in milliseconds instead of time deltas)

    Note: output times are strictly increasing, if time == 0 -> song start

    :param replay: .osr file in question
    :return: list of replay events in absolute time
    :raises ValueError: mid-replay backward time jump > CORRUPT_CAP
    """
    events = [] # Accumulates time events
    abs_time = 0 # Accumulates time in milliseconds
    prev = None # If none -> first frame, otherwise -> last frame's time

    for ev in replay.replay_data:
        abs_time += ev.time_delta

        if abs_time < 0: # Skips accumulation of negative time deltas at the start (from pre-song lead in data, present in every replay)
            continue
        if not (-200 <= ev.x <= 712 and -200 <= ev.y <= 584): # Drops single frame garbage coordinates (102 replay audit, ±10-17k px spikes, > half of label variance). bounds = reachable screen + slack
            continue
        if prev is None and (ev.x, ev.y) == LEAD_IN_POS: # Skips placeholder first frame
            continue
        if prev is not None and abs_time < prev: # Guard for replays that go backwards in time
            if prev - abs_time > CORRUPT_CAP: # If the jitter is too strong then it's most likely corruption, raise an error
                raise ValueError(
                    f"time went backwards {prev - abs_time}ms mid-replay "
                    f"(prev={prev}, now={abs_time}) — probably corrupt")
            abs_time = prev + 1 # Adds one incase the replay went back in time by a very small amount (yes it happens)

        prev = abs_time
        events.append((abs_time, ev.x, ev.y, ev.keys))
    return events


def beatmap_replay_pairs(paths: list[Path]) -> Iterator[tuple[slider.beatmap.Beatmap, str, Replay, Path]]:
    """
    Pairs replay paths with their respective beatmap, beatmap_id and replay files. Paths whose hash is not in the cache are skipped.

    Lazy loads beatmap and replay objects, so it's not necessary to load them all at once.
    :param paths: list of paths, each path pointing to an .osr file
    :return: generator of (beatmap, beatmap_id, replay, path) tuples
    """
    cache = load_cache()

    parsed_maps = {}
    for path in paths:
        replay = Replay.from_path(str(path)) # Grab replay object from replay path
        try:
            beatmap_id = cache['hash_index'][replay.beatmap_hash] # Grab beatmap_id of corresponding beatmap hash from replay
        except KeyError: # Skip unresolved replays
            continue
        if beatmap_id not in parsed_maps:
            try:
                parsed_maps[beatmap_id] = slider.beatmap.Beatmap.from_path(str(OSU_DIR / f"{beatmap_id}.osu")) # Store beatmap object in dictionary
            except FileNotFoundError:
                parsed_maps[beatmap_id] = None # Set beatmap as None incase of missing file
        beatmap = parsed_maps[beatmap_id]
        if beatmap is None: # Skip unresolved beatmaps
            continue
        yield (beatmap, beatmap_id, replay, path) # Using yield for lazy-loading


def cursor_at(times, xs, ys, t):
    if t <= times[0]:
        return xs[0], ys[0]
    if t >= times[-1]:
        return xs[-1], ys[-1]
    i = bisect.bisect_right(times, t)
    t0, t1 = times[i-1], times[i]
    frac = (t - t0) / (t1 - t0)
    x = xs[i-1] + frac * (xs[i] - xs[i-1])
    y = ys[i-1] + frac * (ys[i] - ys[i-1])
    return x, y


def beat_length_at(beatmap, time_td):
    tp = beatmap.timing_point_at(time_td)
    return tp.parent.ms_per_beat if tp.parent is not None else tp.ms_per_beat


def build_grid(beatmap):
    first_obj = beatmap._hit_objects[0]
    last_obj = beatmap._hit_objects[-1]

    lead_in = beat_length_at(beatmap, first_obj.time)
    grid_start = max(0, to_ms(first_obj.time) - lead_in)
    grid_end = to_ms(getattr(last_obj, 'end_time', last_obj.time))

    GRID_TAIL_TICKS = 12
    return np.arange(grid_start, grid_end + GRID_TAIL_TICKS * TICK_MS, TICK_MS)


def resample_cursor(replay, grid):
    events = convert_to_absolute(replay)
    times = np.array([e[0] for e in events])
    xs = np.array([e[1] for e in events])
    ys = np.array([e[2] for e in events])

    x_grid = np.interp(grid, times, xs)
    y_grid = np.interp(grid, times, ys)
    return x_grid, y_grid


def resample_keys(replay, grid):
    events = convert_to_absolute(replay)
    times = np.array([e[0] for e in events])
    keys = np.array([e[3] for e in events])

    SLOT_A = 1 | 4
    SLOT_B = 2 | 8

    idx = np.clip(np.searchsorted(times, grid, side='right') - 1, 0, len(times) - 1)
    sampled = keys[idx]
    held = np.stack([
        (sampled & SLOT_A) != 0,
        (sampled & SLOT_B) != 0,
    ])

    prev = np.concatenate([np.zeros((2, 1), dtype=bool), held[:, :-1]], axis=1)
    onset = held & ~prev
    offset = ~held & prev
    return held, onset, offset


def resample_map_features(beatmap, grid):
    objects = beatmap._hit_objects
    n_ticks = len(grid)

    starts = np.array([to_ms(o.time) for o in objects])
    ends = np.array([to_ms(getattr(o, 'end_time', o.time)) for o in objects])

    target_x = np.zeros(n_ticks)
    target_y = np.zeros(n_ticks)
    is_active = np.zeros(n_ticks, dtype=bool)
    is_slider = np.zeros(n_ticks, dtype=bool)
    is_spinner = np.zeros(n_ticks, dtype=bool)
    time_to_next = np.zeros(n_ticks)

    next_obj_idx = np.searchsorted(starts, grid, side='left')

    for i, obj in enumerate(objects):
        active_mask = (grid >= starts[i]) & (grid <= ends[i])

        if active_mask.any():
            is_active[active_mask] = True

            if isinstance(obj, Spinner):
                is_spinner[active_mask] = True
                target_x[active_mask] = obj.position.x
                target_y[active_mask] = obj.position.y

            elif isinstance(obj, Slider):
                is_slider[active_mask] = True
                active_ticks = grid[active_mask]
                ball_positions = [slider_position(obj, t)[1] for t in active_ticks]
                target_x[active_mask] = [p.x for p in ball_positions]
                target_y[active_mask] = [p.y for p in ball_positions]

            else:
                target_x[active_mask] = obj.position.x
                target_y[active_mask] = obj.position.y

    free_mask = ~is_active
    if free_mask.any():
        idx = np.clip(next_obj_idx[free_mask], 0, len(objects) - 1)
        next_objs = [objects[j] for j in idx]
        target_x[free_mask] = [o.position.x for o in next_objs]
        target_y[free_mask] = [o.position.y for o in next_objs]
        is_slider[free_mask] = [isinstance(o, Slider) for o in next_objs]

    clipped_next = np.clip(next_obj_idx, 0, len(objects) - 1)
    time_to_next = np.maximum(np.where(is_active, 0.0, starts[clipped_next] - grid), 0.0)

    ar = beatmap.approach_rate
    if ar < 5:
        preempt = 1200 + 600 * (5 - ar) / 5
    else:
        preempt = 1200 - 750 * (ar - 5) / 5

    approach_progress = np.clip(1.0 - time_to_next / preempt, 0.0, 1.0)

    return {
        'target_x': target_x,
        'target_y': target_y,
        'is_active': is_active,
        'is_slider': is_slider,
        'is_spinner': is_spinner,
        'time_to_next': time_to_next,
        'approach_progress': approach_progress,
    }


def build_training_example(beatmap, replay):
    grid = build_grid(beatmap)

    x_grid, y_grid = resample_cursor(replay, grid)
    held, onset, offset = resample_keys(replay, grid)
    map_feats = resample_map_features(beatmap, grid)

    target_dx = map_feats['target_x'] - x_grid
    target_dy = map_feats['target_y'] - y_grid

    cursor_dx = np.diff(x_grid, prepend=x_grid[0])
    cursor_dy = np.diff(y_grid, prepend=y_grid[0])

    approach_progress = map_feats['approach_progress']

    return {
        'grid': grid,
        'cursor_x': x_grid, 'cursor_y': y_grid,
        'cursor_dx': cursor_dx, 'cursor_dy': cursor_dy,
        'target_dx': target_dx, 'target_dy': target_dy,
        'time_to_next': map_feats['time_to_next'],
        'is_active': map_feats['is_active'],
        'is_slider': map_feats['is_slider'],
        'is_spinner': map_feats['is_spinner'],
        'key_held': held, 'key_onset': onset, 'key_offset': offset,
        'approach_progress': approach_progress,
    }


def main():
    paths = list(SUITABLE_DIR.rglob('*.osr'))
    beatmap, replay, path = beatmap_replay_pairs(paths)[0]

    ex = build_training_example(beatmap, replay)

    idx = np.where(ex['is_active'] & ~ex['is_slider'] & ~ex['is_spinner'])[0][0]

    reconstructed_x = ex['cursor_x'][idx] + ex['target_dx'][idx]
    reconstructed_y = ex['cursor_y'][idx] + ex['target_dy'][idx]

    print(f"reconstructed: ({reconstructed_x:.2f}, {reconstructed_y:.2f})")

    map_feats = resample_map_features(beatmap, ex['grid'])
    print(f"target_x/y:    ({map_feats['target_x'][idx]:.2f}, {map_feats['target_y'][idx]:.2f})")


if __name__ == "__main__":
    main()