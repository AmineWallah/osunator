import datetime
from datetime import timedelta
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
GRID_TAIL_TICKS = 12 # Added to a grid so that it doesn't cut off 16ms earlier than the last hit object


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

    :param replay: osu! replay file
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


def cursor_at(times: np.ndarray, xs: np.ndarray, ys: np.ndarray, t: float) -> tuple[float, float]:
    """
    Linearly interpolate the cursor's position at a single query time t from the recorded frames; clamps to the first/last frame outside the recorded range.

    :param times: array of timestamps (in milliseconds), has to be increasing
    :param xs: array of x-coordinates
    :param ys: array of y-coordinates
    :param t: precise timestamp that we want the coordinates for (in milliseconds)
    :return: (x, y) coordinates of the cursor at t
    """
    if t <= times[0]:
        return xs[0], ys[0]
    if t >= times[-1]:
        return xs[-1], ys[-1]

    i = bisect.bisect_right(times, t) # Returns the index of the first element that is strictly greater to t
    t0, t1 = times[i-1], times[i] # Times at the left and right of the queried time

    # Calculating the coordinates of the cursor at t
    frac = (t - t0) / (t1 - t0)

    x = xs[i-1] + frac * (xs[i] - xs[i-1])
    y = ys[i-1] + frac * (ys[i] - ys[i-1])
    return x, y


def beat_length_at(beatmap: slider.beatmap.Beatmap, time_td: timedelta) -> float:
    """
    True beat length (ms per beat) in effect at a given time. (check osu wiki for timing point convention)

    :param time_td: query time as a timedelta
    :return: ms per beat (e.g. 500.0 = 120 BPM), always positive
    """
    tp = beatmap.timing_point_at(time_td)
    return tp.parent.ms_per_beat if tp.parent is not None else tp.ms_per_beat


def build_grid(beatmap: slider.beatmap.Beatmap) -> np.ndarray:
    """
    Builds a grid of timestamps (in milliseconds) that will be used to sample the replay data.
    :param beatmap: osu! beatmap object
    :return: Replay grid
    """
    first_obj = beatmap._hit_objects[0]
    last_obj = beatmap._hit_objects[-1]

    lead_in = beat_length_at(beatmap, first_obj.time)
    grid_start = max(0, to_ms(first_obj.time) - lead_in) # First object - lead in so that the grid starts at the first hit-object without including the typical empty space at the start of a map
    grid_end = to_ms(getattr(last_obj, 'end_time', last_obj.time)) # last_obj time so that the grid doesn't crop a slider/spinner object

    return np.arange(grid_start, grid_end + GRID_TAIL_TICKS * TICK_MS, TICK_MS) # GRID_TAIL_TICKS keeps the grid from ending 16ms earlier than the last hit-object


def resample_cursor(replay: osrparse.Replay , grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resamples cursor events from a replay to a 60 ticks per second grid, the grid has to be from the map that the replay refers to

    :param replay: osu! replay object
    :param grid: grid of timestamps (in milliseconds)
    :return: tuple of x and y coordinates of the cursor at each grid tick
    """
    events = convert_to_absolute(replay) # Gets the absolute times of a replay event
    # Time, x and y events respectively
    times = np.array([e[0] for e in events])
    xs = np.array([e[1] for e in events])
    ys = np.array([e[2] for e in events])

    # Join the time, x and y events with their respective grids
    x_grid = np.interp(grid, times, xs)
    y_grid = np.interp(grid, times, ys)
    return x_grid, y_grid


def resample_keys(replay: osrparse.Replay, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Resamples key events from a replay to a 60 ticks per second grid, the grid has to be from the map that the replay refers to

        :param replay: osu! replay object (frames carry a key bitmask:
        M1=1, M2=2, K1=4, K2=8)
        :param grid: grid of timestamps (in milliseconds)
        :return: tuple of held, onset, and offset states at each grid tick
    """
    events = convert_to_absolute(replay) # Gets the absolute times of a replay event
    # Times and key events respectively
    times = np.array([e[0] for e in events])
    keys = np.array([e[3] for e in events])

    SLOT_A = 1 | 4 # Mouse/Keyboard key 1
    SLOT_B = 2 | 8 # Mouse/Keyboard key 2

    idx = np.clip(np.searchsorted(times, grid, side='right') - 1, 0, len(times) - 1) # Re-assigns the index to a correct grid tick because an event can fall in between two
    sampled = keys[idx]
    held = np.stack([
        (sampled & SLOT_A) != 0,
        (sampled & SLOT_B) != 0,
    ]) # (2, n_ticks)

    # edge detection: prev = held shifted one tick (nothing held before t0)
    prev = np.concatenate([np.zeros((2, 1), dtype=bool), held[:, :-1]], axis=1)
    onset = held & ~prev # rising edge: press starts this tick
    offset = ~held & prev # falling edge: release this tick
    return held, onset, offset


def resample_map_features(beatmap: slider.beatmap.Beatmap, grid: np.ndarray) -> dict[str, np.ndarray]:
    """
    For a given map, resample it's features to match a 60 ticks per second grid
    :param beatmap: beatmap to be resampled
    :param grid: grid of the corresponding beatmap
    :return: dictionary of map features, each feature is a numpy array of length n_ticks
    """
    objects = beatmap._hit_objects
    n_ticks = len(grid)

    # Converts map object times from time-deltas to milliseconds
    starts = np.array([to_ms(o.time) for o in objects])
    ends = np.array([to_ms(getattr(o, 'end_time', o.time)) for o in objects])

    # Initialize resampled events to zeroes
    target_x = np.zeros(n_ticks)
    target_y = np.zeros(n_ticks)
    is_active = np.zeros(n_ticks, dtype=bool)
    is_slider = np.zeros(n_ticks, dtype=bool)
    is_spinner = np.zeros(n_ticks, dtype=bool)
    time_to_next = np.zeros(n_ticks)

    next_obj_idx = np.searchsorted(starts, grid, side='left') # per tick: index of the first object starting at-or-after it

    for i, obj in enumerate(objects):
        active_mask = (grid >= starts[i]) & (grid <= ends[i]) # ticks inside THIS object's own time span

        if active_mask.any(): # If at least one grid tick is within the object's time span
            is_active[active_mask] = True

            if isinstance(obj, Spinner): # Mark spinner as active for it's duration
                is_spinner[active_mask] = True
                target_x[active_mask] = obj.position.x
                target_y[active_mask] = obj.position.y

            elif isinstance(obj, Slider): # Mark slider as active for it's duration
                is_slider[active_mask] = True
                active_ticks = grid[active_mask]
                # Calculate slider ball positions at the active ticks
                ball_positions = [slider_position(obj, t)[1] for t in active_ticks]
                target_x[active_mask] = [p.x for p in ball_positions]
                target_y[active_mask] = [p.y for p in ball_positions]

            else:
                # Log circle's positions
                target_x[active_mask] = obj.position.x
                target_y[active_mask] = obj.position.y

    free_mask = ~is_active # free flight (no active object): aim target is the next object's entry point
    if free_mask.any():
        idx = np.clip(next_obj_idx[free_mask], 0, len(objects) - 1) # pin out-of-range tail indices (past-last-object ticks) to the last object
        next_objs = [objects[j] for j in idx]
        target_x[free_mask] = [o.position.x for o in next_objs]
        target_y[free_mask] = [o.position.y for o in next_objs]
        is_slider[free_mask] = [isinstance(o, Slider) for o in next_objs]

    clipped_next = np.clip(next_obj_idx, 0, len(objects) - 1)
    time_to_next = np.maximum(np.where(is_active, 0.0, starts[clipped_next] - grid), 0.0)

    # Calculates for how long an object is visible before its hit time
    ar = beatmap.approach_rate
    if ar < 5: # AR 5 is the baseline (code snippet straight from the osu! wiki)
        preempt = 1200 + 600 * (5 - ar) / 5
    else:
        preempt = 1200 - 750 * (ar - 5) / 5

    approach_progress = np.clip(1.0 - time_to_next / preempt, 0.0, 1.0) # Fraction of how close the approach circle is to the hit-object

    # Dict of map features
    return {
        'target_x': target_x,
        'target_y': target_y,
        'is_active': is_active,
        'is_slider': is_slider,
        'is_spinner': is_spinner,
        'time_to_next': time_to_next,
        'approach_progress': approach_progress,
    }


def build_training_example(beatmap: slider.beatmap.Beatmap, replay: osrparse.Replay) -> dict[str, np.ndarray]:
    """
    Builds training example for a given map and replay.
    :param beatmap: osu! beatmap object
    :param replay: osu! replay object of the map in question
    :return: dictionary of training example features
    """
    # Grid building
    grid = build_grid(beatmap)
    x_grid, y_grid = resample_cursor(replay, grid)
    held, onset, offset = resample_keys(replay, grid)

    # Getting map features
    map_feats = resample_map_features(beatmap, grid)

    # INPUTS: distances from the cursor to the hit-object
    target_dx = map_feats['target_x'] - x_grid
    target_dy = map_feats['target_y'] - y_grid

    # LABELS: distance from a cursor position to the next one on the grid
    cursor_dx = np.diff(x_grid, prepend=x_grid[0])
    cursor_dy = np.diff(y_grid, prepend=y_grid[0])

    # approach progress of a hit-circle to the hit-object (0 = not yet visible, 1 = hit moment)
    approach_progress = map_feats['approach_progress']

    # returns training example
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