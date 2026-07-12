from tqdm import tqdm
from osrparse import Replay
import slider
import bisect
from config import load_cache, OSU_DIR, SUITABLE_DIR
import numpy as np
from slider.beatmap import Circle, Slider, Spinner

CORRUPT_CAP = 500      # ms; backward jump bigger than this in convert_to_absolute = corrupt replay, not jitter
TICK_MS = 1000 / 60    # 60Hz resample grid spacing (~16.667ms)
CHUNK_SECONDS = 10
CHUNK_TICKS = int(CHUNK_SECONDS * 1000 / TICK_MS)
LEAD_IN_POS = (256.0, -500.0)   # osu's fixed pre-song placeholder cursor position


def to_ms(td):
    """Convert a datetime.timedelta (slider's native time unit) to a float, milliseconds.
    Pipeline invariant: everything downstream of this is int/float ms, never timedelta."""
    return td.total_seconds() * 1000


def slider_position(slider_obj, timestamp):
    """Given a slider.beatmap.Slider object and an absolute-ms timestamp within its
    active span, return (timestamp, Position) of the slider ball at that instant.
    Handles repeats: even passes run head->tail, odd passes run tail->head.
    timestamp must already be in the same abs-ms clock as convert_to_absolute() output."""
    start = to_ms(slider_obj.time)
    end = to_ms(slider_obj.end_time)
    total = end - start
    frac = (timestamp - start) / total      # 0..1 over the WHOLE slider (all repeats)
    repeat = slider_obj.repeat

    scaled = frac * repeat          # e.g. 0..3 for repeat=3
    pass_index = int(scaled)        # which pass: 0,1,2,...
    local = scaled - pass_index     # how far along THIS pass: 0..1

    # clamp the exact-endpoint case (frac == 1 makes pass_index == repeat)
    if pass_index >= repeat:
        pass_index = repeat - 1
        local = 1.0

    if pass_index % 2 == 1:         # odd passes run backward
        local = 1.0 - local

    pos = slider_obj.curve(local)
    return (timestamp, pos)


def convert_to_absolute(replay):
    """Turn a Replay's relative time_delta frames into a list of
    (abs_time_ms, x, y, keys) tuples on one absolute-ms clock (song start = 0).

    - Skips lead-in frames (abs_time < 0): these are pre-song placeholder frames
      (e.g. cursor parked at (256,-500)), not real cursor data.
    - Tolerates small backward jitter (<= CORRUPT_CAP ms) from osu's integer-ms
      rounding by clamping forward (prev + 1), keeping the time axis strictly
      increasing (required by bisect/np.interp downstream).
    - Raises ValueError if a backward jump exceeds CORRUPT_CAP: that's not jitter,
      it's a broken/corrupt replay file.
    """
    events = []
    abs_time = 0
    prev = None
    for ev in replay.replay_data:
        abs_time += ev.time_delta
        if abs_time < 0:
            continue
        if not (-200 <= ev.x <= 712 and -200 <= ev.y <= 584):
            continue
        if prev is None and (ev.x, ev.y) == LEAD_IN_POS:
            continue   # still pre-song placeholder even though time is >= 0
        if prev is not None and abs_time < prev:
            if prev - abs_time > CORRUPT_CAP:
                raise ValueError(
                    f"time went backwards {prev - abs_time}ms mid-replay "
                    f"(prev={prev}, now={abs_time}) — probably corrupt")
            abs_time = prev + 1
        prev = abs_time
        events.append((abs_time, ev.x, ev.y, ev.keys))
    return events



def beatmap_replay_pairs(paths):
    cache = load_cache()
    parsed_maps = {}
    for path in paths:
        replay = Replay.from_path(str(path))
        try:
            beatmap_id = cache['hash_index'][replay.beatmap_hash]
        except KeyError:
            continue
        if beatmap_id not in parsed_maps:
            try:
                parsed_maps[beatmap_id] = slider.beatmap.Beatmap.from_path(str(OSU_DIR / f"{beatmap_id}.osu"))
            except FileNotFoundError:
                parsed_maps[beatmap_id] = None
        beatmap = parsed_maps[beatmap_id]
        if beatmap is None:
            continue
        yield (beatmap, beatmap_id, replay, path)


def cursor_at(times, xs, ys, t):
    """Scalar/diagnostic helper: linearly interpolate the cursor's (x, y) position
    at a single query time t, given the recorded (times, xs, ys) columns from
    convert_to_absolute(). Clamps to the first/last frame if t is out of range.

    times must be strictly increasing (guaranteed by convert_to_absolute's clamp).
    For resampling a whole grid of ticks at once, use np.interp directly instead —
    this function re-does the bisect search per call, which is fine for one-off
    queries (e.g. the alignment diagnostic) but wasteful in a per-tick loop."""
    if t <= times[0]:
        return xs[0], ys[0]        # query before first frame -> clamp
    if t >= times[-1]:
        return xs[-1], ys[-1]      # query after last frame  -> clamp
    i = bisect.bisect_right(times, t)   # times[i-1] <= t < times[i]
    t0, t1 = times[i-1], times[i]
    frac = (t - t0) / (t1 - t0)
    x = xs[i-1] + frac * (xs[i] - xs[i-1])
    y = ys[i-1] + frac * (ys[i] - ys[i-1])
    return x, y


def beat_length_at(beatmap, time_td):
    """Return the true beat length (ms per beat) in effect at a given time (timedelta).
    Inherited ("green") timing points store a negative ms_per_beat (their SV
    multiplier, not a real duration) — in that case the real beat length lives on
    tp.parent (the governing uninherited/"red" line). Uninherited points have
    parent=None and ms_per_beat is already the real beat length."""
    tp = beatmap.timing_point_at(time_td)
    return tp.parent.ms_per_beat if tp.parent is not None else tp.ms_per_beat


def build_grid(beatmap):
    """Build the fixed 60Hz tick grid (in absolute ms) that one training example
    spans, for a given parsed beatmap.

    Span decision:
    - start = first hit-object's time, minus one full beat length of lead-in
      (so the model sees the cursor approach the first object, not teleport onto
      it), clamped to 0 so maps with an early first object don't get a negative
      grid start.
    - end = last hit-object's end_time (covers the full tail of a final slider/
      spinner). Circles have no end_time attribute distinct from .time, so we
      fall back to .time for them via getattr.

    Deliberately does NOT extend past the last object into post-map cursor drift
    (players walking the cursor to a results screen, idling, etc.) — there's no
    map-side signal to condition that motion on, so it doesn't belong in the core
    aim-cloning dataset. Revisit separately later if idle/post-map behavior is
    ever wanted.

    Returns a 1D np.ndarray of tick times in absolute ms, spaced TICK_MS apart.
    """
    first_obj = beatmap._hit_objects[0]
    last_obj = beatmap._hit_objects[-1]

    lead_in = beat_length_at(beatmap, first_obj.time)
    grid_start = max(0, to_ms(first_obj.time) - lead_in)
    grid_end = to_ms(getattr(last_obj, 'end_time', last_obj.time))


    GRID_TAIL_TICKS = 12 # cus np.arrange range is exclusive, we add this to avoid the miss on every last hit object
    return np.arange(grid_start, grid_end + GRID_TAIL_TICKS * TICK_MS, TICK_MS)

def resample_cursor(replay, grid):
    """Resample a replay's cursor path onto a fixed tick grid via linear
    interpolation. grid is whatever build_grid(beatmap) returned — the tick
    times this replay's cursor gets evaluated at.

    Returns (x_grid, y_grid): parallel np.ndarrays, same length as grid,
    cursor position at each tick. Out-of-range ticks clamp to the nearest
    real frame (np.interp's default behavior).
    """
    events = convert_to_absolute(replay)
    times = np.array([e[0] for e in events])
    xs = np.array([e[1] for e in events])
    ys = np.array([e[2] for e in events])

    x_grid = np.interp(grid, times, xs)
    y_grid = np.interp(grid, times, ys)
    return x_grid, y_grid

def resample_keys(replay, grid):
    """Resample key state onto the tick grid via zero-order hold (step
    function — keys are categorical, never interpolated).

    M1 and K1 are merged into one 'slot A' channel, M2 and K2 into 'slot B' —
    they're the same logical tap action on different input devices (mouse vs
    keyboard), and which device a player uses doesn't make a replay more or
    less human. Keeping 2 channels (not collapsing to 1) preserves real
    overlap/gap timing between alternating taps, which IS a human signal.

    Returns:
        held:   bool array, shape (2, len(grid)) — [slot_a, slot_b] held state per tick
        onset:  bool array, same shape — 0->1 edges (press starts)
        offset: bool array, same shape — 1->0 edges (release starts)
    """
    events = convert_to_absolute(replay)
    times = np.array([e[0] for e in events])
    keys = np.array([e[3] for e in events])

    SLOT_A = 1 | 4   # M1 | K1
    SLOT_B = 2 | 8   # M2 | K2

    idx = np.clip(np.searchsorted(times, grid, side='right') - 1, 0, len(times) - 1)
    sampled = keys[idx]
    held = np.stack([
        (sampled & SLOT_A) != 0,
        (sampled & SLOT_B) != 0,
    ])   # shape (2, n_ticks)

    prev = np.concatenate([np.zeros((2, 1), dtype=bool), held[:, :-1]], axis=1)
    onset = held & ~prev
    offset = ~held & prev
    return held, onset, offset

def resample_map_features(beatmap, grid):
    """Resample the MAP side of a training example onto the tick grid.

    Returns a dict of parallel np.ndarrays, one entry per grid tick:
        target_x, target_y : absolute position of the current aim target
        is_active           : True if a hit object's own span [time, end_time]
                               currently contains this tick (circle instant,
                               or slider/spinner mid-span)
        is_slider           : True if the active/next target is a slider
        is_spinner          : True if this tick falls inside a spinner's span
                               (mask/exclude from aim learning downstream)
        time_to_next        : ms until the next hit object's start time
                               (0 if currently active on one)
    """
    objects = beatmap._hit_objects
    n_ticks = len(grid)

    # Precompute per-object absolute start/end times once (not per tick).
    starts = np.array([to_ms(o.time) for o in objects])
    ends = np.array([to_ms(getattr(o, 'end_time', o.time)) for o in objects])

    target_x = np.zeros(n_ticks)
    target_y = np.zeros(n_ticks)
    is_active = np.zeros(n_ticks, dtype=bool)
    is_slider = np.zeros(n_ticks, dtype=bool)
    is_spinner = np.zeros(n_ticks, dtype=bool)
    time_to_next = np.zeros(n_ticks)

    # next_obj_idx[i] = index of the first object whose start time is >= grid[i]
    # i.e. "the next object to happen, at or after this tick" — used both to
    # find the free-flight aim target and to compute time_to_next.
    next_obj_idx = np.searchsorted(starts, grid, side='left')

    for i, obj in enumerate(objects):
        # ticks where THIS object is the current active one (its own span)
        active_mask = (grid >= starts[i]) & (grid <= ends[i])

        if active_mask.any():
            is_active[active_mask] = True

            if isinstance(obj, Spinner):
                is_spinner[active_mask] = True
                # spinner has no meaningful aim target; leave target_x/y at
                # whatever free-flight logic below assigns (it'll be masked
                # out downstream anyway via is_spinner)
                target_x[active_mask] = obj.position.x
                target_y[active_mask] = obj.position.y

            elif isinstance(obj, Slider):
                is_slider[active_mask] = True
                active_ticks = grid[active_mask]
                # slider_position is scalar; call once per active tick (a
                # slider's active span is short, tens of ticks at most)
                ball_positions = [slider_position(obj, t)[1] for t in active_ticks]
                target_x[active_mask] = [p.x for p in ball_positions]
                target_y[active_mask] = [p.y for p in ball_positions]

            else:  # Circle
                target_x[active_mask] = obj.position.x
                target_y[active_mask] = obj.position.y

    # Free-flight ticks: not currently active on any object. Aim target is
    # the NEXT object's entry point (its head, whether circle or slider).
    free_mask = ~is_active
    if free_mask.any():
        idx = np.clip(next_obj_idx[free_mask], 0, len(objects) - 1)
        next_objs = [objects[j] for j in idx]
        target_x[free_mask] = [o.position.x for o in next_objs]
        target_y[free_mask] = [o.position.y for o in next_objs]
        is_slider[free_mask] = [isinstance(o, Slider) for o in next_objs]

    # time_to_next: ms until the next object's start (0 while active on one)
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
    """Join cursor + keys + map features into one training example.
    target_dx/dy = INPUT (cursor -> map target). cursor_dx/dy = LABEL (tick-to-tick movement)."""
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
    beatmap,replay,path = beatmap_replay_pairs(paths)[0]

    ex = build_training_example(beatmap, replay)

    # find any tick that's actively on a circle (not slider/spinner)
    idx = np.where(ex['is_active'] & ~ex['is_slider'] & ~ex['is_spinner'])[0][0]

    reconstructed_x = ex['cursor_x'][idx] + ex['target_dx'][idx]
    reconstructed_y = ex['cursor_y'][idx] + ex['target_dy'][idx]

    print(f"reconstructed: ({reconstructed_x:.2f}, {reconstructed_y:.2f})")

    # cross-check against the actual object position at that tick, independently
    map_feats = resample_map_features(beatmap, ex['grid'])
    print(f"target_x/y:    ({map_feats['target_x'][idx]:.2f}, {map_feats['target_y'][idx]:.2f})")





if __name__ == "__main__":
    main()