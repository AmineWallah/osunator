import numpy as np
import slider
from slider.beatmap import Slider, Spinner
from osunator.parsing import build_grid, to_ms

BEATMAP = "/home/amine/.local/share/osu-wine/osu!/Songs/219380 Konuko - Toumei Elegy/Konuko - Toumei Elegy (Awaken) [Ultimate Reverberant Gonkanau].osu"   # ideally the compare-map

bm = slider.beatmap.Beatmap.from_path(BEATMAP)
grid = build_grid(bm)

objs = bm._hit_objects
circles = [o for o in objs if not isinstance(o, (Slider, Spinner))]

active = 0
for o in circles:
    t = to_ms(o.time)
    # same condition resample_map_features applies, span collapsed to a point
    if np.any((grid >= t) & (grid <= t)):
        active += 1

print(f"{active}/{len(circles)} circles produce any active tick")