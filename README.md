# Osunator
Osunator is a replay generator for the rhythm game osu! that takes an .osu file as an input and outputs a replay file (.osr), written with TensorFlow.

The primary goal of this project is to be a proof of concept for being able to replicate a human-like playstyle using an
LSTM model. Provided with a dataset of as few as ~40 .osr files, the model can imitate legitimate player behaviors on 
certain patterns (natural snapping on cross-screen jumps, smooth flowing on streams etc...). With that said, the ground
truth used to evaluate the model is how closely it can replicate a human replay rather than actually performing decently
at the game.

The current model is trained on around ~5.5k replays on ~1.1k different maps from @skihikingkevin's dataset on 
kaggle (thank you so much for sharing!), as well as a few more replays I collected myself. It performs well enough to
score A-Ranks on the 7 to 8 star rating range, and to even perfect lower rated maps.

Note: the replay file's stored score metadata is an intentional placeholder (D rank, zeroed counters)

## Demo
Comparison of a generated replay VS a replay of me playing the game on a same map: (renders made with 
[danser-go](https://github.com/Wieku/danser-go))

<p align="center">
  AI Generated replay
</p>

<p align="center">
  <img src="assets/Osunator%20replay.gif" alt="AI Generated replay">
</p>

<p align="center">
  Human replay
</p>

<p align="center">
  <img src="assets/Amine%20replay.gif" alt="AI Generated replay">
</p>


## Evaluation

These metrics have been calculated from 10 different maps on the test set. ~29 minutes of generated play, 8,432 hit targets.

| Metric                                   | Result                                                       |
|------------------------------------------|--------------------------------------------------------------|
| Roundtrip integrity (`.osr` re-decode)   | 10/10 PASS · 0.000 px position error · ≤0.7 ms timing drift  |
| Target misses                            | 21 / 8,432 (0.25%)                                           |
| Onset recall (±3 ticks)                  | mean 99.1% · min 97%                                         |
| Onset-count error vs human               | -2.5% … +4.5% (unbiased)                                     |
| Hit-timing offset (mean per map)         | -0.89 … +0.70 ticks (sub-tick) · median +0.0t on all 10 maps |
| Moving-speed ratio (gen / human)         | 0.87 - 1.03 (mean 0.96)                                      |
| Longest map                              | 18,304 ticks (~5 min) · zero dead zones                      |

**Known limitations:** 
- generated cursor motion shows a 2-4× higher rate of >90° heading reversals than the human
reference at small movement scales (median 2–5 px), usually micro-jitters, not oscillations.
- spinner ticks are masked from the model for now, which is why it performs very poorly on them.
- cannot generate replays in any map altering mods like Hard Rock, Double Time, Easy and Half Time.


## Docker Setup

[![Docker Hub](https://img.shields.io/docker/v/aminewallah/osunator?logo=docker&label=Docker%20Hub)](https://hub.docker.com/r/aminewallah/osunator)

You can download the docker image of the latest release from the docker hub on the link above.

To run it:
```bash
mkdir -p ~/osunator-out

docker run --rm \
  -v "/path/to/your/osu/Songs/some-mapset":/maps:ro \
  -v ~/osunator-out:/out \
  aminewallah/osunator:latest \
  "/maps/your-map.osu" -o /out
```

The generated `your-map.osr` appears in `~/osunator-out`.

Options (append after the map path):

| Flag | Default | Notes |
|---|---|---|
| `--mod {nomod,hidden,nofail}` | `nofail` | NoFail is the default so replays stay watchable end-to-end even when the play would otherwise fail out. |
| `-t`, `--temperature FLOAT` | `0.0` | Cursor noise. `0` is deterministic (identical output per map); ~`0.1` adds human-plausible variation; above ~`0.3` reads like beginner play. |

On my Ryzen 5 5600, it takes around 30 seconds to generate a 3 minute-ish long map, so performance might vary depending 
on your CPU.

The docker image is also shipped with the CUDA version of TensorFlow, which is why it's much bigger than what it should 
be, I'll try to fix that in the future.


## Running from source

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

**These steps were ran on Linux. if running on Windows, you might need to use WSL2**

```bash
git clone https://github.com/AmineWallah/osunator
cd osunator
uv sync

# model weights, not tracked in git
curl -LO https://github.com/AmineWallah/osunator/releases/download/0.1.0/best_model.keras

mkdir -p out
uv run osunator "path/to/your-map.osu" -o out
```

[releases](https://github.com/AmineWallah/osunator/releases/tag/0.1.0) page and place it in the root of the repo.

`uv sync` installs the dependencies and the project itself into a local
virtual environment; the `osunator` command is the package's entry
point, so no PYTHONPATH or activation gymnastics are needed (`uv run`
handles the environment).

Notes:
- The output directory must already exist (`-o` validates, it doesn't create).
- A CUDA-capable GPU is used automatically if present; otherwise it
  falls back to CPU.
- Run `uv run osunator --help` for the full flag reference.
## Ethical Aspect
Osunator is a research artifact for studying imitation learning on dense human demonstration data, not a cheating tool
and it's built to stay that way.

For transparency, the replays do carry statistical fingerprints (60 ticks per second grid, determinism at temperature 0,
and the motion artifacts listed under the limitations section)

This project doesn't provide any way to submit generated replays to the game servers either, and the lack of hosted
deployment makes replay generation local by design.

## Special thanks
To the oomfies who contributed with their replay data in the early versions of the model, THANK YOU: 
300mm, JaViLuMa, pluk, MrFish, Jop, Robin.