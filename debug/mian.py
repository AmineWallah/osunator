import numpy as np
import matplotlib.pyplot as plt
from training_data import perturb_example

# --- load ONE raw example ---
# adapt this line to however train_model loads them; either:
# from train_model import precompute_raw_examples
# example = precompute_raw_examples()[0]
# or load an .npz from your raw cache directly:
example = dict(np.load(
    "/features/INFERNOBESTMAP - GALNERYUS - RAISE MY SWORD [A THOUSAND FLAMES] (2023-10-08) Osu.npz"))

rng = np.random.default_rng(42)
pert = perturb_example(example, noise_std_px=10.0, rng=rng)  # defaults = lag active

mags = np.hypot(pert['cursor_dx'], pert['cursor_dy'])  # label magnitude per tick, px
lagged = pert['offset_mag'] > 25
clean = ~lagged

print(f"ticks: {len(mags)}  lagged: {lagged.sum()} ({lagged.mean():.0%})")
for name, m in [("clean ", mags[clean]), ("lagged", mags[lagged])]:
    print(f"{name}: mean {m.mean():6.2f}  median {np.median(m):6.2f}  "
          f"p95 {np.percentile(m, 95):6.2f}  max {m.max():7.2f}")

plt.hist(mags[clean], bins=100, alpha=0.6, label="clean ticks", density=True)
plt.hist(mags[lagged], bins=100, alpha=0.6, label="lagged ticks", density=True)
plt.xlabel("|label delta| (px/tick)")
plt.ylabel("density")
plt.legend()
plt.title("perturb_example label magnitudes")
plt.savefig("../label_hist.png", dpi=120)
print("saved label_hist.png")