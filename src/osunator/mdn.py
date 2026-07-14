"""
Mixture Density Network head: the cursor policy's output distribution.

WHY: many ticks have several valid human moves (decelerating vs. cruising,
wiggle left vs. right). MSE's optimum under multiple answers is their
average — none of them — which produced the observed mean-collapse
oscillation. The MDN instead outputs K weighted gaussians over (dx, dy),
so coexisting hypotheses stay separate instead of averaging into garbage.

Training scores the mixture's surprise at the human's move (mdn_nll);
generation draws from it (sample_mdn / CorrelatedSampler). All units are
normalized (z-scored deltas per norm_stats.json).
"""

import numpy as np
import tensorflow as tf

N_MIX = 4               # K: unablated initial choice — revisit if the
                        # spaced-stream snapping implicates mode count
OUT_DIM = 2             # (dx, dy)
SIGMA_FLOOR = 0.01      # normalized units. Blocks the degenerate optimum of
                        # collapsing sigma -> 0 onto one training point
                        # (loss -> -inf): no component can win by memorizing
MDN_PARAMS = N_MIX * (1 + 2 * OUT_DIM)   # flat per-tick layout:
                                         # [K logits | K*2 means | K*2 raw sigmas]


def split_mdn_params(params: tf.Tensor, n_mix=N_MIX) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Split raw MDN head output into logits, means, standard deviations.
    Graph-compatible (runs inside mdn_nll), hence tf ops and the
    runtime-built reshape target: leading dims are dynamic in-graph.

    :param params: raw no-activation Dense output, last axis = K*5, any
        leading shape.
    :return: logits (..., K); mu, sigma (..., K, 2). sigma already positive
        (softplus) and floored — downstream never re-guards.
    """
    # flat [K logits | K*2 means | K*2 raw sigmas] -> logits (..., K),
    # mu and sigma (..., K, 2); target shape built at runtime
    logits = params[..., :n_mix]
    mu = tf.reshape(params[..., n_mix:n_mix * (1 + OUT_DIM)],
                    tf.concat([tf.shape(params)[:-1], [n_mix, OUT_DIM]], axis=0))
    sigma_raw = tf.reshape(params[..., n_mix * (1 + OUT_DIM):],
                           tf.concat([tf.shape(params)[:-1], [n_mix, OUT_DIM]], axis=0))
    sigma = tf.nn.softplus(sigma_raw) + SIGMA_FLOOR   # reals -> positive, floored
    return logits, mu, sigma


def mdn_nll(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Negative log-likelihood of the human's move under the predicted mixture:
    per tick, how surprised is the mixture by what the human actually did.
    Mass parked BETWEEN two real modes scores badly on both, so the gradient
    spreads components to cover each mode honestly — the anti-averaging
    property MSE lacks.

    Per component: log N(y|mu,sigma) = -0.5*sum_d z_d^2 - sum_d log sigma_d
    - log 2pi (diagonal 2D gaussian); mixture = logsumexp(log_pi + log_N).
    logsumexp for stability — exp of very negative log-densities underflows.

    :param y_true: human (dx, dy), normalized, (..., 2).
    :param y_pred: raw MDN params, (..., K*5). (Keras loss arg order.)
    :return: per-tick loss, (batch, time), NOT a scalar — Keras sample_weight
        applies per (batch, time); reducing here silently breaks masking.
    """
    logits, mu, sigma = split_mdn_params(y_pred)
    log_pi = tf.nn.log_softmax(logits, axis=-1)            # (..., K)

    y = tf.expand_dims(y_true, axis=-2)                     # (..., 1, 2) vs (..., K, 2)
    z = (y - mu) / sigma
    log_component = (-0.5 * tf.reduce_sum(tf.square(z), axis=-1)
                     - tf.reduce_sum(tf.math.log(sigma), axis=-1)
                     - tf.math.log(2.0 * np.pi))            # (..., K)

    log_likelihood = tf.reduce_logsumexp(log_pi + log_component, axis=-1)
    return -log_likelihood                                   # (batch, time)


def sample_mdn(params, temperature=1.0, rng=None):
    """
    One (dx, dy) per tick from the mixture — two-stage dice: pick WHICH
    component (softmaxed logits), then a point WITHIN it (mu_k + sigma_k *
    noise). Temperature scales BOTH stages (logits/T, sigma*T) so T is one
    precision-vs-humanness knob; T=1 faithful, T<=0 greedy (dominant mean,
    deterministic).

    NOTE: noise here is independent per call — white, watches as shake.
    Closed-loop generation uses CorrelatedSampler; this stays for T=0 and
    teacher-forced diagnostics.

    :param params: raw MDN params, leading shape + (K*5,).
    :return: samples, leading shape + (2,).
    """
    if rng is None:
        rng = np.random.default_rng()

    logits, mu, sigma = split_mdn_params(tf.convert_to_tensor(params))
    logits, mu, sigma = logits.numpy(), mu.numpy(), sigma.numpy()

    if temperature <= 0:                                     # greedy: dominant mode's mean
        k = np.argmax(logits, axis=-1)
        return np.take_along_axis(mu, k[..., None, None], axis=-2).squeeze(-2)

    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=-1, keepdims=True)     # stable softmax
    pi = np.exp(scaled)
    pi = pi / pi.sum(axis=-1, keepdims=True)

    flat_pi = pi.reshape(-1, pi.shape[-1])
    ks = np.array([rng.choice(pi.shape[-1], p=p) for p in flat_pi]).reshape(pi.shape[:-1])

    mu_k = np.take_along_axis(mu, ks[..., None, None], axis=-2).squeeze(-2)
    sigma_k = np.take_along_axis(sigma, ks[..., None, None], axis=-2).squeeze(-2)
    return mu_k + rng.normal(size=mu_k.shape) * sigma_k * temperature


class CorrelatedSampler:
    """
    MDN sampler with temporally correlated (colored) noise.

    White noise (sample_mdn, T>0) = 60 uncorrelated nudges/sec — shake,
    rejected. Real hand deviation persists across ticks, so the gaussian
    draw is replaced by a persistent Ornstein-Uhlenbeck state (component
    choice stays fresh per tick — only the noise is correlated):

        eps[t] = rho * eps[t-1] + sqrt(1 - rho^2) * white[t]
        sample = mu_k + sigma_k * T * eps[t]

    sqrt(1-rho^2) pins eps's stationary variance at 1: same amplitude as
    white at a given T, different color. Mirrors the training-side OU
    perturbation (recovery from smooth drift <-> production of it).

    Measured: T=0 bit-identical to sample_mdn greedy; noise lag-1 autocorr
    0.938 vs -0.004 white; flip rate 10.8% vs 49.9% (= arccos(rho)/pi);
    std within ~2% of sigma*T for both.

    Lifecycle: eps is per-replay — reset() before each generation, paired
    with the LSTM reset. __init__ calls reset(); init is stationary, not
    zeros (early ticks shouldn't be calmer than steady state).

    LIMITATION: eps advances once per sample() call — single-tick calls
    only. A whole chunk at T>0 would share one noise draw; chunk paths use
    T=0 or plain sample_mdn.
    """

    def __init__(self, temperature, rho=0.94, rng=None):
        self.temperature = temperature
        self.rho = rho
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        """Fresh per-replay noise state, stationary draw (unit variance)."""
        self.eps = self.rng.normal(size=2)

    def sample(self, params):
        """
        One (dx, dy) draw; advances the OU state (T>0 only).

        :param params: raw MDN params for ONE tick, (1, 1, K*5) — see class
            LIMITATION.
        :return: (1, 1, 2), same convention as sample_mdn.
        """
        logits, mu, sigma = split_mdn_params(tf.convert_to_tensor(params))
        logits, mu, sigma = logits.numpy(), mu.numpy(), sigma.numpy()

        if self.temperature <= 0:
            # greedy, deliberately BEFORE the OU update: T=0 must not
            # consume RNG draws or advance state (seeded reproducibility)
            k = np.argmax(logits, axis=-1)
            return np.take_along_axis(mu, k[..., None, None], axis=-2).squeeze(-2)

        # OU update: persistent eps drifts instead of flickering
        w = self.rng.normal(size=2)
        self.eps = self.rho * self.eps + np.sqrt(1 - self.rho ** 2) * w

        # component choice: fresh every tick, exactly as sample_mdn
        scaled = logits / self.temperature
        scaled = scaled - scaled.max(axis=-1, keepdims=True)  # stable softmax
        pi = np.exp(scaled)
        pi = pi / pi.sum(axis=-1, keepdims=True)

        flat_pi = pi.reshape(-1, pi.shape[-1])
        ks = np.array([self.rng.choice(pi.shape[-1], p=p) for p in flat_pi]).reshape(pi.shape[:-1])

        mu_k = np.take_along_axis(mu, ks[..., None, None], axis=-2).squeeze(-2)
        sigma_k = np.take_along_axis(sigma, ks[..., None, None], axis=-2).squeeze(-2)

        return mu_k + sigma_k * self.temperature * self.eps