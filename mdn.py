import numpy as np
import tensorflow as tf

N_MIX = 4               # K: mixture components
OUT_DIM = 2             # (dx, dy)
SIGMA_FLOOR = 0.01      # in normalized (z-scored) units
MDN_PARAMS = N_MIX * (1 + 2 * OUT_DIM)   # logits + means + sigmas = K*5


def split_mdn_params(params, n_mix=N_MIX):
    """(..., K*5) -> logits (..., K), mu (..., K, 2), sigma (..., K, 2).
    sigma passes through softplus then gets floored."""
    logits = params[..., :n_mix]
    mu = tf.reshape(params[..., n_mix:n_mix * (1 + OUT_DIM)],
                    tf.concat([tf.shape(params)[:-1], [n_mix, OUT_DIM]], axis=0))
    sigma_raw = tf.reshape(params[..., n_mix * (1 + OUT_DIM):],
                           tf.concat([tf.shape(params)[:-1], [n_mix, OUT_DIM]], axis=0))
    sigma = tf.nn.softplus(sigma_raw) + SIGMA_FLOOR
    return logits, mu, sigma


def mdn_nll(y_true, y_pred):
    """Negative log-likelihood of y_true (..., 2) under the mixture encoded
    in y_pred (..., K*5). Returns per-tick loss (batch, time) so Keras's
    sample_weight (the padding mask) applies exactly as it did for MSE.

    log N(y|mu,sigma) for diagonal 2D gaussian:
        -0.5 * sum_d ((y_d-mu_d)/sigma_d)^2 - sum_d log sigma_d - log(2*pi)
    mixture: logsumexp over components of (log pi_k + log N_k).
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
    """Draw one (dx, dy) sample per tick from the predicted mixture.
    numpy-side (generation runs tick-by-tick outside the graph anyway).

    temperature scales BOTH the component choice (logits / T) and the
    gaussian spread (sigma * T): T=1 faithful sampling, T->0 approaches
    greedy (dominant mode's mean, no noise) — the precision-vs-humanness
    knob, exposed from day one.
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
    def __init__(self, temperature, rho=0.94, rng=None):
        self.temperature = temperature
        self.rho = rho
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        self.eps = self.rng.normal(size=2)

    def sample(self, params):
        logits, mu, sigma = split_mdn_params(tf.convert_to_tensor(params))
        logits, mu, sigma = logits.numpy(), mu.numpy(), sigma.numpy()

        if self.temperature <= 0:
            k = np.argmax(logits, axis=-1)
            return np.take_along_axis(mu, k[..., None, None], axis=-2).squeeze(-2) # copied this from sample_mdn()

        w = self.rng.normal(size=2)
        self.eps = self.rho * self.eps + np.sqrt(1 - self.rho ** 2) * w

        scaled = logits / self.temperature
        scaled = scaled - scaled.max(axis=-1, keepdims=True)  # stable softmax
        pi = np.exp(scaled)
        pi = pi / pi.sum(axis=-1, keepdims=True)

        flat_pi = pi.reshape(-1, pi.shape[-1])
        ks = np.array([self.rng.choice(pi.shape[-1], p=p) for p in flat_pi]).reshape(pi.shape[:-1])

        mu_k = np.take_along_axis(mu, ks[..., None, None], axis=-2).squeeze(-2)
        sigma_k = np.take_along_axis(sigma, ks[..., None, None], axis=-2).squeeze(-2)

        return mu_k + sigma_k * self.temperature * self.eps