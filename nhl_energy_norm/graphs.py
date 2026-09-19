
from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def standardize(F: jnp.ndarray) -> jnp.ndarray:

    mu = jnp.mean(F, axis=0, keepdims=True)
    sd = jnp.std(F, axis=0, keepdims=True)
    return (F - mu) / (sd + 1e-8)


def position_group_mask(labels) -> np.ndarray:

    labels = np.asarray(labels)
    return (labels[:, None] == labels[None, :]).astype(np.float64)


def rbf_weight_matrix(F: jnp.ndarray, sigma: jnp.ndarray, group_mask=None) -> jnp.ndarray:

    diff = F[:, None, :] - F[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    W = jnp.exp(-sq_dist / (2.0 * sigma ** 2))
    W = W * (1.0 - jnp.eye(F.shape[0]))
    if group_mask is not None:
        W = W * jnp.asarray(group_mask)
    return W


def symmetric_normalized_laplacian(W: jnp.ndarray) -> jnp.ndarray:

    deg = jnp.sum(W, axis=1)
    d_inv_sqrt = 1.0 / jnp.sqrt(deg + 1e-8)
    L = jnp.eye(W.shape[0]) - (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])
    return 0.5 * (L + L.T)


def build_laplacian(F_raw: jnp.ndarray, sigma: jnp.ndarray, group_mask=None) -> jnp.ndarray:

    F = standardize(F_raw)
    W = rbf_weight_matrix(F, sigma, group_mask=group_mask)
    return symmetric_normalized_laplacian(W)


def median_heuristic_sigma(F_raw: jnp.ndarray) -> float:

    F = standardize(F_raw)
    diff = F[:, None, :] - F[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    n = F.shape[0]
    iu = jnp.triu_indices(n, k=1)
    d = jnp.sqrt(sq_dist[iu])
    return float(jnp.median(d))
