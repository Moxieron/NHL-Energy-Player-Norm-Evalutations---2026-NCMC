
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import graphs


class RawParams(NamedTuple):

    log_alpha: jnp.ndarray
    log_beta: jnp.ndarray
    log_gamma: jnp.ndarray
    log_delta: jnp.ndarray
    log_sigma_sim: jnp.ndarray
    log_sigma_role: jnp.ndarray
    log_sigma_sys: jnp.ndarray


_INV_SOFTPLUS_ONE = 0.5413248546129181  # softplus(this) == 1.0


def init_raw_params(sigma_sim0: float = 1.0, sigma_role0: float = 1.0,
                     sigma_sys0: float = 1.0) -> RawParams:
    def inv_softplus(x: float) -> jnp.ndarray:
        return jnp.log(jnp.expm1(jnp.array(float(x))) + 1e-6)

    return RawParams(
        log_alpha=jnp.array(_INV_SOFTPLUS_ONE),
        log_beta=jnp.array(_INV_SOFTPLUS_ONE),
        log_gamma=jnp.array(_INV_SOFTPLUS_ONE),
        log_delta=jnp.array(_INV_SOFTPLUS_ONE),
        log_sigma_sim=inv_softplus(sigma_sim0),
        log_sigma_role=inv_softplus(sigma_role0),
        log_sigma_sys=inv_softplus(sigma_sys0),
    )


def _pos(v: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softplus(v)


def _pos_sigma(v: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softplus(v) + 1e-3  # floor


def build_K(F_sim: jnp.ndarray, F_role: jnp.ndarray, F_sys: jnp.ndarray,
            params: RawParams, sim_group_mask=None, role_group_mask=None) -> jnp.ndarray:

    n = F_sim.shape[0]
    alpha, beta = _pos(params.log_alpha), _pos(params.log_beta)
    gamma, delta = _pos(params.log_gamma), _pos(params.log_delta)

    L_sim = graphs.build_laplacian(F_sim, _pos_sigma(params.log_sigma_sim), group_mask=sim_group_mask)
    L_role = graphs.build_laplacian(F_role, _pos_sigma(params.log_sigma_role), group_mask=role_group_mask)
    L_sys = graphs.build_laplacian(F_sys, _pos_sigma(params.log_sigma_sys))

    return alpha * jnp.eye(n) + beta * L_sim + gamma * L_role + delta * L_sys


def solve_x_star(X: jnp.ndarray, g: jnp.ndarray, sigma_inv_diag: jnp.ndarray,
                  K: jnp.ndarray):

    Xt_Sinv = X.T * sigma_inv_diag[None, :]
    A = Xt_Sinv @ X + K
    A = 0.5 * (A + A.T)
    A_inv = jnp.linalg.inv(A)
    x_star = A_inv @ (Xt_Sinv @ g)
    return x_star, A_inv


def posterior_variance(A_inv: jnp.ndarray) -> jnp.ndarray:
    return jnp.diag(A_inv)


def pairwise_diff_variance(A_inv: jnp.ndarray, i: int, j: int) -> jnp.ndarray:
    return A_inv[i, i] + A_inv[j, j] - 2.0 * A_inv[i, j]
