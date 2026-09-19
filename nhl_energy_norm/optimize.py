from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from . import model


class FitResult(NamedTuple):
    params: model.RawParams
    loss_history: list
    final_train_x_star: jnp.ndarray


def _val_loss(raw_params, X_train, g_train, sigma_inv_train,
              X_val, g_val, sigma_inv_val, F_sim, F_role, F_sys,
              sim_group_mask, role_group_mask) -> jnp.ndarray:
    K = model.build_K(F_sim, F_role, F_sys, raw_params,
                       sim_group_mask=sim_group_mask, role_group_mask=role_group_mask)
    x_star, _ = model.solve_x_star(X_train, g_train, sigma_inv_train, K)
    resid = g_val - X_val @ x_star
    weighted_sq_err = sigma_inv_val * (resid ** 2)
    return jnp.sum(weighted_sq_err) / jnp.sum(sigma_inv_val)


def _zero_grad_fields(grads: model.RawParams, freeze: Tuple[str, ...]) -> model.RawParams:
    d = grads._asdict()
    for name in freeze:
        d[name] = jnp.zeros_like(d[name])
    return model.RawParams(**d)


def fit_hyperparameters(X_train, g_train, sigma_inv_train,
                         X_val, g_val, sigma_inv_val,
                         F_sim, F_role, F_sys,
                         n_steps: int = 500, lr: float = 0.05,
                         init_params: Optional[model.RawParams] = None,
                         freeze: Tuple[str, ...] = (),
                         sim_group_mask=None, role_group_mask=None) -> FitResult:
    if init_params is None:
        s_sim = model.graphs.median_heuristic_sigma(F_sim)
        s_role = model.graphs.median_heuristic_sigma(F_role)
        s_sys = model.graphs.median_heuristic_sigma(F_sys)
        init_params = model.init_raw_params(s_sim, s_role, s_sys)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(init_params)

    loss_and_grad = jax.jit(jax.value_and_grad(
        lambda p: _val_loss(p, X_train, g_train, sigma_inv_train,
                             X_val, g_val, sigma_inv_val, F_sim, F_role, F_sys,
                             sim_group_mask, role_group_mask)
    ))

    params = init_params
    history = []
    for _ in range(n_steps):
        loss, grads = loss_and_grad(params)
        if freeze:
            grads = _zero_grad_fields(grads, freeze)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        history.append(float(loss))

    K_final = model.build_K(F_sim, F_role, F_sys, params,
                             sim_group_mask=sim_group_mask, role_group_mask=role_group_mask)
    x_star_final, _ = model.solve_x_star(X_train, g_train, sigma_inv_train, K_final)
    return FitResult(params=params, loss_history=history, final_train_x_star=x_star_final)


def fit_hyperparameters_kfold(X, g, sigma_inv, F_sim, F_role, F_sys,
                               n_folds: int = 5, n_steps: int = 500, lr: float = 0.05,
                               sim_group_mask=None, role_group_mask=None) -> FitResult:


    m = X.shape[0]
    rng_np = __import__("numpy").random.default_rng(0)
    fold_ids = rng_np.integers(0, n_folds, size=m)

    s_sim = model.graphs.median_heuristic_sigma(F_sim)
    s_role = model.graphs.median_heuristic_sigma(F_role)
    s_sys = model.graphs.median_heuristic_sigma(F_sys)
    init_params = model.init_raw_params(s_sim, s_role, s_sys)

    fold_data = []
    for k in range(n_folds):
        val_idx = fold_ids == k
        train_idx = ~val_idx
        fold_data.append((
            X[train_idx], g[train_idx], sigma_inv[train_idx],
            X[val_idx], g[val_idx], sigma_inv[val_idx],
        ))

    def kfold_loss(params):
        total = jnp.array(0.0)
        for X_tr, g_tr, s_tr, X_va, g_va, s_va in fold_data:
            K = model.build_K(F_sim, F_role, F_sys, params,
                               sim_group_mask=sim_group_mask, role_group_mask=role_group_mask)
            x_star, _ = model.solve_x_star(X_tr, g_tr, s_tr, K)
            resid = g_va - X_va @ x_star
            total = total + jnp.sum(s_va * resid ** 2) / jnp.sum(s_va)
        return total / n_folds

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(init_params)
    loss_and_grad = jax.jit(jax.value_and_grad(kfold_loss))

    params = init_params
    history = []
    for step in range(n_steps):
        loss, grads = loss_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        history.append(float(loss))
        if step % 100 == 0:
            print(f"  step {step}: loss = {float(loss):.4f}")

    K_final = model.build_K(F_sim, F_role, F_sys, params,
                             sim_group_mask=sim_group_mask, role_group_mask=role_group_mask)
    x_star_final, _ = model.solve_x_star(X, g, sigma_inv, K_final)
    return FitResult(params=params, loss_history=history, final_train_x_star=x_star_final)


def ablate(X_train, g_train, sigma_inv_train, X_val, g_val, sigma_inv_val,
           F_sim, F_role, F_sys, fitted: model.RawParams,
           n_steps: int = 300, lr: float = 0.05,
           sim_group_mask=None, role_group_mask=None) -> dict:
    freeze_map = {"sim": "log_beta", "role": "log_gamma", "sys": "log_delta"}
    results = {}
    for drop, field_name in freeze_map.items():
        p = fitted._replace(**{field_name: jnp.array(-30.0)})
        refit = fit_hyperparameters(
            X_train, g_train, sigma_inv_train, X_val, g_val, sigma_inv_val,
            F_sim, F_role, F_sys, n_steps=n_steps, lr=lr, init_params=p, freeze=(field_name,),
            sim_group_mask=sim_group_mask, role_group_mask=role_group_mask,
        )
        results[f"without_{drop}"] = refit.loss_history[-1]

    full_loss = _val_loss(fitted, X_train, g_train, sigma_inv_train,
                           X_val, g_val, sigma_inv_val, F_sim, F_role, F_sys,
                           sim_group_mask, role_group_mask)
    results["full_model"] = float(full_loss)
    return results
