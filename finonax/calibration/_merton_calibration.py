from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from finonax.analytical._merton import merton_call_price


class MertonChain(NamedTuple):
    strikes: jax.Array
    maturities: jax.Array
    prices: jax.Array
    true_params: dict


def generate_synthetic_merton_chain(
    S0, strikes, maturities, r,
    sigma_true, lambda_true, mu_jump_true, sigma_jump_true,
    rng_key, noise_std=0.0,
):
    S0 = jnp.asarray(S0, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    sigma_true = jnp.asarray(sigma_true, dtype=jnp.float64)
    lambda_true = jnp.asarray(lambda_true, dtype=jnp.float64)
    mu_jump_true = jnp.asarray(mu_jump_true, dtype=jnp.float64)
    sigma_jump_true = jnp.asarray(sigma_jump_true, dtype=jnp.float64)

    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    maturities = jnp.asarray(maturities, dtype=jnp.float64)

    K_grid, T_grid = jnp.meshgrid(strikes, maturities, indexing="ij")
    K_flat = K_grid.ravel()
    T_flat = T_grid.ravel()
    n = len(K_flat)

    # Python loop: merton_call_price is scalar K/T (internal n-axis is the series index)
    prices = jnp.array(
        [
            float(
                merton_call_price(
                    S0, K_flat[i], r,
                    sigma_true, T_flat[i],
                    lambda_true, mu_jump_true, sigma_jump_true,
                )
            )
            for i in range(n)
        ],
        dtype=jnp.float64,
    )

    if noise_std > 0.0:
        noise = jax.random.normal(rng_key, shape=(n,), dtype=jnp.float64) * noise_std
        prices = prices + noise

    return MertonChain(
        strikes=K_flat,
        maturities=T_flat,
        prices=prices,
        true_params={
            "sigma": float(sigma_true),
            "lambda_jump": float(lambda_true),
            "mu_jump": float(mu_jump_true),
            "sigma_jump": float(sigma_jump_true),
        },
    )


def calibrate_merton(
    chain, S0, r,
    initial_sigma=0.2,
    initial_lambda=0.5,
    initial_mu_jump=-0.05,
    initial_sigma_jump=0.10,
    num_iterations=1000,
    learning_rate=0.01,
):
    S0 = jnp.asarray(S0, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    n_contracts = len(chain.strikes)

    def softplus_inv(x):
        return jnp.log(jnp.exp(jnp.asarray(x, dtype=jnp.float64)) - 1.0)

    def decode(raw):
        return {
            "sigma": jax.nn.softplus(raw["raw_sigma"]),
            "lambda_jump": jax.nn.softplus(raw["raw_lambda"]),
            "mu_jump": raw["raw_mu_jump"],
            "sigma_jump": jax.nn.softplus(raw["raw_sigma_jump"]),
        }

    def loss_fn(raw):
        p = decode(raw)
        # Python loop unrolled at JIT trace time; n_contracts is a concrete int.
        prices_pred = jnp.stack(
            [
                merton_call_price(
                    S0, chain.strikes[i], r,
                    p["sigma"], chain.maturities[i],
                    p["lambda_jump"], p["mu_jump"], p["sigma_jump"],
                )
                for i in range(n_contracts)
            ]
        )
        return jnp.mean((prices_pred - chain.prices) ** 2)

    # --- Warm start: coarse 3-D grid over (lambda, mu_jump, sigma_jump).
    # Merton calibration is non-convex; the default initial conditions are
    # typically in the wrong basin. A grid forward sweep (no gradients) is
    # cheap and places Adam in the correct basin before gradient descent.
    # The grid covers the equity-relevant parameter range for all three
    # jump parameters; sigma is kept at initial_sigma for the sweep.
    lambda_grid = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    mu_grid = [-0.40, -0.20, -0.10, -0.05, 0.00]
    sigma_jump_grid = [0.05, 0.10, 0.15, 0.20, 0.30]

    best_grid_loss = float("inf")
    best_lambda = initial_lambda
    best_mu = initial_mu_jump
    best_sigma_jump = initial_sigma_jump

    for lam in lambda_grid:
        for mu in mu_grid:
            for sj in sigma_jump_grid:
                raw_try = {
                    "raw_sigma": softplus_inv(initial_sigma),
                    "raw_lambda": softplus_inv(lam),
                    "raw_mu_jump": jnp.asarray(mu, dtype=jnp.float64),
                    "raw_sigma_jump": softplus_inv(sj),
                }
                loss_try = float(loss_fn(raw_try))
                if loss_try < best_grid_loss:
                    best_grid_loss = loss_try
                    best_lambda = lam
                    best_mu = mu
                    best_sigma_jump = sj

    raw_params = {
        "raw_sigma": softplus_inv(initial_sigma),
        "raw_lambda": softplus_inv(best_lambda),
        "raw_mu_jump": jnp.asarray(best_mu, dtype=jnp.float64),
        "raw_sigma_jump": softplus_inv(best_sigma_jump),
    }

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(raw_params)

    @jax.jit
    def step(raw, state):
        loss, grad = jax.value_and_grad(loss_fn)(raw)
        updates, new_state = optimizer.update(grad, state)
        new_raw = optax.apply_updates(raw, updates)
        return new_raw, new_state, loss

    loss_history = []
    params_history = []

    for _ in range(num_iterations):
        raw_params, opt_state, loss = step(raw_params, opt_state)
        loss_history.append(float(loss))
        params_history.append({k: float(v) for k, v in decode(raw_params).items()})

    params = {k: float(v) for k, v in decode(raw_params).items()}
    return params, {"loss": loss_history, "params": params_history}
