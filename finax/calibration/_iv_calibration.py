from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from finax.analytical._black_scholes import bs_call_price


class OptionChain(NamedTuple):
    strikes: jax.Array
    maturities: jax.Array
    prices: jax.Array
    true_ivs: jax.Array


def generate_synthetic_chain(S0, strikes, maturities, r, iv_surface_fn, rng_key, noise_std=0.0):
    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    maturities = jnp.asarray(maturities, dtype=jnp.float64)

    # Cartesian product: K on outer axis, T on inner axis (indexing='ij')
    K_grid, T_grid = jnp.meshgrid(strikes, maturities, indexing="ij")
    K_flat = K_grid.ravel()
    T_flat = T_grid.ravel()
    n = len(K_flat)

    # Python iteration so iv_surface_fn can be any callable, JAX-traced or not
    true_ivs = jnp.array(
        [float(iv_surface_fn(float(K_flat[i]), float(T_flat[i]))) for i in range(n)],
        dtype=jnp.float64,
    )

    # bs_call_price accepts array arguments (all ops are elementwise)
    prices = bs_call_price(
        jnp.asarray(S0, dtype=jnp.float64),
        K_flat,
        jnp.asarray(r, dtype=jnp.float64),
        true_ivs,
        T_flat,
    )

    if noise_std > 0.0:
        noise = jax.random.normal(rng_key, shape=(n,), dtype=jnp.float64) * noise_std
        prices = prices + noise

    return OptionChain(strikes=K_flat, maturities=T_flat, prices=prices, true_ivs=true_ivs)


def calibrate_iv(chain, S0, r, initial_iv=0.2, num_iterations=500, learning_rate=0.01):
    S0 = jnp.asarray(S0, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    n = len(chain.strikes)

    # Softplus parameterisation: sigma = softplus(raw), raw = softplus_inv(sigma)
    # softplus_inv(x) = log(exp(x) - 1)
    init_raw = jnp.log(jnp.exp(jnp.asarray(initial_iv, dtype=jnp.float64)) - 1.0)
    raw_sigma = jnp.full((n,), init_raw, dtype=jnp.float64)

    def loss_fn(raw):
        sigma = jax.nn.softplus(raw)
        prices_pred = bs_call_price(S0, chain.strikes, r, sigma, chain.maturities)
        return jnp.mean((prices_pred - chain.prices) ** 2)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(raw_sigma)

    @jax.jit
    def step(raw, state):
        loss, grad = jax.value_and_grad(loss_fn)(raw)
        updates, new_state = optimizer.update(grad, state)
        new_raw = optax.apply_updates(raw, updates)
        return new_raw, new_state, loss

    loss_history = []
    iv_history = []

    for _ in range(num_iterations):
        raw_sigma, opt_state, loss = step(raw_sigma, opt_state)
        loss_history.append(float(loss))
        iv_history.append(jax.nn.softplus(raw_sigma))

    ivs = jax.nn.softplus(raw_sigma)
    return ivs, {"loss": loss_history, "ivs": iv_history}
