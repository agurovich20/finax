import jax
import jax.numpy as jnp
import pytest

from finonax.analytical import (
    bs_call_price,
    bs_put_price,
    bs_gamma,
    bs_vega,
    bs_call_delta,
    bs_put_delta,
    bs_call_rho,
    bs_put_rho,
    bs_call_theta,
    bs_put_theta,
)

S = jnp.float64(100.0)
K = jnp.float64(100.0)
r = jnp.float64(0.05)
sigma = jnp.float64(0.2)
T = jnp.float64(1.0)


def test_call_reference_price():
    assert jnp.isclose(
        bs_call_price(S, K, r, sigma, T),
        jnp.float64(10.450583572185565),
        atol=1e-10,
    )


def test_put_reference_price():
    assert jnp.isclose(
        bs_put_price(S, K, r, sigma, T),
        jnp.float64(5.573526022256971),
        atol=1e-10,
    )


def test_put_call_parity():
    call = bs_call_price(S, K, r, sigma, T)
    put = bs_put_price(S, K, r, sigma, T)
    lhs = call - put
    rhs = S - K * jnp.exp(-r * T)
    assert jnp.abs(lhs - rhs) < 1e-10


def test_call_delta_atm():
    assert jnp.isclose(bs_call_delta(S, K, r, sigma, T), jnp.float64(0.6368306511), atol=1e-6)


def test_gamma_symmetric():
    gc = bs_gamma(S, K, r, sigma, T)
    # Gamma is identical for call and put; verify against both price functions.
    grad_call = jax.grad(jax.grad(bs_call_price, argnums=0), argnums=0)(S, K, r, sigma, T)
    grad_put = jax.grad(jax.grad(bs_put_price, argnums=0), argnums=0)(S, K, r, sigma, T)
    assert jnp.isclose(gc, grad_call, atol=1e-6)
    assert jnp.isclose(gc, grad_put, atol=1e-6)


def test_greeks_match_grad_call():
    grad_S = jax.grad(bs_call_price, argnums=0)(S, K, r, sigma, T)
    grad_r = jax.grad(bs_call_price, argnums=2)(S, K, r, sigma, T)
    grad_sigma = jax.grad(bs_call_price, argnums=3)(S, K, r, sigma, T)
    grad_T = jax.grad(bs_call_price, argnums=4)(S, K, r, sigma, T)

    assert jnp.abs(bs_call_delta(S, K, r, sigma, T) - grad_S) < 1e-6
    assert jnp.abs(bs_vega(S, K, r, sigma, T) - grad_sigma) < 1e-6
    assert jnp.abs(bs_call_rho(S, K, r, sigma, T) - grad_r) < 1e-6
    # Theta is defined as -dV/dT (time decay convention), so theta == -grad_T.
    assert jnp.abs(bs_call_theta(S, K, r, sigma, T) + grad_T) < 1e-6


def test_greeks_match_grad_put():
    grad_S = jax.grad(bs_put_price, argnums=0)(S, K, r, sigma, T)
    grad_r = jax.grad(bs_put_price, argnums=2)(S, K, r, sigma, T)
    grad_sigma = jax.grad(bs_put_price, argnums=3)(S, K, r, sigma, T)
    grad_T = jax.grad(bs_put_price, argnums=4)(S, K, r, sigma, T)

    assert jnp.abs(bs_put_delta(S, K, r, sigma, T) - grad_S) < 1e-6
    assert jnp.abs(bs_vega(S, K, r, sigma, T) - grad_sigma) < 1e-6
    assert jnp.abs(bs_put_rho(S, K, r, sigma, T) - grad_r) < 1e-6
    assert jnp.abs(bs_put_theta(S, K, r, sigma, T) + grad_T) < 1e-6


def test_jit_and_grad_composable():
    jitted_call = jax.jit(bs_call_price)
    val = jitted_call(S, K, r, sigma, T)
    assert val.shape == ()

    grad_fn = jax.jit(jax.grad(bs_call_price, argnums=0))
    g = grad_fn(S, K, r, sigma, T)
    assert g.shape == ()
