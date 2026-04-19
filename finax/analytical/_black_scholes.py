import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


def _d1_d2(S, K, r, sigma, T):
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return d1, d2


@jax.jit
def bs_call_price(S, K, r, sigma, T):
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)


@jax.jit
def bs_put_price(S, K, r, sigma, T):
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return K * jnp.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


@jax.jit
def bs_gamma(S, K, r, sigma, T):
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return norm.pdf(d1) / (S * sigma * jnp.sqrt(T))


@jax.jit
def bs_vega(S, K, r, sigma, T):
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return S * norm.pdf(d1) * jnp.sqrt(T)


@jax.jit
def bs_call_delta(S, K, r, sigma, T):
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return norm.cdf(d1)


@jax.jit
def bs_put_delta(S, K, r, sigma, T):
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return norm.cdf(d1) - 1.0


@jax.jit
def bs_call_rho(S, K, r, sigma, T):
    _, d2 = _d1_d2(S, K, r, sigma, T)
    return K * T * jnp.exp(-r * T) * norm.cdf(d2)


@jax.jit
def bs_put_rho(S, K, r, sigma, T):
    _, d2 = _d1_d2(S, K, r, sigma, T)
    return -K * T * jnp.exp(-r * T) * norm.cdf(-d2)


@jax.jit
def bs_call_theta(S, K, r, sigma, T):
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return -(S * norm.pdf(d1) * sigma) / (2 * jnp.sqrt(T)) - r * K * jnp.exp(-r * T) * norm.cdf(d2)


@jax.jit
def bs_put_theta(S, K, r, sigma, T):
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return -(S * norm.pdf(d1) * sigma) / (2 * jnp.sqrt(T)) + r * K * jnp.exp(-r * T) * norm.cdf(-d2)
