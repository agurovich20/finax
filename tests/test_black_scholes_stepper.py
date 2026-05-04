import pytest
import jax.numpy as jnp

from finonax import BlackScholes
from finonax.analytical import bs_call_price, bs_put_price
from finonax._spectral import build_derivative_operator


def _make_grid(N=2048, S0=100.0, x_half_extent=5.0):
    x_center = jnp.log(S0)
    x_grid = jnp.linspace(
        x_center - x_half_extent,
        x_center + x_half_extent,
        N,
        endpoint=False,
    )
    S_grid = jnp.exp(x_grid)
    domain_extent = 2 * x_half_extent
    return S_grid, domain_extent


def test_bs_matches_closed_form_atm():
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    N = 2048
    num_steps = 200
    dtau = T / num_steps

    S_grid, domain_extent = _make_grid(N=N)
    stepper = BlackScholes(domain_extent, N, dtau, sigma=sigma, r=r)
    V_spectral = stepper.price(lambda S: jnp.maximum(S - K, 0.0), S_grid, num_steps)

    i_atm = jnp.argmin(jnp.abs(S_grid - 100.0))
    ref = bs_call_price(100.0, K, r, sigma, T)
    assert jnp.allclose(V_spectral[i_atm], ref, atol=1e-3)


def test_bs_matches_closed_form_put():
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    N = 2048
    num_steps = 200
    dtau = T / num_steps

    S_grid, domain_extent = _make_grid(N=N)
    stepper = BlackScholes(domain_extent, N, dtau, sigma=sigma, r=r)
    V_spectral = stepper.price(lambda S: jnp.maximum(K - S, 0.0), S_grid, num_steps)

    i_atm = jnp.argmin(jnp.abs(S_grid - 100.0))
    ref = bs_put_price(100.0, K, r, sigma, T)
    assert jnp.allclose(V_spectral[i_atm], ref, atol=1e-3)


def test_bs_sigma_validation():
    with pytest.raises((ValueError, RuntimeError), match="sigma must be positive"):
        BlackScholes(10.0, 64, 0.01, sigma=0.0, r=0.05)
    with pytest.raises((ValueError, RuntimeError), match="sigma must be positive"):
        BlackScholes(10.0, 64, 0.01, sigma=-0.1, r=0.05)


def test_bs_linear_operator_has_correct_drift():
    sigma = 0.3
    r = 0.07
    domain_extent = 10.0
    N = 64

    stepper = BlackScholes(domain_extent, N, 0.01, sigma=sigma, r=r)
    D = build_derivative_operator(1, domain_extent, N)
    L = stepper._build_linear_operator(D)

    k_scaled = 2 * jnp.pi / domain_extent
    expected_imag = (r - 0.5 * sigma**2) * k_scaled
    assert jnp.allclose(L[0, 1].imag, expected_imag, atol=1e-10)
