import pytest
import jax.numpy as jnp

from finonax import BlackScholes, Merton
from finonax import merton_call_price, merton_put_price
from finonax._spectral import build_derivative_operator

# Canonical Merton parameters
S0 = 100.0
K = 100.0
r = 0.05
sigma = 0.2
T = 1.0
LAMBDA = 1.0
MU_J = -0.1
SIGMA_J = 0.15

N = 2048
NUM_STEPS = 200
X_HALF = 5.0
DOMAIN_EXTENT = 2 * X_HALF


def _make_grid(N=N):
    x_center = jnp.log(S0)
    x_grid = jnp.linspace(
        x_center - X_HALF, x_center + X_HALF, N, endpoint=False
    )
    return jnp.exp(x_grid)


def test_merton_matches_closed_form_atm():
    dtau = T / NUM_STEPS
    S_grid = _make_grid()
    stepper = Merton(
        DOMAIN_EXTENT, N, dtau,
        sigma=sigma, r=r,
        lambda_jump=LAMBDA, mu_jump=MU_J, sigma_jump=SIGMA_J,
    )
    V = stepper.price(lambda S: jnp.maximum(S - K, 0.0), S_grid, NUM_STEPS)

    i_atm = jnp.argmin(jnp.abs(S_grid - S0))
    ref = merton_call_price(S0, K, r, sigma, T, LAMBDA, MU_J, SIGMA_J)
    assert jnp.allclose(V[i_atm], ref, atol=1e-3)


def test_merton_matches_closed_form_put():
    dtau = T / NUM_STEPS
    S_grid = _make_grid()
    stepper = Merton(
        DOMAIN_EXTENT, N, dtau,
        sigma=sigma, r=r,
        lambda_jump=LAMBDA, mu_jump=MU_J, sigma_jump=SIGMA_J,
    )
    V = stepper.price(lambda S: jnp.maximum(K - S, 0.0), S_grid, NUM_STEPS)

    i_atm = jnp.argmin(jnp.abs(S_grid - S0))
    ref = merton_put_price(S0, K, r, sigma, T, LAMBDA, MU_J, SIGMA_J)
    assert jnp.allclose(V[i_atm], ref, atol=1e-3)


def test_merton_reduces_to_bs_when_no_jumps():
    """With lambda=0, the Merton linear operator equals the BS operator exactly."""
    N_small = 512
    dtau = T / 100

    S_grid = _make_grid(N=N_small)
    bs = BlackScholes(DOMAIN_EXTENT, N_small, dtau, sigma=sigma, r=r)
    merton = Merton(
        DOMAIN_EXTENT, N_small, dtau,
        sigma=sigma, r=r,
        lambda_jump=0.0, mu_jump=MU_J, sigma_jump=SIGMA_J,
    )

    payoff = lambda S: jnp.maximum(S - K, 0.0)
    V_bs = bs.price(payoff, S_grid, 100)
    V_merton = merton.price(payoff, S_grid, 100)

    assert jnp.allclose(V_bs, V_merton, atol=1e-10)


def test_merton_linear_operator_has_jump_term():
    """
    L_Merton − L_BS = −λκ·D − λ + λ·φ_p(k) at every wavenumber.

    This catches bugs where the jump term (characteristic function term)
    or the compensating drift correction are accidentally dropped.
    """
    N_small = 64
    domain = 10.0

    merton = Merton(
        domain, N_small, 0.01,
        sigma=sigma, r=r,
        lambda_jump=LAMBDA, mu_jump=MU_J, sigma_jump=SIGMA_J,
    )
    bs = BlackScholes(domain, N_small, 0.01, sigma=sigma, r=r)

    D = build_derivative_operator(1, domain, N_small)
    L_merton = merton._build_linear_operator(D)
    L_bs = bs._build_linear_operator(D)

    kappa = float(jnp.exp(MU_J + 0.5 * SIGMA_J**2) - 1.0)
    k = (D / 1j).real
    phi_p = jnp.exp(1j * MU_J * k - 0.5 * SIGMA_J**2 * k**2)
    expected_diff = -LAMBDA * kappa * D - LAMBDA + LAMBDA * phi_p

    assert jnp.allclose(L_merton - L_bs, expected_diff, atol=1e-12)
