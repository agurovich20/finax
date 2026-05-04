import jax.numpy as jnp

from finonax import merton_call_price, merton_put_price
from finonax.analytical._merton import _merton_series
from finonax.analytical._black_scholes import bs_call_price, bs_put_price

# Canonical Merton parameters (Merton 1976 / textbook)
S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
LAMBDA, MU_J, SIGMA_J = 1.0, -0.1, 0.15

# Reference value computed independently via Python-loop summation at n_max=100
# (pure numpy.float64 arithmetic, no JAX tracing).
# kappa = exp(-0.1 + 0.5*0.15^2) - 1 = exp(-0.08875) - 1 ≈ -0.08493
# lambda_prime = 1 * (1 + kappa) ≈ 0.91507
# The series converges in ~5 significant terms for lambda_prime*T ≈ 0.915.
MERTON_CALL_REF = 12.761288593628757


def test_merton_reduces_to_bs_when_no_jumps():
    """With lambda=0 the Merton series collapses to a single BS term."""
    params = [
        (90.0, 100.0, 0.05, 0.2, 1.0),
        (100.0, 100.0, 0.05, 0.2, 0.5),
        (110.0, 100.0, 0.05, 0.3, 2.0),
        (80.0, 100.0, 0.05, 0.25, 0.25),
    ]
    for S, K, r, sigma, T in params:
        m_call = merton_call_price(S, K, r, sigma, T, 0.0, MU_J, SIGMA_J)
        bs = bs_call_price(S, K, r, sigma, T)
        assert abs(float(m_call) - float(bs)) < 1e-12, (
            f"lambda=0 merton call {float(m_call):.15f} != bs {float(bs):.15f}"
        )


def test_merton_put_call_parity():
    """C - P = S - K·exp(-rT) for Merton put and call."""
    test_cases = [
        (S0, K, r, sigma, T, LAMBDA, MU_J, SIGMA_J),
        (90.0, 100.0, 0.03, 0.25, 0.5, 0.5, -0.05, 0.1),
        (110.0, 100.0, 0.06, 0.15, 2.0, 2.0, 0.0, 0.2),
    ]
    for S, K_, r_, sig, T_, lam, mj, sj in test_cases:
        call = merton_call_price(S, K_, r_, sig, T_, lam, mj, sj)
        put = merton_put_price(S, K_, r_, sig, T_, lam, mj, sj)
        parity = float(call) - float(put)
        expected = float(S - K_ * jnp.exp(-r_ * T_))
        assert abs(parity - expected) < 1e-10, (
            f"Put-call parity violated: C-P={parity:.15f}, S-K*exp(-rT)={expected:.15f}"
        )


def test_merton_reference_value():
    """
    Canonical Merton call price matches the independently computed reference.

    Reference value: 12.761288593628757
    Computed by Python-loop series summation at n_max=100 using math.lgamma
    and math.exp (no JAX), confirming the JAX implementation is correct.
    """
    price = merton_call_price(S0, K, r, sigma, T, LAMBDA, MU_J, SIGMA_J)
    assert abs(float(price) - MERTON_CALL_REF) < 1e-10


def test_merton_series_convergence():
    """n_max=50 vs n_max=100 agree to better than 1e-12 for canonical params."""
    price_50 = _merton_series(S0, K, r, sigma, T, LAMBDA, MU_J, SIGMA_J, 50, bs_call_price)
    price_100 = _merton_series(S0, K, r, sigma, T, LAMBDA, MU_J, SIGMA_J, 100, bs_call_price)
    assert abs(float(price_50) - float(price_100)) < 1e-12
