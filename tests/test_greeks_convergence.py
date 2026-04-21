"""
Parameter-grid Greek accuracy regression tests (M2.3).

Validates all five Greeks across a 5-moneyness x 4-maturity grid.
Measured worst-case relative errors (N=1024, validation/greeks_grid.py):
  delta: 4.07e-04, gamma: 7.19e-04, vega: 7.19e-04,
  rho:   4.00e-04, theta: 2.26e-04

Worst-case overall: 7.19e-04 (gamma/vega, m=0.8, T=0.25).
Uniform relative tolerance 3e-3 (~4x margin on worst case).
Uses relative tolerance throughout; see KNOWN_ISSUES.md for context.
"""
import pytest
import jax.numpy as jnp

from finax import BlackScholes, delta, gamma, vega, rho, theta
from finax.analytical import (
    bs_call_delta,
    bs_gamma,
    bs_vega,
    bs_call_rho,
    bs_call_theta,
)

MONEYNESS_VALUES = [0.8, 0.9, 1.0, 1.1, 1.2]
T_VALUES = [0.25, 0.5, 1.0, 2.0]

S0 = 100.0
r_val = 0.05
sigma_val = 0.2
N = 1024
num_steps = 200
x_half_extent = 3.0
domain_extent = 2 * x_half_extent

RTOL = 3e-3


def _make_grid():
    x_grid = jnp.linspace(
        jnp.log(S0) - x_half_extent,
        jnp.log(S0) + x_half_extent,
        N, endpoint=False,
    )
    S_grid = jnp.exp(x_grid)
    i_s0 = int(jnp.argmin(jnp.abs(S_grid - S0)))
    return S_grid, i_s0


@pytest.mark.parametrize(
    "greek_name",
    ["delta", "gamma", "vega", "rho", "theta"],
)
def test_greek_accuracy_across_parameter_grid(greek_name):
    """Relative error < 3e-3 for all (moneyness, T) combinations."""
    S_grid, i_s0 = _make_grid()

    for T in T_VALUES:
        dtau = T / num_steps
        for m in MONEYNESS_VALUES:
            K = m * S0
            payoff_fn = lambda S, K=K: jnp.maximum(S - K, 0.0)

            stepper = BlackScholes(
                domain_extent, N, dtau, sigma=sigma_val, r=r_val,
            )
            V = stepper.price(payoff_fn, S_grid, num_steps)

            def make_stepper(sig, r, T_arg, K=K):
                return BlackScholes(
                    domain_extent, N, T_arg / num_steps, sigma=sig, r=r,
                )

            if greek_name == "delta":
                finax_val = delta(stepper, V, S_grid, i_s0)
                analytical = float(bs_call_delta(S0, K, r_val, sigma_val, T))
            elif greek_name == "gamma":
                finax_val = gamma(stepper, V, S_grid, i_s0)
                analytical = float(bs_gamma(S0, K, r_val, sigma_val, T))
            elif greek_name == "vega":
                finax_val = vega(
                    make_stepper, payoff_fn, S_grid, num_steps, i_s0,
                    sigma=sigma_val, r=r_val, T=T,
                )
                analytical = float(bs_vega(S0, K, r_val, sigma_val, T))
            elif greek_name == "rho":
                finax_val = rho(
                    make_stepper, payoff_fn, S_grid, num_steps, i_s0,
                    sigma=sigma_val, r=r_val, T=T,
                )
                analytical = float(bs_call_rho(S0, K, r_val, sigma_val, T))
            else:
                finax_val = theta(
                    make_stepper, payoff_fn, S_grid, num_steps, i_s0,
                    sigma=sigma_val, r=r_val, T=T,
                )
                analytical = float(bs_call_theta(S0, K, r_val, sigma_val, T))

            rel_err = abs(finax_val - analytical) / abs(analytical)
            assert rel_err < RTOL, (
                f"{greek_name}: m={m}, T={T}: "
                f"finax={finax_val:.6f}, analytical={analytical:.6f}, "
                f"rel_error={rel_err:.2e} >= {RTOL:.1e}"
            )
