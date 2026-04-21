import jax.numpy as jnp

from finax import BlackScholes, delta, gamma, rho, theta, vega
from finax.analytical import (
    bs_call_delta,
    bs_call_rho,
    bs_call_theta,
    bs_gamma,
    bs_vega,
)

# Canonical parameters shared across all five tests
S0 = 100.0
K = 100.0
r_val = 0.05
sigma_val = 0.2
T_val = 1.0
N = 2048
num_steps = 200
x_half_extent = 3.0
domain_extent = 2 * x_half_extent

ATOL = 5e-4
# Vega is O(38), ~60x larger than delta; absolute error scales with magnitude.
# Measured error at N=2048: 6.4e-4, so 5e-4 is too tight; use 1e-3.
ATOL_VEGA = 1e-3


def _make_grid():
    x_center = jnp.log(S0)
    x_grid = jnp.linspace(
        x_center - x_half_extent,
        x_center + x_half_extent,
        N, endpoint=False,
    )
    S_grid = jnp.exp(x_grid)
    i_atm = int(jnp.argmin(jnp.abs(S_grid - S0)))
    return S_grid, i_atm


def _make_stepper_factory():
    def make_stepper(sigma, r, T):
        return BlackScholes(
            domain_extent=domain_extent,
            num_points=N,
            dtau=T / num_steps,
            sigma=sigma,
            r=r,
        )
    return make_stepper


def _priced_V(S_grid):
    stepper = BlackScholes(
        domain_extent=domain_extent,
        num_points=N,
        dtau=T_val / num_steps,
        sigma=sigma_val,
        r=r_val,
    )
    return stepper, stepper.price(
        lambda S: jnp.maximum(S - K, 0.0), S_grid, num_steps
    )


def test_delta_matches_closed_form():
    S_grid, i_atm = _make_grid()
    stepper, V = _priced_V(S_grid)

    finax_delta = delta(stepper, V, S_grid, i_atm)
    analytical = float(bs_call_delta(S0, K, r_val, sigma_val, T_val))

    assert abs(finax_delta - analytical) < ATOL, (
        f"delta: finax={finax_delta:.6f}, analytical={analytical:.6f}, "
        f"error={abs(finax_delta - analytical):.2e}"
    )


def test_gamma_matches_closed_form():
    S_grid, i_atm = _make_grid()
    stepper, V = _priced_V(S_grid)

    finax_gamma = gamma(stepper, V, S_grid, i_atm)
    analytical = float(bs_gamma(S0, K, r_val, sigma_val, T_val))

    assert abs(finax_gamma - analytical) < ATOL, (
        f"gamma: finax={finax_gamma:.6f}, analytical={analytical:.6f}, "
        f"error={abs(finax_gamma - analytical):.2e}"
    )


def test_vega_matches_closed_form():
    S_grid, i_atm = _make_grid()
    payoff = lambda S: jnp.maximum(S - K, 0.0)

    finax_vega = vega(
        _make_stepper_factory(), payoff, S_grid, num_steps, i_atm,
        sigma=sigma_val, r=r_val, T=T_val,
    )
    analytical = float(bs_vega(S0, K, r_val, sigma_val, T_val))

    assert abs(finax_vega - analytical) < ATOL_VEGA, (
        f"vega: finax={finax_vega:.6f}, analytical={analytical:.6f}, "
        f"error={abs(finax_vega - analytical):.2e}"
    )


def test_rho_matches_closed_form():
    S_grid, i_atm = _make_grid()
    payoff = lambda S: jnp.maximum(S - K, 0.0)

    finax_rho = rho(
        _make_stepper_factory(), payoff, S_grid, num_steps, i_atm,
        sigma=sigma_val, r=r_val, T=T_val,
    )
    analytical = float(bs_call_rho(S0, K, r_val, sigma_val, T_val))

    assert abs(finax_rho - analytical) < ATOL, (
        f"rho: finax={finax_rho:.6f}, analytical={analytical:.6f}, "
        f"error={abs(finax_rho - analytical):.2e}"
    )


def test_theta_matches_closed_form():
    S_grid, i_atm = _make_grid()
    payoff = lambda S: jnp.maximum(S - K, 0.0)

    finax_theta = theta(
        _make_stepper_factory(), payoff, S_grid, num_steps, i_atm,
        sigma=sigma_val, r=r_val, T=T_val,
    )
    analytical = float(bs_call_theta(S0, K, r_val, sigma_val, T_val))

    assert abs(finax_theta - analytical) < ATOL, (
        f"theta: finax={finax_theta:.6f}, analytical={analytical:.6f}, "
        f"error={abs(finax_theta - analytical):.2e}"
    )
