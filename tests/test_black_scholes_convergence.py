import pytest
import jax.numpy as jnp

from finonax import BlackScholes
from finonax.analytical import bs_call_price


@pytest.mark.parametrize(
    "N,expected_max_error",
    [
        (512, 6e-3),   # measured 2.15e-3 at x_half_extent=3.0; 3x margin
        (1024, 2e-3),  # measured 5.37e-4 at x_half_extent=3.0; 4x margin
        (2048, 5e-4),  # measured 1.34e-4 at x_half_extent=3.0; 4x margin
    ],
)
def test_atm_call_convergence(N, expected_max_error):
    """Verify convergence behavior observed during M1.3 is stable.

    If any of these tolerances fail, the error has grown from
    previously-measured values, indicating a regression.
    """
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    x_half_extent = 3.0

    x_center = jnp.log(S0)
    x_grid = jnp.linspace(
        x_center - x_half_extent,
        x_center + x_half_extent,
        N, endpoint=False,
    )
    S_grid = jnp.exp(x_grid)
    num_steps = 200
    dtau = T / num_steps

    stepper = BlackScholes(
        domain_extent=2 * x_half_extent,
        num_points=N,
        dtau=dtau,
        sigma=sigma,
        r=r,
    )
    V = stepper.price(
        lambda S: jnp.maximum(S - K, 0.0),
        S_grid,
        num_steps,
    )
    i_atm = jnp.argmin(jnp.abs(S_grid - S0))
    finax_price = float(V[i_atm])
    analytical_price = float(bs_call_price(S0, K, r, sigma, T))
    error = abs(finax_price - analytical_price)

    assert error < expected_max_error, (
        f"N={N}: error {error:.2e} exceeds "
        f"tolerance {expected_max_error:.2e}. "
        f"finonax={finax_price:.6f}, analytical={analytical_price:.6f}."
    )
