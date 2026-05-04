import jax.numpy as jnp

from finonax import Merton, delta, gamma, rho, theta, vega
from finonax.analytical import merton_call_price

S0 = 100.0
K = 100.0
r_val = 0.05
sigma_val = 0.2
T_val = 1.0
lambda_jump = 1.0
mu_jump = -0.1
sigma_jump = 0.15

N = 2048
num_steps = 200
x_half_extent = 3.0
domain_extent = 2 * x_half_extent

ATOL_DELTA = 5e-3
ATOL_GAMMA = 1e-3
ATOL_VEGA = 5e-3
ATOL_RHO = 5e-3
ATOL_THETA = 5e-3


def _make_grid():
    x_grid = jnp.linspace(
        jnp.log(S0) - x_half_extent,
        jnp.log(S0) + x_half_extent,
        N, endpoint=False,
    )
    S_grid = jnp.exp(x_grid)
    i_atm = int(jnp.argmin(jnp.abs(S_grid - S0)))
    return S_grid, i_atm


def _make_stepper(sig, r, T):
    return Merton(
        domain_extent=domain_extent,
        num_points=N,
        dtau=T / num_steps,
        sigma=sig,
        r=r,
        lambda_jump=lambda_jump,
        mu_jump=mu_jump,
        sigma_jump=sigma_jump,
    )


def _priced_V(S_grid):
    stepper = _make_stepper(sigma_val, r_val, T_val)
    payoff = lambda S: jnp.maximum(S - K, 0.0)
    V = stepper.price(payoff, S_grid, num_steps)
    return stepper, V


def _mcp(*args):
    return float(merton_call_price(*args, lambda_jump, mu_jump, sigma_jump))


def test_merton_delta_matches_fd():
    S_grid, i_atm = _make_grid()
    stepper, V = _priced_V(S_grid)

    h = 1.0
    delta_fd = (_mcp(S0 + h, K, r_val, sigma_val, T_val) -
                _mcp(S0 - h, K, r_val, sigma_val, T_val)) / (2 * h)

    result = delta(stepper, V, S_grid, i_atm)

    assert abs(result - delta_fd) < ATOL_DELTA, (
        f"delta: finonax={result:.6f}, fd={delta_fd:.6f}, "
        f"error={abs(result - delta_fd):.2e}"
    )


def test_merton_gamma_matches_fd():
    S_grid, i_atm = _make_grid()
    stepper, V = _priced_V(S_grid)

    h = 1.0
    gamma_fd = (
        _mcp(S0 + h, K, r_val, sigma_val, T_val)
        - 2 * _mcp(S0, K, r_val, sigma_val, T_val)
        + _mcp(S0 - h, K, r_val, sigma_val, T_val)
    ) / h ** 2

    result = gamma(stepper, V, S_grid, i_atm)

    assert abs(result - gamma_fd) < ATOL_GAMMA, (
        f"gamma: finonax={result:.6f}, fd={gamma_fd:.6f}, "
        f"error={abs(result - gamma_fd):.2e}"
    )


def test_merton_vega_matches_fd():
    S_grid, i_atm = _make_grid()
    payoff = lambda S: jnp.maximum(S - K, 0.0)

    h = 1e-3
    vega_fd = (_mcp(S0, K, r_val, sigma_val + h, T_val) -
               _mcp(S0, K, r_val, sigma_val - h, T_val)) / (2 * h)

    result = vega(_make_stepper, payoff, S_grid, num_steps, i_atm,
                  sigma=sigma_val, r=r_val, T=T_val)

    assert abs(result - vega_fd) < ATOL_VEGA, (
        f"vega: finonax={result:.6f}, fd={vega_fd:.6f}, "
        f"error={abs(result - vega_fd):.2e}"
    )


def test_merton_rho_matches_fd():
    S_grid, i_atm = _make_grid()
    payoff = lambda S: jnp.maximum(S - K, 0.0)

    h = 1e-3
    rho_fd = (_mcp(S0, K, r_val + h, sigma_val, T_val) -
              _mcp(S0, K, r_val - h, sigma_val, T_val)) / (2 * h)

    result = rho(_make_stepper, payoff, S_grid, num_steps, i_atm,
                 sigma=sigma_val, r=r_val, T=T_val)

    assert abs(result - rho_fd) < ATOL_RHO, (
        f"rho: finonax={result:.6f}, fd={rho_fd:.6f}, "
        f"error={abs(result - rho_fd):.2e}"
    )


def test_merton_theta_matches_fd():
    S_grid, i_atm = _make_grid()
    payoff = lambda S: jnp.maximum(S - K, 0.0)

    h = 1e-3
    theta_fd = -(_mcp(S0, K, r_val, sigma_val, T_val + h) -
                 _mcp(S0, K, r_val, sigma_val, T_val - h)) / (2 * h)

    result = theta(_make_stepper, payoff, S_grid, num_steps, i_atm,
                   sigma=sigma_val, r=r_val, T=T_val)

    assert abs(result - theta_fd) < ATOL_THETA, (
        f"theta: finonax={result:.6f}, fd={theta_fd:.6f}, "
        f"error={abs(result - theta_fd):.2e}"
    )
