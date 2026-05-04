import jax
import jax.numpy as jnp

from finonax._spectral import build_derivative_operator, fft, ifft


# ---- Spectral Greeks (delta, gamma) ----

def delta(stepper, V, S_grid, i):
    """Compute ∂V/∂S at index i using spectral differentiation in log-price.

    Arguments:
      - stepper: a BackwardStepper instance. Only its spatial metadata
        (domain_extent, num_points) is used.
      - V: 1D array of option values, shape (num_points,).
      - S_grid: 1D array of spot prices, shape (num_points,).
      - i: integer index into S_grid at which to evaluate delta.

    Returns:
      - delta: scalar, ∂V/∂S at S_grid[i].
    """
    V_hat = fft(V[None, :], num_spatial_dims=1)
    D = build_derivative_operator(1, stepper.domain_extent, stepper.num_points)
    dVdx = ifft(D * V_hat, num_spatial_dims=1, num_points=stepper.num_points)
    return float(dVdx[0, i] / S_grid[i])


def gamma(stepper, V, S_grid, i):
    """Compute ∂²V/∂S² at index i using spectral differentiation in log-price.

    Arguments:
      - stepper: a BackwardStepper instance (spatial metadata only).
      - V: 1D array of option values, shape (num_points,).
      - S_grid: 1D array of spot prices, shape (num_points,).
      - i: integer index into S_grid at which to evaluate gamma.

    Returns:
      - gamma: scalar, ∂²V/∂S² at S_grid[i].
    """
    V_hat = fft(V[None, :], num_spatial_dims=1)
    D = build_derivative_operator(1, stepper.domain_extent, stepper.num_points)
    dVdx_hat = D * V_hat
    d2Vdx2_hat = D * dVdx_hat
    dVdx = ifft(dVdx_hat, num_spatial_dims=1, num_points=stepper.num_points)
    d2Vdx2 = ifft(d2Vdx2_hat, num_spatial_dims=1, num_points=stepper.num_points)
    S = S_grid[i]
    # Chain rule: ∂²V/∂S² = (1/S²)(∂²V/∂x² - ∂V/∂x)
    return float((d2Vdx2[0, i] - dVdx[0, i]) / (S ** 2))


# ---- Autodiff Greeks (vega, rho, theta) ----

def vega(stepper_factory, payoff_fn, S_grid, num_steps, i, *, sigma, r, T):
    """Compute ∂V/∂σ via autodiff through the PDE stepper.

    Arguments:
      - stepper_factory: callable (sigma, r, T) -> BackwardStepper. Works with
        any BackwardStepper subclass (BlackScholes, Merton, etc.). Called once
        with concrete values to validate parameters eagerly. Any model
        parameters beyond sigma and r (e.g., lambda_jump, mu_jump, sigma_jump
        for Merton) must be closed over by the factory.
      - payoff_fn: callable S -> V at τ=0.
      - S_grid: 1D array of spot prices, shape (num_points,).
      - num_steps: Python int, number of forward-τ steps.
      - i: integer index into S_grid.
      - sigma, r, T: keyword-only floats specifying the differentiable model
        parameters.

    Returns:
      - vega: scalar, ∂V/∂σ at S_grid[i].
    """
    stepper_factory(sigma, r, T)

    def price_at_i(s):
        stepper = stepper_factory(s, r, T)
        return stepper.price(payoff_fn, S_grid, num_steps)[i]

    return float(jax.grad(price_at_i)(float(sigma)))


def rho(stepper_factory, payoff_fn, S_grid, num_steps, i, *, sigma, r, T):
    """Compute ∂V/∂r via autodiff through the PDE stepper.

    Arguments: same as vega; differentiates with respect to r. Any model
    parameters beyond sigma and r must be closed over by the factory.

    Returns:
      - rho: scalar, ∂V/∂r at S_grid[i].
    """
    stepper_factory(sigma, r, T)

    def price_at_i(r_val):
        stepper = stepper_factory(sigma, r_val, T)
        return stepper.price(payoff_fn, S_grid, num_steps)[i]

    return float(jax.grad(price_at_i)(float(r)))


def theta(stepper_factory, payoff_fn, S_grid, num_steps, i, *, sigma, r, T):
    """Compute -∂V/∂T via autodiff through the PDE stepper.

    Returns theta in the trader convention: theta = -∂V/∂T (time decay,
    negative for long options), matching the convention in finonax.analytical.

    Arguments: same as vega; differentiates with respect to T. Any model
    parameters beyond sigma and r must be closed over by the factory.

    Returns:
      - theta: scalar, -∂V/∂T at S_grid[i].
    """
    stepper_factory(sigma, r, T)

    def price_at_i(T_val):
        stepper = stepper_factory(sigma, r, T_val)
        return stepper.price(payoff_fn, S_grid, num_steps)[i]

    return float(-jax.grad(price_at_i)(float(T)))
