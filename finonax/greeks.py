import jax
import jax.numpy as jnp

from finonax._base_stepper import BackwardStepper
from finonax._spectral import build_derivative_operator, fft, ifft
from finonax.nonlin_fun import BaseNonlinearFun


# ---- Private helpers for autodiff Greeks ----
#
# BlackScholes.__init__ contains `if sigma <= 0: raise ValueError(...)`.
# When JAX traces through a grad computation, sigma is an abstract tracer;
# Python `if` on a tracer raises ConcretizationTypeError. _BSForAD is
# identical to BlackScholes minus that check, used only inside vega/rho/theta.

class _ZeroNonlin(BaseNonlinearFun):
    def __call__(self, u_hat):
        return jnp.zeros_like(u_hat)


class _BSForAD(BackwardStepper):
    sigma: float
    r: float

    def __init__(self, domain_extent, num_points, dtau, *, sigma, r):
        self.sigma = sigma
        self.r = r
        super().__init__(
            num_spatial_dims=1,
            domain_extent=domain_extent,
            num_points=num_points,
            dtau=dtau,
            num_channels=1,
            order=0,
        )

    def _build_linear_operator(self, D):
        return (
            0.5 * self.sigma**2 * D**2
            + (self.r - 0.5 * self.sigma**2) * D
            - self.r
        )

    def _build_nonlinear_fun(self, D):
        return _ZeroNonlin(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            dealiasing_fraction=1.0,
        )

    def price(self, payoff_fn, S_grid, num_steps):
        V_init = payoff_fn(S_grid)[None, :]

        def body(V, _):
            return self(V), None

        V_final, _ = jax.lax.scan(body, V_init, None, length=num_steps)
        return V_final[0]


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
      - stepper_factory: callable (sigma, r, T) -> BackwardStepper. Called
        once with concrete values to validate parameters and extract grid
        metadata (domain_extent, num_points).
      - payoff_fn: callable S -> V at τ=0.
      - S_grid: 1D array of spot prices, shape (num_points,).
      - num_steps: Python int, number of forward-τ steps.
      - i: integer index into S_grid.
      - sigma, r, T: keyword-only floats specifying the model parameters.

    Returns:
      - vega: scalar, ∂V/∂σ at S_grid[i].
    """
    _ref = stepper_factory(sigma, r, T)
    domain_extent, N = _ref.domain_extent, _ref.num_points

    def price_at_i(s):
        stepper = _BSForAD(domain_extent, N, T / num_steps, sigma=s, r=r)
        return stepper.price(payoff_fn, S_grid, num_steps)[i]

    return float(jax.grad(price_at_i)(float(sigma)))


def rho(stepper_factory, payoff_fn, S_grid, num_steps, i, *, sigma, r, T):
    """Compute ∂V/∂r via autodiff through the PDE stepper.

    Arguments: same as vega, differentiating with respect to r.

    Returns:
      - rho: scalar, ∂V/∂r at S_grid[i].
    """
    _ref = stepper_factory(sigma, r, T)
    domain_extent, N = _ref.domain_extent, _ref.num_points

    def price_at_i(r_val):
        stepper = _BSForAD(domain_extent, N, T / num_steps, sigma=sigma, r=r_val)
        return stepper.price(payoff_fn, S_grid, num_steps)[i]

    return float(jax.grad(price_at_i)(float(r)))


def theta(stepper_factory, payoff_fn, S_grid, num_steps, i, *, sigma, r, T):
    """Compute -∂V/∂T via autodiff through the PDE stepper.

    Returns theta in the trader convention: theta = -∂V/∂T (time decay,
    negative for long options), matching the convention in finonax.analytical.

    Arguments: same as vega, differentiating with respect to T.

    Returns:
      - theta: scalar, -∂V/∂T at S_grid[i].
    """
    _ref = stepper_factory(sigma, r, T)
    domain_extent, N = _ref.domain_extent, _ref.num_points

    def price_at_i(T_val):
        stepper = _BSForAD(domain_extent, N, T_val / num_steps, sigma=sigma, r=r)
        return stepper.price(payoff_fn, S_grid, num_steps)[i]

    return float(-jax.grad(price_at_i)(float(T)))
