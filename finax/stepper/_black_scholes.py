import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex

from finax._base_stepper import BackwardStepper
from finax.nonlin_fun import BaseNonlinearFun


class _ZeroNonlin(BaseNonlinearFun):
    def __call__(self, u_hat):
        return jnp.zeros_like(u_hat)


class BlackScholes(BackwardStepper):
    """
    Spectral solver for the log-transformed Black-Scholes PDE, evolving
    forward in time-to-maturity τ = T − t from a terminal payoff at τ=0.

    **The PDE**

    Under the risk-neutral measure the Black-Scholes PDE in calendar
    coordinates (t, S) is:

        ∂V/∂t + (1/2)σ²S² ∂²V/∂S² + rS ∂V/∂S − rV = 0,   V(T, S) = payoff(S).

    Substituting x = log(S) and τ = T − t converts this to a
    *constant-coefficient* advection-diffusion-reaction equation:

        ∂V/∂τ = (1/2)σ² ∂²V/∂x² + (r − (1/2)σ²) ∂V/∂x − rV

    The log transform removes the variable-coefficient S² factor and
    maps the semi-infinite domain S ∈ (0, ∞) to x ∈ (−∞, ∞), which
    we truncate to a large periodic window [x_min, x_max].

    **Fourier-space linear operator**

    The Fourier-space linear operator (using D = ik_scaled as the
    derivative operator) is:

        𝓛(k) = (1/2)σ² D² + (r − σ²/2) D − r
              = −(1/2)σ²k² + i(r − σ²/2)k − r

    The imaginary drift term i(r − σ²/2)k is a meaningful departure
    from physics PDEs where 𝓛 is typically purely real.

    **Time integration accuracy**

    With order=0 (default), ETDRK0 computes exp(𝓛 Δτ) exactly in
    Fourier space. There is no time-discretization error for this
    linear PDE — only spatial truncation (finite N) and domain
    truncation (finite window) errors.

    **Payoff-kink caveat (Gibbs phenomenon)**

    Standard option payoffs (calls, puts) have a non-differentiable
    kink at S = K (x = log K). The Fourier series of the initial
    condition oscillates near this kink (Gibbs phenomenon), introducing
    a pointwise error of O(1/N) that does not decrease with smaller Δτ,
    only with larger N or with grid alignment (placing K on a grid
    node). At N = 512, ATM prices are accurate to ~6e-3; at N = 1024,
    ~1.5e-3; at N = 2048, ~4e-4. The dominant error is not the
    kink at S = K but the wrap-around discontinuity of the option
    payoff at the periodic domain boundary, where the call payoff
    jumps from its maximum value back to zero. Accuracy scales as
    O(1/N²) and is improved more by increasing N than by widening
    the domain (a wider domain with fixed N actually degrades
    accuracy, since grid points per unit log-price decrease).
    """

    sigma: float
    r: float

    def __init__(
        self,
        domain_extent: float,
        num_points: int,
        dtau: float,
        *,
        sigma: float,
        r: float,
        order: int = 0,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.sigma = sigma
        self.r = r

        super().__init__(
            num_spatial_dims=1,
            domain_extent=domain_extent,
            num_points=num_points,
            dtau=dtau,
            num_channels=1,
            order=order,
            num_circle_points=num_circle_points,
            circle_radius=circle_radius,
        )

    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "1 (N//2)+1"],
    ) -> Complex[Array, "1 (N//2)+1"]:
        D = derivative_operator
        return (
            0.5 * self.sigma**2 * D**2
            + (self.r - 0.5 * self.sigma**2) * D
            - self.r
        )

    def _build_nonlinear_fun(self, derivative_operator):
        return _ZeroNonlin(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            dealiasing_fraction=1.0,
        )

    def price(self, payoff_fn, S_grid, num_steps):
        """
        Price an option with the given payoff function.

        Evaluates the payoff on S_grid, then evolves forward in τ
        via num_steps stepper applications, compiled as a single
        jax.lax.scan. This means the compiled trace size is
        independent of num_steps, which matters when the method
        is composed with jax.grad or jax.vmap over a parameter
        grid.

        Arguments:
          - payoff_fn: callable S -> V at τ=0 (i.e. the terminal payoff).
            Must accept a 1D array of spot prices and return a 1D array.
          - S_grid: 1D jnp.array of spot prices. Must be a uniform grid
            in log(S) — i.e. S_grid = exp(jnp.linspace(x_min, x_max, N,
            endpoint=False)) consistent with the stepper's domain_extent.
          - num_steps: integer number of forward-τ steps (must be a
            Python int, not a traced JAX value).

        Returns:
          - V: 1D jnp.array of option values on S_grid, shape (num_points,),
            after evolving forward in τ by num_steps * dtau.
        """
        V_init = payoff_fn(S_grid)[None, :]

        def body(V, _):
            return self(V), None

        V_final, _ = jax.lax.scan(body, V_init, None, length=num_steps)
        return V_final[0]
