import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex

from finonax._base_stepper import BackwardStepper
from finonax.nonlin_fun import BaseNonlinearFun


class _ZeroNonlin(BaseNonlinearFun):
    def __call__(self, u_hat):
        return jnp.zeros_like(u_hat)


class Merton(BackwardStepper):
    """
    Spectral solver for the log-transformed Merton jump-diffusion PIDE,
    evolving forward in time-to-maturity τ = T − t from a terminal payoff.

    **The PIDE**

    Merton's model adds a compound Poisson jump process to the log-price:

        dS/S = (r − λκ) dt + σ dW + (J − 1) dN

    where N is Poisson with intensity λ, J is a log-normal jump multiplier
    with log-mean μ_J and log-std σ_J, and κ = exp(μ_J + σ_J²/2) − 1 is
    the compensating drift that keeps the discounted price a martingale.

    In log-price coordinates x = log(S) and τ = T − t, the PIDE is:

        ∂V/∂τ = (1/2)σ² ∂²V/∂x²
               + (r − σ²/2 − λκ) ∂V/∂x
               − (r + λ) V
               + λ ∫ V(x + y) p(y) dy

    where p(y) = N(μ_J, σ_J²) is the log-jump density and the integral is
    a convolution in x.

    **Fourier-space linear operator**

    The Fourier transform of the convolution ∫ V(x + y) p(y) dy is
    φ_p(k) · V̂(k), where the characteristic function of p is:

        φ_p(k) = exp(i μ_J k − σ_J² k²/2)

    The full linear operator in Fourier space (using D = ik_scaled) is:

        𝓛(k) = (1/2)σ² D² + (r − σ²/2 − λκ) D − (r + λ) + λ φ_p(k)

    Despite being derived from a PIDE with a non-local integral operator,
    𝓛 is **diagonal** in the Fourier basis. ETDRK0 therefore remains
    *exact in time* for this operator, for the same reason as
    Black-Scholes — the entire dynamics, including the jump term, fold
    into a single diagonal exponential exp(𝓛 Δτ) per step.

    **Parameter conventions**

    - λ is jump intensity in jumps per year.
    - μ_J and σ_J are the mean and standard deviation of the
      log-jump size distribution N(μ_J, σ_J²). They are NOT the mean
      and std of the jump multiplier J itself.
    - Setting λ=0 recovers Black-Scholes exactly.

    **Caveats**

    The same payoff-kink (Gibbs) and periodic-domain boundary
    caveats as BlackScholes apply here. See BlackScholes docstring.
    Accuracy at N=2048 is ~1e-3 for ATM options under the canonical
    Merton parameters. The jump term does not introduce additional
    spectral error beyond the payoff kink.
    """

    sigma: float
    r: float
    lambda_jump: float
    mu_jump: float
    sigma_jump: float

    def __init__(
        self,
        domain_extent: float,
        num_points: int,
        dtau: float,
        *,
        sigma: float,
        r: float,
        lambda_jump: float,
        mu_jump: float,
        sigma_jump: float,
        order: int = 0,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if lambda_jump < 0:
            raise ValueError(f"lambda_jump must be non-negative, got {lambda_jump}")
        if sigma_jump <= 0:
            raise ValueError(f"sigma_jump must be positive, got {sigma_jump}")

        self.sigma = sigma
        self.r = r
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump

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
        # Real scaled wavenumbers: D = i·k_scaled, so k_scaled = D / i
        k = (D / 1j).real
        kappa = jnp.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1.0
        # Characteristic function of the log-jump density N(mu_jump, sigma_jump²)
        phi_p = jnp.exp(1j * self.mu_jump * k - 0.5 * self.sigma_jump**2 * k**2)
        return (
            0.5 * self.sigma**2 * D**2
            + (self.r - 0.5 * self.sigma**2 - self.lambda_jump * kappa) * D
            - (self.r + self.lambda_jump)
            + self.lambda_jump * phi_p
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

        Arguments and return value are identical to BlackScholes.price.
        See that method's docstring for full documentation.
        """
        V_init = payoff_fn(S_grid)[None, :]

        def body(V, _):
            return self(V), None

        V_final, _ = jax.lax.scan(body, V_init, None, length=num_steps)
        return V_final[0]
