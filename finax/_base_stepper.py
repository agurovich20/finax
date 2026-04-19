from abc import ABC, abstractmethod

import equinox as eqx
from jaxtyping import Array, Complex, Float

from finax._spectral import (
    build_derivative_operator,
    fft,
    ifft,
    spatial_shape,
    wavenumber_shape,
)
from finax.etdrk import ETDRK0, ETDRK1, ETDRK2, ETDRK3, ETDRK4, BaseETDRK
from finax.nonlin_fun import BaseNonlinearFun


class BackwardStepper(eqx.Module, ABC):
    """
    Base class for backward-in-time timesteppers for finance PDEs of the
    semi-linear form

        V_τ = 𝓛V + 𝒩(V)

    where τ = T − t is time-to-maturity, 𝓛 is a linear differential
    operator, and 𝒩 is a nonlinear differential operator.

    The user supplies the terminal payoff as the initial condition at
    τ=0, and this class advances forward in τ. From the user's
    perspective this means marching backward in calendar time t from
    expiry T to present.

    This sign convention differs from typical physics PDE solvers
    (which march forward in calendar time from an initial condition)
    but the machinery is identical — sign changes are absorbed into
    how subclasses define 𝓛 in `_build_linear_operator`.

    A subclass must implement `_build_linear_operator` (returning the
    Fourier-diagonal linear operator) and `_build_nonlinear_fun`
    (returning a BaseNonlinearFun instance). For purely linear PDEs
    like log-transformed Black-Scholes, pass order=0; the nonlinear
    function is then not used at runtime but must still be returned
    from `_build_nonlinear_fun` — return a trivial BaseNonlinearFun
    subclass whose __call__ returns zeros.

    Save attributes specific to the concrete PDE BEFORE calling the
    parent constructor, because it invokes the abstract methods during
    initialization.

    !!! note "Spatial dimensionality"

        finax's spectral core supports 1D and 2D problems — which
        covers Black-Scholes, Merton jump diffusion, and Heston
        stochastic volatility. Higher-dimensional finance PDEs such
        as basket options on many underlyings are subject to the
        curse of dimensionality and will be addressed in a future
        neural branch of finax (Deep BSDE, PINN, Fourier Neural
        Operators). See ROADMAP.md. Passing num_spatial_dims > 2
        raises ValueError.
    """

    num_spatial_dims: int
    domain_extent: float
    num_points: int
    num_channels: int
    dtau: float
    dx: float

    _integrator: BaseETDRK

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dtau: float,
        *,
        num_channels: int,
        order: int,
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        """
        Initialize a BackwardStepper for a backward-in-time finance PDE
        of the semi-linear form

            V_τ = 𝓛V + 𝒩(V)

        where τ = T − t is time-to-maturity. See the class docstring for
        the conceptual model; this docstring covers the constructor
        arguments only.

        A subclass must implement `_build_linear_operator` (returning
        the Fourier-diagonal linear operator 𝓛) and `_build_nonlinear_fun`
        (returning a BaseNonlinearFun instance). Under Fourier
        pseudo-spectral discretization, 𝓛 is diagonal and `_build_linear_operator`
        returns its diagonal entries; 𝒩 is evaluated in physical space
        via BaseNonlinearFun.

        The implementation uses Exponential Time Differencing Runge-Kutta
        (ETDRK) schemes of order 0 through 4 (Cox & Matthews 2002; Kassam
        & Trefethen 2005).

        Save attributes specific to the concrete PDE before calling this
        parent constructor, because it invokes the abstract methods during
        initialization.

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. Must be 1 or 2;
            finax's spectral core does not support higher-dimensional PDEs (see
            ROADMAP.md for the planned neural branch).
        - `domain_extent`: The size of the domain `L`; in higher dimensions
            the domain is assumed to be a scaled hypercube `Ω = (0, L)ᵈ`.
        - `num_points`: The number of points `N` used to discretize the
            domain. This **includes** the left boundary point and **excludes**
            the right boundary point. In higher dimensions; the number of points
            in each dimension is the same. Hence, the total number of degrees of
            freedom is `Nᵈ`.
        - `dtau`: The time-to-maturity step size Δτ between two consecutive
            states (in years).
        - `num_channels`: The number of channels `C` in the state vector/tensor.
            For most problems, like simple linear PDEs this will be one (because
            the option value field in a Black-Scholes PDE is a scalar field).
            Some other problems like reaction-diffusion equations with multiple
            species will have more than one channel. This information is only
            used to check the shape of the input state vector in the `__call__`
            method. (keyword-only)
        - `order`: The order of the ETDRK method to use. Must be one of {0, 1,
            2, 3, 4}. The option `0` only solves the linear part of the
            equation. Hence, only use this for linear PDEs. For nonlinear PDEs,
            a higher order method tends to be more stable and accurate. `2` is
            often a good compromise in single-precision. Use `4` together with
            double precision (`jax.config.update("jax_enable_x64", True)`) for
            highest accuracy. (keyword-only)
        - `num_circle_points`: How many points to use in the complex contour
            integral method to compute the coefficients of the exponential time
            differencing Runge Kutta method. Default: 16.
        - `circle_radius`: The radius of the contour used to compute the
            coefficients of the exponential time differencing Runge Kutta
            method. Default: 1.0.
        """
        if num_spatial_dims not in (1, 2):
            raise ValueError(
                f"finax's spectral core supports num_spatial_dims in "
                f"{{1, 2}} only, got {num_spatial_dims}. "
                f"See ROADMAP.md for the planned neural branch that "
                f"handles higher-dimensional finance PDEs."
            )

        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent
        self.num_points = num_points
        self.dtau = dtau
        self.num_channels = num_channels

        # Uses the convention that N does **not** include the right boundary
        # point
        self.dx = domain_extent / num_points

        derivative_operator = build_derivative_operator(
            num_spatial_dims, domain_extent, num_points
        )

        linear_operator = self._build_linear_operator(derivative_operator)
        single_channel_shape = (1,) + wavenumber_shape(
            self.num_spatial_dims, self.num_points
        )  # Same operator for each channel (i.e., we broadcast)
        multi_channel_shape = (self.num_channels,) + wavenumber_shape(
            self.num_spatial_dims, self.num_points
        )  # Different operator for each channel
        if linear_operator.shape not in (single_channel_shape, multi_channel_shape):
            raise ValueError(
                f"""Expected linear operator to have shape
                 {single_channel_shape} or {multi_channel_shape}, got
                 {linear_operator.shape}."""
            )
        nonlinear_fun = self._build_nonlinear_fun(derivative_operator)

        if order == 0:
            self._integrator = ETDRK0(
                self.dtau,
                linear_operator,
            )
        elif order == 1:
            self._integrator = ETDRK1(
                self.dtau,
                linear_operator,
                nonlinear_fun,
                num_circle_points=num_circle_points,
                circle_radius=circle_radius,
            )
        elif order == 2:
            self._integrator = ETDRK2(
                self.dtau,
                linear_operator,
                nonlinear_fun,
                num_circle_points=num_circle_points,
                circle_radius=circle_radius,
            )
        elif order == 3:
            self._integrator = ETDRK3(
                self.dtau,
                linear_operator,
                nonlinear_fun,
                num_circle_points=num_circle_points,
                circle_radius=circle_radius,
            )
        elif order == 4:
            self._integrator = ETDRK4(
                self.dtau,
                linear_operator,
                nonlinear_fun,
                num_circle_points=num_circle_points,
                circle_radius=circle_radius,
            )
        else:
            raise NotImplementedError(f"Order {order} not implemented.")

    @abstractmethod
    def _build_linear_operator(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Assemble the L operator in Fourier space.

        **Arguments:**

        - `derivative_operator`: The derivative operator, shape `( D, ...,
            N//2+1 )`. The ellipsis are (D-1) axis of size N (**not** of size
            N//2+1).

        **Returns:**

        - `L`: The linear operator, shape `( C, ..., N//2+1 )`.
        """
        pass

    @abstractmethod
    def _build_nonlinear_fun(
        self,
        derivative_operator: Complex[Array, "D ... (N//2)+1"],
    ) -> BaseNonlinearFun:
        """
        Build the function that evaluates nonlinearity in physical space,
        transforms to Fourier space, and evaluates derivatives there.

        **Arguments:**

        - `derivative_operator`: The derivative operator, shape `( D, ...,
            N//2+1 )`.

        **Returns:**

        - `nonlinear_fun`: A function that evaluates the nonlinearities in
            time space, transforms to Fourier space, and evaluates the
            derivatives there. Should be a subclass of `BaseNonlinearFun`.
        """
        pass

    def step(self, u: Float[Array, "C ... N"]) -> Float[Array, "C ... N"]:
        """
        Perform one step of the time integration.

        **Arguments:**

        - `u`: The state vector, shape `(C, ..., N,)`.

        **Returns:**

        - `u_next`: The state vector after one step, shape `(C, ..., N,)`.
        """
        u_hat = fft(u, num_spatial_dims=self.num_spatial_dims)
        u_next_hat = self.step_fourier(u_hat)
        u_next = ifft(
            u_next_hat,
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
        )
        return u_next

    def step_fourier(
        self, u_hat: Complex[Array, "C ... (N//2)+1"]
    ) -> Complex[Array, "C ... (N//2)+1"]:
        """
        Perform one step of the time integration in Fourier space. Oftentimes,
        this is more efficient than `step` since it avoids back and forth
        transforms.

        **Arguments:**

        - `u_hat`: The (real) Fourier transform of the state vector

        **Returns:**

        - `u_next_hat`: The (real) Fourier transform of the state vector
            after one step
        """
        return self._integrator.step_fourier(u_hat)

    def __call__(
        self,
        u: Float[Array, "C ... N"],
    ) -> Float[Array, "C ... N"]:
        """
        Perform one step of the time integration for a single state.

        **Arguments:**

        - `u`: The state vector, shape `(C, ..., N,)`.

        **Returns:**

        - `u_next`: The state vector after one step, shape `(C, ..., N,)`.

        !!! tip
            Use `jax.lax.scan` over this call to efficiently produce
            temporal trajectories.

        !!! info
            For batched operation, use `jax.vmap` on this function.
        """
        expected_shape = (self.num_channels,) + spatial_shape(
            self.num_spatial_dims, self.num_points
        )
        if u.shape != expected_shape:
            raise ValueError(
                f"""Expected shape {expected_shape}, got {u.shape}. For batched
                 operation use `jax.vmap` on this function."""
            )
        return self.step(u)
