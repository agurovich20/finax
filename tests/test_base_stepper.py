import pytest
import jax.numpy as jnp

from finonax import BackwardStepper
from finonax.nonlin_fun import BaseNonlinearFun


class ZeroNonlin(BaseNonlinearFun):
    def __call__(self, u_hat):
        return jnp.zeros_like(u_hat)


class TrivialDecay(BackwardStepper):
    decay_rate: float

    def __init__(
        self,
        num_spatial_dims: int,
        domain_extent: float,
        num_points: int,
        dtau: float,
        *,
        decay_rate: float,
    ):
        self.decay_rate = decay_rate
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            domain_extent=domain_extent,
            num_points=num_points,
            dtau=dtau,
            num_channels=1,
            order=0,
        )

    def _build_linear_operator(self, derivative_operator):
        shape = (1,) + derivative_operator.shape[1:]
        return jnp.full(shape, -self.decay_rate + 0j, dtype=jnp.complex128)

    def _build_nonlinear_fun(self, derivative_operator):
        return ZeroNonlin(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            dealiasing_fraction=1.0,
        )


def test_trivial_decay_matches_analytical():
    N = 64
    L_domain = 2 * jnp.pi
    dtau = 0.01
    decay_rate = 1.0

    x = jnp.linspace(0, L_domain, N, endpoint=False)
    V_0 = jnp.cos(x)[None, :]

    stepper = TrivialDecay(1, L_domain, N, dtau, decay_rate=decay_rate)

    V = V_0
    for _ in range(100):
        V = stepper(V)

    V_analytical = V_0 * jnp.exp(-1.0)
    assert jnp.allclose(V, V_analytical, atol=1e-6)


def test_dtau_attribute_is_set():
    stepper = TrivialDecay(1, 1.0, 32, dtau=0.05, decay_rate=0.5)
    assert stepper.dtau == 0.05
    with pytest.raises(AttributeError):
        _ = stepper.dt


def test_shape_validation_raises():
    stepper = TrivialDecay(1, 1.0, 64, dtau=0.01, decay_rate=1.0)
    with pytest.raises(ValueError):
        stepper(jnp.ones((1, 32)))
    with pytest.raises(ValueError):
        stepper(jnp.ones((2, 64)))


def test_num_spatial_dims_validation():
    with pytest.raises(ValueError, match="ROADMAP.md"):
        TrivialDecay(3, 1.0, 32, dtau=0.01, decay_rate=1.0)
