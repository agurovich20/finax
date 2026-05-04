import jax.numpy as jnp

from finonax.etdrk import ETDRK0
from finonax.etdrk._utils import roots_of_unity


def test_etdrk0_pure_exponential_decay():
    N = 64
    dt = 0.01
    linear_operator = jnp.full((1, N // 2 + 1), -1.0 + 0j)
    stepper = ETDRK0(dt=dt, linear_operator=linear_operator)

    u_hat = jnp.ones((1, N // 2 + 1), dtype=jnp.complex128)

    # One step: exp(-0.01)
    u_hat = stepper.step_fourier(u_hat)
    expected_1 = jnp.ones((1, N // 2 + 1), dtype=jnp.complex128) * jnp.exp(-0.01)
    assert jnp.allclose(u_hat, expected_1, atol=1e-12)

    # 99 more steps (100 total): exp(-1.0)
    for _ in range(99):
        u_hat = stepper.step_fourier(u_hat)
    expected_100 = jnp.ones((1, N // 2 + 1), dtype=jnp.complex128) * jnp.exp(-1.0)
    assert jnp.allclose(u_hat, expected_100, atol=1e-10)


def test_roots_of_unity_on_circle():
    roots = roots_of_unity(16)

    # All roots must lie on the unit circle.
    assert jnp.allclose(jnp.abs(roots), jnp.ones(16), atol=1e-14)

    # None may be purely real (imaginary part must be nonzero for every root).
    assert jnp.min(jnp.abs(roots.imag)) > 0

    # M-th roots of unity sum to zero (defining cancellation property).
    assert jnp.allclose(jnp.sum(roots), 0.0 + 0.0j, atol=1e-14)

    # Angular span must exceed π — catches half-circle clustering.
    angles = jnp.angle(roots)
    assert jnp.max(angles) - jnp.min(angles) > jnp.pi
