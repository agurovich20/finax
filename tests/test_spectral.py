import numpy as np
import jax.numpy as jnp

from finax._spectral import (
    build_derivative_operator,
    fft,
    ifft,
    wavenumber_shape,
)


def test_fft_ifft_roundtrip():
    rng = np.random.default_rng(0)
    for num_spatial_dims in [1, 2]:
        N = 32
        shape = (1,) + (N,) * num_spatial_dims
        field = jnp.array(rng.standard_normal(shape))
        field_back = ifft(
            fft(field, num_spatial_dims=num_spatial_dims),
            num_spatial_dims=num_spatial_dims,
            num_points=N,
        )
        assert jnp.allclose(field, field_back, atol=1e-6), (
            f"Roundtrip failed for num_spatial_dims={num_spatial_dims}"
        )


def test_derivative_of_sine_1d():
    N = 64
    L = 2 * float(jnp.pi)
    x = jnp.linspace(0, L, N, endpoint=False)
    u = jnp.sin(3 * x)[None, :]  # shape (1, N)

    der_op = build_derivative_operator(1, L, N)  # shape (1, N//2+1)
    u_hat = fft(u, num_spatial_dims=1)
    u_der_hat = der_op * u_hat
    u_der = ifft(u_der_hat, num_spatial_dims=1, num_points=N)

    expected = 3 * jnp.cos(3 * x)[None, :]
    assert jnp.allclose(u_der, expected, atol=1e-4)


def test_wavenumber_shape_1d():
    assert wavenumber_shape(1, 32) == (17,)


def test_build_derivative_operator_shape_1d():
    op = build_derivative_operator(1, 1.0, 32)
    assert op.shape == (1, 17)
    assert op.dtype.kind == "c"


def test_derivative_operator_values():
    L = 2 * float(jnp.pi)
    op = build_derivative_operator(1, L, 8)  # shape (1, 5)
    expected = 1j * jnp.array([0, 1, 2, 3, 4], dtype=jnp.float64)
    assert jnp.allclose(op[0], expected, atol=1e-12)
