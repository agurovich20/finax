from typing import TypeVar

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

C = TypeVar("C")
D = TypeVar("D")
N = TypeVar("N")


def build_wavenumbers(
    num_spatial_dims: int,
    num_points: int,
    *,
    indexing: str = "ij",
) -> Float[Array, "D ... (N//2)+1"]:
    """
    Setup an array containing integer coordinates of wavenumbers associated with
    a "num_spatial_dims"-dimensional rfft (real-valued FFT)
    `jax.numpy.fft.rfftn`.

    **Arguments:**

    - `num_spatial_dims`: The number of spatial dimensions.
    - `num_points`: The number of points in each spatial dimension.
    - `indexing`: The indexing scheme to use for `jax.numpy.meshgrid`.
        Either `"ij"` or `"xy"`. Default is `"ij"`.

    **Returns:**

    - `wavenumbers`: An array of wavenumber integer coordinates, shape
        `(D, ..., (N//2)+1)`.
    """
    right_most_wavenumbers = jnp.fft.rfftfreq(num_points, 1 / num_points)
    other_wavenumbers = jnp.fft.fftfreq(num_points, 1 / num_points)

    wavenumber_list = [
        other_wavenumbers,
    ] * (num_spatial_dims - 1) + [
        right_most_wavenumbers,
    ]

    wavenumbers = jnp.stack(
        jnp.meshgrid(*wavenumber_list, indexing=indexing),
    )

    return wavenumbers


def build_scaled_wavenumbers(
    num_spatial_dims: int,
    domain_extent: float,
    num_points: int,
    *,
    indexing: str = "ij",
) -> Float[Array, "D ... (N//2)+1"]:
    """
    Setup an array containing **scaled** wavenumbers associated with a
    "num_spatial_dims"-dimensional rfft (real-valued FFT) `jax.numpy.fft.rfftn`.
    Scaling is done by `2 * pi / L`.

    **Arguments:**

    - `num_spatial_dims`: The number of spatial dimensions.
    - `domain_extent`: The domain extent.
    - `num_points`: The number of points in each spatial dimension.
    - `indexing`: The indexing scheme to use for `jax.numpy.meshgrid`.
        Either `"ij"` or `"xy"`. Default is `"ij"`.

    **Returns:**

    - `wavenumbers`: An array of wavenumber integer coordinates, shape
        `(D, ..., (N//2)+1)`.

    !!! info
        These correctly scaled wavenumbers are used to set up derivative
        operators via `1j * wavenumbers`.
    """
    scale = 2 * jnp.pi / domain_extent
    wavenumbers = build_wavenumbers(num_spatial_dims, num_points, indexing=indexing)
    return scale * wavenumbers


def build_derivative_operator(
    num_spatial_dims: int,
    domain_extent: float,
    num_points: int,
    *,
    indexing: str = "ij",
) -> Complex[Array, "D ... (N//2)+1"]:
    """
    Setup the derivative operator in Fourier space.

    **Arguments:**

    - `num_spatial_dims`: The number of spatial dimensions `d`.
    - `domain_extent`: The size of the domain `L`; in higher dimensions
        the domain is assumed to be a scaled hypercube `Ω = (0, L)ᵈ`.
    - `num_points`: The number of points `N` used to discretize the
        domain. This **includes** the left boundary point and **excludes** the
        right boundary point. In higher dimensions; the number of points in each
        dimension is the same. Hence, the total number of degrees of freedom is
        `Nᵈ`.
    - `indexing`: The indexing scheme to use for `jax.numpy.meshgrid`.

    **Returns:**

    - `derivative_operator`: The derivative operator in Fourier space
        (complex-valued array)
    """
    return 1j * build_scaled_wavenumbers(
        num_spatial_dims, domain_extent, num_points, indexing=indexing
    )


def space_indices(num_spatial_dims: int) -> tuple[int, ...]:
    """
    Returns the axes indices within a state array that correspond to the spatial
    axes.

    !!! example
        For a 2D field array, the spatial indices are `(-2, -1)`.

    **Arguments:**

    - `num_spatial_dims`: The number of spatial dimensions.

    **Returns:**

    - `indices`: The indices of the spatial axes.
    """
    return tuple(range(-num_spatial_dims, 0))


def spatial_shape(num_spatial_dims: int, num_points: int) -> tuple[int, ...]:
    """
    Returns the shape of a spatial field array (without its leading channel
    axis). This follows the `Exponax` convention that the resolution is
    identical in each dimension.

    !!! example
        For a 2D field array with 64 points in each dimension, the spatial shape
        is `(64, 64)`. For a 3D field array with 32 points in each dimension,
        the spatial shape is `(32, 32, 32)`.

    **Arguments:**

    - `num_spatial_dims`: The number of spatial dimensions.
    - `num_points`: The number of points in each spatial dimension.

    **Returns:**

    - `shape`: The shape of the spatial field array.
    """
    return (num_points,) * num_spatial_dims


def wavenumber_shape(num_spatial_dims: int, num_points: int) -> tuple[int, ...]:
    """
    Returns the spatial shape of a field in Fourier space (assuming the usage of
    `exponax.fft` which internally performs a real-valued fft
    `jax.numpy.fft.rfftn`).

    !!! example
        For a 2D field array with 64 points in each dimension, the wavenumber shape
        is `(64, 33)`. For a 3D field array with 32 points in each dimension,
        the spatial shape is `(32, 32, 17)`. For a 1D field array with 51 points,
        the wavenumber shape is `(26,)`.

    **Arguments:**

    - `num_spatial_dims`: The number of spatial dimensions.
    - `num_points`: The number of points in each spatial dimension.

    **Returns:**

    - `shape`: The shape of the spatial axes of a state array in Fourier space.
    """
    return (num_points,) * (num_spatial_dims - 1) + (num_points // 2 + 1,)


def fft(
    field: Float[Array, "C ... N"],
    *,
    num_spatial_dims: int | None = None,
) -> Complex[Array, "C ... (N//2)+1"]:
    """
    Perform a **real-valued** FFT of a field. This function is designed for
    states in `Exponax` with a leading channel axis and then one, two, or three
    subsequent spatial axes, **each of the same length** N.

    Only accepts real-valued input fields and performs a real-valued FFT. Hence,
    the last axis of the returned field is of length N//2+1.

    !!! warning
        The argument `num_spatial_dims` can only be correctly inferred if the
        array follows the Exponax convention, e.g., no leading batch axis. For a
        batched operation, use `jax.vmap` on this function.

    **Arguments:**

    - `field`: The state to transform.
    - `num_spatial_dims`: The number of spatial dimensions, i.e., how many
        spatial axes follow the channel axis. Can be inferred from the array if
        it follows the Exponax convention. For example, it is not allowed to
        have a leading batch axis, in such a case use `jax.vmap` on this
        function.

    **Returns:**

    - `field_hat`: The transformed field, shape `(C, ..., N//2+1)`.

    !!! info
        Internally uses `jax.numpy.fft.rfftn` with the default settings for the
        `norm` argument with `norm="backward"`. This means that the forward FFT
        (this function) does not apply any normalization to the result, only the
        [`exponax.ifft`][] function applies normalization. To extract the
        amplitude of the coefficients divide by
        `exponax.spectral.build_scaling_array`.
    """
    if num_spatial_dims is None:
        num_spatial_dims = field.ndim - 1

    return jnp.fft.rfftn(field, axes=space_indices(num_spatial_dims))


def ifft(
    field_hat: Complex[Array, "C ... (N//2)+1"],
    *,
    num_spatial_dims: int | None = None,
    num_points: int | None = None,
) -> Float[Array, "C ... N"]:
    """
    Perform the inverse **real-valued** FFT of a field. This is the inverse
    operation of `exponax.fft`. This function is designed for states in
    `Exponax` with a leading channel axis and then one, two, or three following
    spatial axes. In state space all spatial axes have the same length N (here
    called `num_points`).

    Requires a complex-valued field in Fourier space with the last axis of
    length N//2+1.

    !!! info
        The number of points (N, or `num_points`) must be provided if the number
        of spatial dimensions is 1. Otherwise, it can be inferred from the shape
        of the field.

    !!! warning
        The argument `num_spatial_dims` can only be correctly inferred if the
        array follows the Exponax convention, e.g., no leading batch axis. For a
        batched operation, use `jax.vmap` on this function.

    **Arguments:**

    - `field_hat`: The transformed field, shape `(C, ..., N//2+1)`.
    - `num_spatial_dims`: The number of spatial dimensions, i.e., how many
        spatial axes follow the channel axis. Can be inferred from the array if
        it follows the Exponax convention. For example, it is not allowed to
        have a leading batch axis, in such a case use `jax.vmap` on this
        function.
    - `num_points`: The number of points in each spatial dimension. Can be
        inferred if `num_spatial_dims` >= 2

    **Returns:**

    - `field`: The state in physical space, shape `(C, ..., N,)`.

    !!! info
        Internally uses `jax.numpy.fft.irfftn` with the default settings for the
        `norm` argument with `norm="backward"`. This means that the forward FFT
        [`exponax.fft`][] function does not apply any normalization to the
        input, only the inverse FFT (this function) applies normalization.
        Hence, if you want to define a state in Fourier space and inversely
        transform it, consider using [`exponax.spectral.build_scaling_array`][]
        to correctly scale the complex values before transforming them back.
    """
    if num_spatial_dims is None:
        num_spatial_dims = field_hat.ndim - 1

    if num_points is None:
        if num_spatial_dims >= 2:
            num_points = field_hat.shape[-2]
        else:
            raise ValueError("num_points must be provided if num_spatial_dims == 1.")
    return jnp.fft.irfftn(
        field_hat,
        s=spatial_shape(num_spatial_dims, num_points),
        axes=space_indices(num_spatial_dims),
    )
