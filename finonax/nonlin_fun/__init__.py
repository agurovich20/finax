from abc import ABC, abstractmethod

import equinox as eqx
from jaxtyping import Array, Complex


class BaseNonlinearFun(eqx.Module, ABC):
    num_spatial_dims: int
    num_points: int
    dealiasing_fraction: float

    def __init__(
        self,
        num_spatial_dims: int,
        num_points: int,
        *,
        dealiasing_fraction: float,
    ):
        self.num_spatial_dims = num_spatial_dims
        self.num_points = num_points
        self.dealiasing_fraction = dealiasing_fraction

    @abstractmethod
    def __call__(
        self,
        u_hat: Complex[Array, "C ... (N//2)+1"],
    ) -> Complex[Array, "C ... (N//2)+1"]:
        pass
