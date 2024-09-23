"""
kan_rbf.py: KAN with RBF
----------------------------------------------------

* Authors: Quoc Nguyen (minhquoc0712@gmail.com)
* Date: 2024-09-23
* Version: 0.0.1

"""

from typing import List, Optional

from einops import einsum

import jax
import jax.numpy as jnp
from flax import nnx


def radial_basis_function(x: jax.Array, grid, h):
    """
    x: b, d_in
    grid: c
    return: b, d_in, c
    """
    return jnp.exp(-(((x[..., None] - grid) / h) ** 2))


class KANRBFLayer(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_update_bias: bool = True,
        use_layernorm: bool = True,
        base_activation=nnx.silu,
        spline_weight_init_scale: float = 0.1,
        denominator: Optional[float] = None,
    ) -> None:
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.use_layernorm = use_layernorm
        self.use_base_update = use_base_update

        self.grid = nnx.Variable(jnp.linspace(grid_min, grid_max, num_grids))
        self.denominator = (
            denominator if denominator else (grid_max - grid_min) / (num_grids - 1)
        )
        coef_key = rngs.coef()
        self.coef = nnx.Param(
            jax.random.truncated_normal(
                coef_key,
                lower=-2.0,
                upper=-2.0,
                shape=(out_features, in_features, num_grids),
            )
            * jnp.array(spline_weight_init_scale)
        )

        if self.use_base_update:
            self.base_linear = nnx.Linear(
                in_features=in_features,
                out_features=out_features,
                use_bias=base_update_bias,
                rngs=rngs,
            )
            self.base_activation = base_activation

        if use_layernorm:
            assert in_features > 1, "Do not use layernorms on 1D inputs."
            self.layernorm = nnx.LayerNorm(in_features, epsilon=1e-5, rngs=rngs)

    def __call__(self, x: jax.Array, use_layernorm: bool = True) -> jax.Array:
        """
        x: b, d_in
        """
        if self.use_layernorm and use_layernorm:
            rbf = radial_basis_function(
                self.layernorm(x), grid=self.grid, h=self.denominator
            )
        else:
            rbf = radial_basis_function(x, grid=self.grid, h=self.denominator)
        out = jnp.einsum("bik, oik -> bo", rbf, self.coef.value)

        if self.use_base_update:
            res = self.base_linear(self.base_activation(x))
            out = res + out

        return out

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_points: int = 100,
        num_extrapolate_bins: int = 3,
    ):
        x = jnp.linspace(
            self.grid_min - num_extrapolate_bins * self.denominator,
            self.grid_max + num_extrapolate_bins * self.denominator,
            num_points,
        )  # num_points
        c = self.coef.value[output_index, input_index]  # num_grids
        y = radial_basis_function(
            x, grid=self.grid, h=self.denominator
        )  # num_points, num_grids
        y = einsum(y, c, "i k, k -> i")

        return x, y


class KANRBF(nnx.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        rngs: nnx.Rngs,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_layernorm: bool = True,
        use_base_update: bool = True,
        base_update_bias: bool = True,
        base_activation=nnx.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        self.layers = [
            KANRBFLayer(
                in_features=in_features,
                out_features=out_features,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_layernorm=use_layernorm,
                use_base_update=use_base_update,
                base_update_bias=base_update_bias,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
                rngs=rngs,
            )
            for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:])
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
        return x
