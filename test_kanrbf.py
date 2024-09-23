"""
test_kanrbf.py: Test correctness of new implementation with FastKAN
-------------------------------------------------------------------

* Authors: Quoc Nguyen (minhquoc0712@gmail.com)
* Date: 2024-09-23
* Version: 0.0.1

"""

import matplotlib.pyplot as plt
from einops import rearrange

import torch
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from fastkan import FastKANLayer
from kan_rbf import KANRBFLayer


input_dim = 2
output_dim = 3
grid_min = -2.0
grid_max = 2.0
num_grids = 8
use_layernorm = True
use_base_update = True

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (6, input_dim))

kan_layer = FastKANLayer(
    input_dim=input_dim,
    output_dim=output_dim,
    grid_min=grid_min,
    grid_max=grid_max,
    num_grids=num_grids,
    use_base_update=use_base_update,
    use_layernorm=use_layernorm,
)
y1 = kan_layer(torch.Tensor(np.array(x)))

kan_rbf_layer = KANRBFLayer(
    in_features=input_dim,
    out_features=output_dim,
    use_base_update=use_base_update,
    base_update_bias=True,
    use_layernorm=use_layernorm,
    rngs=nnx.Rngs(key),
)
kan_rbf_layer.grid.value = jnp.array(kan_layer.rbf.grid.numpy())
coef = kan_layer.spline_linear.weight.detach().numpy()
coef = rearrange(coef, "d2 (d1 c) -> d2 d1 c", d1=input_dim)
kan_rbf_layer.coef.value = jnp.array(coef)

if use_base_update:
    kan_rbf_layer.base_linear.kernel.value = jnp.array(
        kan_layer.base_linear.weight.detach().numpy()
    ).T
    kan_rbf_layer.base_linear.bias.value = jnp.array(
        kan_layer.base_linear.bias.detach().numpy()
    )

y2 = kan_rbf_layer(x)
assert np.allclose(y1.detach().numpy(), y2, atol=1e-6), f"{y1} != {y2}"
print("Test correctness of new implementation with FastKAN passed!")

plt.figure()
for i in range(input_dim):
    for j in range(output_dim):
        x, y = kan_rbf_layer.plot_curve(i, j)
        plt.plot(x, y, label=r"$\phi_{" + f"{i},{j}" + r"}$")
plt.legend(loc="upper right")
plt.show()
