import math
import jax
from typing import Callable, Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp
from jaxrl_m.common.common import default_init


def pytorch_init(fan_in: float):
    """
    Default init for PyTorch Linear layer weights and biases:
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """
    bound = math.sqrt(1 / fan_in)

    def _init(key, shape, dtype):
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
    activate_final: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class MLPResNetBlock(nn.Module):
    features: int
    act: Callable
    dropout_rate: float = None
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        residual = x
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(self.features * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.features)(x)

        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + x


class MLPResNet(nn.Module):
    num_blocks: int
    out_dim: int
    dropout_rate: float = None
    use_layer_norm: bool = False
    hidden_dim: int = 256
    activations: Callable = nn.swish

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=default_init())(x)
        for _ in range(self.num_blocks):
            x = MLPResNetBlock(
                self.hidden_dim,
                act=self.activations,
                use_layer_norm=self.use_layer_norm,
                dropout_rate=self.dropout_rate,
            )(x, train=train)

        x = self.activations(x)
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        return x


class FilmFullMLP(nn.Module):
    hidden_dim: int
    dropout_rate: Optional[float] = None 

    @nn.compact
    def __call__(self, feature, context, train: bool = False): 
        f_d, c_d, h_d = feature.shape[-1], context.shape[-1], self.hidden_dim
        film = nn.Dense(
            2 * self.hidden_dim * 3,
            kernel_init=pytorch_init(c_d),
            bias_init=pytorch_init(c_d),
        )
        linear1 = nn.Dense(
            self.hidden_dim, kernel_init=pytorch_init(f_d), bias_init=pytorch_init(f_d)
        )
        linear2 = nn.Dense(
            self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d)
        )
        linear3 = nn.Dense(
            self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=pytorch_init(h_d)
        )

        gamma, beta = jnp.split(
            film(context).reshape(context.shape[0], 3, -1), 2, axis=-1
        )
        
        # First hidden layer
        out = nn.relu(gamma[:, 0] * linear1(feature) + beta[:, 0])
        if self.dropout_rate is not None and self.dropout_rate > 0:
            out = nn.Dropout(rate=self.dropout_rate)(out, deterministic=not train)
        
        # Second hidden layer
        out = nn.relu(gamma[:, 1] * linear2(out) + beta[:, 1])
        if self.dropout_rate is not None and self.dropout_rate > 0:
            out = nn.Dropout(rate=self.dropout_rate)(out, deterministic=not train)
        
        # Third hidden layer
        out = nn.relu(gamma[:, 2] * linear3(out) + beta[:, 2])
        if self.dropout_rate is not None and self.dropout_rate > 0:
            out = nn.Dropout(rate=self.dropout_rate)(out, deterministic=not train)
        
        return out
