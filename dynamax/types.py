from typing import Optional, Union
from typing_extensions import Protocol
from jaxtyping import Array, Float
import jax._src.random as prng
import jax

# PRNGKey = prng.KeyArray  # deprecated in jax version 0.4.16
PRNGKey = jax.Array  

Scalar = Union[float, Float[Array, ""]] # python float or scalar jax device array with dtype float
