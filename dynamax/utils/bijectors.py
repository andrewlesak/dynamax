import tensorflow_probability.substrates.jax.bijectors as tfb
import jax.numpy as jnp

# From https://www.tensorflow.org/probability/examples/
# TensorFlow_Probability_Case_Study_Covariance_Estimation
class PSDToRealBijector(tfb.Chain):

    def __init__(self,
                 validate_args=False,
                 validate_event_size=False,
                 parameters=None,
                 name=None):

        bijectors = [
            tfb.Invert(tfb.FillTriangular()),
            tfb.TransformDiagonal(tfb.Invert(tfb.Exp())),
            tfb.Invert(tfb.CholeskyOuterProduct()),
        ]
        super().__init__(bijectors, validate_args, validate_event_size, parameters, name)


class RealToPSDBijector(tfb.Chain):

    def __init__(self,
                 validate_args=False,
                 validate_event_size=False,
                 parameters=None,
                 name=None):

        bijectors = [
            tfb.CholeskyOuterProduct(),
            tfb.TransformDiagonal(tfb.Exp()),
            tfb.FillTriangular(),
        ]
        super().__init__(bijectors, validate_args, validate_event_size, parameters, name)


class RealToSymmetricBijector(tfb.Chain):
    """Bijector that converts a real matrix into a symmetric matrix."""

    def __init__(self, validate_args=False, name="RealToSymmetricBijector"):
        super().__init__([
            tfb.Inline(
                forward_fn=lambda x: 0.5 * (x + jnp.swapaxes(x, -1, -2)),  # Symmetrization
                inverse_fn=lambda y: y,  # No-op for inverse, symmetric stays symmetric
                forward_min_event_ndims=2,
                inverse_min_event_ndims=2
            )
        ], validate_args=validate_args, name=name)


class BoundedBijector(tfb.Bijector):
    """Bijector that enforces both lower and upper bounds on parameters."""
    def __init__(
        self, 
        lower_bound: float = -jnp.inf,
        upper_bound: float = jnp.inf, 
        validate_args: bool = False, 
        name: str = "BoundedBijector"):

        if lower_bound >= upper_bound:
            raise ValueError("lower_bound must be less than upper_bound")
        
        super().__init__(forward_min_event_ndims=0, validate_args=validate_args, name=name)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def _forward(self, x):
        return jnp.clip(x, self.lower_bound, self.upper_bound)

    def _inverse(self, y):
        return y

    def forward_log_det_jacobian(self, x):
        return jnp.zeros_like(x)