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


class ToeplitzToRealBijector(tfb.Bijector):
    """
    Bijector that converts a Toeplitz matrix of shape (m, n) to a real vector.
    Problem: while this code works in basic tests, during EM for a HMM the shapes are not handled correctly and
    breaks. This may be a result of not using tensorflow objects. Need to revisit this later and construct
    this bijector in tensorflow in similar manner to https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/bijectors/FillTriangular 
    """
    def __init__(self, 
                 m, 
                 n, 
                 validate_args=False, 
                 name="ToeplitzToRealBijector"):
        super(ToeplitzToRealBijector, self).__init__(
            forward_min_event_ndims=2,  # forward transform works on matrices
            inverse_min_event_ndims=1,  # inverse works on vectors
            validate_args=validate_args, 
            name=name)
        
        self.m = m  # number of rows in Toeplitz matrix
        self.n = n  # number of columns in Toeplitz matrix

    def _forward(self, y):
        """
        Forward transformation: maps a Toeplitz matrix of shape (m, n) to a real vector of size m+n-1.
        y: Toeplitz matrix of shape (..., m, n)
        """
        first_row = y[..., 0, :]   # First row of the Toeplitz matrix (size n)
        first_col = y[..., 1:, 0]  # First column of the Toeplitz matrix (size m-1, excluding the first element)
        # Concatenate the first row and the remaining part of the first column
        return jnp.concatenate([first_row, first_col], axis=-1)

    def _inverse(self, x):
        """
        Inverse transformation: maps a real vector of size m+n-1 to a Toeplitz matrix of shape (m, n).
        x: real vector of size (..., m+n-1)
        """
        first_row = x[..., :self.n]  # First row of the Toeplitz matrix (size n)
        first_col = jnp.concatenate([x[..., :1], x[..., self.n:]], axis=-1)  # First column, including 00 element

        # Force the toeplitz operation to run on CPU
        #def toeplitz_cpu(c, r):
        #    with jax.default_device(jax.devices('cpu')[0]):
        #        return jax.scipy.linalg.toeplitz(c=c, r=r)
        
        # Map across the batch dimensions using vmap
        #return vmap(toeplitz_cpu)(c=first_col, r=first_row)

        # Map across the batch dimensions using vmap
        return vmap(jax.scipy.linalg.toeplitz)(c=first_col, r=first_row)

    def _forward_log_det_jacobian(self, y):
        # No volume change for the Toeplitz transformation (log |det J| = 0)
        return jnp.zeros(y.shape[:-2])

    def _inverse_log_det_jacobian(self, x):
        # No volume change for the inverse transformation (log |det J| = 0)
        return jnp.zeros(x.shape[:-1])