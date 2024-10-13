import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import vmap
from jaxtyping import Float, Array
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.parameters import ParameterProperties
from dynamax.types import Scalar
from dynamax.utils.utils import pytree_sum
from dynamax.utils.bijectors import RealToPSDBijector
from tensorflow_probability.substrates import jax as tfp
from typing import NamedTuple, Optional, Tuple, Union

tfd = tfp.distributions
tfb = tfp.bijectors

# -- Defines closed form linear regression HMM (aka linear model HMM (LM-HMM)) --

class ParamsLinearRegressionHMMEmissions(NamedTuple):
    weights: Union[Float[Array, "state_dim emission_dim input_dim"], ParameterProperties]
    biases: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]
    covs: Union[Float[Array, "state_dim emission_dim emission_dim"], ParameterProperties]


class ParamsLinearRegressionHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsLinearRegressionHMMEmissions


class LinearRegressionHMMEmissions(HMMEmissions):
    def __init__(self,
                 num_states,
                 input_dim,
                 emission_dim):
        """_summary_

        Args:
            initial_probabilities (_type_): _description_
            transition_matrix (_type_): _description_
            emission_matrices (_type_): _description_
            emission_biases (_type_): _description_
            emission_covariance_matrices (_type_): _description_
        """
        self.num_states = num_states
        self.input_dim = input_dim
        self.emission_dim = emission_dim

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def initialize(self,
                   key=jr.PRNGKey(0),
                   method="prior",
                   emission_weights=None,
                   emission_biases=None,
                   emission_covariances=None,
                   emissions=None):
        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            from sklearn.cluster import KMeans
            key, subkey = jr.split(key)  # Create a random seed for SKLearn.
            sklearn_key = jr.randint(subkey, shape=(), minval=0, maxval=2147483647)  # Max int32 value.
            km = KMeans(self.num_states, random_state=int(sklearn_key)).fit(emissions.reshape(-1, self.emission_dim))
            _emission_weights = jnp.zeros((self.num_states, self.emission_dim, self.input_dim))
            _emission_biases = jnp.array(km.cluster_centers_)
            _emission_covs = jnp.tile(jnp.eye(self.emission_dim)[None, :, :], (self.num_states, 1, 1))

        elif method.lower() == "prior":
            # TODO: Use an MNIW prior
            key1, key2, key = jr.split(key, 3)
            _emission_weights = 0.01 * jr.normal(key1, (self.num_states, self.emission_dim, self.input_dim))
            _emission_biases = jr.normal(key2, (self.num_states, self.emission_dim))
            _emission_covs = jnp.tile(jnp.eye(self.emission_dim), (self.num_states, 1, 1))
        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsLinearRegressionHMMEmissions(
            weights=default(emission_weights, _emission_weights),
            biases=default(emission_biases, _emission_biases),
            covs=default(emission_covariances, _emission_covs))
        props = ParamsLinearRegressionHMMEmissions(
            weights=ParameterProperties(),
            biases=ParameterProperties(),
            covs=ParameterProperties(constrainer=RealToPSDBijector()))
        return params, props

    def distribution(self, params, state, inputs):
        prediction = params.weights[state] @ inputs
        prediction +=  params.biases[state]
        return tfd.MultivariateNormalFullCovariance(prediction, params.covs[state])

    def log_prior(self, params):
        return 0.0

    # Expectation-maximization (EM) code
    def collect_suff_stats(self, params, posterior, emissions, inputs=None):
        expected_states = posterior.smoothed_probs
        sum_w = jnp.einsum("tk->k", expected_states)
        sum_x = jnp.einsum("tk,ti->ki", expected_states, inputs)
        sum_y = jnp.einsum("tk,ti->ki", expected_states, emissions)
        sum_xxT = jnp.einsum("tk,ti,tj->kij", expected_states, inputs, inputs)
        sum_xyT = jnp.einsum("tk,ti,tj->kij", expected_states, inputs, emissions)
        sum_yyT = jnp.einsum("tk,ti,tj->kij", expected_states, emissions, emissions)
        return dict(sum_w=sum_w, sum_x=sum_x, sum_y=sum_y, sum_xxT=sum_xxT, sum_xyT=sum_xyT, sum_yyT=sum_yyT)

    def initialize_m_step_state(self, params, props):
        return None

    def m_step(self, params, props, batch_stats, m_step_state):
        def _single_m_step(stats):
            sum_w = stats['sum_w']
            sum_x = stats['sum_x']
            sum_y = stats['sum_y']
            sum_xxT = stats['sum_xxT']
            sum_xyT = stats['sum_xyT']
            sum_yyT = stats['sum_yyT']

            # Make block matrices for stacking features (x) and bias (1)
            sum_x1x1T = jnp.block(
                [[sum_xxT,                   jnp.expand_dims(sum_x, 1)],
                 [jnp.expand_dims(sum_x, 0), jnp.expand_dims(sum_w, (0, 1))]]
            )
            sum_x1yT = jnp.vstack([sum_xyT, sum_y])

            # Solve for the optimal A, b, and Sigma
            Ab = jnp.linalg.solve(sum_x1x1T, sum_x1yT).T
            Sigma = 1 / sum_w * (sum_yyT - Ab @ sum_x1yT)
            Sigma = 0.5 * (Sigma + Sigma.T)                 # for numerical stability
            return Ab[:, :-1], Ab[:, -1], Sigma

        emission_stats = pytree_sum(batch_stats, axis=0)
        As, bs, Sigmas = vmap(_single_m_step)(emission_stats)
        params = params._replace(weights=As, biases=bs, covs=Sigmas)
        return params, m_step_state


class LinearRegressionHMM(HMM):
    r"""An HMM whose emissions come from a linear regression with state-dependent weights.
    This is also known as a *switching linear regression* model.

    Let $y_t \in \mathbb{R}^N$ and $u_t \in \mathbb{R}^M$ denote vector-valued emissions
    and inputs at time $t$, respectively. In this model, the emission distribution is,

    $$p(y_t \mid z_t, u_t, \theta) = \mathcal{N}(y_{t} \mid W_{z_t} u_t + b_{z_t}, \Sigma_{z_t})$$

    with *emission weights* $W_k \in \mathbb{R}^{N \times M}$, *emission biases* $b_k \in \mathbb{R}^N$,
    and *emission covariances* $\Sigma_k \in \mathbb{R}_{\succeq 0}^{N \times N}$.

    The emissions parameters are $\theta = \{W_k, b_k, \Sigma_k\}_{k=1}^K$.

    We do not place a prior on the emission parameters.

    *Note: in the future we add a* matrix-normal-inverse-Wishart_ *prior (see pg 576).*

    .. _matrix-normal-inverse-Wishart: https://github.com/probml/pml2-book

    :param num_states: number of discrete states $K$
    :param input_dim: input dimension $M$
    :param emission_dim: emission dimension $N$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.

    """
    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 emission_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = LinearRegressionHMMEmissions(num_states, input_dim, emission_dim)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        return (self.input_dim,)

    def initialize(self,
                   key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_weights: Optional[Float[Array, "num_states emission_dim input_dim"]]=None,
                   emission_biases: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_covariances:  Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            initial_probs: manually specified initial state probabilities.
            transition_matrix: manually specified transition matrix.
            emission_weights: manually specified emission weights.
            emission_biases: manually specified emission biases.
            emission_covariances: manually specified emission covariances.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.

        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases, emission_covariances=emission_covariances, emissions=emissions)
        return ParamsLinearRegressionHMM(**params), ParamsLinearRegressionHMM(**props)



# -- Define constrained linear regression/model HMM (aka cLM-HMM) --
# weights matrix is Toeplitz with optional flag for symmetric (thus the dynamic_dim)
# coupled_IO_fit allows for fitting both permutations of coupled input/output data

class ParamsConstrainedLinearRegressionSharedSphericalGaussianHMMEmissions(NamedTuple):
    flat_weights: Union[Float[Array, "state_dim dynamic_dim"], ParameterProperties]
    biases: Union[Float[Array, "state_dim"], ParameterProperties]
    scales: Union[Float[Array, "1"], ParameterProperties]


class ParamsConstrainedLinearRegressionSharedSphericalGaussianHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsConstrainedLinearRegressionSharedSphericalGaussianHMMEmissions


class ConstrainedLinearRegressionSharedSphericalGaussianHMMEmissions(HMMEmissions):
    def __init__(self,
                 num_states,
                 input_dim,
                 emission_dim,
                 symmetric=False,
                 coupled_IO_fit=False,
                 m_step_optimizer=optax.adam(1e-2),
                 m_step_num_iters=50):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        """_summary_

        Args:
            num_states: Number of states in the HMM
            input_dim: Dimensionality of inputs
            emission_dim: Dimensionality of emissions (should be equal to input_dim)
            symmetric: If True, use Symmetric Toeplitz, otherwise use standard Toeplitz
            coupled_IO_fit: If True, fit both permutations of coupled input/output data simultaneously
        """
        if coupled_IO_fit:
            assert input_dim==emission_dim, f"Input dim ({input_dim}) must be the same as emission dim ({emission_dim}) for fitting both permutations of coupled input/output data"
        if symmetric:
            assert input_dim==emission_dim, f"Input dim ({input_dim}) must be the same as emission dim ({emission_dim}) for symmetric matrix"
        self.num_states = num_states
        self.input_dim = input_dim
        self.emission_dim = emission_dim
        self.symmetric = symmetric

        # dynamic dimension: emission_dim if symmetric, otherwise emission_dim + input_dim - 1
        self.dynamic_dim = self.emission_dim if self.symmetric else self.emission_dim + self.input_dim - 1

    @property
    def emission_shape(self):
        return (self.emission_dim,)
    
    @property
    def inputs_shape(self):
        return (self.input_dim,)

    def initialize(self,
                   key=jr.PRNGKey(0),
                   method="random",
                   emission_flat_weights=None,
                   emission_biases=None,
                   emission_scales=None,
                   emissions=None,
                   inputs=None):
        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            from sklearn.cluster import KMeans
            raise Exception("{} method not implemented yet".format(method))

        elif method.lower() == "prior":
            # TODO: Use an MNIW prior
            raise Exception("{} method not implemented yet".format(method))
        
        elif method.lower() == "random":
            key1, key2, key3 = jr.split(key, 3)
            _emission_flat_weights = 0.01 * jr.normal(key1, (self.num_states, self.dynamic_dim))
            _emission_biases = jr.normal(key2, (self.num_states,))
            _emission_scales = jnp.ones((1,))
            
        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsConstrainedLinearRegressionSharedSphericalGaussianHMMEmissions(
            flat_weights=default(emission_flat_weights, _emission_flat_weights),
            biases=default(emission_biases, _emission_biases),
            scales=default(emission_scales, _emission_scales))

        props = ParamsConstrainedLinearRegressionSharedSphericalGaussianHMMEmissions(
            flat_weights=ParameterProperties(),
            biases=ParameterProperties(),
            scales=ParameterProperties(constrainer=tfb.Softplus()))  # positive value constraint
        return params, props        

    def get_weights(self, flat_Wks):
        if self.symmetric:
            # symmetric Toeplitz matrix constructed from single row
            first_row = flat_Wks[..., :self.emission_dim]  
            return vmap(jax.scipy.linalg.toeplitz)(c=first_row, r=first_row) 
        else:
            # standard Toeplitz matrix constructed from single row and column
            first_row = flat_Wks[..., :self.input_dim]  # first row of the Toeplitz matrix (size input_dim)
            first_col = jnp.concatenate([flat_Wks[..., :1], flat_Wks[..., self.input_dim:]], axis=-1)  # first column
            return vmap(jax.scipy.linalg.toeplitz)(c=first_col, r=first_row)

    def distribution(self, params, state, inputs):
        # get weights
        Wks = self.get_weights(params.flat_weights)
        prediction = jnp.matmul(inputs, Wks[state])
        prediction += params.biases[state] * jnp.ones((self.emission_dim,))
        return tfd.MultivariateNormalDiag(prediction, params.scales[0] * jnp.ones((self.emission_dim,)))
    
    def log_prior(self, params):
        return 0.0

    def permute(self, params, perm):
        """Permute the emissions parameters based on a permutation of the latent states."""
        permuted_flat_weights = params.flat_weights[perm]
        permuted_biases = params.biases[perm]
        return ParamsConstrainedLinearRegressionSharedSphericalGaussianHMMEmissions(
            flat_weights=permuted_flat_weights, 
            biases=permuted_biases,
            scales=params.scales)  # scales don't need permutation as its shared across states


class ConstrainedLinearRegressionSharedSphericalGaussianHMM(HMM):
    r"""An HMM whose emissions come from a linear regression with state-dependent Toeplitz weights.
    We constrain the bias vector to be a constant vector $\vec{b}^{(k)} = b^{(k)}\vec{1}$ and parameterize the
    Toeplitz weights as a flat vector of length input_dim + emission_dim - 1 (or emission_dim if symmetric).
    The covariance is spherical and tied across all states $\boldsymbol{\Sigma}^{(k)} = \sigma^2 \mathbf{I}$.

    Let $\vec{y}_t \in \mathbb{R}^D$ and $\vec{u}_t \in \mathbb{R}^M$ denote vector-valued emissions
    and inputs at time $t$, respectively. In this model, the emission distribution is,

    $$p(\vec{y}_t \mid z_t, \vec{u}_t, \theta) = \mathcal{N}(\vec{y}_{t} \mid \mathbf{W}^{(k)} \vec{u}_t + b^{(k)}\vec{1}, \sigma^2 \mathbf{I})$$

    with *emission weights* $\mathbf{W}^{(k)} \in \mathbb{R}^{D \times M}$, *emission biases* $b^{(k)} \in \mathbb{R}$,
    and *emission covariances* $\Sigma_k = \sigma^2 \mathbf{I}^{D \times D}$ for $\sigma \in \mathbb{R}_{\succeq 0}$.

    The emissions parameters are $\theta = \{\mathbf{W}^{(k)}, b^{(k)}, \sigma\}_{k=1}^K$.

    We do not place a prior on the emission parameters.

    We also allow for fitting both permutations of input and output in the case of coupled data. This way, we ensure that each permutation
    is assigned to the same state (which should be the case if the data is coupled, as it does not matter what is the input vs output).
    If `coupled_IO_fit` = True, we pass the stacked emissions as $\vec{y}^{(2)}_t=(\vec{y}_t,\vec{u}_t)$ and inputs $\vec{u}^{(2)}_t=(\vec{u}_t,\vec{y}_t)$ that we later separate 
    in the function `_compute_conditional_logliks()` in "abstractions.py". Current implementation requires the model specified emission and 
    input dims to be equal, and be half of the dims of the passed stacked emissions and inputs (i.e. we split the stacked input/output data in half).
    In this case, the emission distribution is 
    $$p(\vec{y}^{(2)}_t \mid z_t, \vec{u}^{(2)}_t, \theta) = p(\vec{y}_t \mid z_t, \vec{u}_t, \theta) p(\vec{u}_t \mid z_t, \vec{y}_t, \theta) = 
    \mathcal{N}(\vec{y}_{t} \mid \mathbf{W}^{(k)} \vec{u}_t + b^{(k)}\vec{1}, \sigma^2 \mathbf{I}) \mathcal{N}(\vec{u}_{t} \mid \mathbf{W}^{(k)} \vec{y}_t + b^{(k)}\vec{1}, \sigma^2 \mathbf{I})$$


    :param num_states: number of discrete states $K$
    :param input_dim: input dimension $M$
    :param emission_dim: emission dimension $N$
    :param symmetric: optionally constrain Toeplitz weights matrix to be symmetric
    :param coupled_IO_fit: fit both permutations of coupled input/output data 
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.

    """
    def __init__(self,
                 num_states: int,
                 inputs_dim: int,
                 emissions_dim: int,
                 symmetric: bool=False,
                 coupled_IO_fit: bool=False,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 m_step_optimizer: optax.GradientTransformation=optax.adam(1e-2),
                 m_step_num_iters: int=50):
        self.emissions_dim = emissions_dim
        self.inputs_dim = inputs_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = ConstrainedLinearRegressionSharedSphericalGaussianHMMEmissions(num_states, inputs_dim, emissions_dim, 
                                                                                            symmetric=symmetric,
                                                                                            coupled_IO_fit=coupled_IO_fit,
                                                                                            m_step_optimizer=m_step_optimizer,
                                                                                            m_step_num_iters=m_step_num_iters)
        super().__init__(num_states, initial_component, transition_component, emission_component)
    
    @property
    def emission_shape(self):
        return (self.emissions_dim,)

    @property
    def inputs_shape(self):
        return (self.inputs_dim,)
    
    def initialize(self,
                   key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="random",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_flat_weights: Optional[Float[Array, "num_states dynamic_dim"]]=None,
                   emission_biases: Optional[Float[Array, "num_states"]]=None,
                   emission_scales:  Optional[Float[Array, "1"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None,
                   inputs:  Optional[Float[Array, "num_timesteps input_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        """Initialize the model parameters and their corresponding properties.

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            initial_probs: manually specified initial state probabilities.
            transition_matrix: manually specified transition matrix.
            emission_flat_weights: manually specified parameterized emission weights.
            emission_biases: manually specified emission biases.
            emission_covariances: manually specified emission covariances.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.

        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_flat_weights=emission_flat_weights, emission_biases=emission_biases, emission_scales=emission_scales, emissions=emissions, inputs=inputs)
        return ParamsConstrainedLinearRegressionSharedSphericalGaussianHMM(**params), ParamsConstrainedLinearRegressionSharedSphericalGaussianHMM(**props)
    