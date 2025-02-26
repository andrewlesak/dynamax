import jax.numpy as jnp
import jax.random as jr
import optax
from jax import lax
from jax.tree_util import tree_map
from jaxtyping import Float, Array
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.hidden_markov_model.models.linreg_hmm import LinearRegressionHMMEmissions, ParamsLinearRegressionHMMEmissions
from dynamax.hidden_markov_model.models.linreg_hmm import LinearRegressionSharedSphericalGaussianHMMEmissions, ParamsLinearRegressionSharedSphericalGaussianHMMEmissions
from dynamax.parameters import ParameterProperties
from dynamax.types import Scalar
from dynamax.utils.bijectors import RealToPSDBijector
from tensorflow_probability.substrates import jax as tfp
from typing import NamedTuple, Optional, Tuple, Union

tfd = tfp.distributions
tfb = tfp.bijectors


class ParamsLinearAutoregressiveHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsLinearRegressionHMMEmissions


class LinearAutoregressiveHMMEmissions(LinearRegressionHMMEmissions):
    def __init__(self,
                 num_states,
                 emission_dim,
                 num_lags=1):
        self.num_lags = num_lags
        self.emission_dim = emission_dim
        input_dim = num_lags * emission_dim
        super().__init__(num_states, input_dim, emission_dim)

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
            _emission_weights = jnp.zeros((self.num_states, self.emission_dim, self.emission_dim * self.num_lags))
            _emission_biases = jnp.array(km.cluster_centers_)
            _emission_covs = jnp.tile(jnp.eye(self.emission_dim)[None, :, :], (self.num_states, 1, 1))

        elif method.lower() == "prior":
            # technically there's an MNIW prior, but that's a bit complicated...
            key1, key2, key = jr.split(key, 3)
            _emission_weights = jnp.zeros((self.num_states, self.emission_dim, self.emission_dim * self.num_lags))
            _emission_weights = _emission_weights.at[:, :, :self.emission_dim].set(0.95 * jnp.eye(self.emission_dim))
            _emission_weights += 0.01 * jr.normal(key1, (self.num_states, self.emission_dim, self.emission_dim * self.num_lags))
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


class LinearAutoregressiveHMM(HMM):
    r"""An autoregressive HMM whose emissions are a linear function of the previous emissions with state-dependent weights.
    This is also known as a *switching vector autoregressive* model.

    Let $y_t \in \mathbb{R}^N$ denote vector-valued emissions at time $t$.
    In this model, the emission distribution is,

    $$p(y_t \mid y_{1:t-1}, z_t, \theta) = \mathcal{N}(y_{t} \mid \sum_{\ell = 1}^L W_{z_t, \ell} y_{t-\ell} + b_{z_t}, \Sigma_{z_t})$$

    with *emission weights* $W_{k,\ell} \in \mathbb{R}^{N \times N}$ for each *lag* $\ell=1,\ldots,L$,
    *emission biases* $b_k \in \mathbb{R}^N$,
    and *emission covariances* $\Sigma_k \in \mathbb{R}_{\succeq 0}^{N \times N}$.

    The emissions parameters are $\theta = \{\{W_{k,\ell}\}_{\ell=1}^L, b_k, \Sigma_k\}_{k=1}^K$.

    We do not place a prior on the emission parameters.

    *Note: in the future we add a* matrix-normal-inverse-Wishart_ *prior (see pg 576).*

    .. _matrix-normal-inverse-Wishart: https://github.com/probml/pml2-book

    :param num_states: number of discrete states $K$
    :param emission_dim: emission dimension $N$
    :param num_lags: number of lags $L$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.

    """
    def __init__(self,
                 num_states: int,
                 emission_dim: int,
                 num_lags: int=1,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0):
        self.emission_dim = emission_dim
        self.num_lags = num_lags
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = LinearAutoregressiveHMMEmissions(num_states, emission_dim, num_lags=num_lags)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        """Return a pytree matching the pytree of tuples specifying the shape(s)
        of a single time step's inputs.
        """
        return (self.num_lags * self.emission_dim,)

    def initialize(self,
                   key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_weights: Optional[Float[Array, "num_states emission_dim emission_dim_times_num_lags"]]=None,
                   emission_biases: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_covariances:  Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        r"""Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            initial_probs: manually specified initial state probabilities.
            transition_matrix: manually specified transition matrix.
            emission_weights: manually specified emission weights. The weights are stored as matrices $W_k = [W_{k,1}, \ldots, W_{k,L}] \in \mathbb{R}^{N \times N \cdot L}$.
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
        return ParamsLinearAutoregressiveHMM(**params), ParamsLinearAutoregressiveHMM(**props)

    def sample(self,
               params: HMMParameterSet,
               key: jr.PRNGKey,
               num_timesteps: int,
               prev_emissions: Optional[Float[Array, "num_lags emission_dim"]]=None,
    ) -> Tuple[Float[Array, "num_timesteps state_dim"], Float[Array, "num_timesteps emission_dim"]]:
        r"""Sample states $z_{1:T}$ and emissions $y_{1:T}$ given parameters $\theta$.

        Args:
            params: model parameters $\theta$
            key: random number generator
            num_timesteps: number of timesteps $T$
            prev_emissions: (optionally) preceding emissions $y_{-L+1:0}$. Defaults to zeros.

        Returns:
            latent states and emissions

        """
        if prev_emissions is None:
            # Default to zeros
            prev_emissions = jnp.zeros((self.num_lags, self.emission_dim))

        def _step(carry, key):
            prev_state, prev_emissions = carry
            key1, key2 = jr.split(key, 2)
            state = self.transition_distribution(params, prev_state).sample(seed=key2)
            emission = self.emission_distribution(params, state, inputs=jnp.ravel(prev_emissions)).sample(seed=key1)
            next_prev_emissions = jnp.vstack([emission, prev_emissions[:-1]])
            return (state, next_prev_emissions), (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_state = self.initial_distribution(params).sample(seed=key1)
        initial_emission = self.emission_distribution(params, initial_state, inputs=jnp.ravel(prev_emissions)).sample(seed=key2)
        initial_prev_emissions = jnp.vstack([initial_emission, prev_emissions[:-1]])

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        _, (next_states, next_emissions) = lax.scan(
            _step, (initial_state, initial_prev_emissions), next_keys)

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return states, emissions

    def compute_inputs(self,
                       emissions: Float[Array, "num_timesteps emission_dim"],
                       prev_emissions: Optional[Float[Array, "num_lags emission_dim"]]=None
    ) -> Float[Array, "num_timesteps emission_dim_times_num_lags"]:
        r"""Helper function to compute the matrix of lagged emissions.

        Args:
            emissions: $(T \times N)$ array of emissions
            prev_emissions: $(L \times N)$ array of previous emissions. Defaults to zeros.

        Returns:
            $(T \times N \cdot L)$ array of lagged emissions. These are the inputs to the fitting functions.
        """
        if prev_emissions is None:
            # Default to zeros
            prev_emissions = jnp.zeros((self.num_lags, self.emission_dim))

        padded_emissions = jnp.vstack((prev_emissions, emissions))
        num_timesteps = len(emissions)
        return jnp.column_stack([padded_emissions[lag:lag+num_timesteps]
                                 for lag in reversed(range(self.num_lags))])



## ------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------
## make class with shared spherical cov using generic optimizer linear regression hmm -
## with full weights matrix, optionally tied and constant biases ----------------------
## ------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------

class ParamsGenOptSharedSphericalGaussianLinearAutoregressiveHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsLinearRegressionSharedSphericalGaussianHMMEmissions

class GenOptSharedSphericalGaussianLinearAutoregressiveHMMEmissions(LinearRegressionSharedSphericalGaussianHMMEmissions):
    def __init__(self,
                 num_states,
                 emission_dim,
                 num_lags=1,
                 tied_bias=False,
                 const_bias=False,
                 m_step_optimizer=optax.adam(1e-2),
                 m_step_num_iters=50):
        self.num_lags = num_lags
        self.emission_dim = emission_dim
        input_dim = num_lags * emission_dim
        super().__init__(num_states, input_dim, emission_dim, tied_bias=tied_bias, const_bias=const_bias, 
                         m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
    
    # if initialize is unspecified, defaults to initialization procedure in called HMMEmissions class
    # def initialize():  

class GenOptSharedSphericalGaussianLinearAutoregressiveHMM(HMM):
    r"""_summary_
    """
    def __init__(self,
                 num_states: int,
                 emission_dim: int,
                 num_lags: int=1,
                 tied_bias: bool=False,
                 const_bias: bool=False,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 m_step_optimizer: optax.GradientTransformation=optax.adam(1e-2),
                 m_step_num_iters: int=50):
        self.emission_dim = emission_dim
        self.num_lags = num_lags
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = GenOptSharedSphericalGaussianLinearAutoregressiveHMMEmissions(
            num_states, 
            emission_dim, 
            num_lags=num_lags, 
            tied_bias=tied_bias,
            const_bias=const_bias,
            m_step_optimizer=m_step_optimizer, 
            m_step_num_iters=m_step_num_iters
        )
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        """Return a pytree matching the pytree of tuples specifying the shape(s)
        of a single time step's inputs.
        """
        return (self.num_lags * self.emission_dim,)

    def initialize(self,
                   key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="random",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_weights: Optional[Float[Array, "num_states emission_dim emission_dim_times_num_lags"]]=None,
                   emission_biases: Optional[Float[Array, "dynamic_bias_state_dim dynamic_bias_emission_dim"]]=None,
                   emission_scales: Optional[Float[Array, "1"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        r"""_summary_
        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases, emission_scales=emission_scales, emissions=emissions)
        return ParamsGenOptSharedSphericalGaussianLinearAutoregressiveHMM(**params), ParamsGenOptSharedSphericalGaussianLinearAutoregressiveHMM(**props)

    # def sample():  # add sample function eventually

    def compute_inputs(self,
                       emissions: Float[Array, "num_timesteps emission_dim"],
                       prev_emissions: Optional[Float[Array, "num_lags emission_dim"]]=None
    ) -> Float[Array, "num_timesteps emission_dim_times_num_lags"]:
        r"""Helper function to compute the matrix of lagged emissions.

        Args:
            emissions: $(T \times N)$ array of emissions
            prev_emissions: $(L \times N)$ array of previous emissions. Defaults to zeros.

        Returns:
            $(T \times N \cdot L)$ array of lagged emissions. These are the inputs to the fitting functions.
        """
        if prev_emissions is None:
            # Default to zeros
            prev_emissions = jnp.zeros((self.num_lags, self.emission_dim))

        padded_emissions = jnp.vstack((prev_emissions, emissions))
        num_timesteps = len(emissions)
        return jnp.column_stack([padded_emissions[lag:lag+num_timesteps]
                                 for lag in reversed(range(self.num_lags))])



## ------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------
## ARHMM with optionally lagged inputs (separate lag for emissions AND for inputs)
## ------------------------------------------------------------------------------------
## ------------------------------------------------------------------------------------

class ParamsGenOptInputDrivenSharedSphericalGaussianLinearAutoregressiveHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsLinearRegressionSharedSphericalGaussianHMMEmissions

class GenOptInputDrivenSharedSphericalGaussianLinearAutoregressiveHMMEmissions(LinearRegressionSharedSphericalGaussianHMMEmissions):
    def __init__(self,
                 num_states,
                 emission_dim,
                 input_dim=0,
                 num_lags=1,
                 num_input_lags=0,
                 tied_bias=False,
                 const_bias=False,
                 m_step_optimizer=optax.adam(1e-2),
                 m_step_num_iters=50):
        self.num_emission_lags = num_lags
        self.num_input_lags = num_input_lags
        self.input_dim = input_dim

        super_input_dim = emission_dim * num_lags + input_dim * (num_input_lags + 1)
        super().__init__(num_states, super_input_dim, emission_dim, tied_bias=tied_bias, const_bias=const_bias, 
                         m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
    
    # if initialize is unspecified, defaults to initialization procedure in called HMMEmissions class
    # def initialize():  

class GenOptInputDrivenSharedSphericalGaussianLinearAutoregressiveHMM(HMM):
    r"""_summary_
    """
    def __init__(self,
                 num_states: int,
                 emission_dim: int,
                 input_dim: int=0,
                 num_lags: int=1,
                 num_input_lags: int=0,
                 tied_bias: bool=False,
                 const_bias: bool=False,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 m_step_optimizer: optax.GradientTransformation=optax.adam(1e-2),
                 m_step_num_iters: int=50):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.num_lags = num_lags
        self.num_input_lags = num_input_lags

        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = GenOptInputDrivenSharedSphericalGaussianLinearAutoregressiveHMMEmissions(
            num_states, 
            emission_dim, 
            input_dim=input_dim,
            num_lags=num_lags, 
            num_input_lags=num_input_lags,
            tied_bias=tied_bias,
            const_bias=const_bias,
            m_step_optimizer=m_step_optimizer, 
            m_step_num_iters=m_step_num_iters
        )
        super().__init__(num_states, initial_component, transition_component, emission_component)
    
    @property
    def inputs_shape(self):
        """Return a pytree matching the pytree of tuples specifying the shape(s)
        of a single time step's inputs.
        """
        return (self.emission_dim * self.num_lags + self.input_dim * (self.num_input_lags + 1),)

    def initialize(self,
                   key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="random",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_weights: Optional[Float[Array, "num_states emission_dim super_input_dim"]]=None,
                   emission_biases: Optional[Float[Array, "dynamic_bias_state_dim dynamic_bias_emission_dim"]]=None,
                   emission_scales: Optional[Float[Array, "1"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None,
                   inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        r"""_summary_
        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases, emission_scales=emission_scales, emissions=emissions, inputs=inputs)
        return ParamsGenOptInputDrivenSharedSphericalGaussianLinearAutoregressiveHMM(**params), ParamsGenOptInputDrivenSharedSphericalGaussianLinearAutoregressiveHMM(**props)

    def compute_super_inputs(self,
                             emissions: Float[Array, "num_timesteps emission_dim"],
                             inputs: Float[Array, "num_timesteps input_dim"],
                             prev_emissions: Optional[Float[Array, "num_lags emission_dim"]]=None,
                             prev_inputs: Optional[Float[Array, "num_input_lags_plus_one input_dim"]]=None
    ) -> Float[Array, "num_timesteps emission_dim_times_num_lags"]:
        r"""Helper function to compute the matrix of lagged emissions and inputs and combine into super inputs.

        Args:
            emissions: $(T \times N)$ array of emissions
            inputs: $(T \times M)$ array of inputs
            prev_emissions: $(L \times N)$ array of previous emissions. Defaults to zeros.
            prev_inputs: $(L'+1 \times M+)$ array of previous inputs (includes 0 lag). Defaults to zeros.

        Returns:
            $(T \times N \cdot L + M \cdot (L'+1))$ array of lagged emissions. These are the inputs to the fitting functions.
        """
        if prev_emissions is None:
            # Default to zeros
            prev_emissions = jnp.zeros((self.num_lags, self.emission_dim))

        if prev_inputs is None:
            # Default to zeros for previous inputs (will include lag = 0)
            prev_inputs = jnp.zeros((self.num_input_lags, self.input_dim))

        padded_emissions = jnp.vstack((prev_emissions, emissions))
        padded_inputs = jnp.vstack((prev_inputs, inputs))
        num_timesteps = len(emissions)
    
        # compute lagged emissions and lagged inputs (but keep input at lag 0!)
        lagged_emissions = jnp.column_stack([padded_emissions[lag:lag+num_timesteps] for lag in reversed(range(self.num_lags))])
        lagged_inputs = jnp.column_stack([padded_inputs[lag:lag+num_timesteps] for lag in reversed(range(self.num_input_lags + 1))])

        # combine into "super inputs" to pass into fitting function
        return jnp.column_stack([lagged_emissions, lagged_inputs])
