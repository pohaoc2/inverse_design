import numpy as np
from typing import Optional, List, Dict, Union, Tuple, Callable, Any, Literal
import logging
from copy import deepcopy
from inverse_design.rf.rf import RF
from inverse_design.rf.drf import DRF

class ABCSMCRF:
    """
    Implementation of ABC Sequential Monte Carlo with Random Forests.
    
    This class implements Algorithms 5 and 6 from the paper, suitable for
    iteratively improving parameter inference using RF methods.
    """
    
    def __init__(
        self,
        n_iterations: int = 5,
        n_particles: int = 1000,
        rf_type: Literal['RF', 'DRF'] = 'DRF',
        n_trees: int = 500,
        min_samples_leaf: int = 5,
        n_try: Optional[int] = None,
        random_state: Optional[int] = None,
        criterion: Literal['CART', 'MMD'] = 'CART',
        prior_sampler: Optional[Callable] = None,
        perturbation_kernel: Optional[Callable] = None,
        prior_pdf: Optional[Callable] = None
    ):
        """
        Initialize the ABC-SMC-RF model.
        
        Parameters:
        -----------
        n_iterations : int, default=5
            Number of SMC iterations.
        n_particles : int, default=1000
            Number of particles per iteration.
        rf_type : {'RF', 'DRF'}, default='DRF'
            Type of random forest to use.
            - 'RF': uses ABC-RF (for univariate parameter inference)
            - 'DRF': uses ABC-DRF (for multivariate parameter inference)
        n_trees : int, default=500
            Number of trees in each random forest.
        min_samples_leaf : int, default=5
            Minimum number of samples required to be at a leaf node.
        n_try : int, optional
            Number of statistics to consider when looking for the best split.
        random_state : int, optional
            Controls the randomization in the random forests.
        criterion : {'CART', 'MMD'}, default='CART'
            The function to measure the quality of a split (for DRF only).
        prior_sampler : Callable, optional
            Function to sample from the prior distribution.
            If None, a uniform prior in [0, 1] for each parameter is assumed.
        perturbation_kernel : Callable, optional
            Function to perturb parameters between iterations.
            If None, parameters are perturbed by adding Gaussian noise.
        prior_pdf : Callable, optional
            Function to evaluate the prior density.
            If None, a uniform prior in [0, 1] for each parameter is assumed.
        """
        self.n_iterations = n_iterations
        self.n_particles = n_particles
        self.rf_type = rf_type
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.n_try = n_try
        self.random_state = random_state
        self.criterion = criterion
        
        # Initialize RNGs
        self.rng = np.random.RandomState(random_state)
        
        # Set default functions if not provided
        self.prior_sampler = prior_sampler if prior_sampler is not None else self._default_prior_sampler
        self.perturbation_kernel = perturbation_kernel if perturbation_kernel is not None else self._default_perturbation_kernel
        self.prior_pdf = prior_pdf if prior_pdf is not None else self._default_prior_pdf
        
        # Storage for results
        self.parameter_samples = []
        self.weights = []
        self.statistics = []
        self.rf_models = []
        
    def _default_prior_sampler(self, n_samples: int) -> np.ndarray:
        """
        Default prior sampler (uniform in [0, 1] for each parameter).
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate.
            
        Returns:
        --------
        samples : np.ndarray of shape (n_samples, n_parameters)
            Sampled parameters.
        """
        # Default to 1D parameter if not yet determined
        n_parameters = 1
        if hasattr(self, 'n_parameters'):
            n_parameters = self.n_parameters
            
        return self.rng.uniform(0, 1, size=(n_samples, n_parameters))
    
    def _default_perturbation_kernel(self, parameters: np.ndarray) -> np.ndarray:
        """
        Default perturbation kernel (Gaussian noise).
        
        Parameters:
        -----------
        parameters : np.ndarray of shape (n_parameters,)
            Parameters to perturb.
            
        Returns:
        --------
        perturbed : np.ndarray of shape (n_parameters,)
            Perturbed parameters.
        """
        # Scale the perturbation based on the iteration
        if not hasattr(self, 'current_iteration'):
            scale = 0.1
        else:
            # Reduce scale as iterations progress
            scale = 0.1 / np.sqrt(1 + self.current_iteration)
            
        return parameters + scale * self.rng.normal(0, 1, size=parameters.shape)
    
    def _default_prior_pdf(self, parameters: np.ndarray) -> float:
        """
        Default prior PDF (uniform in [0, 1] for each parameter).
        
        Parameters:
        -----------
        parameters : np.ndarray of shape (n_parameters,)
            Parameters to evaluate.
            
        Returns:
        --------
        density : float
            Prior density at the given parameters.
        """
        # Check if all parameters are in [0, 1]
        if np.all((parameters >= 0) & (parameters <= 1)):
            return 1.0
        else:
            return 0.0
            
    def fit(self, observed_statistics: np.ndarray, simulator: Callable) -> None:
        """
        Run the ABC-SMC-RF algorithm.
        
        Parameters:
        -----------
        observed_statistics : np.ndarray of shape (n_statistics,)
            Observed summary statistics.
        simulator : Callable
            Function that simulates data given parameters and returns summary statistics.
            It should take a parameter vector and return a statistics vector.
        """
        self.observed_statistics = observed_statistics
        self.simulator = simulator
        
        # Determine dimensions from observed statistics
        self.n_statistics = observed_statistics.shape[0]
        
        # Run iterations
        for t in range(self.n_iterations):
            self.current_iteration = t
            logging.info(f"Running ABC-SMC-RF iteration {t+1}/{self.n_iterations}")
            
            # First iteration: sample from prior
            if t == 0:
                self._first_iteration()
            # Subsequent iterations: sample from previous posterior and perturb
            else:
                self._subsequent_iteration()
                
            # Build random forest with current samples
            self._build_rf_model(t)
            
            # Compute weights for current samples
            self._compute_weights(t)
            
            logging.info(f"Iteration {t+1} completed with {len(self.parameter_samples[t])} particles")
            
        return self
    
    def _first_iteration(self) -> None:
        """
        Execute the first iteration of ABC-SMC-RF (sampling from prior).
        """
        # Sample parameters from prior
        parameters = self.prior_sampler(self.n_particles)
        
        # Determine parameter dimension
        if parameters.ndim == 1:
            parameters = parameters.reshape(-1, 1)
        self.n_parameters = parameters.shape[1]
        
        # Simulate statistics for each parameter set
        statistics = np.zeros((self.n_particles, self.n_statistics))
        for i in range(self.n_particles):
            statistics[i] = self.simulator(parameters[i])
            
        # Store results
        self.parameter_samples.append(parameters)
        self.statistics.append(statistics)
        
    def _subsequent_iteration(self) -> None:
        """
        Execute a subsequent iteration of ABC-SMC-RF (sampling from previous posterior).
        """
        prev_parameters = self.parameter_samples[-1]
        prev_weights = self.weights[-1]
        
        # Allocate arrays for new samples
        parameters = np.zeros((self.n_particles, self.n_parameters))
        statistics = np.zeros((self.n_particles, self.n_statistics))
        
        # Sample and perturb until we have enough particles
        i = 0
        while i < self.n_particles:
            # Sample from previous posterior
            idx = self.rng.choice(len(prev_parameters), p=prev_weights)
            theta_star = prev_parameters[idx]
            
            # Perturb the parameters
            theta_candidate = self.perturbation_kernel(theta_star)
            
            # Check if the perturbed parameters have non-zero prior density
            prior_density = self.prior_pdf(theta_candidate)
            if prior_density <= 0:
                continue
            
            # Simulate data with the new parameters
            try:
                sim_statistics = self.simulator(theta_candidate)
                
                # Store results
                parameters[i] = theta_candidate
                statistics[i] = sim_statistics
                i += 1
            except Exception as e:
                logging.warning(f"Simulation failed for parameters {theta_candidate}: {e}")
                continue
        
        # Store results
        self.parameter_samples.append(parameters)
        self.statistics.append(statistics)
        
    def _build_rf_model(self, t: int) -> None:
        """
        Build a random forest model for the current iteration.
        
        Parameters:
        -----------
        t : int
            Current iteration index.
        """
        parameters = self.parameter_samples[t]
        statistics = self.statistics[t]
        
        # Choose the appropriate RF type
        if self.rf_type == 'RF':
            if parameters.shape[1] > 1:
                logging.warning("Multiple parameters detected but using ABC-RF (univariate). "
                               "Consider using rf_type='DRF' for multivariate inference.")
                
            # Create separate models for each parameter
            models = []
            for p in range(parameters.shape[1]):
                model = RF(
                    n_trees=self.n_trees,
                    min_samples_leaf=self.min_samples_leaf,
                    n_try=self.n_try,
                    random_state=self.rng.randint(0, 2**31)
                )
                model.fit(parameters[:, p], statistics)
                models.append(model)
                
            self.rf_models.append(models)
                
        else:  # DRF
            model = DRF(
                n_trees=self.n_trees,
                min_samples_leaf=self.min_samples_leaf,
                n_try=self.n_try,
                random_state=self.rng.randint(0, 2**31),
                criterion=self.criterion
            )
            model.fit(parameters, statistics)
            self.rf_models.append([model])
            
    def _compute_weights(self, t: int) -> None:
        """
        Compute weights for the current samples.
        
        Parameters:
        -----------
        t : int
            Current iteration index.
        """
        if self.rf_type == 'RF':
            # Combine weights from separate RF models
            combined_weights = np.ones(self.n_particles)
            
            for p, model in enumerate(self.rf_models[t]):
                weights_p = model.predict_weights(self.observed_statistics)
                combined_weights *= weights_p
                
            # Renormalize
            if np.sum(combined_weights) > 0:
                combined_weights /= np.sum(combined_weights)
                
            self.weights.append(combined_weights)
                
        else:  # DRF
            model = self.rf_models[t][0]
            weights = model.predict_weights(self.observed_statistics)
            self.weights.append(weights)
            
    def posterior_sample(self, n_samples: int = 1000) -> np.ndarray:
        """
        Generate samples from the final posterior distribution.
        
        Parameters:
        -----------
        n_samples : int, default=1000
            Number of samples to generate.
            
        Returns:
        --------
        samples : np.ndarray of shape (n_samples, n_parameters)
            Samples from the posterior distribution.
        """
        if not self.parameter_samples:
            raise ValueError("No samples available. Call fit() first.")
            
        final_parameters = self.parameter_samples[-1]
        final_weights = self.weights[-1]
        
        idx = self.rng.choice(len(final_parameters), size=n_samples, p=final_weights)
        return final_parameters[idx]
    
    def get_iteration_results(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the parameters and weights from a specific iteration.
        
        Parameters:
        -----------
        t : int
            Iteration index (0-based).
            
        Returns:
        --------
        parameters : np.ndarray
            Parameter samples from the specified iteration.
        weights : np.ndarray
            Corresponding weights.
        """
        if t < 0 or t >= len(self.parameter_samples):
            raise ValueError(f"Invalid iteration index {t}. Must be between 0 and {len(self.parameter_samples)-1}.")
            
        return self.parameter_samples[t], self.weights[t]
    
    def get_variable_importance(self, t: Optional[int] = None) -> np.ndarray:
        """
        Get the importance of each summary statistic.
        
        Parameters:
        -----------
        t : int, optional
            Iteration index (0-based). If None, the last iteration is used.
            
        Returns:
        --------
        importances : np.ndarray
            Importance of each summary statistic.
        """
        if not self.rf_models:
            raise ValueError("No models available. Call fit() first.")
            
        if t is None:
            t = len(self.rf_models) - 1
            
        if t < 0 or t >= len(self.rf_models):
            raise ValueError(f"Invalid iteration index {t}. Must be between 0 and {len(self.rf_models)-1}.")
            
        if self.rf_type == 'RF':
            # Combine importances from separate RF models
            combined_importances = np.zeros(self.n_statistics)
            
            for model in self.rf_models[t]:
                importances = model.variable_importance()
                combined_importances += importances
                
            # Normalize
            if np.sum(combined_importances) > 0:
                combined_importances /= np.sum(combined_importances)
                
            return combined_importances
                
        else:  # DRF
            model = self.rf_models[t][0]
            return model.variable_importance()