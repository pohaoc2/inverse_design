import numpy as np
from typing import Optional, List, Dict, Union, Tuple, Callable, Any, Literal
import logging
from copy import deepcopy
from inverse_design.rf.rf import RF
from inverse_design.rf.drf import DRF
import multiprocessing as mp
from tqdm import tqdm
import concurrent.futures
import os
from pathlib import Path
import re
import pandas as pd
from inverse_design.examples.run_simulations import run_simulations
from inverse_design.common.enum import Target
#from inverse_design.examples.simulation_metrics import SimulationMetrics

class ABCSMCRF:
    """
    Implementation of ABC Sequential Monte Carlo with Random Forests.
    
    This class implements Algorithms 5 and 6 from the paper, suitable for
    iteratively improving parameter inference using RF methods.
    """
    
    def __init__(
        self,
        n_iterations: int = 5,
        sobol_power: int = 2,
        rf_type: Literal['RF', 'DRF'] = 'DRF',
        n_trees: int = 500,
        min_samples_leaf: int = 5,
        param_ranges: Dict[str, Tuple[float, float]] = None,
        n_try: Optional[int] = None,
        random_state: Optional[int] = None,
        criterion: Literal['CART', 'MMD'] = 'CART',
        perturbation_kernel: Optional[Callable] = None,
        prior_pdf: Optional[Callable] = None
    ):
        """
        Initialize the ABC-SMC-RF model.
        
        Parameters:
        -----------
        n_iterations : int, default=5
            Number of SMC iterations.
        sobol_power : int, default=2
            Power of the Sobol sequence.
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
        perturbation_kernel : Callable, optional
            Function to perturb parameters between iterations.
            If None, parameters are perturbed by adding Gaussian noise.
        prior_pdf : Callable, optional
            Function to evaluate the prior density.
            If None, a uniform prior in [0, 1] for each parameter is assumed.
        """
        self.n_iterations = n_iterations
        self.sobol_power = sobol_power
        self.rf_type = rf_type
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.n_try = n_try
        self.random_state = random_state
        self.criterion = criterion
        
        # Initialize RNGs
        self.rng = np.random.RandomState(random_state)
        
        self.param_ranges = param_ranges
        self.perturbation_kernel = perturbation_kernel if perturbation_kernel is not None else self._default_perturbation_kernel
        self.prior_pdf = prior_pdf if prior_pdf is not None else self._default_prior_pdf
        
        self.parameter_samples = []
        self.weights = []
        self.statistics = []
        self.rf_models = []
        
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

    def _run_parallel_simulations(self, input_dir: str, output_dir: str, jar_path: str) -> List[str]:
        """Run ARCADE simulations in parallel."""
        from inverse_design.examples.run_simulations import run_simulations
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Run simulations
        run_simulations(
            input_dir=input_dir+"/inputs",
            output_dir=output_dir,
            jar_path=jar_path,
            max_workers=int(mp.cpu_count()/2),
            start_index=1
        )

    def fit(
        self, 
        observed_statistics: List[Target], 
        input_dir: str, 
        output_dir: str, 
        jar_path: str,
        timestamps: List[str]
    ) -> None:
        """
        Run the ABC-SMC-RF algorithm with ARCADE simulations.
        
        Parameters:
        -----------
        observed_statistics : List[Target]
            List of Target objects
        input_dir : str
            Directory containing input XML files
        output_dir : str
            Directory for simulation outputs
        jar_path : str
            Path to ARCADE jar file
        timestamps : List[str]
            List of timestamps to analyze (e.g., ["000000", "000720", ...])
        """
        self.observed_statistics = observed_statistics
        self.n_statistics = len(observed_statistics)
        # Run iterations
        for t in range(self.n_iterations):
            self.current_iteration = t
            logging.info(f"Running ABC-SMC-RF iteration {t+1}/{self.n_iterations}")
            
            # First iteration: sample from prior
            if t == 0:
                self._first_iteration(input_dir, output_dir, jar_path, timestamps)
            # Subsequent iterations: sample from previous posterior and perturb
            else:
                self._subsequent_iteration(input_dir, output_dir, jar_path, timestamps)
                
            # Build random forest with current samples
            self._build_rf_model(t)
            # Compute weights for current samples
            self._compute_weights(t)
            
            logging.info(f"Iteration {t+1} completed with {len(self.parameter_samples[t])} particles")
            
        return self

    def _analyze_simulation_results(self, output_dir: str, dir_postfix: str, timestamps: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze simulation results and extract statistics and parameters.
        
        Parameters:
        -----------
        output_dir : str
            Base output directory
        dir_postfix : str
            Directory suffix for current iteration
        timestamps : List[str]
            List of timestamps to analyze
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames of statistics and valid parameters
        """
        from inverse_design.analyze.save_aggregated_results import SimulationMetrics
        
        # Analyze simulation results
        metrics_calculator = SimulationMetrics(output_dir + dir_postfix)
        metrics_calculator.analyze_all_simulations(timestamps)
        metrics_calculator.extract_and_save_parameters(output_dir + dir_postfix + "/inputs")

        # Extract statistics from analysis results
        sim_folders = sorted(
            [f for f in Path(output_dir + dir_postfix).glob("inputs/input_*")],
            key=lambda x: int(re.search(r"input_(\d+)", x.name).group(1)),
        )
        statistics = pd.read_csv(f"{output_dir}/{dir_postfix}/final_metrics.csv")
        valid_parameters = pd.read_csv(f"{output_dir}/{dir_postfix}/all_param_df.csv")
        statistics.drop(columns=["input_folder", "states"], inplace=True)
        valid_parameters.drop(columns=["input_folder"], inplace=True)
        
        return statistics, valid_parameters

    def _first_iteration(self, input_dir: str, output_dir: str, jar_path: str, timestamps: List[str]) -> None:
        """Execute the first iteration of ABC-SMC-RF (sampling from prior)."""
        # Generate input files using prior sampler
        from inverse_design.utils.create_input_files import generate_perturbed_parameters
        
        # Generate input XML files
        dir_postfix = f"iter_{self.current_iteration}"
        generate_perturbed_parameters(
            sobol_power=self.sobol_power,
            param_ranges=self.param_ranges,
            output_dir=input_dir + dir_postfix,
            template_path="test_abc_smc_rf.xml"
        )
        # Run simulations in parallel
        self._run_parallel_simulations(input_dir + dir_postfix, output_dir + dir_postfix, jar_path)
        
        statistics, valid_parameters = self._analyze_simulation_results(output_dir, dir_postfix, timestamps)
        target_stats = np.array([statistics[target.metric.value] for target in self.observed_statistics])
        target_stats = target_stats.reshape(-1, self.n_statistics)
        self.parameter_samples.append(np.array(valid_parameters))
        self.statistics.append(target_stats)

    def _subsequent_iteration(self, input_dir: str, output_dir: str, jar_path: str, timestamps: List[str]) -> None:
        """
        Execute a subsequent iteration of ABC-SMC-RF (sampling from previous posterior).
        """
        from inverse_design.utils.create_input_files import generate_input_files
        prev_parameters = self.parameter_samples[-1]
        prev_weights = self.weights[-1]
        
        # Generate candidate parameters
        n_candidates = 2 ** self.sobol_power
        parameters = np.zeros((n_candidates, prev_parameters.shape[1]))
        
        i = 0
        while i < n_candidates:
            # Sample from previous posterior
            print(f"weights: {len(prev_weights)}")
            idx = self.rng.choice(len(prev_parameters), p=prev_weights)
            theta_star = prev_parameters[idx]
            # Perturb the parameters
            theta_candidate = self.perturbation_kernel(theta_star, self.current_iteration, max_iterations=self.n_iterations, param_ranges=self.param_ranges)
            # Check if the perturbed parameters have non-zero prior density
            prior_density = self.prior_pdf(theta_candidate, param_ranges=self.param_ranges)
            if prior_density > 0:
                parameters[i] = theta_candidate
                i += 1
        dir_postfix = f"iter_{self.current_iteration}"
        generate_input_files(
            param_names=self.param_ranges.keys(),
            param_values=parameters,
            output_dir=input_dir + dir_postfix,
            template_path="test_abc_smc_rf.xml"
        )
        self._run_parallel_simulations(input_dir + dir_postfix, output_dir + dir_postfix, jar_path)

        # Analyze results and get statistics and parameters
        statistics, valid_parameters = self._analyze_simulation_results(output_dir, dir_postfix, timestamps)
        target_stats = np.array([statistics[target.metric.value] for target in self.observed_statistics])
        target_stats = target_stats.reshape(-1, self.n_statistics)
        if len(valid_parameters) < n_candidates:
            logging.warning(f"Only {len(valid_parameters)} valid simulations out of {n_candidates} attempts")

        # Take only the first n_particles
        # Store results
        self.parameter_samples.append(parameters)
        self.statistics.append(target_stats)
        
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
                    random_state=self.rng.randint(42)
                )
                model.fit(parameters[:, p], statistics)
                models.append(model)
                
            self.rf_models.append(models)
                
        else:  # DRF
            model = DRF(
                n_trees=self.n_trees,
                min_samples_leaf=self.min_samples_leaf,
                n_try=self.n_try,
                random_state=self.rng.randint(42),
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
