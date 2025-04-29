from typing import Optional, List, Dict, Union, Tuple, Callable, Any, Literal
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
import graphviz
import logging
import multiprocessing as mp
import concurrent.futures
import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from inverse_design.rf.rf import RF
from inverse_design.rf.drf import DRF
from inverse_design.examples.run_simulations import run_simulations
from inverse_design.common.enum import Target
from inverse_design.rf.abc_smc_rf_base import ABCSMCRFBase

class ABCSMCRF(ABCSMCRFBase):
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
        subsample_ratio: float = 0.5,
        perturbation_kernel: Optional[Callable] = None,
        prior_pdf: Optional[Callable] = None,
        config_params: Optional[Dict[str, Dict[str, str]]] = None
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
        config_params: Optional[Dict[str, Dict[str, str]]] = None
            The configuration to perturb.
        """
        super().__init__(n_iterations, rf_type, n_trees, min_samples_leaf, n_try, random_state, criterion)
        self.sobol_power = sobol_power
        self.param_ranges = param_ranges
        self.subsample_ratio = subsample_ratio
        self.perturbation_kernel = perturbation_kernel if perturbation_kernel is not None else self._default_perturbation_kernel
        self.prior_pdf = prior_pdf if prior_pdf is not None else self._default_prior_pdf
        self.config_params = config_params
        self.scaler = None
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
        all_output_names = [f"input_{i}" for i in range(1, 2 ** self.sobol_power + 1)]
        if os.path.exists(output_dir):
            current_output_names = [f.name for f in Path(output_dir + f"/inputs").glob("input_*")]
            missing_output_indices = [int(f.split("_")[-1].split(".")[0]) for f in all_output_names if f not in current_output_names]
            if len(missing_output_indices) == 0:
                print(f"ARCADE simulations for {output_dir} already exist, and have {len(current_output_names)} outputs (target = {2 ** self.sobol_power}), skipping generation")
                return
            else:
                print(f"ARCADE simulations for {output_dir} exist but have {len(current_output_names)} outputs, expected {2 ** self.sobol_power}")
        else:
            os.makedirs(output_dir, exist_ok=True)
            missing_output_indices = list(range(1, 2 ** self.sobol_power + 1))
        # Run simulations
        
        run_simulations(
            input_dir=input_dir+"/inputs",
            output_dir=output_dir,
            jar_path=jar_path,
            max_workers=int(mp.cpu_count()/2),
            running_index=missing_output_indices
        )

    def fit(
        self, 
        target_names: List[str],
        target_values: List[float],
        input_dir: str, 
        output_dir: str, 
        jar_path: str,
        timestamps: List[str]
    ) -> None:
        """
        Run the ABC-SMC-RF algorithm with ARCADE simulations.
        
        Parameters:
        -----------
        target_names : List[str]
            List of target names
        target_values : List[float]
            List of target values
        input_dir : str
            Directory containing input XML files
        output_dir : str
            Directory for simulation outputs
        jar_path : str
            Path to ARCADE jar file
        timestamps : List[str]
            List of timestamps to analyze (e.g., ["000000", "000720", ...])
        """
        self.target_names = target_names
        self.target_values = target_values
        self.n_statistics = len(target_names)
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

    def _remove_invalid_rows(self, statistics: np.ndarray, valid_parameters: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Remove rows containing NaN or infinite values.
        """
        indices = np.where(np.isnan(statistics).any(axis=1) | np.isinf(statistics).any(axis=1))[0]
        statistics = np.delete(statistics, indices, axis=0)
        valid_parameters = valid_parameters.drop(valid_parameters.index[indices])
        return statistics, valid_parameters

    def normalize_statistics(self, statistics: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
        """
        Normalize statistics to be between 0 and 1.
        """
        scaler = StandardScaler()
        scaler.fit(statistics)
        return scaler.transform(statistics), scaler

    def _analyze_simulation_results(self, input_dir: str, output_dir: str, dir_postfix: str, timestamps: List[str]) -> Tuple[np.ndarray, np.ndarray]:
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
        sim_folders = sorted(
            [f for f in Path(output_dir + dir_postfix).glob("inputs/input_*")],
            key=lambda x: int(re.search(r"input_(\d+)", x.name).group(1)),
        )
        sim_folders = sim_folders[:2 ** self.sobol_power]
        # Analyze simulation results

        if not os.path.exists(f"{output_dir}/{dir_postfix}/final_metrics.csv"):
            metrics_calculator = SimulationMetrics(output_dir + dir_postfix, input_dir + dir_postfix)
            metrics_calculator.analyze_all_simulations(timestamps, sim_folders)
            metrics_calculator.extract_and_save_parameters(sim_folders)
            statistics = pd.read_csv(f"{output_dir}/{dir_postfix}/final_metrics.csv")
            valid_parameters = pd.read_csv(f"{output_dir}/{dir_postfix}/all_param_df.csv")
            statistics.drop(columns=["input_folder", "states"], inplace=True)
            valid_parameters.drop(columns=["input_folder"], inplace=True)
        else:
            statistics = pd.read_csv(f"{output_dir}/{dir_postfix}/final_metrics.csv")
            valid_parameters = pd.read_csv(f"{output_dir}/{dir_postfix}/all_param_df.csv")
            statistics.drop(columns=["input_folder", "states"], inplace=True)
            valid_parameters.drop(columns=["input_folder"], inplace=True)
        if len(valid_parameters.columns) != len(self.param_ranges.keys()):
            valid_parameters = valid_parameters[self.param_ranges.keys()]

        if self.config_params["point_based"]:
            valid_parameters.drop(columns=["CAPILLARY_DENSITY"], inplace=True)
        else:
            valid_parameters.drop(columns=["DISTANCE_TO_CENTER"], inplace=True)
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
            config_params=self.config_params,
            output_dir=input_dir + dir_postfix,
        )
        # Run simulations in parallel
        self._run_parallel_simulations(input_dir + dir_postfix, output_dir + dir_postfix, jar_path)
        
        statistics, valid_parameters = self._analyze_simulation_results(input_dir, output_dir, dir_postfix, timestamps)
        # Debug: Check for infinite values
        target_stats_array = np.array([statistics[target_name] for target_name in self.target_names]).T
        target_stats_array, valid_parameters = self._remove_invalid_rows(target_stats_array, valid_parameters)
        target_stats, self.scaler = self.normalize_statistics(target_stats_array)
        self.target_values = self.scaler.transform(np.array(self.target_values).reshape(-1, len(self.target_values))).flatten()
        self.parameter_samples.append(np.array(valid_parameters))
        self.parameter_columns = list(valid_parameters.columns)
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
        parameters = np.zeros((n_candidates, prev_parameters.shape[1]), dtype=object)
        
        i = 0
        while i < n_candidates:
            # Sample from previous posterior
            idx = self.rng.choice(len(prev_parameters), p=prev_weights)
            theta_star = prev_parameters[idx]
            # Perturb the parameters
            theta_candidate = self.perturbation_kernel(theta_star,
                                                        self.parameter_columns,
                                                        self.current_iteration,
                                                        max_iterations=self.n_iterations,
                                                        param_ranges=self.param_ranges,
                                                        config_params=self.config_params
            )
            updated_param_ranges = {param_name: (min_val, max_val) for param_name, (min_val, max_val) in self.param_ranges.items() if param_name in self.parameter_columns}
            prior_density = self.prior_pdf(theta_candidate, self.parameter_columns, param_ranges=updated_param_ranges, config_params=self.config_params)
            if prior_density > 0:
                parameters[i] = theta_candidate
                i += 1

        dir_postfix = f"iter_{self.current_iteration}"
        if self.config_params["point_based"]:
            # Create new parameters array with one extra column for x_center
            added_params = np.zeros((n_candidates, parameters.shape[1] + 1), dtype=object)
            for i, param in enumerate(parameters):
                x_center = int((6 * self.config_params["radius_bound"] - 3) / 2)
                # Insert x_center at index 0
                added_params[i] = np.insert(param, 0, f"{x_center-1}:{x_center+1}")
            # Update parameters with the new array that includes x_center
            parameters = added_params
            
            new_param_ranges = {"X_SPACING": (1, 15)}
            for key, value in self.param_ranges.items():
                if key != "X_SPACING":
                    new_param_ranges[key] = value
            param_names = new_param_ranges.keys()

        generate_input_files(
            param_names=self.parameter_columns,
            param_values=parameters,
            output_dir=input_dir + dir_postfix,
            config_params=self.config_params,
        )
        self._run_parallel_simulations(input_dir + dir_postfix, output_dir + dir_postfix, jar_path)

        # Analyze results and get statistics and parameters
        statistics, valid_parameters = self._analyze_simulation_results(input_dir, output_dir, dir_postfix, timestamps)
        target_stats = np.array([statistics[target_name] for target_name in self.target_names]).T
        target_stats, valid_parameters = self._remove_invalid_rows(target_stats, valid_parameters)
        target_stats = self.scaler.transform(target_stats)
        if len(valid_parameters) < n_candidates:
            logging.warning(f"Only {len(valid_parameters)} valid simulations out of {n_candidates} attempts")
        self.parameter_samples.append(np.array(valid_parameters))
        self.statistics.append(target_stats)
            
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
            weights = model.predict_weights(self.target_values)
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
    