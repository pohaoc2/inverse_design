from typing import Optional, List, Dict, Union, Tuple, Callable, Any, Literal
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
import logging
import multiprocessing as mp
import concurrent.futures
import os
import re
import pandas as pd
import numpy as np
from inverse_design.rf.rf import RF
from inverse_design.rf.drf import DRF
from inverse_design.examples.run_simulations import run_simulations
from inverse_design.common.enum import Target
#from inverse_design.examples.simulation_metrics import SimulationMetrics
import graphviz

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
        subsample_ratio: float = 0.5,
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
        self.subsample_ratio = subsample_ratio
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
        start_index = 1
        # Check if the input directory exists, if exists, skip the generation
        if os.path.exists(output_dir):
            num_outputs = len([f for f in Path(output_dir + f"/inputs").glob("input_*")])
            if num_outputs >= 2 ** self.sobol_power:
                print(f"ARCADE simulations for {output_dir} already exist, and have {num_outputs} outputs (target = {2 ** self.sobol_power}), skipping generation")
                return
            else:
                print(f"ARCADE simulations for {output_dir} exist but have {num_outputs} outputs, expected {2 ** self.sobol_power}")
                start_index = num_outputs + 1
        else:
            os.makedirs(output_dir, exist_ok=True)
            # Run simulations
        run_simulations(
            input_dir=input_dir+"/inputs",
            output_dir=output_dir,
            jar_path=jar_path,
            max_workers=int(mp.cpu_count()/2),
            start_index=start_index
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
        sim_folders = sorted(
            [f for f in Path(output_dir + dir_postfix).glob("inputs/input_*")],
            key=lambda x: int(re.search(r"input_(\d+)", x.name).group(1)),
        )
        sim_folders = sim_folders[:2 ** self.sobol_power]
        # Analyze simulation results

        if not os.path.exists(f"{output_dir}/{dir_postfix}/final_metrics.csv"):
            metrics_calculator = SimulationMetrics(output_dir + dir_postfix)
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
            template_path="sample_input_v3.xml"
        )
        # Run simulations in parallel
        self._run_parallel_simulations(input_dir + dir_postfix, output_dir + dir_postfix, jar_path)
        
        statistics, valid_parameters = self._analyze_simulation_results(output_dir, dir_postfix, timestamps)
        target_stats = np.array([statistics[target_name] for target_name in self.target_names]).T
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
            template_path="sample_input_v3.xml"
        )
        self._run_parallel_simulations(input_dir + dir_postfix, output_dir + dir_postfix, jar_path)

        # Analyze results and get statistics and parameters
        statistics, valid_parameters = self._analyze_simulation_results(output_dir, dir_postfix, timestamps)
        target_stats = np.array([statistics[target_name] for target_name in self.target_names]).T
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
                criterion=self.criterion,
                subsample_ratio=self.subsample_ratio
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
            
        return self.parameter_samples[t], self.statistics[t], self.weights[t]
    
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

    def plot_tree(self, iteration: int = -1, tree_index: int = 0, max_depth: int = 3, 
                    feature_names: Optional[List[str]] = None) -> None:
        """
        Plot a visualization of a tree from the random forest with weighted leaf nodes and color bar.
        
        Parameters:
        -----------
        iteration : int, default=-1
            Which iteration's model to visualize (-1 means last iteration)
        tree_index : int, default=0
            Which tree to visualize
        max_depth : int, default=3
            Maximum depth of the tree to display
        feature_names : List[str], optional
            Names of the features/statistics
        """
        # Get the model from the specified iteration
        if iteration < 0:
            iteration = len(self.rf_models) + iteration
            
        if iteration < 0 or iteration >= len(self.rf_models):
            print(f"Invalid iteration index {iteration}. Available iterations: 0-{len(self.rf_models)-1}")
            return
            
        models = self.rf_models[iteration]
        
        # Handle model selection based on RF type
        if self.rf_type == 'DRF':
            if len(models) == 0:
                print("No models available for this iteration")
                return
                
            model = models[0]
            if tree_index >= len(model.trees):
                print(f"Invalid tree index {tree_index}. Available trees: 0-{len(model.trees)-1}")
                return
                
            tree = model.trees[tree_index]
        else:  # RF
            if tree_index >= len(models):
                print(f"Invalid model index {tree_index}. Available models: 0-{len(models)-1}")
                return
                
            model = models[tree_index]
            if len(model.trees) == 0:
                print("No trees available for this model")
                return
                
            tree = model.trees[0]

        # Feature names handling
        n_stats = 0
        for node in tree['nodes']:
            if not node.get('leaf', True) and 'split_stat_idx' in node:
                n_stats = max(n_stats, node['split_stat_idx'] + 1)
        
        if feature_names is None:
            feature_names = [f'statistic_{i}' for i in range(n_stats)]
        elif len(feature_names) < n_stats:
            feature_names = feature_names + [f'statistic_{i}' for i in range(len(feature_names), n_stats)]

        # Create DOT data
        dot = graphviz.Digraph(comment='Decision Tree')
        dot.attr(rankdir='TB')
        dot.attr('node', shape='box', style='rounded,filled')

        # Create a node lookup dictionary for faster access
        node_dict = {node['id']: node for node in tree['nodes']}
        
        # Calculate weights for all leaf nodes
        leaf_weights = []
        for node in tree['nodes']:
            if node.get('leaf', True):
                samples = node.get('samples', [])
                if len(samples) > 0:
                    if self.rf_type == 'DRF':
                        weights = self.weights[iteration][samples]
                        avg_weight = np.mean(weights)
                    else:  # RF
                        weights = np.ones(len(samples))
                        for model in self.rf_models[iteration]:
                            weights *= model.predict_weights(self.target_values)[samples]
                        avg_weight = np.mean(weights)
                    leaf_weights.append(avg_weight)

        # Normalize weights to [0, 1] for color scaling
        def get_color_for_weight(weight):
            """Convert weight to a color between red (low) and green (high)"""
            if not leaf_weights:
                return '#lightgreen'
            
            # Normalize weight to [0, 1]
            normalized = (weight - min_weight) / weight_range
            
            # Convert to RGB
            red = int(255 * (1 - normalized))
            green = int(255 * normalized)
            return f'#{red:02x}{green:02x}00'

        if leaf_weights:
            min_weight = min(leaf_weights)
            max_weight = max(leaf_weights)
            weight_range = max_weight - min_weight if max_weight > min_weight else 1.0
            
            # Create color bar legend
            with dot.subgraph(name='cluster_legend') as legend:
                legend.attr(label='Weight Legend')
                legend.attr(style='rounded')
                legend.attr(color='black')
                
                # Create color bar with 5 segments
                n_segments = 5
                for i in range(n_segments):
                    normalized = i / (n_segments - 1)
                    weight = min_weight + normalized * weight_range
                    color = get_color_for_weight(weight)
                    
                    # Create a node for this color segment
                    legend.node(
                        f'legend_{i}',
                        f'{weight:.2e}',
                        shape='box',
                        style='filled',
                        fillcolor=color,
                        fontcolor='black' if normalized > 0.5 else 'white'
                    )
                    
                    # Connect segments
                    if i > 0:
                        legend.edge(f'legend_{i-1}', f'legend_{i}', style='invis')
                
                # Force legend nodes to be arranged horizontally
                legend.attr(rankdir='LR')
                legend.attr(rank='sink')
        
        # Create main graph cluster
        with dot.subgraph(name='cluster_main') as main:
            main.attr(label='Decision Tree')
            
            def add_node_to_graph(node_id, depth=0):
                if node_id in processed_nodes:
                    return
                
                processed_nodes.add(node_id)
                node = node_dict[node_id]
                dot_node_id = f"node_{node_id}"
                
                # Calculate sample count for this node
                sample_count = len(node.get('samples', []))
                
                if node.get('leaf', True):
                    samples = node.get('samples', [])
                    if len(samples) > 0:
                        if self.rf_type == 'DRF':
                            weights = self.weights[iteration][samples]
                            avg_weight = np.mean(weights)
                        else:
                            weights = np.ones(len(samples))
                            for model in self.rf_models[iteration]:
                                weights *= model.predict_weights(self.target_values)[samples]
                            avg_weight = np.mean(weights)
                        
                        color = get_color_for_weight(avg_weight)
                        label = f"Leaf Node\nsamples = {sample_count}\nweight = {avg_weight:.3e}"
                    else:
                        color = '#lightgray'
                        label = f"Leaf Node\nsamples = {sample_count}"
                    
                    main.node(dot_node_id, label, fillcolor=color, style='filled,rounded')
                else:
                    if 'split_stat_idx' in node and node['split_stat_idx'] < len(feature_names):
                        stat_name = feature_names[node['split_stat_idx']]
                    else:
                        stat_name = f"feature_{node.get('split_stat_idx', 'unknown')}"
                        
                    threshold = node.get('threshold', 0)
                    label = f"{stat_name} ≤ {threshold:.3f}\nsamples = {sample_count}"
                    main.node(dot_node_id, label, fillcolor='lightblue', style='filled,rounded')

            def add_tree_recursive(node_id, depth=0, visited=None):
                if visited is None:
                    visited = set()
                
                if node_id in visited or depth > max_depth:
                    return
                
                visited.add(node_id)
                add_node_to_graph(node_id, depth)
                
                node = node_dict[node_id]
                dot_node_id = f"node_{node_id}"
                
                if not node.get('leaf', True) and 'left_child' in node and 'right_child' in node:
                    left_child = node['left_child']
                    right_child = node['right_child']
                    
                    left_edge = (node_id, left_child, 'yes')
                    right_edge = (node_id, right_child, 'no')
                    
                    if left_edge not in processed_edges:
                        processed_edges.add(left_edge)
                        left_dot_id = f"node_{left_child}"
                        main.edge(dot_node_id, left_dot_id, 'yes')
                        add_tree_recursive(left_child, depth + 1, visited.copy())
                    
                    if right_edge not in processed_edges:
                        processed_edges.add(right_edge)
                        right_dot_id = f"node_{right_child}"
                        main.edge(dot_node_id, right_dot_id, 'no')
                        add_tree_recursive(right_child, depth + 1, visited.copy())

            # Process all nodes and their connections
            processed_nodes = set()
            processed_edges = set()
            
            # Start with the root node (ID 0)
            add_tree_recursive(0)

        # Render the tree
        output_path = f"tree_iter_{iteration}_index_{tree_index}"
        try:
            dot.render(output_path, format="png", cleanup=True)
            print(f"Tree visualization with color bar saved as {output_path}.png")
        except Exception as e:
            print(f"Error rendering tree: {e}")
            print("\nGraphviz DOT source:")
            print(dot.source)