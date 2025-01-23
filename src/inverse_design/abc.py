# Approximate Bayesian Computation for inverse design

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .config import BDMConfig, ABCConfig
import logging
from enum import Enum
from dataclasses import dataclass
import hydra
from omegaconf import DictConfig, OmegaConf
import scipy
from . import vis
from . import evaluate

class Metric(Enum):
    DENSITY = "density"
    TIME_TO_EQUILIBRIUM = "time_to_equilibrium"
    # Add more metrics as needed, for example:
    # CLUSTERING = "clustering"
    # EDGE_RATIO = "edge_ratio"
    # etc.

@dataclass
class Target:
    metric: Metric
    value: float
    weight: float = 1.0  # For multi-objective weighting
    
class ABC:
    def __init__(self, bdm_config: BDMConfig, abc_config: ABCConfig, targets: Union[Target, List[Target]]):
        """
        Initialize ABC for inverse design
        Args:
            bdm_config: Base configuration for BDM model
            abc_config: Configuration for ABC algorithm
            targets: Single target or list of targets to achieve
        """
        self.base_config = bdm_config
        self.targets = [targets] if isinstance(targets, Target) else targets
        
        # Get ABC parameters from config
        # Calculate power of 2 that gives us at least num_samples
        self.sobol_power = abc_config.sobol_power
        self.epsilon = abc_config.epsilon
        self.parameter_ranges = abc_config.parameter_ranges
        self.output_frequency = abc_config.output_frequency
        
        # Initialize Sobol sequence generator
        self.sobol_engine = scipy.stats.qmc.Sobol(
            d=len(self.parameter_ranges),  # dimension = number of parameters
            scramble=True  # randomize the sequence
        )
        
        # Generate all samples at initialization
        self.samples = self.sobol_engine.random_base2(m=self.sobol_power)
        self.num_samples = len(self.samples)
        self.current_sample_idx = 0
        
        self.metric_functions = {
            Metric.DENSITY: self._calculate_density,
            Metric.TIME_TO_EQUILIBRIUM: self._calculate_time_to_equilibrium,
        }
        self.all_accept_metrics = {
            'all': [],
            'accepted': []
        }
    
    def _calculate_density(self, grid_states: List, time_points: List[float], 
                         target_time: Optional[float] = None) -> float:
        """Calculate cell density from grid states
        Args:
            grid_states: List of Grid objects at different time points
            time_points: List of corresponding time points
            target_time: Time point at which to calculate density (if None, uses last time point)
        Returns:
            Cell density (percentage) at target time or last time point [0, 100]
        """
        if target_time is None:
            grid = grid_states[-1]
        else:
            idx = np.argmin(np.abs(np.array(time_points) - target_time))
            grid = grid_states[idx]
            
        return grid.num_cells / (grid.lattice_size**2) * 100
    
    def _calculate_time_to_equilibrium(self, grid_states: List, time_points: List[float],
                                     threshold: float = 0.05) -> float:
        """Calculate time to reach equilibrium based on cell density fluctuations
        Args:
            grid_states: List of Grid objects at different time points
            time_points: List of corresponding time points
            threshold: Threshold for density fluctuation (default: 0.05)
        Returns:
            Time point when steady state is reached (or max time if not reached)
        """
        densities = []
        for i in range(len(grid_states)):
            density = self._calculate_density(grid_states[:i+1], time_points[:i+1])
            densities.append(density)
        
        window_size = 5  # Number of time points to check for stability
        for i in range(window_size, len(time_points)):
            window = densities[i-window_size:i]
            mean_density = np.mean(window)
            
            # Handle the case where mean_density is 0
            if mean_density == 0:
                fluctuation = np.max(np.abs(np.array(window)))
            else:
                fluctuation = np.max(np.abs(np.array(window) - mean_density) / mean_density)
            
            if fluctuation < threshold:
                return time_points[i]
        
        # If equilibrium is not reached, return the final time point
        return time_points[-1]
    
    def calculate_metrics(self, grid_states: List, time_points: List[float],
                         target_time: Optional[float] = None) -> Dict[Metric, float]:
        """Calculate all required metrics from grid states
        Args:
            grid_states: List of Grid objects at different time points
            time_points: List of corresponding time points
            target_time: Time point at which to calculate metrics (only used for density)
        Returns:
            Dictionary mapping metrics to their calculated values
        """
        metrics = {}
        for target in self.targets:
            if target.metric == Metric.DENSITY:
                metrics[target.metric] = self._calculate_density(
                    grid_states, time_points, target_time
                )
            elif target.metric == Metric.TIME_TO_EQUILIBRIUM:
                metrics[target.metric] = self._calculate_time_to_equilibrium(
                    grid_states, time_points
                )
        return metrics
    
    def calculate_distance(self, metrics: Dict[Metric, float]) -> float:
        """
        Calculate weighted distance between calculated metrics and targets
        Args:
            metrics: Dictionary of calculated metric values
        Returns:
            Total weighted distance using normalized metrics
        """
        total_distance = 0.0
        total_weight = sum(target.weight for target in self.targets)
        
        # Define normalization factors for each metric
        normalization_factors = {
            Metric.DENSITY: 100.0,  # density is in percentage (0-100)
            Metric.TIME_TO_EQUILIBRIUM: self.base_config.output.max_time,  # normalize by max simulation time
            # Add other metrics' normalization factors as needed
        }
        
        for target in self.targets:
            value = metrics[target.metric]
            # Normalize the distance by the appropriate factor
            normalized_distance = abs(value - target.value) / normalization_factors[target.metric]
            total_distance += (normalized_distance * target.weight) / total_weight
            
        return total_distance
    
    def sample_parameters(self) -> Dict:
        """Sample parameters using Sobol sequence for better space-filling properties
        Returns:
            Dictionary of parameter names and their sampled values
        """
        if self.current_sample_idx >= len(self.samples):
            raise ValueError("All samples have been used. Initialize a new ABC instance if needed.")
        
        # Get next point from pre-generated Sobol sequence
        sample = self.samples[self.current_sample_idx]
        self.current_sample_idx += 1
        
        # Scale the sample to the parameter ranges
        params = {}
        for (param_name, ranges), value in zip(self.parameter_ranges.items(), sample):
            min_val = ranges.min
            max_val = ranges.max
            scaled_value = min_val + (max_val - min_val) * value
            params[param_name] = scaled_value
            
        return params
    
    def run_inference(self, target_time: Optional[float] = None) -> Tuple[List[Dict], List[float], Dict]:
        """Run ABC inference to find parameters that achieve targets"""
        from .bdm import BDM
        
        log = logging.getLogger(__name__)
        accepted_params = []
        distances = []
        
        target_str = ", ".join([f"{t.metric.value}: {t.value}" for t in self.targets])
        log.info(f"Starting ABC inference for targets: {target_str}, Total samples: {self.num_samples}")
        self.all_accept_metrics = {
            'all': [],
            'accepted': []
        }
        for i in range(self.num_samples):
            params = self.sample_parameters()
            
            config = self.base_config.copy()
            config.rates.proliferate = params['proliferate']
            config.rates.death = params['death']
            config.rates.migrate = params['migrate']
            
            model = BDM(config)
            time_points, _, grid_states = model.step()
            metrics = self.calculate_metrics(grid_states, time_points, target_time)
            self.all_accept_metrics['all'].append(metrics)
            distance = self.calculate_distance(metrics)
            
            if distance < self.epsilon:
                accepted_params.append(params)
                distances.append(distance)
                self.all_accept_metrics['accepted'].append(metrics)
                
            if (i + 1) % self.output_frequency == 0:
                log.info(f"Processed {i + 1} samples, accepted {len(accepted_params)}")
        log.info(f"Inference complete. Accepted {len(accepted_params)} parameter sets")
        
        return accepted_params, distances
    
    def plot_results(self, accepted_params: List[Dict], pdf_results: Dict, 
                    save_dir: str = "."):
        """Plot all results using the visualization module
        Args:
            accepted_params: List of accepted parameter dictionaries
            pdf_results: Results from _estimate_pdfs
            save_dir: Directory to save plots
        """
        # Plot parameter distributions
        vis.plot_parameter_distributions(
            accepted_params, 
            pdf_results,
            f"{save_dir}/parameter_distributions.png"
        )
        x_ranges = {
            Metric.DENSITY: (0, 100),
            Metric.TIME_TO_EQUILIBRIUM: (0, self.base_config.output.max_time)
        }
        # Plot metric distributions
        for target in self.targets:
            vis.plot_metric_distribution(
                self.all_accept_metrics,
                target.metric,
                self.targets,
                f"{save_dir}/{target.metric.value}_distribution.png",
                x_range=x_ranges[target.metric]
            )

    def get_best_parameters(self, target_time: Optional[float] = None) -> Tuple[Dict, Dict]:
        """
        Get the parameter set that best matches the targets and parameter PDFs
        Args:
            target_time: Time point at which to evaluate metrics
        Returns:
            best_params: Best parameter set
            param_pdfs: Dictionary of parameter PDFs
        """
        
        accepted_params, distances = self.run_inference(target_time)
        
        if not accepted_params:
            raise ValueError("No parameter sets were accepted. Try increasing epsilon or number of samples.")
            
        best_idx = np.argmin(distances)
        return accepted_params[best_idx]

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def main(cfg: DictConfig):
        # Load configurations
        bdm_config = BDMConfig.from_dictconfig(cfg.bdm)
        abc_config = ABCConfig.from_dictconfig(cfg.abc)
        
        # Define targets
        targets = [Target(Metric.DENSITY, 70.0), Target(Metric.TIME_TO_EQUILIBRIUM, 1200.0)]
        
        # Initialize ABC
        abc = ABC(bdm_config, abc_config, targets)
        
        try:
            # Create output directory with timestamp
            from datetime import datetime
            import os
            import json
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"abc_results_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Run inference
            accepted_params, distances = abc.run_inference()
            
            # Get best parameters
            best_idx = np.argmin(distances)
            best_params = accepted_params[best_idx]
            # Print best parameters
            print("\nBest Parameters:")
            for param_name, value in best_params.items():
                print(f"{param_name}: {value:.3f}")

            # Save configuration and results
            config_dict = {
                'timestamp': timestamp,
                'targets': [{'metric': t.metric.value, 'value': t.value, 'weight': t.weight} for t in targets],
                'best_parameters': best_params,
                'bdm_config': OmegaConf.to_container(cfg.bdm, resolve=True),
                'abc_config': OmegaConf.to_container(cfg.abc, resolve=True)
            }
            
            with open(os.path.join(output_dir, 'config.json'), 'w') as f:
                json.dump(config_dict, f, indent=4)
            

            # Calculate PDFs and compare sampling methods
            parameter_pdfs = evaluate.estimate_pdfs(accepted_params)
            sampling_results = evaluate.compare_sampling_methods(abc, parameter_pdfs, num_samples=np.power(2, abc_config.sobol_power-1))
            
            # Save sampling results
            with open(os.path.join(output_dir, 'sampling_results.json'), 'w') as f:
                # Convert any numpy values to float and Metric enum to string for JSON serialization
                json_results = {}
                for method, results in sampling_results.items():
                    json_results[method] = {}
                    for k, v in results.items():
                        if k == 'metrics':
                            # Handle metrics dictionary separately to convert Metric enum keys
                            json_results[method][k] = {
                                metric.value if hasattr(metric, 'value') else str(metric): float(value)
                                for metric, value in v.items()
                            }
                        else:
                            # Handle other values
                            json_results[method][k] = float(v) if isinstance(v, (np.float32, np.float64)) else v
                json.dump(json_results, f, indent=4)
            
            print("\nSampling Method Comparison:")
            for method, results in sampling_results.items():
                print(f"\n{method}:")
                if method == 'independent_means':
                    print(f"  Distance: {results['distance']:.4f}")
                    print("  Metrics:")
                    for metric_name, value in results['metrics'].items():
                        print(f"    {metric_name.value}: {value:.4f}")
                else:  # joint_sampling
                    print(f"  Mean Distance: {results['distance_mean']:.4f}")
                    print(f"  Distance Std: {results['distance_std']:.4f}")
                    print("  Mean Metrics:")
                    for metric_name, value in results['metrics'].items():
                        print(f"    {metric_name.value}: {value:.4f}")
            
            # Plot and save results
            abc.plot_results(accepted_params, parameter_pdfs, save_dir=output_dir)
            
            print(f"\nResults saved to: {output_dir}")
            
        except ValueError as e:
            print(f"Error: {e}")
    
    main()
    

