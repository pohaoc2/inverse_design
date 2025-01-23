# Approximate Bayesian Computation for inverse design

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from bdm import BDM
from config import BDMConfig, ABCConfig
import logging
from enum import Enum
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

class Metric(Enum):
    DENSITY = "density"
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
        self.num_samples = abc_config.num_samples
        self.epsilon = abc_config.epsilon
        self.parameter_ranges = abc_config.parameter_ranges
        self.output_frequency = abc_config.output_frequency
        self.metric_functions = {
            Metric.DENSITY: self._calculate_density,
        }
        self.all_accept_metrics = {
            'all': [],
            'accepted': []
        }
    
    def _calculate_density(self, grid_states: List, time_points: List[float], 
                         target_time: Optional[float] = None) -> float:
        """Calculate cell density from grid states"""
        if target_time is None:
            grid = grid_states[-1]
        else:
            idx = np.argmin(np.abs(np.array(time_points) - target_time))
            grid = grid_states[idx]
            
        return grid.num_cells / (grid.lattice_size**2) * 100
    
    def calculate_metrics(self, grid_states: List, time_points: List[float],
                         target_time: Optional[float] = None) -> Dict[Metric, float]:
        """
        Calculate all required metrics from grid states
        Args:
            grid_states: List of Grid objects at different time points
            time_points: List of corresponding time points
            target_time: Time point at which to calculate metrics
        Returns:
            Dictionary mapping metrics to their calculated values
        """
        return {
            target.metric: self.metric_functions[target.metric](
                grid_states, time_points, target_time
            )
            for target in self.targets
        }
    
    def calculate_distance(self, metrics: Dict[Metric, float]) -> float:
        """
        Calculate weighted distance between calculated metrics and targets
        Args:
            metrics: Dictionary of calculated metric values
        Returns:
            Total weighted distance
        """
        total_distance = 0.0
        total_weight = sum(target.weight for target in self.targets)
        
        for target in self.targets:
            value = metrics[target.metric]
            distance = abs(value - target.value)
            total_distance += (distance * target.weight) / total_weight
            
        return total_distance
    
    def sample_parameters(self) -> Dict:
        """Sample parameters from prior distributions"""
        return {
            param: np.random.uniform(ranges.min, ranges.max)
            for param, ranges in self.parameter_ranges.items()
        }
    
    def run_inference(self, target_time: Optional[float] = None) -> Tuple[List[Dict], List[float], Dict]:
        """
        Run ABC inference to find parameters that achieve targets
        Args:
            target_time: Time point at which to evaluate metrics
        Returns:
            accepted_params: List of parameter sets that achieved targets
            distances: List of distances from targets for accepted parameters
            parameter_pdfs: Dictionary mapping parameter names to their KDE estimates
        """
        log = logging.getLogger(__name__)
        accepted_params = []
        distances = []
        
        target_str = ", ".join([f"{t.metric.value}: {t.value}" for t in self.targets])
        log.info(f"Starting ABC inference for targets: {target_str}")
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
        
        # Calculate PDFs using kernel density estimation
        parameter_pdfs = self._estimate_pdfs(accepted_params)
        
        return accepted_params, distances, parameter_pdfs
    
    def _estimate_pdfs(self, accepted_params: List[Dict]) -> Dict:
        """
        Estimate PDFs for each parameter using kernel density estimation
        Args:
            accepted_params: List of accepted parameter dictionaries
        Returns:
            Dictionary mapping parameter names to their KDE objects
        """
        if not accepted_params:
            return {}
            
        # Extract parameter values into arrays
        param_arrays = {
            param: np.array([p[param] for p in accepted_params])
            for param in accepted_params[0].keys()
        }
        
        # Compute KDE for each parameter
        param_pdfs = {}
        for param_name, values in param_arrays.items():
            kde = stats.gaussian_kde(values)
            param_pdfs[param_name] = kde
            
        return param_pdfs
    
    def get_best_parameters(self, target_time: Optional[float] = None) -> Tuple[Dict, Dict]:
        """
        Get the parameter set that best matches the targets and parameter PDFs
        Args:
            target_time: Time point at which to evaluate metrics
        Returns:
            best_params: Best parameter set
            param_pdfs: Dictionary of parameter PDFs
        """
        accepted_params, distances, param_pdfs = self.run_inference(target_time)
        
        if not accepted_params:
            raise ValueError("No parameter sets were accepted. Try increasing epsilon or number of samples.")
            
        best_idx = np.argmin(distances)
        return accepted_params[best_idx], param_pdfs

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def main(cfg: DictConfig):
        # Load configurations
        bdm_config = BDMConfig.from_dictconfig(cfg.bdm)
        abc_config = ABCConfig.from_dictconfig(cfg.abc)
        
        # Define targets
        targets = [Target(Metric.DENSITY, 70.0)]
        
        # Initialize ABC
        abc = ABC(bdm_config, abc_config, targets)
        
        try:
            # Run inference
            accepted_params, distances, param_pdfs = abc.run_inference()
            
            # Print best parameters
            best_idx = np.argmin(distances)
            best_params = accepted_params[best_idx]
            print("\nBest Parameters:")
            for param_name, value in best_params.items():
                print(f"{param_name}: {value:.3f}")
            
            # Plot density distributions
            plt.figure(figsize=(8, 6))
            all_densities = [metrics[Metric.DENSITY] for metrics in abc.all_accept_metrics['all']]
            accepted_densities = [metrics[Metric.DENSITY] for metrics in abc.all_accept_metrics['accepted']]
            
            # Calculate common bins
            bins = np.histogram_bin_edges(all_densities, bins=10)
            
            # Plot histograms with common bins
            plt.hist(all_densities, bins=bins, alpha=0.3, density=True, label='All samples', color='blue')
            plt.hist(accepted_densities, bins=bins, alpha=0.3, density=True, label='Accepted samples', color='orange')
            
            # Add KDE curves
            x_all = np.linspace(0, 100, 200)
            if len(all_densities) > 1:
                kde_all = stats.gaussian_kde(all_densities)
                
                plt.plot(x_all, kde_all(x_all), color='blue', linewidth=2, label='All samples (KDE)')
            
            if len(accepted_densities) > 1:
                kde_accepted = stats.gaussian_kde(accepted_densities)
                plt.plot(x_all, kde_accepted(x_all), color='orange', linewidth=2, label='Accepted samples (KDE)')
            
            plt.axvline(targets[0].value, color='r', linestyle='--', label='Target density')
            plt.xlabel('Density (%)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Densities')
            plt.xlim(0, 100)
            plt.legend()
            plt.savefig("density_distribution.png")
            plt.close()
            
            # Analyze and plot PDFs
            plt.figure(figsize=(12, 4))
            for i, (param_name, kde) in enumerate(param_pdfs.items(), 1):
                # Generate points for plotting using config ranges
                param_range = abc_config.parameter_ranges[param_name]
                x = np.linspace(param_range.min, param_range.max, 1000)
                density = kde(x)
                
                # Calculate statistics
                mean = kde.resample(1000).mean()
                std = kde.resample(1000).std()
                
                # Print statistics
                print(f"\nParameter: {param_name}")
                print(f"Mean: {mean:.3f}")
                print(f"Std: {std:.3f}")
                
                # Plot PDF
                plt.subplot(1, 3, i)
                plt.plot(x, density)
                plt.title(f"{param_name}")
                plt.xlabel("Value")
                plt.ylabel("kde density")
                plt.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
                plt.axvline(mean + std, color='g', linestyle=':', label=f'±1 SD')
                plt.axvline(mean - std, color='g', linestyle=':')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig("abc_results.png")
            plt.close()
        except ValueError as e:
            print(f"Error: {e}")
    
    main()
    

