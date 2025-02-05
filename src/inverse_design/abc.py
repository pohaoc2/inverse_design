# Approximate Bayesian Computation for inverse design
import logging
from typing import Dict, List, Optional, Union
import scipy
import numpy as np
import pandas as pd
from .config import BDMConfig, ABCConfig, ARCADEConfig
from .metrics import Metric, Target, MetricsFactory
from .models.models import ModelFactory
from .models.parameters import ParameterFactory
from .utils import get_samples_data


class ABC:
    def __init__(
        self,
        model_config: Union[BDMConfig, ARCADEConfig],
        abc_config: ABCConfig,
        targets: Union[Target, List[Target]],
    ):
        """
        Initialize ABC for inverse design
        Args:
            model_config: Base configuration for the model (BDM, ARCADE, etc.)
            abc_config: Configuration for ABC algorithm
            targets: Single target or list of targets to achieve
        """
        self.model_config = model_config
        self.abc_config = abc_config
        self.targets = [targets] if isinstance(targets, Target) else targets

        # Get ABC parameters from config
        self.sobol_power = abc_config.sobol_power
        self.epsilon = abc_config.epsilon
        self.parameter_ranges = abc_config.parameter_ranges
        self.output_frequency = abc_config.output_frequency

        # Initialize Sobol sequence generator
        self.sobol_engine = scipy.stats.qmc.Sobol(
            d=len(self.parameter_ranges),  # dimension = number of parameters
            scramble=True,  # randomize the sequence
        )

        # Generate all samples at initialization
        self.samples = self.sobol_engine.random_base2(m=self.sobol_power)
        self.num_samples = len(self.samples)
        self.current_sample_idx = 0

        self.model_type = abc_config.model_type  # e.g., "BDM" or "ARCADE"
        self.max_time = self.model_config.output.max_time

        self.parameter_handler = ParameterFactory.create_parameter_handler(self.model_type)

    def calculate_distance(
        self, metrics: Dict[Metric, float], normalization_factors: Dict[str, float]
    ) -> float:
        """
        Calculate weighted distance between calculated metrics and targets
        Args:
            metrics: Dictionary of calculated metric values
            normalization_factors: Dictionary of normalization factors for each metric
        Returns:
            Total weighted distance using normalized metrics
        """
        total_distance = 0.0
        total_weight = sum(target.weight for target in self.targets)

        for target in self.targets:
            value = metrics[target.metric]
            norm_factor = normalization_factors.get(target.metric.value.lower(), 1.0)
            normalized_distance = abs(value - target.value) / norm_factor
            total_distance += (normalized_distance * target.weight) / total_weight

        return total_distance

    def calculate_metrics(
        self, target_time: Optional[float] = None, metrics_calculator=None
    ) -> Dict[Metric, float]:
        """Calculate all required metrics from grid states
        Args:
            target_time: Time point at which to calculate metrics (only used for density)
            metrics_calculator: Instance of Metrics class to calculate metrics
        Returns:
            Dictionary mapping metrics to their calculated values
        """
        if metrics_calculator is None:
            raise ValueError("Metrics calculator must be provided")

        metrics = {}
        for target in self.targets:
            try:
                if target.metric == Metric.DENSITY:
                    kwargs = {"target_time": target_time}
                elif target.metric == Metric.TIME_TO_EQUILIBRIUM:
                    kwargs = {"threshold": self.model_config.metrics.equilibrium_threshold}
                else:
                    kwargs = {}

                metrics[target.metric] = metrics_calculator.calculate_metric(
                    target.metric, **kwargs
                )
            except ValueError as e:
                raise ValueError(
                    f"Cannot calculate {target.metric.value} for model type {self.model_type}: {str(e)}"
                )

        return metrics

    def sample_parameters(self) -> Dict:
        """Sample parameters using Sobol sequence for better space-filling properties
        Returns:
            Dictionary of parameter names and their sampled values
        """
        if self.current_sample_idx >= len(self.samples):
            raise ValueError("All samples have been used. Initialize a new ABC instance if needed.")

        sample = self.samples[self.current_sample_idx]
        self.current_sample_idx += 1

        params = {}
        for (param_name, ranges), value in zip(self.parameter_ranges.items(), sample):
            min_val = ranges.min
            max_val = ranges.max
            scaled_value = min_val + (max_val - min_val) * value
            params[param_name] = scaled_value

        return params

    def run_inference(self, target_time: Optional[float] = None) -> Dict:
        """Run ABC inference to find parameters that achieve targets

        Args:
            target_time: Optional time point for density calculation

        Returns:
            Dictionary containing all samples with their metrics and acceptance status
        """
        log = logging.getLogger(__name__)

        target_str = ", ".join([f"{t.metric.value}: {t.value}" for t in self.targets])
        log.info(
            f"Starting ABC inference for targets: {target_str}, Total samples: {self.num_samples}"
        )

        self.param_metrics_distances_results = []

        for i in range(self.num_samples):
            params = self.sample_parameters()

            config = self.model_config.copy()
            config = self.parameter_handler.update_config(config, params)

            model = ModelFactory.create_model(self.model_type, config)
            time_points, _, grid_states = model.step()

            metrics_calculator = MetricsFactory.create_metrics(
                self.model_type, grid_states, time_points, self.max_time
            )

            metrics = self.calculate_metrics(target_time, metrics_calculator)
            distance = self.calculate_distance(metrics, metrics_calculator.normalization_factors)

            # Format sample data using parameter handler with additional info
            accepted = distance < self.epsilon
            sample_data = self.parameter_handler.format_sample_data(
                params, metrics, distance=distance, accepted=accepted
            )
            self.param_metrics_distances_results.append(sample_data)

            if (i + 1) % self.output_frequency == 0:
                accepted_count = sum(
                    1 for sample in self.param_metrics_distances_results if sample["accepted"]
                )
                log.info(f"Processed {i + 1} samples, accepted {accepted_count}")

        accepted_count = sum(
            1 for sample in self.param_metrics_distances_results if sample["accepted"]
        )
        log.info(f"Inference complete. Accepted {accepted_count} parameter sets")

        return self.param_metrics_distances_results

    def save_results(self, save_path: str = "all_samples_metrics.csv") -> pd.DataFrame:
        """Save all samples data to a CSV file

        Args:
            save_path: Path where to save the CSV file

        Returns:
            pandas DataFrame containing all samples data
        """
        return get_samples_data(
            param_metrics_distances_results=self.param_metrics_distances_results,
            model_type=self.model_type,
            save_path=save_path,
        )
