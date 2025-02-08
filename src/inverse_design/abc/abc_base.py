from typing import Dict, List, Optional, Union
import pandas as pd
from abc import ABC, abstractmethod
from inverse_design.conf.config import BDMConfig, ABCConfig, ARCADEConfig
from inverse_design.common.enum import Metric, Target
from inverse_design.utils.utils import get_samples_data
from inverse_design.models.parameters import ParameterFactory

class ABCBase(ABC):
    def __init__(
        self,
        model_config: Union[BDMConfig, ARCADEConfig],
        abc_config: ABCConfig,
        targets: Union[Target, List[Target]],
    ):
        """Common initialization for ABC classes"""
        self.model_config = model_config
        self.abc_config = abc_config
        self.targets = [targets] if isinstance(targets, Target) else targets
        
        # Get ABC parameters from config
        self.epsilon = abc_config.epsilon
        self.output_frequency = abc_config.output_frequency
        self.model_type = abc_config.model_type
        self.max_time = self.model_config.output.max_time
        self.parameter_handler = ParameterFactory.create_parameter_handler(self.model_type)
        self.param_metrics_distances_results = []

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

    def calculate_all_metrics(
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

                metrics[target.metric] = metrics_calculator.get_calculate_method(
                    target.metric, **kwargs
                )
            except ValueError as e:
                raise ValueError(
                    f"Cannot calculate {target.metric.value} for model type {self.model_type}: {str(e)}"
                )

        return metrics

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