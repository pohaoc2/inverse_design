from abc import ABC, abstractmethod
from typing import List, Optional, Set, Dict
from enum import Enum
import numpy as np
from dataclasses import dataclass


class Metric(Enum):
    DENSITY = "density"
    TIME_TO_EQUILIBRIUM = "time_to_equilibrium"
    CLUSTER_SIZE = "cluster_size"  # Example additional metric
    MIGRATION_SPEED = "migration_speed"  # Example additional metric


@dataclass
class Target:
    """
    Target class for multi-objective optimization
    Attributes:
        metric: The metric to target (density, time to equilibrium, etc.)
        value: The desired value for this metric
        weight: Weight for multi-objective optimization (default: 1.0)
    """

    metric: Metric
    value: float
    weight: float = 1.0


class Metrics(ABC):
    """Abstract base class for model metrics calculations"""

    @abstractmethod
    def get_available_metrics(self) -> Set[Metric]:
        """Returns set of metrics available for this model"""
        pass

    def __init__(self, grid_states: List, time_points: List[float]):
        """
        Initialize Metrics calculator
        Args:
            grid_states: List of states at different time points
            time_points: List of corresponding time points
        """
        self.grid_states = grid_states
        self.time_points = time_points
        self._available_metrics = self.get_available_metrics()

    def calculate_metric(self, metric: Metric, **kwargs) -> float:
        """Calculate specified metric with given parameters
        Args:
            metric: The metric to calculate
            **kwargs: Additional parameters needed for the calculation
        Returns:
            Calculated metric value
        Raises:
            ValueError: If metric is not available for this model
        """
        if metric not in self._available_metrics:
            raise ValueError(f"Metric {metric.value} is not available for this model")

        # Call the appropriate calculation method based on the metric
        method_name = f"calculate_{metric.value}"
        if not hasattr(self, method_name):
            raise NotImplementedError(f"Calculation method for {metric.value} not implemented")

        calculation_method = getattr(self, method_name)
        return calculation_method(**kwargs)


class MetricsBDM(Metrics):
    """Class for calculating various metrics from BDM grid states"""

    def get_available_metrics(self) -> Set[Metric]:
        return {Metric.DENSITY, Metric.TIME_TO_EQUILIBRIUM}

    # Normalization factors specific to BDM
    normalization_factors = {
        "density": 100.0,  # density is in percentage (0-100)
        "time_to_equilibrium": None,  # To be set based on config or max_time
    }

    def __init__(self, grid_states: List, time_points: List[float], max_time: float):
        super().__init__(grid_states, time_points)
        self.normalization_factors["time_to_equilibrium"] = max_time

    def calculate_density(self, target_time: Optional[float] = None) -> float:
        if target_time is None:
            grid = self.grid_states[-1]
        else:
            idx = np.argmin(np.abs(np.array(self.time_points) - target_time))
            grid = self.grid_states[idx]

        return grid.num_cells / (grid.lattice_size**2) * 100

    def calculate_time_to_equilibrium(self, threshold: float = 0.05) -> float:
        densities = []
        for i in range(len(self.grid_states)):
            density = self.calculate_density(self.time_points[i])
            densities.append(density)

        window_size = 5  # Number of time points to check for stability
        for i in range(window_size, len(self.time_points)):
            window = densities[i - window_size : i]
            mean_density = np.mean(window)

            # Handle the case where mean_density is 0
            if mean_density == 0:
                fluctuation = np.max(np.abs(np.array(window)))
            else:
                fluctuation = np.max(np.abs(np.array(window) - mean_density) / mean_density)

            if fluctuation < threshold:
                return self.time_points[i]

        # If equilibrium is not reached, return the final time point
        return self.time_points[-1]


class MetricsARCADE(Metrics):
    """Class for calculating various metrics from ARCADE states"""

    def get_available_metrics(self) -> Set[Metric]:
        return {Metric.DENSITY, Metric.CLUSTER_SIZE, Metric.MIGRATION_SPEED}

    # Normalization factors specific to ARCADE
    normalization_factors = {
        "density": 100.0,  # Example normalization factor
        "time_to_equilibrium": None,  # To be set based on config or max_time
    }

    def __init__(self, grid_states: List, time_points: List[float], max_time: float):
        super().__init__(grid_states, time_points)
        self.normalization_factors["time_to_equilibrium"] = max_time

    def calculate_density(self, target_time: Optional[float] = None) -> float:
        """Implementation specific to ARCADE model"""
        pass

    def calculate_cluster_size(self) -> float:
        # ... ARCADE-specific implementation ...
        pass

    def calculate_migration_speed(self) -> float:
        # ... ARCADE-specific implementation ...
        pass


class MetricsFactory:
    @staticmethod
    def create_metrics(
        model_type: str, grid_states: List, time_points: List[float], max_time: float
    ) -> Metrics:
        if model_type == "BDM":
            return MetricsBDM(grid_states, time_points, max_time)
        elif model_type == "ARCADE":
            return MetricsARCADE(grid_states, time_points, max_time)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
