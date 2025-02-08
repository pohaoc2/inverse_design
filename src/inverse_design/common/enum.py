from enum import Enum
from dataclasses import dataclass

class Metric(Enum):
    DENSITY = "density"
    TIME_TO_EQUILIBRIUM = "time_to_equilibrium"
    GROWTH_RATE = "growth_rate"
    SYMMETRY = "symmetry"
    ACTIVITY = "activity"

    def __eq__(self, other):
        # Compare by name and value to handle different enum instances
        if hasattr(other, 'name') and hasattr(other, 'value'):
            return self.name == other.name and self.value == other.value
        return False

    def __hash__(self):
        return hash((self.name, self.value))


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