import logging
from typing import Dict, Optional
import scipy
from .abc_base import ABCBase
from .models.models import ModelFactory
from .metrics import MetricsFactory

class ABCWithModel(ABCBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize Sobol sampling
        self.sobol_power = self.abc_config.sobol_power
        self.parameter_ranges = self.abc_config.parameter_ranges

        self.sobol_engine = scipy.stats.qmc.Sobol(
            d=len(self.parameter_ranges),
            scramble=True,
        )
        self.samples = self.sobol_engine.random_base2(m=self.sobol_power)
        self.num_samples = len(self.samples)
        self.current_sample_idx = 0

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