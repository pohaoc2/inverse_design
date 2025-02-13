import logging
from typing import Dict
import pandas as pd
from inverse_design.abc.abc_base import ABCBase


class ABCPrecomputed(ABCBase):
    def __init__(self, *args, param_file: str, metrics_file: str, **kwargs):
        """
        Initialize ABC for pre-computed results
        Args:
            para_file: Path to file containing model parameters
            metrics_file: Path to file containing model metrics
            *args, **kwargs: Arguments passed to ABCBase
        """
        super().__init__(*args, **kwargs)
        self.log = logging.getLogger(__name__)
        self.param_file = param_file
        self.metrics_file = metrics_file
        self.param_df = pd.read_csv(param_file)
        self.metrics_df = pd.read_csv(metrics_file)

        # Check lengths of param_df and metrics_df
        if len(self.param_df) != len(self.metrics_df):
            self.log.warning("Length mismatch: param_df has %d samples, metrics_df has %d samples", len(self.param_df), len(self.metrics_df))
            min_length = min(len(self.param_df), len(self.metrics_df))
            self.param_df = self.param_df.iloc[:min_length]
            self.metrics_df = self.metrics_df.iloc[:min_length]
        self.num_samples = len(self.param_df)
        


    def run_inference(
        self,
    ) -> Dict:
        """Run ABC inference on pre-computed results"""
        

        target_str = ", ".join([f"{t.metric.value}: {t.value}" for t in self.targets])
        self.log.info(
            f"Starting ABC inference on {self.num_samples} pre-computed samples for targets: {target_str}"
        )

        for i, row in self.param_df.iterrows():
            # Convert param Series to dict, excluding 'file_name'
            params = row.drop("file_name").to_dict()

            # Convert metrics Series to dict, only including target metrics
            metrics_row = self.metrics_df.iloc[i]
            metrics = {target.metric: metrics_row[target.metric.value] for target in self.targets}

            distance = self.calculate_distance(metrics)
            accepted = distance < self.epsilon
            sample_data = self.parameter_handler.format_sample_data(
                params, metrics, distance=distance, accepted=accepted
            )
            self.param_metrics_distances_results.append(sample_data)

            if (i + 1) % self.output_frequency == 0:
                accepted_count = sum(
                    1 for sample in self.param_metrics_distances_results if sample["accepted"]
                )
                self.log.info(f"Processed {i + 1} samples, accepted {accepted_count}")
        #for result in self.param_metrics_distances_results:
            #print("distance: ", result["distance"])
        return self.param_metrics_distances_results

    def _extract_parameters(self, row):
        """Extract parameters from pre-computed results row"""
        # Implementation depends on your data format
        pass

    def _extract_grid_states(self, row):
        """Extract grid states from pre-computed results row"""
        # Implementation depends on your data format
        pass

    def _extract_time_points(self, row):
        """Extract time points from pre-computed results row"""
        # Implementation depends on your data format
        pass
