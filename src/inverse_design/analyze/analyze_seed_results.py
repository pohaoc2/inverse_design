import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from inverse_design.analyze.analyze_utils import analyze_metric_percentiles
from typing import List
from inverse_design.analyze.cell_metrics import CellMetrics
from inverse_design.analyze.spatial_metrics import SpatialMetrics
from inverse_design.analyze.analyze_utils import remove_outliers


class SeedAnalyzer:
    def __init__(self, base_output_dir: str):
        """Initialize the seed analyzer.

        Args:
            base_output_dir: Base directory containing all simulation output folders
        """
        self.base_output_dir = Path(base_output_dir)
        self.cell_metrics = CellMetrics()
        self.spatial_metrics = SpatialMetrics()

    def filter_valid_metrics(self, metrics_list):
        """Filter out seeds with NaN or inf values from a list of metrics"""
        valid_indices = []
        for i in range(len(metrics_list[0])):
            has_invalid = any(np.isnan(metric[i]) or np.isinf(metric[i]) for metric in metrics_list)
            if not has_invalid:
                valid_indices.append(i)
        return valid_indices, [[metric[i] for i in valid_indices] for metric in metrics_list]

    def calculate_seed_metrics(self, cells_t1, cells_t2, time_difference, metrics_dict):
        """Calculate and append metrics for a single seed.

        Args:
            cells_t1: Cell data at time t1
            cells_t2: Cell data at time t2
            time_difference: Time difference between t1 and t2
            metrics_dict: Dictionary to store metrics

        Returns:
            Updated metrics dictionary
        """
        metrics_dict["growth_rates"].append(
            self.cell_metrics.calculate_growth_rate(cells_t1, cells_t2, time_difference)
        )
        metrics_dict["avg_volumes_t1"].append(self.cell_metrics.calculate_average_volume(cells_t1))
        metrics_dict["avg_volumes_t2"].append(self.cell_metrics.calculate_average_volume(cells_t2))
        metrics_dict["n_cells_t1"].append(len(cells_t1))
        metrics_dict["n_cells_t2"].append(len(cells_t2))
        metrics_dict["act_t1"].append(self.cell_metrics.calculate_activity(cells_t1))
        metrics_dict["act_t2"].append(self.cell_metrics.calculate_activity(cells_t2))
        metrics_dict["seed_count"] += 1
        metrics_dict["doub_time"].append(
            self.cell_metrics.calculate_doubling_time(len(cells_t1), len(cells_t2), time_difference)
        )
        metrics_dict["states_t1"].append(self.cell_metrics.calculate_cell_states(cells_t1))
        metrics_dict["states_t2"].append(self.cell_metrics.calculate_cell_states(cells_t2))
        return metrics_dict

    def process_parameter_seeds(self, cells_data_t1, cells_data_t2, time_difference):
        """
        Process metrics across different seeds for a single parameter set.

        Args:
            cells_data_t1: List of (info, cells) tuples at time t1
            cells_data_t2: List of (info, cells) tuples at time t2
            time_difference: Time difference between t1 and t2

        Returns:
            Dictionary containing metrics for each experiment group and name
        """
        metrics_by_exp = {}

        for (info_t1, cells_t1), (info_t2, cells_t2) in zip(cells_data_t1, cells_data_t2):
            if (
                info_t1["exp_group"] != info_t2["exp_group"]
                or info_t1["exp_name"] != info_t2["exp_name"]
            ):
                continue

            exp_key = (info_t1["exp_group"], info_t1["exp_name"])
            if exp_key not in metrics_by_exp:
                metrics_by_exp[exp_key] = {
                    "growth_rates": [],
                    "avg_volumes_t1": [],
                    "avg_volumes_t2": [],
                    "n_cells_t1": [],
                    "n_cells_t2": [],
                    "act_t1": [],
                    "act_t2": [],
                    "seed_count": 0,
                    "doub_time": [],
                    "states_t1": [],
                    "states_t2": [],
                }

            metrics_by_exp[exp_key] = self.calculate_seed_metrics(
                cells_t1, cells_t2, time_difference, metrics_by_exp[exp_key]
            )

        # Filter invalid metrics
        for exp_key in metrics_by_exp:
            # Get all metrics except seed_count and states
            metrics_to_check = []
            metrics_keys = []
            for key, value in metrics_by_exp[exp_key].items():
                if key not in ["seed_count", "states_t1", "states_t2"] and isinstance(value, list):
                    metrics_to_check.append(value)
                    metrics_keys.append(key)

            valid_indices, filtered_metrics = self.filter_valid_metrics(metrics_to_check)

            # Update all metrics with filtered values
            for idx, key in enumerate(metrics_keys):
                metrics_by_exp[exp_key][key] = filtered_metrics[idx]

            metrics_by_exp[exp_key]["seed_count"] = len(valid_indices)

        return metrics_by_exp

    def analyze_seeds_for_folders(self, input_folders, t1: str, t2: str, timestamps: List[str]):
        """Analyze metrics across seeds for given input folders.

        Args:
            input_folders: List of input folder names to analyze
            t1: First timepoint
            t2: Second timepoint
            timestamps: List of all timestamps

        Returns:
            Dictionary containing metrics for each folder's seeds
        """
        all_seed_metrics = {}

        for folder_name in input_folders:
            folder_path = self.base_output_dir / folder_name

            # Load cell data
            cells_data_t1 = self.cell_metrics.load_cells_data(folder_path, t1)
            cells_data_t2 = self.cell_metrics.load_cells_data(folder_path, t2)
            time_difference = int(t2) - int(t1)

            # Process seeds for this parameter set
            metrics_by_exp = self.process_parameter_seeds(
                cells_data_t1, cells_data_t2, time_difference
            )
            exp_name = list(metrics_by_exp.keys())[0]
            all_seed_metrics[folder_name] = metrics_by_exp[exp_name]
        return all_seed_metrics

    def plot_seed_comparisons(
        self,
        csv_file: str,
        metric_name: str,
        percentile: float = 10,
        t1: str = "000000",
        t2: str = "010080",
        timestamps: List[str] = None,
        metrics_to_plot: List[str] = None,
        save_file: str = None,
    ):
        """Plot violin plots comparing seed distributions for top and bottom performers.

        Args:
            csv_file: Path to aggregated metrics CSV file
            metric_name: Name of metric to analyze
            t1: First timepoint
            t2: Second timepoint
            timestamps: List of timestamps for analysis
        """
        # Get top and bottom folders
        top_n_input_file, bottom_n_input_file, _ = analyze_metric_percentiles(
            csv_file, metric_name, percentile, verbose=False
        )
        # Get seed metrics for each group
        top_seed_metrics = self.analyze_seeds_for_folders(top_n_input_file, t1, t2, timestamps)
        bottom_seed_metrics = self.analyze_seeds_for_folders(
            bottom_n_input_file, t1, t2, timestamps
        )

        if timestamps is None:
            timestamps = [t1, t2]

        # Create plot data from seed metrics
        plot_data = []
        for group, seed_metrics in [("top", top_seed_metrics), ("bottom", bottom_seed_metrics)]:
            for folder, folder_metrics in seed_metrics.items():
                for metric in metrics_to_plot:
                    values = folder_metrics[metric]
                    for value in values:
                        plot_data.append(
                            {"metric": metric, "value": value, "group": group, "folder": folder}
                        )

        # Convert to DataFrame and remove outliers for each metric separately
        plot_df = pd.DataFrame(plot_data)
        cleaned_plot_data = []
        for metric in metrics_to_plot:
            metric_data = plot_df[plot_df["metric"] == metric].copy()
            cleaned_data, _, _ = remove_outliers(
                metric_data, 
                verbose=True
            )
            cleaned_plot_data.append(cleaned_data)
            
        # Combine cleaned data back into single DataFrame
        plot_df = pd.concat(cleaned_plot_data, axis=0)

        print(f"number of top {percentile}% folders: {len(top_n_input_file)}")
        print(f"number of bottom {percentile}% folders: {len(bottom_n_input_file)}")

        # Create plots
        plt.figure(figsize=(4 * len(metrics_to_plot), 8))

        for idx, metric in enumerate(metrics_to_plot):
            plt.subplot(1, len(metrics_to_plot), idx + 1)
            metric_data = plot_df[plot_df["metric"] == metric]

            # Create violin plot
            sns.violinplot(data=metric_data, x="group", y="value", order=["top", "bottom"])

            # Add individual points
            sns.stripplot(
                data=metric_data,
                x="group",
                y="value",
                order=["top", "bottom"],
                color="black",
                alpha=0.8,
                size=3,
                jitter=0.2,
            )

            plt.title(f"{metric} Distribution")
            plt.xticks(rotation=45)

            # Add statistics
            top_vals = metric_data[metric_data["group"] == "top"]["value"]
            bottom_vals = metric_data[metric_data["group"] == "bottom"]["value"]

            stats_text = (
                f"Top {percentile}%:\nMean: {top_vals.mean():.2f}\nStd: {top_vals.std():.2f}\n\n"
            )
            stats_text += f"Bottom {percentile}%:\nMean: {bottom_vals.mean():.2f}\nStd: {bottom_vals.std():.2f}"

            plt.text(
                0.95,
                0.95,
                stats_text,
                transform=plt.gca().transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.8),
            )

        plt.tight_layout()
        if save_file is not None:
            plt.savefig(save_file)
        else:
            plt.show()


if __name__ == "__main__":
    # Example usage
    base_dir = "ARCADE_OUTPUT/STEM_CELL/STEM_CELL"
    csv_file = f"{base_dir}/simulation_metrics.csv"
    percentile = 10
    metrics_to_plot = ["doub_time", "act_t2", "n_cells_t2"]
    analyzer = SeedAnalyzer(base_dir)
    metric_name = "doub_time_std"
    analyzer.plot_seed_comparisons(
        csv_file=csv_file,
        metric_name=metric_name,
        percentile=percentile,
        metrics_to_plot=metrics_to_plot,
        t1="000000",
        t2="010080",
        save_file=f"{base_dir}/seed_comparisons_{metric_name}.png"
    )
