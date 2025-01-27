# utils.py
import os
from typing import List, Dict
import pandas as pd


def get_samples_data(
    param_metrics_distances_results: List[Dict],
    model_type: str,
    save_path: str = "all_samples_metrics.csv",
):
    """Save all samples data to a CSV file

    Args:
        param_metrics_distances_results: List of dictionaries containing all samples metrics
        model_type: Type of model used (e.g., "BDM", "ARCADE")
        save_path: Path where to save the CSV file
    """
    # Define columns based on model type
    if model_type == "BDM":
        columns = [
            "proliferate",
            "death",
            "migratory",
            "cell_density",
            "time_to_eq",
            "distance",
            "accepted",
        ]
    elif model_type == "ARCADE":
        columns = [
            "division",
            "death",
            "motility",
            "adhesion",
            "cell_density",
            "cluster_size",
            "distance",
            "accepted",
        ]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    all_samples_df = pd.DataFrame(param_metrics_distances_results, columns=columns)
    all_samples_df.to_csv(save_path, index=False)

    return all_samples_df
