import pandas as pd
import numpy as np
from pathlib import Path
import json

def get_parameters_from_json(input_folder, parameter_list):
    """Extract parameters from json file."""
    # Find the base configuration file by looking for a .json file without CELL or LOCATION in the name
    json_files = list(Path(input_folder).glob("*.json"))
    config_file = next(f for f in json_files if "CELL" not in f.name and "LOCATION" not in f.name)

    with open(config_file, "r") as f:
        data = json.load(f)

    params = {}
    # Create mapping dynamically from parameter names
    param_mapping = {
        param: ("populations", "cancerous", param) for param in parameter_list
    }
    for param in parameter_list:
        path = param_mapping[param]
        value = data
        for key in path:
            value = value[key]

        params[param] = value

    return params


def analyze_metric_percentiles(csv_file, metric_name, percentile=10, verbose=True):
    """Analyze metric percentiles from simulation results.
    
    Args:
        csv_file: Path to simulation metrics CSV file
        metric_name: Name of metric to analyze
        percentile: Percentile threshold for top/bottom performers
        verbose: Whether to print analysis results
        
    Returns:
        Tuple of (top_n_input_files, bottom_n_input_files, df)
    """
    df = pd.read_csv(csv_file)
    
    # Calculate percentile thresholds
    top_threshold = np.percentile(df[metric_name], percentile)
    bottom_threshold = np.percentile(df[metric_name], 100 - percentile)
    
    # Get top and bottom performers
    top_n = df[df[metric_name] <= top_threshold]
    bottom_n = df[df[metric_name] >= bottom_threshold]
    
    if verbose:
        print(f"\nAnalyzing {metric_name}:")
        print(f"Top {percentile}% threshold: {top_threshold:.3f}")
        print(f"Bottom {percentile}% threshold: {bottom_threshold:.3f}")
        print(f"Number of top performers: {len(top_n)}")
        print(f"Number of bottom performers: {len(bottom_n)}")
    
    return (
        top_n["input_folder"].tolist(),
        bottom_n["input_folder"].tolist(),
        df
    ) 

def collect_parameter_data(input_files, parameter_base_folder, parameter_list, labels=None):
    """
    Collect parameter data from JSON files for a list of input folders.

    Args:
        input_files: List of input folder names
        parameter_base_folder: Base folder containing parameter files
        parameter_list: List of parameters to extract
        labels: Optional list of labels corresponding to input_files

    Returns:
        pandas.DataFrame: DataFrame containing the parameter values, labels, and folder names
    """
    params_list = []
    for i, input_folder in enumerate(input_files):
        input_folder_path = Path(parameter_base_folder) / input_folder
        params = get_parameters_from_json(input_folder_path, parameter_list)
        # Add folder name to params
        params["input_folder"] = input_folder
        if labels is not None:
            params["percentile_label"] = labels[i]
        params_list.append(params)
    # sort params_list by numeric value in input_folder name
    params_list.sort(key=lambda x: int(x["input_folder"].split("_")[1]))
    return pd.DataFrame(params_list)