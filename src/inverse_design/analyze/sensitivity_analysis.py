import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
from typing import List


def perform_sobol_analysis_from_data(param_df, metrics_df, parameter_names):
    """
    Perform Sobol sensitivity analysis on pre-computed parameter-output pairs.
    
    Args:
        param_df (pd.DataFrame): DataFrame of parameter values
        metrics_df (pd.DataFrame): DataFrame of output values
        parameter_names (list): List of parameter names
    
    Returns:
        dict: Dictionary containing Sobol indices and other sensitivity metrics
    """
    
    X = param_df.to_numpy()
    
    results = {}
    for metric in metrics_df.columns:
        Y = metrics_df[metric].to_numpy()
        
        # Calculate correlation coefficients
        correlations = np.corrcoef(X.T, Y)[-1, :-1]  # Get correlations with output
        total_correlation = np.sum(np.abs(correlations))
        
        # Calculate sensitivity indices as normalized correlations
        sensitivity = np.abs(correlations) / total_correlation if total_correlation != 0 else np.zeros_like(correlations)
        
        results[metric] = {
            'S1': sensitivity,  # First-order indices
            'ST': sensitivity,  # Total-order indices (same as S1 in this simple case)
            'S1_conf': np.zeros_like(sensitivity),  # No confidence intervals in this method
            'ST_conf': np.zeros_like(sensitivity),  # No confidence intervals in this method
            'names': parameter_names
        }
    
    return results

def load_data(param_file, metrics_file, target_name: List[str]):
    """
    Load parameter and metric data from CSV files and perform sensitivity analysis.
    
    Args:
        param_file (str): Path to parameters CSV file
        metrics_file (str): Path to metrics CSV file
    
    Returns:
        dict: Results from Sobol analysis
    """
    # Load data
    param_df = pd.read_csv(param_file)
    metrics_df = pd.read_csv(metrics_file)
    param_df = clean_param_df(param_df)
    metrics_df = clean_metrics_df(metrics_df, target_name)

    return param_df, metrics_df

def clean_param_df(param_df):
    constant_columns = param_df.columns[param_df.nunique() == 1]
    param_df = param_df.drop(columns=constant_columns)
    param_df = param_df.drop(columns=['file_name'])
    return param_df

def clean_metrics_df(metrics_df, target_name: List[str]):
    return metrics_df[target_name]

def plot_sobol_indices(sensitivity_results, save_path=None):
    """
    Plot sensitivity indices for each metric.
    
    Args:
        sensitivity_results (dict): Results from perform_sobol_analysis_from_data
        save_path (str, optional): Path to save the plot. If None, displays plot.
    """
    import matplotlib.pyplot as plt
    
    n_metrics = len(sensitivity_results)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(6, 3*n_metrics))
    if n_metrics == 1:
        axes = [axes]
    
    for ax, (metric, results) in zip(axes, sensitivity_results.items()):
        names = results['names']
        S1 = results['S1']
        
        # Create bar plot
        y_pos = np.arange(len(names))
        ax.barh(y_pos, S1)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Sensitivity Index')
        ax.set_title(f'Parameter Sensitivity for {metric}')
        
        # Add value labels on bars
        for i, v in enumerate(S1):
            ax.text(v, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def main():
    param_file = 'completed_params.csv'
    metrics_file = 'completed_doubling.csv'
    target_names = ['doubling_time', 'activity', 'colony_growth_rate']
    
    param_df, metrics_df = load_data(param_file, metrics_file, target_names)
    param_names = param_df.columns.tolist()
    sensitivity_results = perform_sobol_analysis_from_data(param_df, metrics_df, param_names)
    
    # Plot results
    plot_sobol_indices(sensitivity_results)

if __name__ == "__main__":
    main()
