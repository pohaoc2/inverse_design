import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
from typing import List
import json
from scipy.stats import spearmanr



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
    return param_df


def clean_metrics_df(metrics_df, target_name: List[str]):
    return metrics_df[target_name]


def perform_sensitivity_analysis(param_df, metrics_df, parameter_names, save_sensitivity_json=None):
    """
    Perform sensitivity analysis using Mutual Information for strength of relationship
    and Spearman correlation for direction of effect.

    Args:
        param_df (pd.DataFrame): DataFrame of parameter values
        metrics_df (pd.DataFrame): DataFrame of output values
        parameter_names (list): List of parameter names

    Returns:
        dict: Dictionary containing sensitivity metrics
    """
    from sklearn.feature_selection import mutual_info_regression
    from scipy.stats import spearmanr
    
    X = param_df.to_numpy()
    
    results = {}
    for metric in metrics_df.columns:
        Y = metrics_df[metric].to_numpy()

        # Calculate Mutual Information scores (without normalization)
        mi_scores = mutual_info_regression(X, Y, random_state=42)

        # Calculate Spearman correlations for direction
        spearman_corr = np.array([spearmanr(X[:, i], Y)[0] for i in range(X.shape[1])])
        results[metric] = {
            "MI": mi_scores,
            "spearman": spearman_corr,
            "names": parameter_names
        }
    if save_sensitivity_json:
        results_json = {}
        for metric, result in results.items():
            results_json[metric] = {
                "MI": result["MI"].tolist(),
                "spearman": result["spearman"].tolist(),
                "names": result["names"]
            }

        with open(save_sensitivity_json, 'w') as f:
            json.dump(results_json, f)

    return results


def plot_sensitivity_analysis(sensitivity_results, save_path=None):
    """
    Plot sensitivity analysis results showing both absolute and relative importance.
    For each metric, creates two subplots:
    1. Absolute MI values with direction (Spearman)
    2. Normalized MI values (relative importance) with direction (Spearman)

    Args:
        sensitivity_results (dict): Results from perform_sensitivity_analysis
        save_path (str, optional): Path to save the plot
    """
    import matplotlib.pyplot as plt

    n_metrics = len(sensitivity_results)
    fig, axes = plt.subplots(2, n_metrics, figsize=(8 * n_metrics, 12))
    if n_metrics == 1:
        axes = axes.reshape(2, 1)

    max_mi = 0
    for col, (metric, results) in enumerate(sensitivity_results.items()):
        mi_scores = results["MI"]
        if np.max(mi_scores) > max_mi:
            max_mi = np.max(mi_scores)

    for col, (metric, results) in enumerate(sensitivity_results.items()):
        names = results["names"]
        mi_scores = results["MI"]
        spearman_corr = results["spearman"]
        # Calculate normalized MI scores
        mi_scores_norm = mi_scores / np.sum(mi_scores) if np.sum(mi_scores) > 0 else mi_scores
        for row, (mi_vals, title_suffix) in enumerate([(mi_scores, "Absolute MI"), 
                                                      (mi_scores_norm, "Relative MI")]):
            ax = axes[row, col]
            if row == 0:
                ax.set_xlim(0, max_mi+0.01)
            # Sort parameters by importance
            sorted_idx = np.argsort(mi_vals)
            pos = np.arange(len(names))

            # Create horizontal bar plot
            bars = ax.barh(pos, mi_vals[sorted_idx])
            
            # Color bars based on Spearman correlation
            for idx, bar in enumerate(bars):
                corr = spearman_corr[sorted_idx[idx]]
                # Red for positive correlation, Blue for negative
                color = 'red' if corr > 0 else 'blue'
                # Opacity based on correlation strength
                alpha = min(abs(corr) + 0.2, 1.0)  # minimum opacity of 0.2
                bar.set_color(color)
                bar.set_alpha(alpha)
            
            # Add parameter names
            ax.set_yticks(pos)
            ax.set_yticklabels([names[i] for i in sorted_idx])
            
            # Add value labels on bars
            for i, (mi, corr) in enumerate(zip(mi_vals[sorted_idx], spearman_corr[sorted_idx])):
                label = f'MI={mi:.3f}, ρ={corr:.2f}'
                ax.text(mi+0.01, i, label, va='center')

            # Customize plot
            ax.set_xlabel('Mutual Information Score')
            ax.set_title(f'{metric} - {title_suffix}\n' + 
                        'Red: Positive effect, Blue: Negative effect\n' +
                        'Color intensity: Strength of directional relationship')

    # Add a note about interpretation at the bottom of the figure
    fig.text(0.1,-0.05, 
             'Note: Absolute MI shows overall importance across all metrics (including non-linear effects)\n' +
             'Relative MI shows importance within each metric\n' +
             'ρ (Spearman) shows direction of monotonic relationship',
             fontsize=12, ha='left')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def align_dataframes(param_df, metrics_df):
    """
    Align parameter and metrics dataframes based on input folders.
    
    Args:
        param_df (pd.DataFrame): DataFrame containing parameters
        metrics_df (pd.DataFrame): DataFrame containing metrics
    
    Returns:
        tuple: (aligned_param_df, aligned_metrics_df)
    """
    if len(param_df) != len(metrics_df):
        print(f"Warning: param_df and metrics_df have different lengths: {len(param_df)} != {len(metrics_df)}")
        input_folders = metrics_df['input_folder'].unique()
        param_df = param_df[param_df['input_folder'].isin(input_folders)]
        param_df = param_df.drop(columns=["input_folder"])
        metrics_df = metrics_df.drop(columns=["input_folder"])
    return param_df, metrics_df


def main():
    parameter_base_folder = "ARCADE_OUTPUT/STEM_CELL_META_SIGNAL_HETEROGENEITY"
    input_folder = parameter_base_folder + "/inputs"
    csv_file = f"{parameter_base_folder}/final_metrics.csv"

    param_file = f"{parameter_base_folder}/all_param_df.csv"
    metrics_file = f"{parameter_base_folder}/final_metrics.csv"
    target_metrics = ['symmetry', 'cycle_length', 'act']
    alignment_columns = ['input_folder']
    
    param_df, metrics_df = load_data(param_file, metrics_file, target_metrics + alignment_columns)
    param_df, metrics_df = align_dataframes(param_df, metrics_df)
    param_names = param_df.columns.tolist()

    save_sensitivity_json = f"{parameter_base_folder}/sensitivity_analysis.json"
    sensitivity_results = perform_sensitivity_analysis(param_df, metrics_df, param_names, save_sensitivity_json)
    plot_sensitivity_analysis(sensitivity_results, save_path=f"{parameter_base_folder}/sensitivity_analysis.png")

if __name__ == "__main__":
    main()
