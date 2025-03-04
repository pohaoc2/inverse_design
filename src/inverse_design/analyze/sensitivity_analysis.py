import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.util import extract_group_names
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


def perform_mi_analysis(param_df, metrics_df, parameter_names, save_mi_json=None):
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
    if save_mi_json:
        results_json = {}
        for metric, result in results.items():
            results_json[metric] = {
                "MI": result["MI"].tolist(),
                "spearman": result["spearman"].tolist(),
                "names": result["names"]
            }

        with open(save_mi_json, 'w') as f:
            json.dump(results_json, f)

    return results


def plot_mi_analysis(mi_results, save_path=None):
    """
    Plot sensitivity analysis results showing both absolute and relative importance.
    For each metric, creates two subplots:
    1. Absolute MI values with direction (Spearman)
    2. Normalized MI values (relative importance) with direction (Spearman)

    Args:
        mi_results (dict): Results from perform_mi_analysis
        save_path (str, optional): Path to save the plot
    """
    import matplotlib.pyplot as plt

    n_metrics = len(mi_results)
    fig, axes = plt.subplots(2, n_metrics, figsize=(8 * n_metrics, 12))
    if n_metrics == 1:
        axes = axes.reshape(2, 1)

    max_mi = 0
    for col, (metric, results) in enumerate(mi_results.items()):
        mi_scores = results["MI"]
        if np.max(mi_scores) > max_mi:
            max_mi = np.max(mi_scores)

    for col, (metric, results) in enumerate(mi_results.items()):
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
    Ensures both DataFrames have matching rows in the same order.
    
    Args:
        param_df (pd.DataFrame): DataFrame containing parameters
        metrics_df (pd.DataFrame): DataFrame containing metrics
    
    Returns:
        tuple: (aligned_param_df, aligned_metrics_df)
    """
    if len(param_df) != len(metrics_df):
        print(f"Warning: param_df and metrics_df have different lengths: {len(param_df)} != {len(metrics_df)}")
    
    # Get common input folders and sort them
    common_folders = sorted(
        set(param_df['input_folder']) & set(metrics_df['input_folder'])
    )
    
    # Filter and sort both DataFrames by input_folder
    param_df = param_df[param_df['input_folder'].isin(common_folders)].copy()
    metrics_df = metrics_df[metrics_df['input_folder'].isin(common_folders)].copy()
    
    param_df = param_df.set_index('input_folder').loc[common_folders].reset_index()
    metrics_df = metrics_df.set_index('input_folder').loc[common_folders].reset_index()
    
    # Drop input_folder columns
    param_df = param_df.drop(columns=["input_folder"])
    metrics_df = metrics_df.drop(columns=["input_folder"])
    
    return param_df, metrics_df


def perform_sobol_analysis(param_df, metrics_df, parameter_names, calc_second_order=True, save_sobol_json=None):
    """
    Perform Sobol sensitivity analysis.

    Args:
        param_df (pd.DataFrame): DataFrame of parameter values
        metrics_df (pd.DataFrame): DataFrame of output values
        parameter_names (list): List of parameter names
        n_samples (int): Number of samples for Sobol analysis
        save_sobol_json (str, optional): Path to save results as JSON

    Returns:
        dict: Dictionary containing Sobol sensitivity indices
    """
    # Define the problem dictionary for SALib
    problem = {
        'num_vars': len(parameter_names),
        'names': parameter_names,
        'bounds': [[param_df[param].min(), param_df[param].max()] for param in parameter_names]
    }
    
    results = {}
    for metric in metrics_df.columns:
        Y = metrics_df[metric].to_numpy()
        n_del_samples = Y.size % (2*len(parameter_names) + 2) if calc_second_order else Y.size % (len(parameter_names) + 2)
        Y = Y[:-n_del_samples]
        Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=False)
        if calc_second_order:
            results[metric] = {
                'S1': Si['S1'],  # First-order indices
                'ST': Si['ST'],  # Total-order indices
                'S2': Si['S2'],  # Second-order indices
                'names': parameter_names
            }
        else:
            results[metric] = {
                'S1': Si['S1'],  # First-order indices
                'ST': Si['ST'],  # Total-order indices
                'names': parameter_names
            }

    if save_sobol_json:
        results_json = {}
        for metric, result in results.items():
            results_json[metric] = {
                'S1': result['S1'].tolist(),
                'ST': result['ST'].tolist(),
                'S2': result['S2'].tolist() if calc_second_order else None,
                'names': result['names']
            }
        
        with open(save_sobol_json, 'w') as f:
            json.dump(results_json, f)

    return results

def plot_sobol_analysis(sobol_results, metrics_name, calc_second_order=True, save_path=None):
    """
    Plot Sobol sensitivity analysis results with indices sorted in decreasing order.

    Args:
        sobol_results (dict): Results from perform_sobol_analysis
        metrics_name (str): Name of the metric to plot
        save_path (str, optional): Path to save the plot
    """
    import matplotlib.pyplot as plt

    # Extract and clean indices (replace negative values with 0)
    S1 = np.maximum(sobol_results[metrics_name]['S1'], 0)
    ST = np.maximum(sobol_results[metrics_name]['ST'], 0)
    S2 = np.maximum(sobol_results[metrics_name]['S2'], 0) if calc_second_order else None
    names = sobol_results[metrics_name]['names']

    # Sort indices and names
    S1_sorted_idx = np.argsort(S1)
    ST_sorted_idx = np.argsort(ST)

    S1_sorted = S1[S1_sorted_idx]
    ST_sorted = ST[ST_sorted_idx]
    names_S1 = [names[i] for i in S1_sorted_idx]
    names_ST = [names[i] for i in ST_sorted_idx]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    bars_S1 = axes[0].barh(range(len(names)), S1_sorted)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names_S1)
    axes[0].set_xlabel('First-Order Index')
    axes[0].set_title(f'First-Order Sobol Indices for {metrics_name}')
    
    for i, bar in enumerate(bars_S1):
        width = bar.get_width()
        axes[0].text(width + 0.01, i, f'{width:.3f}', 
                    va='center', ha='left')
    
    bars_ST = axes[1].barh(range(len(names)), ST_sorted)
    axes[1].set_yticks(range(len(names)))
    axes[1].set_yticklabels(names_ST)
    axes[1].set_xlabel('Total-Order Index')
    axes[1].set_title(f'Total-Order Sobol Indices for {metrics_name}')
    
    for i, bar in enumerate(bars_ST):
        width = bar.get_width()
        axes[1].text(width + 0.01, i, f'{width:.3f}', 
                    va='center', ha='left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
    

def main():
    parameter_base_folder = "ARCADE_OUTPUT/STEM_CELL/MS_PRIOR_N1024/"
    param_file = f"{parameter_base_folder}/all_param_df.csv"
    metrics_file = f"{parameter_base_folder}/final_metrics.csv"
    target_metrics = ['symmetry', 'cycle_length', 'act']
    alignment_columns = ['input_folder']
    
    param_df, metrics_df = load_data(param_file, metrics_file, target_metrics + alignment_columns)
    param_df, metrics_df = align_dataframes(param_df, metrics_df)
    param_names = param_df.columns.tolist()

    # Perform both types of sensitivity analysis
    save_mi_json = f"{parameter_base_folder}/mi_analysis.json"
    save_sobol_json = f"{parameter_base_folder}/sobol_analysis.json"
    if 0:
        mi_results = perform_mi_analysis(param_df, metrics_df, param_names, save_mi_json)
        plot_mi_analysis(mi_results, save_path=f"{parameter_base_folder}/mi_analysis.png")

    if 1:
        sobol_results = perform_sobol_analysis(param_df, metrics_df, param_names,
                                            calc_second_order=False,
                                            save_sobol_json=save_sobol_json)
        metrics_name = 'symmetry'
        plot_sobol_analysis(sobol_results,
                            metrics_name=metrics_name,
                            calc_second_order=False,
                            save_path=f"{parameter_base_folder}/sobol_analysis_{metrics_name}.png")
    

if __name__ == "__main__":
    main()
