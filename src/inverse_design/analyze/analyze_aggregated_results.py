import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def analyze_metric_percentiles(csv_file_path, metrics_name: str, verbose: bool = True):
    """
    Analyze rows where metrics fall into the bottom and top 10% of their distributions.
    
    Args:
        csv_file_path: Path to the aggregated results CSV file
        metrics_name: Name of the metric to analyze
    """
    df = pd.read_csv(csv_file_path)
    
    
    # Calculate 10th and 90th percentiles for the metric
    lower_bound = df[metrics_name].quantile(0.1)
    upper_bound = df[metrics_name].quantile(0.9)
    
    bottom_10 = df[df[metrics_name] <= lower_bound]
    top_10 = df[df[metrics_name] >= upper_bound]
    top_10_input_file = top_10["input_folder"].unique()
    bottom_10_input_file = bottom_10["input_folder"].unique()
    if verbose:
        print(f"\nAnalysis for {metrics_name}")
        print("=" * 80)
        print(f"\nBottom 10% (≤ {lower_bound:.3f}):")
        print(bottom_10)
        print(f"\nTop 10% (≥ {upper_bound:.3f}):")
        print(top_10)

    return top_10_input_file, bottom_10_input_file

def get_parameters_from_json(input_folder, parameter_list):
    """Extract parameters from json file."""
    json_file = Path(input_folder) / "uniform_kde_aggressive.json"
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    params = {}
    param_mapping = {
        "CELL_VOLUME_SIGMA": ("populations", "cancerous", "CELL_VOLUME_SIGMA"),
        "APOPTOSIS_AGE_SIGMA": ("populations", "cancerous", "APOPTOSIS_AGE_SIGMA"),
        "NECROTIC_FRACTION": ("populations", "cancerous", "NECROTIC_FRACTION"),
        "ACCURACY": ("populations", "cancerous", "ACCURACY"),
        "AFFINITY": ("populations", "cancerous", "AFFINITY")
    }
    
    for param in parameter_list:
        path = param_mapping[param]
        value = data
        for key in path:
            value = value[key]
        params[param] = value
    
    return params

def collect_parameter_data(input_files, parameter_base_folder, parameter_list):
    """
    Collect parameter data from JSON files for a list of input folders.
    
    Args:
        input_files: List of input folder names
        parameter_base_folder: Base folder containing parameter files
        parameter_list: List of parameters to extract
    
    Returns:
        pandas.DataFrame: DataFrame containing the parameter values
    """
    params_list = []
    
    for input_folder in input_files:
        full_path = Path(parameter_base_folder) / input_folder
        params = get_parameters_from_json(full_path, parameter_list)
        params_list.append(params)
    
    return pd.DataFrame(params_list)

def analyze_and_plot_parameters(top_10_input_file, bottom_10_input_file, parameter_list, parameter_base_folder):
    """
    Analyze and plot parameter distributions for top and bottom 10% cases.
    """
    
    # Collect parameters for both groups using the new function
    top_df = collect_parameter_data(top_10_input_file, parameter_base_folder, parameter_list)
    bottom_df = collect_parameter_data(bottom_10_input_file, parameter_base_folder, parameter_list)
    
    # Plot distributions
    n_params = len(parameter_list)
    fig, axes = plt.subplots(1, n_params, figsize=(4*n_params, 4))
    
    for idx, param in enumerate(parameter_list):
        ax = axes[idx]
        
        # Plot density distributions
        sns.kdeplot(data=top_df[param], ax=ax, label='Top 10%', color='blue')
        sns.kdeplot(data=bottom_df[param], ax=ax, label='Bottom 10%', color='red')
        
        ax.set_title(f'{param} Distribution')
        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{parameter_base_folder}/parameter_distributions.png')
    plt.show()
    
    # Print summary statistics
    print("\nParameter Statistics:")
    print("=" * 80)
    print("\nTop 10% - Mean values:")
    print(top_df.mean())
    print("\nBottom 10% - Mean values:")
    print(bottom_df.mean())

def analyze_pca_parameters(csv_file_path, parameter_list, parameter_base_folder, metrics_names):
    """
    Perform PCA on parameters and visualize top and bottom 10% cases in 2D.
    
    Args:
        csv_file_path: Path to the aggregated results CSV file
        parameter_list: List of parameters to analyze
        parameter_base_folder: Base folder containing parameter files
        metrics_names: List of metrics used to determine top/bottom 10%
    """
    # Get top and bottom 10% input folders
    top_10_input_file, bottom_10_input_file = analyze_metric_percentiles(
        csv_file_path, metrics_names, verbose=False
    )
    
    # Collect parameters for both groups using the new function
    top_df = collect_parameter_data(top_10_input_file, parameter_base_folder, parameter_list)
    bottom_df = collect_parameter_data(bottom_10_input_file, parameter_base_folder, parameter_list)
    
    # Combine data for PCA
    all_data = pd.concat([top_df, bottom_df])
    
    # Create labels (1 for top 10%, 0 for bottom 10%)
    labels = np.array([1] * len(top_df) + [0] * len(bottom_df))
    
    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(all_data)
    
    # Perform PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Plot points
    plt.scatter(data_pca[labels==1, 0], data_pca[labels==1, 1], 
               c='blue', label='Top 10%', alpha=0.6)
    plt.scatter(data_pca[labels==0, 0], data_pca[labels==0, 1], 
               c='red', label='Bottom 10%', alpha=0.6)
    
    # Add parameter vectors
    for i, param in enumerate(parameter_list):
        plt.arrow(0, 0,
                 pca.components_[0, i] * 3,
                 pca.components_[1, i] * 3,
                 color='black', alpha=0.5)
        plt.text(pca.components_[0, i] * 3.2,
                pca.components_[1, i] * 3.2,
                param,
                color='black',
                ha='center',
                va='center')
    
    # Add plot details
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('PCA of Parameters for Top and Bottom 10% Cases')
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add origin lines
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print explained variance ratios
    print("\nPCA Explained Variance Ratios:")
    print("=" * 80)
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio:.1%}")

    # Create a DataFrame to make it easier to read
    loadings_df = pd.DataFrame(pca.components_, 
                            index=[f"PC{i+1}" for i in range(len(pca.components_))], 
                            columns=parameter_list)

    # Show the contribution of each parameter in PC1
    print("Contribution of each parameter to PC1:")
    print(loadings_df.loc["PC1"])
    print(loadings_df.loc["PC2"])

def plot_doubling_time_relationship(csv_file_path):
    """
    Plot the relationship between doubling time and its standard deviation.
    
    Args:
        csv_file_path: Path to the aggregated results CSV file
    """
    # Load the data
    df = pd.read_csv(csv_file_path)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with density coloring
    scatter = plt.scatter(df['doub_time'], df['doub_time_std'], 
                         alpha=0.6, c=df['colony_g_rate'], cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Colony Growth Rate', rotation=270, labelpad=15)
    
    # Add trend line
    z = np.polyfit(df['doub_time'], df['doub_time_std'], 1)
    p = np.poly1d(z)
    plt.plot(df['doub_time'], p(df['doub_time']), 
             "r--", alpha=0.8, label=f'Trend line (slope: {z[0]:.2f})')
    
    # Calculate correlation coefficient
    corr = df['doub_time'].corr(df['doub_time_std'])
    
    # Add labels and title
    plt.xlabel('Doubling Time')
    plt.ylabel('Doubling Time Std')
    plt.title('Relationship between Doubling Time and its Standard Deviation\n' + 
             f'Correlation coefficient: {corr:.2f}')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 80)
    print("\nCorrelation coefficient:", corr)
    print("\nTrend line equation: y = {:.2f}x + {:.2f}".format(z[0], z[1]))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Specify your parameters
    csv_file = "ARCADE_OUTPUT/SMALL_STD_ONLY_VOLUME/simulation_metrics.csv"
    #csv_file = "ARCADE_OUTPUT/MANUAL_VOLUME_APOTOSIS/simulation_metrics.csv"
    metrics_name = "doub_time_std"
    parameter_list = ["CELL_VOLUME_SIGMA", "APOPTOSIS_AGE_SIGMA", "NECROTIC_FRACTION", 
                     "ACCURACY", "AFFINITY"]
    parameter_base_folder = "ARCADE_OUTPUT/SMALL_STD_ONLY_VOLUME"
    #parameter_base_folder = "ARCADE_OUTPUT/MANUAL_VOLUME_APOTOSIS"

    top_10_input_file, bottom_10_input_file = analyze_metric_percentiles(csv_file, metrics_name, verbose=False)
    
    
    # Run analyses
    analyze_and_plot_parameters(top_10_input_file, bottom_10_input_file, parameter_list, parameter_base_folder)
    #analyze_pca_parameters(csv_file, parameter_list, parameter_base_folder, metrics_names)
    
    # Plot doubling time relationship
    plot_doubling_time_relationship(csv_file)
