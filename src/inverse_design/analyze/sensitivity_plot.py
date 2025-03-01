import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_sensitivity_bubble(param_file, metrics_file, target_metric='symmetry', save_file=None):
    """
    Create a bubble plot showing parameter relationships.
    
    Args:
        param_file (str): Path to parameters CSV file
        metrics_file (str): Path to metrics CSV file
        target_metric (str): Target metric to analyze (default: 'symmetry')
        save_file (str): Path to save the figure (optional)
    """
    # Read data
    params_df = pd.read_csv(param_file)
    metrics_df = pd.read_csv(metrics_file)
    
    param_cols = [col for col in params_df.columns if col != 'input_folder']
    params_std = params_df[param_cols].std()
    varying_cols = params_std[params_std.abs() > 1e-10].index
    if len(varying_cols) != 2:
        raise ValueError(f"Expected 2 varying parameters, found {len(varying_cols)}")
    
    # Create bubble plot
    plt.figure(figsize=(10, 8))
    
    # Get standard deviation column
    std_col = f"{target_metric}_std"
    if std_col not in metrics_df.columns:
        raise ValueError(f"Standard deviation column {std_col} not found in metrics file")
    
    # Normalize sizes between 50 and 500 based on standard deviation
    min_std = min(metrics_df[std_col])
    max_std = max(metrics_df[std_col])
    normalized_sizes = 50 + (metrics_df[std_col] - min_std) * (500 - 50) / (max_std - min_std)
    
    # Create scatter plot with colormap for metric value and size for std
    scatter = plt.scatter(params_df[varying_cols[0]], 
                         params_df[varying_cols[1]], 
                         s=normalized_sizes,
                         c=metrics_df[target_metric],
                         cmap='viridis',
                         alpha=0.6)
    
    plt.xlabel(varying_cols[0])
    plt.ylabel(varying_cols[1])
    plt.title(f'Parameter Space: Impact on {target_metric}\nBubble size shows standard deviation')
    
    # Add colorbar for metric values
    cbar = plt.colorbar(scatter)
    cbar.set_label(f'{target_metric} value')
    
    # Add size legend with empty circles for standard deviation
    legend_elements = [plt.scatter([], [], 
                                 s=50 + (s - min_std) * (500 - 50) / (max_std - min_std),
                                 label=f'{s:.3f}', 
                                 alpha=0.6,
                                 facecolor='none',
                                 edgecolor='black')
                      for s in np.linspace(min_std, max_std, 3)]
    plt.legend(handles=legend_elements, title=f'Standard Deviation', 
              labelspacing=2, title_fontsize=10)
    
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()

def main():
    target_metric = "cycle_length"
    parameter_base_folder = f"ARCADE_OUTPUT/SENSITIVITY/{target_metric}"
    
    save_file = f'{parameter_base_folder}/sensitivity_bubble.png'
    plot_sensitivity_bubble(f'{parameter_base_folder}/all_param_df.csv', 
                       f'{parameter_base_folder}/final_metrics.csv',
                       target_metric=target_metric,
                       save_file=save_file)

if __name__ == "__main__":
    main()
