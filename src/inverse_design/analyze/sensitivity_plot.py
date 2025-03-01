import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from inverse_design.analyze.sensitivity_analysis import align_dataframes

def get_varying_parameters(params_df, threshold=1e-10):
    """
    Identify parameters that vary in the dataset above a given threshold.

    Args:
        params_df (pd.DataFrame): DataFrame containing parameter values
        threshold (float): Minimum standard deviation to consider a parameter as varying

    Returns:
        list: List of column names for parameters that vary above the threshold
    """
    param_cols = [col for col in params_df.columns if col != "input_folder"]
    params_std = params_df[param_cols].std()
    return list(params_std[params_std.abs() > threshold].index)


def plot_sensitivity_bubble(param_df, metrics_df, target_metric="symmetry", save_file=None):
    """
    Create a bubble plot showing parameter relationships.

    Args:
        param_df (pd.DataFrame): DataFrame containing parameters
        metrics_df (pd.DataFrame): DataFrame containing metrics
        target_metric (str): Target metric to analyze (default: 'symmetry')
        save_file (str): Path to save the figure (optional)
    """
    # Get varying parameters
    varying_cols = get_varying_parameters(param_df)
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
    scatter = plt.scatter(
        param_df[varying_cols[0]],
        param_df[varying_cols[1]],
        s=normalized_sizes,
        c=metrics_df[target_metric],
        cmap="viridis",
        alpha=0.6,
    )

    plt.xlabel(varying_cols[0])
    plt.ylabel(varying_cols[1])
    plt.title(f"Parameter Space: Impact on {target_metric}\nBubble size shows standard deviation")

    # Add colorbar for metric values
    cbar = plt.colorbar(scatter)
    cbar.set_label(f"{target_metric} value")

    # Add size legend with empty circles for standard deviation
    legend_elements = [
        plt.scatter(
            [],
            [],
            s=50 + (s - min_std) * (500 - 50) / (max_std - min_std),
            label=f"{s:.3f}",
            alpha=0.6,
            facecolor="none",
            edgecolor="black",
        )
        for s in np.linspace(min_std, max_std, 3)
    ]
    plt.legend(
        handles=legend_elements, title=f"Standard Deviation", labelspacing=2, title_fontsize=10
    )

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()


def plot_partial_dependence(
    param_df, metrics_df, target_metric="symmetry", save_file=None
):
    """
    Create partial dependence plots showing relationship between parameters and metric.
    For each subplot:
        - First subplot: Shows param_1 vs metric, with separate lines for each unique param_2 value
        - Second subplot: Shows param_2 vs metric, with separate lines for each unique param_1 value
    
    Args:
        param_df (pd.DataFrame): DataFrame containing parameters
        metrics_df (pd.DataFrame): DataFrame containing metrics
        target_metric (str): Target metric to analyze
        save_file (str): Path to save the figure (optional)
    """
    df = pd.concat([param_df, metrics_df], axis=1)
    varying_cols = get_varying_parameters(param_df)
    if len(varying_cols) != 2:
        raise ValueError(f"Expected 2 varying parameters, found {len(varying_cols)}")
    
    param_1, param_2 = varying_cols
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    markers = ['o', 'x', '+', 's', 'D', 'P', 'H', 'X', 'd', 'p', 'h', 'v', '8']
    
    for i, (main_param, other_param) in enumerate([(param_1, param_2), (param_2, param_1)]):
        unique_values = sorted(df[other_param].unique())
        palette = sns.color_palette(n_colors=len(unique_values))
        
        for j, value in enumerate(unique_values):
            mask = np.isclose(df[other_param], value, rtol=1e-10)
            subset = df[mask].copy()
            subset = subset.sort_values(by=main_param)
            axes[i].plot(
                subset[main_param],
                subset[target_metric],
                '-',
                marker=markers[j % len(markers)],
                color=palette[j],
                label=f"{other_param}={value:.3f}",
                markersize=4,
                alpha=0.6
            )
        
        axes[i].set_xlabel(main_param)
        axes[i].set_ylabel(target_metric)
        axes[i].set_title(f"Relationship between {target_metric}\nand {main_param}")
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def main():
    target_metric = "symmetry"
    parameter_base_folder = f"ARCADE_OUTPUT/SENSITIVITY/{target_metric}"
    save_file = f"{parameter_base_folder}/sensitivity_bubble.png"
    param_df = pd.read_csv(f"{parameter_base_folder}/all_param_df.csv")
    metrics_df = pd.read_csv(f"{parameter_base_folder}/final_metrics.csv")
    param_df, metrics_df = align_dataframes(param_df, metrics_df)


    if 1:
        plot_sensitivity_bubble(
            param_df,
            metrics_df,
            target_metric=target_metric,
            save_file=save_file,
        )
    save_file = f"{parameter_base_folder}/sensitivity_partial_dependence.png"
    if 1:
        plot_partial_dependence(
            param_df,
            metrics_df,
            target_metric=target_metric,
            save_file=save_file,
        )


if __name__ == "__main__":
    main()
