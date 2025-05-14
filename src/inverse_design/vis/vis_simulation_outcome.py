import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from utils.utils import remove_outliers

def load_and_prepare_data(param_file, metrics_file):
    # Load parameter and metrics data
    params_df = pd.read_csv(param_file)
    metrics_df = pd.read_csv(metrics_file)

    # Clean up parameter data
    params_df = params_df.drop("file_name", axis=1)
    metrics_list = ["doub_time", "act_t2", "colony_g_rate"]
    # Select std columns from metrics
    std_cols = [col for col in metrics_df.columns if "std" in col]
    std_cols = [col for col in metrics_df.columns if col in metrics_list]
    metrics_std_df = metrics_df[std_cols]

    # Normalize both dataframes to [0,1] range
    scaler = MinMaxScaler()
    params_normalized = pd.DataFrame(scaler.fit_transform(params_df), columns=params_df.columns)
    metrics_normalized = pd.DataFrame(
        scaler.fit_transform(metrics_std_df), columns=metrics_std_df.columns
    )

    return params_normalized, metrics_normalized


def plot_pairwise_relationships(metrics_df):
    # Calculate number of metrics
    n_metrics = len(metrics_df.columns)

    # Create a grid of subplots
    fig, axes = plt.subplots(n_metrics, n_metrics, figsize=(2 * n_metrics, 2 * n_metrics))

    # For each pair of metrics
    for i, metric1 in enumerate(metrics_df.columns):
        for j, metric2 in enumerate(metrics_df.columns):
            ax = axes[i, j]

            if i != j:
                # Calculate correlation and slope
                correlation = metrics_df[metric1].corr(metrics_df[metric2])
                slope, intercept = np.polyfit(metrics_df[metric1], metrics_df[metric2], 1)

                # Create scatter plot
                ax.scatter(metrics_df[metric1], metrics_df[metric2], alpha=0.5)

                # Add trend line
                x_range = np.array([metrics_df[metric1].min(), metrics_df[metric1].max()])
                ax.plot(
                    x_range,
                    slope * x_range + intercept,
                    "r--",
                    label=f"slope={slope:.2f}\nr={correlation:.2f}",
                )

                ax.legend(fontsize="small")
            else:
                # On diagonal, show density plot
                sns.kdeplot(data=metrics_df[metric1], ax=ax)
                ax.set_xlabel("")
                ax.set_ylabel("")

            # Only show labels on edge plots
            if i == n_metrics - 1:
                ax.set_xlabel(metric2)
            if j == 0:
                ax.set_ylabel(
                    metric1
                )  # Changed from metric2 to metric1 for correct y-axis labeling

            # Remove ticks for cleaner look
            ax.tick_params(labelsize="small")

    plt.tight_layout()
    plt.savefig("metric_pairwise_relationships.png")
    # plt.show()


def calculate_chaos_metric(metrics_df):
    # First normalize all std values globally to [0,1] range
    scaler = MinMaxScaler()
    normalized_stds = pd.DataFrame(
        scaler.fit_transform(metrics_df), columns=metrics_df.columns, index=metrics_df.index
    )

    # For each row (input), calculate Shannon entropy
    shannon_entropies = []
    for idx in normalized_stds.index:
        # Get distribution for this input
        if idx == 5:
            break
        p = normalized_stds.loc[idx]
        # Calculate Shannon entropy: -sum(p * log(p))
        # Adding small epsilon to avoid log(0)
        print(p)
        entropy = -np.sum(p * np.log(p + 1e-10))
        print(entropy)

        # Weight the entropy by the mean std value to account for absolute magnitude
        mean_std = metrics_df.loc[idx].mean()
        weighted_entropy = entropy * mean_std
        print(weighted_entropy)
        print("--------------------------------")
        shannon_entropies.append(weighted_entropy)
    # Create DataFrame with results
    chaos_df = pd.DataFrame(
        {
            "chaos_metric": shannon_entropies,
        },
        index=metrics_df.index,
    )

    # Plot distribution of chaos metric
    plt.figure(figsize=(10, 6))
    sns.histplot(data=chaos_df, x="chaos_metric", kde=True)
    plt.title("Distribution of Chaos Metric\n(Higher values indicate more chaotic behavior)")
    plt.xlabel("Chaos Metric (Weighted Shannon Entropy)")
    plt.ylabel("Count")
    # plt.show()

    return chaos_df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import RectangleSelector, Button
import matplotlib
matplotlib.use('TkAgg')

def plot_pairwise_scatter(df, metric_names, metrics_ranges, point_size=50, alpha=0.8, grid=True, save_path=None):
    """
    Create three pairwise scatter plots to visualize relationships between three numerical features.
    Interactive selection is enabled - click and drag in any plot to select points.
    Selected points will be highlighted in all plots.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the features to plot
    metric_names : list or tuple of str
        Three column names from df to use as [feature_A, feature_B, feature_C]
    metrics_ranges : dict
        A dictionary mapping metric names to their ranges
    point_size : int, optional
        Size of the scatter points
    alpha : float, optional
        Transparency of points (0 to 1)
    grid : bool, optional
        Whether to show grid lines
    """
    # Validate inputs
    if len(metric_names) != 3:
        raise ValueError("metric_names must contain exactly three feature names")
    
    for name in metric_names:
        if name not in df.columns:
            raise ValueError(f"Feature '{name}' not found in DataFrame columns")
        if not pd.api.types.is_numeric_dtype(df[name]):
            raise ValueError(f"Feature '{name}' must be numeric")
    
    # Extract features
    feature_A, feature_B, feature_C = metric_names
    
    # Create the figure and axis
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    ax = ax.flatten()
    
    # Store scatter plots for later reference
    scatters = []
    
    # Create three pairwise scatter plots
    # A vs B
    scatter1 = ax[0].scatter(df[feature_A], df[feature_B],
                         s=point_size,
                         alpha=alpha,
                         edgecolors='k',
                         picker=True)
    ax[0].set_xlabel(feature_A, fontsize=12)
    ax[0].set_ylabel(feature_B, fontsize=12)
    ax[0].set_title(f'{feature_A} vs {feature_B}', fontsize=14)
    ax[0].set_xlim(metrics_ranges[feature_A][0], metrics_ranges[feature_A][1])
    ax[0].set_ylim(metrics_ranges[feature_B][0], metrics_ranges[feature_B][1])
    if grid:
        ax[0].grid(True, linestyle='--', alpha=0.7)
    scatters.append(scatter1)
    
    # B vs C
    scatter2 = ax[1].scatter(df[feature_C], df[feature_B],
                         s=point_size,
                         alpha=alpha,
                         edgecolors='k',
                         picker=True)
    ax[1].set_xlabel(feature_C, fontsize=12)
    ax[1].set_ylabel(feature_B, fontsize=12)
    ax[1].set_title(f'{feature_C} vs {feature_B}', fontsize=14)
    ax[1].set_xlim(metrics_ranges[feature_C][0], metrics_ranges[feature_C][1])
    ax[1].set_ylim(metrics_ranges[feature_B][0], metrics_ranges[feature_B][1])
    if grid:
        ax[1].grid(True, linestyle='--', alpha=0.7)
    scatters.append(scatter2)
    
    # A vs C
    scatter3 = ax[2].scatter(df[feature_A], df[feature_C],
                         s=point_size,
                         alpha=alpha,
                         edgecolors='k',
                         picker=True)
    ax[2].set_xlabel(feature_A, fontsize=12)
    ax[2].set_ylabel(feature_C, fontsize=12)
    ax[2].set_title(f'{feature_A} vs {feature_C}', fontsize=14)
    ax[2].set_xlim(metrics_ranges[feature_A][0], metrics_ranges[feature_A][1])
    ax[2].set_ylim(metrics_ranges[feature_C][0], metrics_ranges[feature_C][1])
    
    if grid:
        ax[2].grid(True, linestyle='--', alpha=0.7)
    scatters.append(scatter3)
    
    # Create a heatmap of the correlation matrix
    corr_mask = np.triu(np.ones_like(df[metric_names].corr(), dtype=bool), k=1)
    sns.heatmap(df[metric_names].corr(), annot=True, cmap='Reds', ax=ax[3], mask=corr_mask)
    ax[3].set_title('Correlation Heatmap', fontsize=14)

    # Store the data for selection
    data = {
        'A': df[feature_A].values,
        'B': df[feature_B].values,
        'C': df[feature_C].values
    }
    
    # Selected indices (shared across all plots)
    selected_indices = set()

    # Function to reset selection
    def reset_selection(event):
        selected_indices.clear()
        
        # Reset colors for all scatter plots
        for scatter in scatters:
            scatter.set_facecolors(['#1f77b4'] * len(df))
        
        # Force redraw
        fig.canvas.draw_idle()
        print("Selection reset")
    


    # Function to handle selection
    def on_select(eclick, erelease):
        if eclick.inaxes != erelease.inaxes:
            return
        
        # Get the selection rectangle coordinates
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Determine which plot was selected
        ax_idx = ax.tolist().index(eclick.inaxes)
        if ax_idx >= 3:  # Skip if selection was in correlation plot
            return
            
        # Determine which features were plotted
        if ax_idx == 0:  # A vs B
            x_feat, y_feat = 'A', 'B'
        elif ax_idx == 1:  # B vs C
            x_feat, y_feat = 'B', 'C'
        else:  # C vs A
            x_feat, y_feat = 'C', 'A'
        
        # Find points within the selection rectangle
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Create mask for points in the selection
        mask = ((data[x_feat] >= x_min) & (data[x_feat] <= x_max) &
                (data[y_feat] >= y_min) & (data[y_feat] <= y_max))
        
        # Get indices of selected points
        new_selected = set(np.where(mask)[0])
        
        # Toggle selection
        for idx in new_selected:
            if idx in selected_indices:
                selected_indices.remove(idx)
            else:
                selected_indices.add(idx)
        
        # Create updated mask
        updated_mask = np.zeros(len(df), dtype=bool)
        for idx in selected_indices:
            updated_mask[idx] = True
        
        # Update colors for all scatter plots
        for scatter in scatters:
            colors = np.array(['#1f77b4'] * len(df))  # Default blue color
            colors[updated_mask] = '#d62728'  # Red for selected points
            scatter.set_facecolors(colors)
        
        # Force redraw
        fig.canvas.draw_idle()

    
    # Create rectangle selectors with explicit button press/release handlers
    selectors = []
    for i in range(3):
        selector = RectangleSelector(
            ax[i], on_select,
            useblit=True,
            button=[1],  # Only use left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        selectors.append(selector)
    
    # Keep a reference to the selectors to prevent garbage collection
    fig.selectors = selectors

    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.12, wspace=0.2, hspace=0.15)

    fig.patch.set_visible(False)
    if save_path:
        plt.savefig(save_path)
    else:
        # Add a reset button
        reset_ax = plt.axes([0.90, 0.005, 0.08, 0.05]) # [x, y, width, height]
        reset_button = Button(reset_ax, 'Reset', color='lightgoldenrodyellow', hovercolor='0.975')
        # Connect the reset button to the function
        reset_button.on_clicked(reset_selection)
        plt.show()
    return fig, ax, selectors



def main():
    param_file = "inputs/kde_sampled_inputs_small_std/kde_sampled_parameters_log.csv"
    data_dir = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_stded/"
    target_metrics = ["doub_time", "act_ratio", "symmetry"]
    metrics_ranges = {"doub_time": (20, 210), "act_ratio": (-0.01, 1.01), "symmetry": (0.5, 1.01)}
    for iter in range(2, 3):
        metrics_file = os.path.join(data_dir, f"iter_{iter}/final_metrics.csv")
        metrics_df = pd.read_csv(metrics_file)

        valid_df = metrics_df[target_metrics].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_df) > 0:
            valid_df = remove_outliers(valid_df, 1.5)
            print(f"Removed {len(metrics_df) - len(valid_df)} outliers in df")
        # Plot
        # Create the jointplot
        g = sns.jointplot(data=valid_df, x='doub_time', y='act_ratio', kind='hex', color='skyblue', height=8)

        # Access the main axes from the JointGrid
        ax_main = g.ax_joint

        # Add lines to the main plot
        ax_main.axvline(x=32, color='red', linestyle='--', label='Target doub_time = 32')
        ax_main.axhline(y=0.9, color='orange', linestyle='--', label='Infeasible act_ratio = 0.9')
        ax_main.axhline(y=0.3, color='orange', linestyle='--', label='Infeasible act_ratio = 0.3')

        # Add arrows
        ax_main.annotate('', xy=(35, 0.95), xytext=(35, 0.9), arrowprops=dict(arrowstyle='->', color='orange'))
        ax_main.annotate('', xy=(35, 0.25), xytext=(35, 0.3), arrowprops=dict(arrowstyle='->', color='orange'))

        # Add legend
        ax_main.legend()
        plt.savefig(f"{data_dir}/iter_{iter}/joint_plot.png")
        plt.show()
        #plot_pairwise_scatter(valid_df, target_metrics, metrics_ranges, save_path=f"{data_dir}/iter_{iter}/pairwise_scatter.png")


if __name__ == "__main__":
    main()
