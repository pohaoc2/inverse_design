# Visualization of the BDM model
import os
from typing import Dict, Tuple, Optional
from typing import List, Any
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from inverse_design.models.bdm.grid import Grid
from inverse_design.common.enum import Metric, Target


def plot_grid(grid: Grid, time_point: float):
    """Visualize the grid with blue squares for cells and white squares for empty locations"""
    fig, ax = plt.subplots(figsize=(10, 10))
    lattice_size = grid.lattice_size

    # Set equal aspect ratio and limits
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, lattice_size - 0.5)
    ax.set_ylim(-0.5, lattice_size - 0.5)

    for i in range(lattice_size):
        for j in range(lattice_size):
            location = grid.lattice[i, j]
            color = "blue" if location.has_cell() else "white"
            square = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor=color, edgecolor="black")
            ax.add_patch(square)
    cell_density = grid.num_cells / (lattice_size**2) * 100
    ax.set_title(f"Time: {time_point}, Cell density: {cell_density:.2f}%")

    # Create 'plots' directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/grid_{int(time_point)}.png")
    plt.close()


def plot_cell_density(
    time_points: List[float],
    cell_densities: List[float],
    red_dot_time: float,
    red_dot_density: float,
):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(time_points, cell_densities, "ko-")
    ax.plot(red_dot_time, red_dot_density, "ro")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cell density")
    ax.set_title("Cell density over time")
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/cell_density_{int(red_dot_time)}.png")
    plt.close()


def plot_metric_distribution(
    all_metrics_values: List[float],
    accepted_metrics_values: List[float],
    metric: Metric,
    targets: List[Target],
    save_path: str,
    x_range: Optional[Tuple[float, float]] = None,
):
    """Plot distribution of a metric for all and accepted samples
    Args:
        all_metrics: Dictionary containing 'all' and 'accepted' metrics
        metric: The metric to plot
        targets: List of targets to find the corresponding target value
        save_path: Path to save the plot
        x_range: Optional tuple of (min, max) for x-axis limits
    """
    # Find target value for this metric
    target_value = next(t.value for t in targets if t.metric == metric)

    plt.figure(figsize=(8, 6))

    bins = np.histogram_bin_edges(all_metrics_values, bins=10)

    # Plot histograms with common bins
    plt.hist(
        all_metrics_values, bins=bins, alpha=0.3, density=True, label="All samples", color="blue"
    )
    plt.hist(
        accepted_metrics_values,
        bins=bins,
        alpha=0.3,
        density=True,
        label="Accepted samples",
        color="orange",
    )

    # Add KDE curves if enough samples
    if x_range is None:
        x_min, x_max = min(all_metrics_values), max(all_metrics_values)
    else:
        x_min, x_max = x_range
    x_plot = np.linspace(x_min, x_max, 200)

    if len(all_metrics_values) > 1:
        kde_all = stats.gaussian_kde(all_metrics_values)
        plt.plot(x_plot, kde_all(x_plot), color="blue", linewidth=2, label="All samples (KDE)")

    if len(accepted_metrics_values) > 1:
        kde_accepted = stats.gaussian_kde(accepted_metrics_values)
        plt.plot(
            x_plot,
            kde_accepted(x_plot),
            color="orange",
            linewidth=2,
            label="Accepted samples (KDE)",
        )

    plt.axvline(target_value, color="r", linestyle="--", label=f"Target {metric.value}")

    plt.xlabel(f"{metric.value}")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {metric.value}")
    if x_range:
        plt.xlim(x_range)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_parameter_kde(pdf_results: Dict, abc_config: Dict, save_path: str):
    """Plot KDE distributions for each parameter with HPD interval and mode
    Args:
        pdf_results: Results from _estimate_pdfs containing KDE objects
        abc_config: ABC configuration containing parameter ranges (prior)
        save_path: Path to save the plot
    """
    param_names = pdf_results["param_names"]
    n_params = len(param_names)
    fig, axes = plt.subplots(2, n_params, figsize=(15, 8), height_ratios=[1, 2])

    for i in range(n_params):
        param_name = param_names[i]
        kde = pdf_results["independent"][param_name]
        
        # Get prior range
        prior_min = abc_config.parameter_ranges[param_name].min
        prior_max = abc_config.parameter_ranges[param_name].max

        # Plot prior in top row (uniform probability)
        prior_prob = 1.0 / (prior_max - prior_min)
        axes[0, i].fill_between(
            [prior_min, prior_max], [prior_prob, prior_prob], alpha=0.5, color="gray"
        )
        axes[0, i].axhline(y=prior_prob, color="gray", linestyle=":", alpha=0.5)
        axes[0, i].set_title(f"{param_name} Prior")
        axes[0, i].set_xlabel("Value")
        if i == 0:
            axes[0, i].set_ylabel("Prior Probability")

        # Plot posterior in bottom row
        x = np.linspace(prior_min, prior_max, 200)
        y = kde(x)
        
        # Convert to probability mass per bin
        bin_width = x[1] - x[0]  # Width of each bin
        y = y * bin_width  # Convert density to probability mass per bin
        
        # Normalize to ensure total probability = 1
        y = y / np.sum(y)
        
        # Verify total probability is 1
        total_prob = np.sum(y)
        
        # For prior, also convert to probability mass per bin
        prior_prob = (1.0 / (prior_max - prior_min)) * bin_width
        
        # Find mode (highest probability point)
        mode_idx = np.argmax(y)
        mode = x[mode_idx]
        
        # Calculate 95% HPD interval
        def compute_hpd_interval(x, y, alpha=0.95):
            # y is already normalized to probability
            sorted_indices = np.argsort(y)[::-1]  # Sort in descending order
            x_sorted = x[sorted_indices]
            y_sorted = y[sorted_indices]
            
            # Calculate cumulative probability
            cumsum = np.zeros_like(y_sorted)
            for i in range(len(y_sorted)):
                if i == 0:
                    cumsum[i] = 0
                else:
                    width = x_sorted[i] - x_sorted[i-1]
                    height = (y_sorted[i] + y_sorted[i-1]) / 2
                    cumsum[i] = cumsum[i-1] + width * height
            
            # Normalize cumsum
            cumsum = cumsum / cumsum[-1]
            
            # Find the threshold that gives us alpha probability mass
            threshold_idx = np.searchsorted(cumsum, alpha)
            if threshold_idx >= len(y_sorted):
                threshold_idx = len(y_sorted) - 1
            threshold = y_sorted[threshold_idx]
            
            # Find all points above the threshold
            hpd_mask = y >= threshold
            
            # Get the bounds
            hpd_x = x[hpd_mask]
            if len(hpd_x) == 0:  # If no points found, use the mode
                hpd_min = x[np.argmax(y)]
                hpd_max = hpd_min
            else:
                hpd_min = np.min(hpd_x)
                hpd_max = np.max(hpd_x)
            
            return hpd_min, hpd_max, threshold

        hpd_min, hpd_max, threshold = compute_hpd_interval(x, y)
        
        # Plot posterior probability distribution
        axes[1, i].plot(x, y, "b-", label="Posterior")
        
        # Plot mode and HPD interval
        axes[1, i].axvline(mode, color="r", linestyle="--", label="Mode")
        axes[1, i].axvline(hpd_min, color="g", linestyle=":", alpha=0.7, label="95% HPD")
        axes[1, i].axvline(hpd_max, color="g", linestyle=":", alpha=0.7)
        
        # Fill HPD interval
        hpd_mask = (x >= hpd_min) & (x <= hpd_max)
        axes[1, i].fill_between(x[hpd_mask], y[hpd_mask], alpha=0.2, color="g")

        axes[1, i].set_title(f"Mode={mode:.3f}\nHPD: [{hpd_min:.3f}, {hpd_max:.3f}]")
        axes[1, i].set_xlabel("Value")
        if i == 0:
            axes[1, i].set_ylabel("Posterior Probability")
            axes[1, i].legend(loc="upper right")

        xlim = (
            prior_min - 0.1 * (prior_max - prior_min),
            prior_max + 0.1 * (prior_max - prior_min),
        )
        axes[0, i].set_xlim(xlim)
        axes[1, i].set_xlim(xlim)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1)
    plt.savefig(save_path)
    plt.close()


def plot_parameter_correlations(accepted_params: List[Dict], pdf_results: Dict, save_path: str):
    """Plot parameter correlations and marginal distributions
    Args:
        accepted_params: List of accepted parameter dictionaries
        pdf_results: Results from _estimate_pdfs
        save_path: Path to save the plot
    """
    param_names = pdf_results["param_names"]
    n_params = len(param_names)

    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    fig.suptitle("Parameter Correlations")

    # Convert accepted params to array for easier plotting
    X = np.array([list(p.values()) for p in accepted_params])

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            if i == j:  # Diagonal: show marginal distributions
                ax.hist(X[:, i], bins=20, density=True, alpha=0.6)
                kde = pdf_results["independent"][param_names[i]]
                x = np.linspace(X[:, i].min(), X[:, i].max(), 100)
                ax.plot(x, kde(x))
                if i == n_params - 1:
                    ax.set_xlabel(param_names[i])
                if j == 0:
                    ax.set_ylabel(param_names[i])
            else:  # Off-diagonal: show scatter plots
                ax.scatter(X[:, j], X[:, i], alpha=0.5, s=10)
                if i == n_params - 1:
                    ax.set_xlabel(param_names[j])
                if j == 0:
                    ax.set_ylabel(param_names[i])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_combined_grid_and_density(
    grid: Grid,
    time_points: List[float],
    cell_densities: List[float],
    red_dot_time_point: float,
    target_density: float,
    target_time_point: float,
):
    """Visualize the grid and cell density in a combined plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the grid
    lattice_size = grid.lattice_size
    ax1.set_aspect("equal")
    ax1.set_xlim(-0.5, lattice_size - 0.5)
    ax1.set_ylim(-0.5, lattice_size - 0.5)

    for i in range(lattice_size):
        for j in range(lattice_size):
            location = grid.lattice[i, j]
            color = "blue" if location.has_cell() else "white"
            square = plt.Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor=color, edgecolor="black")
            ax1.add_patch(square)

    cell_density = grid.num_cells / (lattice_size**2) * 100
    ax1.set_title(f"Grid at Time: {red_dot_time_point}, Cell density: {cell_density:.2f}%")
    red_dot_density = cell_densities[time_points.index(red_dot_time_point)]

    # Plot the cell density over time
    ax2.plot(time_points, cell_densities, "ko-", label="Cell Density")
    ax2.plot(red_dot_time_point, red_dot_density, "ro", label="Current Cell Density")
    ax2.axvline(x=target_time_point, color="g", linestyle="--", label="Target Equilibrium Time")
    ax2.axhline(
        y=target_density, color="b", linestyle="--", label="Target Equilibrium Cell Density"
    )
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Cell Density")
    ax2.set_title("Cell Density Over Time")
    ax2.set_xlim(0, 2500)
    ax2.set_ylim(0, 100)
    ax2.legend()

    # Create 'plots' directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"plots/combined_grid_and_density_{int(red_dot_time_point)}.png")
    plt.close()


def plot_abc_results(
    accepted_params: List[Dict],
    pdf_results: Dict,
    all_metrics: List[Dict],
    accepted_metrics: List[Dict],
    targets: List[Target],
    abc_config: Any,
    model_config: Any,
    save_dir: str = ".",
):
    """Plot all ABC inference results
    Args:
        accepted_params: List of accepted parameter dictionaries
        pdf_results: Results from PDF estimation
        all_sample_metrics: Dictionary containing all and accepted metrics
        targets: List of target metrics
        model_config: Model configuration containing output settings
        save_dir: Directory to save plots
    """
    # Plot parameter distributions and correlations
    plot_parameter_correlations(
        accepted_params, pdf_results, os.path.join(save_dir, "parameter_correlations.png")
    )
    plot_parameter_kde(
        pdf_results, abc_config, os.path.join(save_dir, "parameter_distributions.png")
    )

    # Plot metric distributions
    x_ranges = {
        Metric.DENSITY: (0, 100),
        Metric.TIME_TO_EQUILIBRIUM: (0, model_config.output.max_time),
    }
    for target in targets:
        plot_metric_distribution(
            all_metrics,
            accepted_metrics,
            target.metric,
            targets,
            os.path.join(save_dir, f"{target.metric.value}_distribution.png"),
            x_range=x_ranges[target.metric],
        )


def plot_joint_distribution(accepted_params: List[Dict], save_path: str):
    """Plot joint distribution of parameters using corner plot
    Args:
        accepted_params: List of accepted parameter dictionaries
        save_path: Path to save the plot
    """
    # Convert accepted params to numpy array
    param_names = list(accepted_params[0].keys())
    X = np.array([[p[name] for name in param_names] for p in accepted_params])
    n_params = len(param_names)
    
    # Create corner plot
    fig, axes = plt.subplots(n_params, n_params, figsize=(3*n_params, 3*n_params))
    
    # Loop through parameter pairs
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:  # Diagonal: show marginal distributions
                # Calculate KDE for marginal distribution
                kde = stats.gaussian_kde(X[:, i])
                x_range = np.linspace(np.min(X[:, i]), np.max(X[:, i]), 100)
                density = kde(x_range)
                
                # Convert to probability mass per bin
                bin_width = x_range[1] - x_range[0]
                prob_mass = density * bin_width
                prob_mass = prob_mass / np.sum(prob_mass)
                
                # Plot marginal distribution
                ax.plot(x_range, prob_mass, 'b-')
                ax.fill_between(x_range, prob_mass, alpha=0.3)
                
                # Calculate and plot mode and HPD
                mode_idx = np.argmax(prob_mass)
                mode = x_range[mode_idx]
                
                # Calculate HPD
                sorted_idx = np.argsort(prob_mass)[::-1]
                cumsum = np.cumsum(prob_mass[sorted_idx])
                hpd_idx = cumsum <= 0.95
                hpd_range = x_range[sorted_idx[hpd_idx]]
                hpd_min, hpd_max = np.min(hpd_range), np.max(hpd_range)
                
                # Add vertical lines for mode and HPD
                ax.axvline(mode, color='r', linestyle='--', alpha=0.5)
                ax.axvline(hpd_min, color='g', linestyle=':', alpha=0.5)
                ax.axvline(hpd_max, color='g', linestyle=':', alpha=0.5)
                
                # Add title with statistics
                ax.set_title(f'{param_names[i]}\nMode: {mode:.2f}\nHPD: [{hpd_min:.2f}, {hpd_max:.2f}]')
                
            elif i > j:  # Lower triangle: show joint distributions
                # Calculate 2D KDE
                try:
                    kde = stats.gaussian_kde(np.vstack([X[:, j], X[:, i]]))
                    
                    # Create grid of points
                    x_range = np.linspace(np.min(X[:, j]), np.max(X[:, j]), 50)
                    y_range = np.linspace(np.min(X[:, i]), np.max(X[:, i]), 50)
                    xx, yy = np.meshgrid(x_range, y_range)
                    
                    # Evaluate KDE on grid
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    z = kde(positions).reshape(xx.shape)
                    
                    # Convert to probability mass per bin
                    bin_area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
                    z = z * bin_area
                    z = z / np.sum(z)
                    
                    # Plot joint distribution
                    im = ax.imshow(z, extent=[np.min(x_range), np.max(x_range),
                                            np.min(y_range), np.max(y_range)],
                                 aspect='auto', origin='lower', cmap='viridis')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax)
                    
                    # Add scatter plot of actual points
                    ax.scatter(X[:, j], X[:, i], c='white', alpha=0.1, s=1)
                    
                except np.linalg.LinAlgError:
                    ax.text(0.5, 0.5, 'Insufficient\ndata for\nKDE', 
                          ha='center', va='center', transform=ax.transAxes)
            
            else:  # Upper triangle: leave empty
                ax.axis('off')
            
            # Set labels
            if i == n_params-1:
                ax.set_xlabel(param_names[j])
            if j == 0:
                ax.set_ylabel(param_names[i])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pca_visualization(accepted_params_list: List[List[Dict]], save_path: str):
    """Visualize high-dimensional parameter space in 2D using PCA for multiple parameter groups
    Args:
        accepted_params_list: List of lists of accepted parameter dictionaries, each inner list represents a group
        save_path: Path to save the plot
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Convert all params to numpy array
    param_names = list(accepted_params_list[0][0].keys())
    X_groups = []
    for accepted_params in accepted_params_list:
        X = np.array([[p[name] for name in param_names] for p in accepted_params])
        X_groups.append(X)
    
    # Combine all data for PCA
    X_combined = np.vstack(X_groups)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Split PCA results back into groups
    start_idx = 0
    X_pca_groups = []
    for X in X_groups:
        end_idx = start_idx + len(X)
        X_pca_groups.append(X_pca[start_idx:end_idx])
        start_idx = end_idx
    
    # Calculate explained variance ratios
    explained_var_ratio = pca.explained_variance_ratio_
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot each group with different colors
    colors = ["red", "blue"] #plt.cm.tab10(np.linspace(0, 1, len(X_pca_groups)))
    
    for group_idx, (X_pca_group, color) in enumerate(zip(X_pca_groups, colors)):
        plt.scatter(X_pca_group[:, 0], X_pca_group[:, 1], 
                   c=[color], alpha=0.6, s=20, label=f'Group {group_idx + 1}')
    
    plt.xlabel(f'PC1 ({explained_var_ratio[0]:.1%} variance explained)')
    plt.ylabel(f'PC2 ({explained_var_ratio[1]:.1%} variance explained)')
    plt.title('PCA Projection of Parameter Space')
    plt.legend()
    
    # Add text box with total explained variance
    text = f'Total variance explained: {np.sum(explained_var_ratio):.1%}'
    plt.text(0.95, 0.95, text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
