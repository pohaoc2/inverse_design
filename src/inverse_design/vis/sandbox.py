import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates
from scipy.stats import gaussian_kde
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure
warnings.filterwarnings('ignore')

# Load the data
base_dir = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast"
prior_df = pd.read_csv(f"{base_dir}/iter_0/all_param_df.csv")
posterior_df = pd.read_csv(f"{base_dir}/iter_4/all_param_df.csv")
drop_cols = ["input_folder", "X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"]
prior_df.drop(columns=drop_cols, inplace=True)
posterior_df.drop(columns=drop_cols, inplace=True)
param_names = prior_df.columns.tolist()


# 1. Corner Plot (Pair Plot) for subset of parameters
def create_corner_plot(prior_data, posterior_data, param_subset=None):
    """Create corner plot for a subset of parameters"""
    if param_subset is None:
        param_subset = param_names[:5]  # Show first 5 parameters
    
    fig, axes = plt.subplots(len(param_subset), len(param_subset), 
                            figsize=(12, 12), tight_layout=True)
    
    for i, param1 in enumerate(param_subset):
        for j, param2 in enumerate(param_subset):
            ax = axes[i, j]
            
            if i == j:  # Diagonal: marginal distributions
                ax.hist(prior_data[param1], bins=30, alpha=0.5, 
                       color='blue', density=True, label='Prior')
                ax.hist(posterior_data[param1], bins=30, alpha=0.5, 
                       color='orange', density=True, label='Posterior')
                ax.set_ylabel('Density')
                ax.legend()
            elif i > j:  # Lower triangle: scatter plots
                ax.scatter(prior_data[param2], prior_data[param1], 
                          alpha=0.3, s=1, color='blue', label='Prior')
                ax.scatter(posterior_data[param2], posterior_data[param1], 
                          alpha=0.3, s=1, color='orange', label='Posterior')
                ax.set_xlabel(param2)
                ax.set_ylabel(param1)
            else:  # Upper triangle: clear
                ax.axis('off')
    
    plt.suptitle('Corner Plot: Prior vs Posterior')
    return fig

# 2. Marginal Distribution Comparison
def plot_marginal_distributions(prior_data, posterior_data):
    """Plot marginal distributions for all parameters"""
    n_cols = 4
    n_rows = int(np.ceil(len(param_names) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for i, param in enumerate(param_names):
        ax = axes[i]
        
        # Plot histograms
        ax.hist(prior_data[param], bins=30, alpha=0.5, 
               color='blue', density=True, label='Prior')
        ax.hist(posterior_data[param], bins=30, alpha=0.5, 
               color='orange', density=True, label='Posterior')
        
        # Add vertical lines for means
        ax.axvline(prior_data[param].mean(), color='blue', 
                   linestyle='--', alpha=0.7)
        ax.axvline(posterior_data[param].mean(), color='orange', 
                   linestyle='--', alpha=0.7)
        
        ax.set_title(param)
        ax.set_ylabel('Density')
        ax.legend()
    
    # Remove empty subplots
    for i in range(len(param_names), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Marginal Distributions: Prior vs Posterior', y=1.02)
    return fig

# 3. Principal Component Analysis (PCA) Visualization
def plot_pca_analysis(prior_data, posterior_data):
    """Perform PCA and visualize in 2D"""
    # Standardize the data
    scaler = StandardScaler()
    
    # Fit on combined data
    combined_data = np.vstack([prior_data.values, posterior_data.values])
    scaler.fit(combined_data)
    
    prior_scaled = scaler.transform(prior_data.values)
    posterior_scaled = scaler.transform(posterior_data.values)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(combined_data)
    
    prior_pca = pca.transform(prior_scaled)
    posterior_pca = pca.transform(posterior_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA scatter plot
    ax1.scatter(prior_pca[:, 0], prior_pca[:, 1], alpha=0.5, 
               color='blue', s=1, label='Prior')
    ax1.scatter(posterior_pca[:, 0], posterior_pca[:, 1], alpha=0.5, 
               color='orange', s=1, label='Posterior')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.legend()
    ax1.set_title('PCA Projection')
    
    # Component loadings
    loadings = pca.components_.T
    ax2.bar(range(len(param_names)), loadings[:, 0], alpha=0.7, 
           label='PC1')
    ax2.bar(range(len(param_names)), loadings[:, 1], alpha=0.7, 
           label='PC2')
    ax2.set_xticks(range(len(param_names)))
    ax2.set_xticklabels(param_names, rotation=45, ha='right')
    ax2.set_ylabel('Loading')
    ax2.legend()
    ax2.set_title('PCA Component Loadings')
    
    plt.tight_layout()
    return fig

# 4. Correlation Matrix Heatmap
def plot_correlation_matrix(prior_data, posterior_data):
    """Plot correlation matrices for prior and posterior"""
    prior_corr = prior_data.corr()
    posterior_corr = posterior_data.corr()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Prior correlation matrix
    sns.heatmap(prior_corr, cmap='coolwarm', center=0,
                ax=ax1, cbar_kws={'label': 'Correlation'})
    ax1.set_title('Prior Parameter Correlations')
    
    # Posterior correlation matrix
    sns.heatmap(posterior_corr, cmap='coolwarm', center=0,
                ax=ax2, cbar_kws={'label': 'Correlation'})
    ax2.set_title('Posterior Parameter Correlations')
    
    plt.tight_layout()
    return fig

# 4b. Correlation Matrix Heatmap for each peak
def plot_correlation_matrix_for_peaks(posterior_data, param_subset=None, n_components=2):
    """Plot correlation matrices for prior and posterior for each peak"""
    if param_subset is None:
        param_subset = param_names
    
    # Prepare data
    data = posterior_data[param_subset].values
    
    # Perform PCA and find peaks
    _, peak_positions, _, _, _, _, _, peak_points = perform_pca_and_find_peaks(data, n_components)

    # Create a figure with subplots
    n_peaks = len(peak_positions)
    fig, axes = plt.subplots(1, n_peaks, figsize=(8 * n_peaks, 8))
    
    # Plot the correlation matrix for each peak
    for i, peak_point in enumerate(peak_points):
        # Create a dataframe with the peak points and the parameter names
        peak_point = pd.DataFrame(peak_point, columns=param_subset)
        corr_matrix = peak_point.corr()
        # Mask the upper triangle only (not the diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_matrix = corr_matrix.mask(mask)
        if i == len(peak_points) - 1:
            sns.heatmap(corr_matrix, cmap='coolwarm', center=0, ax=axes[i], fmt=".2f", cbar=True)
        else:
            sns.heatmap(corr_matrix, cmap='coolwarm', center=0, ax=axes[i], fmt=".2f", cbar=False)
        axes[i].set_title(f'Correlation Matrix for Peak {i+1}')
    
    plt.tight_layout()
    return fig
    
# 5. Trace plots for parameter convergence
def plot_trace_analysis(prior_data, posterior_data):
    """Plot trace-like analysis showing parameter evolution"""
    n_cols = 4
    n_rows = int(np.ceil(len(param_names) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for i, param in enumerate(param_names):
        ax = axes[i]
        
        # Create synthetic "chain" by randomly ordering samples
        indices = np.arange(len(prior_data))
        np.random.shuffle(indices)
        
        prior_trace = prior_data[param].values[indices]
        posterior_trace = posterior_data[param].values[indices]
        
        # Plot running averages
        x = np.arange(len(prior_trace))
        prior_running_mean = np.cumsum(prior_trace) / (x + 1)
        posterior_running_mean = np.cumsum(posterior_trace) / (x + 1)
        
        ax.plot(x[::50], prior_running_mean[::50], color='blue', 
               alpha=0.7, label='Prior')
        ax.plot(x[::50], posterior_running_mean[::50], color='orange', 
               alpha=0.7, label='Posterior')
        
        ax.set_title(f'{param} - Running Mean')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.legend()
    
    # Remove empty subplots
    for i in range(len(param_names), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Running Means: Prior vs Posterior', y=1.02)
    return fig

# 6. Summary statistics comparison
def plot_summary_comparison(prior_data, posterior_data):
    """Plot summary statistics comparison"""
    stats_df = pd.DataFrame({
        'Parameter': param_names,
        'Prior Mean': prior_data.mean().values,
        'Prior Std': prior_data.std().values,
        'Posterior Mean': posterior_data.mean().values,
        'Posterior Std': posterior_data.std().values
    })
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean comparison
    x = np.arange(len(param_names))
    width = 0.35
    ax1.bar(x - width/2, stats_df['Prior Mean'], width, 
           label='Prior', alpha=0.7)
    ax1.bar(x + width/2, stats_df['Posterior Mean'], width, 
           label='Posterior', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_names, rotation=45, ha='right')
    ax1.set_ylabel('Mean')
    ax1.set_title('Parameter Means')
    ax1.legend()
    
    # Standard deviation comparison
    ax2.bar(x - width/2, stats_df['Prior Std'], width, 
           label='Prior', alpha=0.7)
    ax2.bar(x + width/2, stats_df['Posterior Std'], width, 
           label='Posterior', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(param_names, rotation=45, ha='right')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Parameter Standard Deviations')
    ax2.legend()
    
    # Mean difference (Posterior - Prior)
    mean_diff = stats_df['Posterior Mean'] - stats_df['Prior Mean']
    ax3.bar(x, mean_diff, color='green', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(param_names, rotation=45, ha='right')
    ax3.set_ylabel('Mean Difference')
    ax3.set_title('Change in Mean (Posterior - Prior)')
    
    # Uncertainty reduction (Prior Std / Posterior Std)
    uncertainty_reduction = stats_df['Prior Std'] / stats_df['Posterior Std']
    ax4.bar(x, uncertainty_reduction, color='purple', alpha=0.7)
    ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(param_names, rotation=45, ha='right')
    ax4.set_ylabel('Uncertainty Reduction Ratio')
    ax4.set_title('Uncertainty Reduction (Prior Std / Posterior Std)')
    
    plt.tight_layout()
    return fig

# 7. Posterior Pair Density Plot
def plot_posterior_pair_density(posterior_data, param_subset=None):
    """Create a pair plot showing posterior density relationships between parameters using KDE contours"""
    if param_subset is None:
        param_subset = param_names[:5]  # Show first 5 parameters by default
    
    n_params = len(param_subset)
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    
    # Create meshgrid for each pair of parameters
    for i, param1 in enumerate(param_subset):
        for j, param2 in enumerate(param_subset):
            ax = axes[i, j]
            
            if i == j:  # Diagonal: KDE of single parameter
                
                kde = gaussian_kde(posterior_data[param1])
                x_range = np.linspace(posterior_data[param1].min(), posterior_data[param1].max(), 100)
                ax.plot(x_range, kde(x_range), color='blue')
                ax.fill_between(x_range, kde(x_range), alpha=0.3, color='blue')
                ax.set_ylabel(param1)
            
            elif i > j:  # Lower triangle: KDE contour plot
                # Create meshgrid for the contour plot
                x = np.linspace(posterior_data[param2].min(), posterior_data[param2].max(), 100)
                y = np.linspace(posterior_data[param1].min(), posterior_data[param1].max(), 100)
                X, Y = np.meshgrid(x, y)
                positions = np.vstack([X.ravel(), Y.ravel()])
                
                # Calculate KDE
                values = np.vstack([posterior_data[param2], posterior_data[param1]])
                kernel = gaussian_kde(values)
                Z = np.reshape(kernel(positions).T, X.shape)
                
                # Plot contour
                contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
                ax.set_xlabel(param2)
                ax.set_ylabel(param1)
                
                # Add colorbar to the last plot
                if i == n_params-1 and j == n_params-2:
                    plt.colorbar(contour, ax=ax, label='Density')
            
            else:  # Upper triangle: clear
                ax.axis('off')
    
    plt.suptitle('Posterior Parameter Pair Densities', y=1.02)
    plt.tight_layout()
    return fig

def find_density_peaks(Z, X, Y, threshold_ratio=0.1, neighborhood_size=5):
    """
    Find local maxima in density plot and their closest data points.
    
    Args:
        Z: 2D array of density values
        X, Y: 2D arrays of coordinates
        pca_result: PCA transformed data points
        data: Original data points
        threshold_ratio: Ratio of maximum density to use as threshold (default: 0.1)
        neighborhood_size: Size of neighborhood for local maximum detection (default: 5)
    
    Returns:
        peak_positions: Array of peak coordinates
        closest_indices: Indices of closest data points to each peak
        closest_points: Original data points closest to each peak
    """
    # Find local maxima in the 2D density
    neighborhood = np.ones((neighborhood_size, neighborhood_size))
    local_maxima = maximum_filter(Z, footprint=neighborhood) == Z
    threshold = threshold_ratio * np.max(Z)
    peaks_mask = local_maxima & (Z > threshold)
    
    # Get peak coordinates
    peak_coords = np.where(peaks_mask)
    peak_positions = np.column_stack((X[peak_coords], Y[peak_coords]))
    
    return peak_positions

def perform_pca_and_find_peaks(data, n_components=2, threshold_ratio=0.1, neighborhood_size=5):
    """
    Perform PCA and find density peaks in the reduced space.
    
    Args:
        data: DataFrame or array containing the data
        n_components: Number of PCA components to use (default: 2)
        threshold_ratio: Ratio of maximum density to use as threshold (default: 0.1)
        neighborhood_size: Size of neighborhood for local maximum detection (default: 5)
    
    Returns:
        tuple: (pca_result, peak_positions, point_colors, pca)
            - pca_result: PCA transformed data
            - peak_positions: Array of peak coordinates
            - point_colors: List of colors for each point based on closest peak
            - pca: Fitted PCA object
    """
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)
    
    # Find density peaks in PCA space
    x = pca_result[:, 0]
    y = pca_result[:, 1]
    
    # Create meshgrid for contour plot
    x_grid = np.linspace(x.min(), x.max(), 100)
    y_grid = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # Calculate KDE
    kernel = gaussian_kde(np.vstack([x, y]))
    Z = np.reshape(kernel(positions).T, X.shape)
    
    # Find peaks
    peak_positions = find_density_peaks(Z, X, Y, threshold_ratio, neighborhood_size)
    # Assign colors to each point based on closest peak
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray']
    point_colors = []
    peak_point_indices = []
    for point in pca_result[:, :2]:
        distances = np.sqrt(np.sum((peak_positions - point)**2, axis=1))
        closest_peak = np.argmin(distances)
        point_colors.append(colors[closest_peak % len(colors)])
        peak_point_indices.append(closest_peak)
    peak_point_indices = np.array(peak_point_indices)
    # Find points belong to each peak, shape (n_peaks, n_points_per_peak, n_params)
    peak_points = []
    for i, peak in enumerate(peak_positions):
        peak_points.append(data[peak_point_indices == i])


    return pca_result, peak_positions, point_colors, pca, Z, X, Y, peak_points

def plot_pca_with_peaks(posterior_data, prior_data, param_subset=None, n_components=2, fig=None, ax=None):
    """
    Create a PCA plot with density contours and peaks.
    
    Args:
        posterior_data: DataFrame containing posterior samples
        prior_data: DataFrame containing prior samples
        param_subset: List of parameters to include in PCA (default: all parameters)
        n_components: Number of PCA components to plot (default: 2)
        fig: Optional figure object
        ax: Optional axis object
    
    Returns:
        tuple: (fig, ax) - Figure and axis objects
    """
    if param_subset is None:
        param_subset = param_names
    
    # Prepare data
    data = posterior_data[param_subset].values
    
    # Perform PCA and find peaks
    pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(data, n_components)
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot scatter and contours
    ax.scatter(pca_result[:, 0], pca_result[:, 1], c=point_colors, alpha=0.1, s=10)
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, ax=ax, label='Density')

    # Plot peaks
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray']
    for i, peak in enumerate(peak_positions):
        print(f"========= Peak {i+1} coordinates: {peak} =========")
        # Find the closest point to the peak
        distances = np.sqrt(np.sum((pca_result[:, :2] - peak)**2, axis=1))
        closest_idx = np.argmin(distances)
        point = data[closest_idx]
        for param_name, param_value in zip(param_subset, point):
            print(f"{param_name}: {param_value}")
        point_in_pca = pca_result[closest_idx]
        ax.scatter(peak[0], peak[1], c=colors[i % len(colors)], s=100, 
                  edgecolor='black', linewidth=2, label=f'Peak {i+1}')
        
        # Plot x the closest point to the peak
        ax.scatter(point_in_pca[0], point_in_pca[1], color=colors[i % len(colors)], s=100, alpha=0.8,
                      marker='x', label=f'Closest to Peak {i+1}', zorder=5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title('PCA Density Plot with Local Maxima')
    ax.legend()
    
    return fig, ax

def plot_all_peak_percentage_changes(prior_data, posterior_data, param_subset=None, n_components=2):
    """
    Create percentage change bar plots for all peaks in a single figure.
    
    Args:
        posterior_data: DataFrame containing posterior samples
        prior_data: DataFrame containing prior samples
        param_subset: List of parameters to include in PCA (default: all parameters)
        n_components: Number of PCA components to plot (default: 2)
    
    Returns:
        tuple: (fig, ax) - Figure and axis objects
    """
    if param_subset is None:
        param_subset = param_names
    
    # Prepare data
    data = posterior_data[param_subset].values
    
    # Perform PCA and find peaks
    pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(data, n_components)
    # Calculate prior means for percentage change
    prior_means = prior_data[param_subset].mean().values
    
    # Create figure with subplots
    n_peaks = len(peak_positions)
    fig, axes = plt.subplots(n_peaks, 1, figsize=(15, 5 * n_peaks))
    if n_peaks == 1:
        axes = [axes]
    
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    # Plot each peak's percentage changes
    for i, ax in enumerate(axes):
        # Find closest point to peak
        distances = np.sqrt(np.sum((pca_result[:, :2] - peak_positions[i])**2, axis=1))
        closest_idx = np.argmin(distances)
        point = data[closest_idx]
        color = colors[i % len(colors)]
        
        # Calculate percentage changes
        pct_changes = ((point - prior_means) / prior_means) * 100
        
        # Create bar plot
        bars = ax.bar(range(len(param_subset)), pct_changes, 
                     alpha=0.8, color=color, edgecolor='black', linewidth=1)
        
        # Add value labels
        for j, (bar, pct_change, orig_val) in enumerate(zip(bars, pct_changes, point)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -5),
                   f'{orig_val:.1e}',
                   ha='center', va='bottom' if height >= 0 else 'top',
                   rotation=45, fontsize=9)
        
        # Customize plot
        ax.set_xticks(range(len(param_subset)))
        if i == n_peaks - 1:
            ax.set_xticklabels(param_subset, rotation=45, ha='right', fontsize=10)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel('Percentage Change (%)', fontsize=12)
        ax.set_title(f'Peak {i+1} Parameter Changes', fontsize=14)
        
        # Add reference lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        ax.axhline(y=-50, color='gray', linestyle='--', alpha=0.7)
        ax.axhline(y=100, color='gray', linestyle=':', alpha=0.7)
        ax.axhline(y=-100, color='gray', linestyle=':', alpha=0.7)
        
        # Set consistent y-axis limits
        all_pct_changes = [((point - prior_means) / prior_means) * 100 for point in data]
        y_min = min(min(pct) for pct in all_pct_changes) - 10
        y_max = max(max(pct) for pct in all_pct_changes) + 10
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    return fig

# 8. Parallel Coordinates Plot with Percentage Change
def plot_parallel_coordinates_with_change(prior_data, posterior_data, param_subset=None, alpha=0.1, n_components=2):
    """
    Create a parallel coordinates plot showing percentage change from reference values,
    with color encoding based on PCA clusters of the original data.
    
    Args:
        prior_data: DataFrame containing prior samples
        posterior_data: DataFrame containing posterior samples
        param_subset: List of parameters to plot (default: first 5 parameters)
        sample_size: Number of samples to plot (default: 1000)
        alpha: Transparency of lines (default: 0.1)
        n_components: Number of PCA components to use (default: 2)
    """
    if param_subset is None:
        param_subset = param_names[:5]
    
    # Perform PCA on original data first
    _, peak_positions, point_colors, _, _, _, _, peak_points = perform_pca_and_find_peaks(posterior_data[param_subset].values, n_components)
    # Calculate percentage change after PCA clustering
    ref_values = prior_data[param_subset].mean()
    posterior_df_copy = posterior_data[param_subset].copy()
    for param in param_subset:
        posterior_df_copy[param] = ((posterior_df_copy[param] - ref_values[param]) / ref_values[param]) * 100
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Plot parallel coordinates with color encoding
    for i, (_, row) in enumerate(posterior_df_copy.iterrows()):
        values = row[param_subset].values
        ax.plot(range(len(param_subset)), values, 
                color=point_colors[i], alpha=alpha, linewidth=0.5)
    
    # Add reference line at 0% change
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, 
               label='Reference (0% change)', linewidth=1.5)
    
    ax.set_ylabel('Percentage Change (%)', fontsize=10)
    ax.set_xticks(range(len(param_subset)))
    ax.set_xticklabels(param_subset, rotation=45, ha='right', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Create custom legend for peaks
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'gray']
    legend_elements = []
    for i, peak_points_i in enumerate(peak_points):
        if len(peak_points_i) > 0:  # Only add to legend if peak has points
            legend_elements.append(plt.Line2D([0], [0], color=colors[i % len(colors)], 
                                            label=f'Peak {i+1} ({len(peak_points_i)} points)',
                                            linewidth=2))
    
    # Add reference line to legend
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--',
                                    label='Reference (0% change)', linewidth=1.5))
    
    # Add legend
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Create all plots
print("Generating visualization plots...")


param_subset_small = ["PROLIFERATION_ENERGY_MU", "CONVERSION_FRACTION_MU", "ATP_PRODUCTION_RATE_MU", "GLUCOSE_UPTAKE_RATE_MU"]#, "CELL_VOLUME_MU"]

param_subset = prior_df.columns.tolist()
param_subset = [param for param in param_subset if not param.endswith("_SIGMA")]
#param_subset.remove("OXYGEN_CONCENTRATION")
param_subset.remove("CAPILLARY_DENSITY")
#param_subset.remove("GLUCOSE_CONCENTRATION")

if not os.path.exists(f'{base_dir}/posterior_plots'):
    os.makedirs(f'{base_dir}/posterior_plots')
"""
# 1. Corner plot
corner_fig = create_corner_plot(prior_df, posterior_df, param_subset_small)
plt.figure(corner_fig.number)
plt.savefig(f'{base_dir}/posterior_plots/corner_plot.png', dpi=300, bbox_inches='tight')

# 2. Marginal distributions
marginal_fig = plot_marginal_distributions(prior_df, posterior_df)
plt.figure(marginal_fig.number)
plt.savefig(f'{base_dir}/posterior_plots/marginal_distributions.png', dpi=300, bbox_inches='tight')

# 3. PCA analysis
pca_fig = plot_pca_analysis(prior_df, posterior_df)
plt.figure(pca_fig.number)
plt.savefig(f'{base_dir}/posterior_plots/pca_analysis.png', dpi=300, bbox_inches='tight')

# 4. Correlation matrices
corr_fig = plot_correlation_matrix(prior_df, posterior_df)
plt.figure(corr_fig.number)
plt.savefig(f'{base_dir}/posterior_plots/correlation_matrices.png', dpi=300, bbox_inches='tight')

# 5. Trace analysis
trace_fig = plot_trace_analysis(prior_df, posterior_df)
plt.figure(trace_fig.number)
plt.savefig(f'{base_dir}/posterior_plots/trace_analysis.png', dpi=300, bbox_inches='tight')
#plt.show()

# 6. Summary comparison
summary_fig = plot_summary_comparison(prior_df, posterior_df)
plt.figure(summary_fig.number)
plt.savefig(f'{base_dir}/posterior_plots/summary_comparison.png', dpi=300, bbox_inches='tight')

# 7. Posterior pair density plot
posterior_pair_fig = plot_posterior_pair_density(posterior_df, param_subset_small)
plt.figure(posterior_pair_fig.number)
plt.savefig(f'{base_dir}/posterior_plots/posterior_pair_density.png', dpi=300, bbox_inches='tight')

# 8. Parallel coordinates plot with percentage change
parallel_fig = plot_parallel_coordinates_with_change(
    prior_df, 
    posterior_df, 
    param_subset, 
    alpha=0.5          # Adjust transparency
)
plt.figure(parallel_fig.number)
plt.savefig(f'{base_dir}/posterior_plots/parallel_coordinates_change.png', dpi=300, bbox_inches='tight')
"""
# 9. PCA density plot
pca_density_fig = plot_pca_with_peaks(posterior_df, prior_df)#, param_subset)
#plt.savefig(f'{base_dir}/posterior_plots/pca_density.png', dpi=300, bbox_inches='tight')
"""
# 10. plot_peak_percentage_changes
peak_percentage_fig = plot_all_peak_percentage_changes(prior_df, posterior_df, param_subset)
plt.savefig(f'{base_dir}/posterior_plots/peak_percentage_changes.png', dpi=300, bbox_inches='tight')

# 11. Correlation matrix for each peak
peak_corr_fig = plot_correlation_matrix_for_peaks(posterior_df, param_subset)
plt.figure(peak_corr_fig.number)
plt.savefig(f'{base_dir}/posterior_plots/peak_correlation_matrices.png', dpi=300, bbox_inches='tight')
"""
"""
plt.figure(figsize=(20, 18))
sns.heatmap(posterior_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Posterior Parameters')
plt.show()
"""

print("All visualizations have been created and saved!")