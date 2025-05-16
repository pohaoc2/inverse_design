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
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)
    
    # Plot 1: PCA scatter with density contours
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

    # Get the peak positions and closest points
    peak_positions = find_density_peaks(Z, X, Y)
    # Find points within 10% of the peak
    peak_points = []
    for peak in peak_positions:
        distances = np.sqrt(np.sum((pca_result[:, :2] - peak)**2, axis=1))
        peak_points.append(data[distances < 0.1])
    
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

# 7b. PCA Density Plot
def plot_pca_density(posterior_data, prior_data, param_subset=None, n_components=2):
    """
    Create a density plot in PCA space showing the main directions of variation.
    
    Args:
        posterior_data: DataFrame containing posterior samples
        prior_data: DataFrame containing prior samples
        param_subset: List of parameters to include in PCA (default: all parameters)
        n_components: Number of PCA components to plot (default: 2)
    """
    if param_subset is None:
        param_subset = param_names
    
    # Prepare data
    data = posterior_data[param_subset].values
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_scaled)
    
    # Plot 1: PCA scatter with density contours
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
    
    # Find peaks and closest points
    peak_positions = find_density_peaks(
        Z, X, Y
    )
    # Find closest points to each peak
    closest_indices = []
    closest_points = []
    for peak in peak_positions:
        distances = np.sqrt(np.sum((pca_result[:, :2] - peak)**2, axis=1))
        closest_idx = np.argmin(distances)
        closest_indices.append(closest_idx)
        closest_points.append(data[closest_idx])

    # Create figure with subplots
    n_peaks = len(peak_positions)
    fig = plt.figure(figsize=(20, 4 * n_peaks))
    
    # Create a more flexible gridspec
    gs = fig.add_gridspec(n_peaks, 2, width_ratios=[2.5, 1])
    
    # PCA plot (spans left column for all rows)
    ax_pca = fig.add_subplot(gs[:, 0])
    
    # Plot scatter and contours
    ax_pca.scatter(x, y, alpha=0.1, s=10, color='blue')
    contour = ax_pca.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    
    # Plot peaks and their closest points
    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
    for i, (peak, closest_idx) in enumerate(zip(peak_positions, closest_indices)):
        color = colors[i % len(colors)]
        ax_pca.scatter(peak[0], peak[1], color=color, s=100, alpha=0.8, 
                      label=f'Peak {i+1}', zorder=5, edgecolors='black', linewidth=2)
        ax_pca.scatter(x[closest_idx], y[closest_idx], color=color, s=100, alpha=0.8,
                      marker='x', label=f'Closest to Peak {i+1}', zorder=5)
    
    ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
    ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
    ax_pca.set_title('PCA Density Plot with Local Maxima', fontsize=16)
    ax_pca.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.colorbar(contour, ax=ax_pca, label='Density', shrink=0.7)
    
    # Calculate prior means for percentage change
    prior_means = prior_data[param_subset].mean().values
    
    # Create separate bar plots for each peak on the right side
    for i, (peak, point) in enumerate(zip(peak_positions, closest_points)):
        # Each peak gets a subplot in the right columns
        ax = fig.add_subplot(gs[i, 1:])  # Spans columns 1 and 2
        
        # Calculate percentage changes
        pct_changes = ((point - prior_means) / prior_means) * 100
        
        # Create bar plot
        bars = ax.bar(range(len(param_subset)), pct_changes, 
                     alpha=0.8, color=colors[i % len(colors)], edgecolor='black', linewidth=1)
        
        # Add value labels
        for j, (bar, pct_change, orig_val) in enumerate(zip(bars, pct_changes, point)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -5),
                   f'{orig_val:.1e}',
                   ha='center', va='bottom' if height >= 0 else 'top',
                   rotation=45, fontsize=9)
        
        # Customize plot
        ax.set_xticks(range(len(param_subset)))
        if i == len(peak_positions) - 1:
            ax.set_xticklabels(param_subset, rotation=45, ha='right', fontsize=10)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel('Percentage Change (%)', fontsize=12)
        
        # Add reference lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        ax.axhline(y=-50, color='gray', linestyle='--', alpha=0.7)
        ax.axhline(y=100, color='gray', linestyle=':', alpha=0.7)
        ax.axhline(y=-100, color='gray', linestyle=':', alpha=0.7)
        
        # Set consistent y-axis limits across all subplots
        all_pct_changes = [((point - prior_means) / prior_means) * 100 for point in closest_points]
        y_min = min(min(pct) for pct in all_pct_changes) - 10
        y_max = max(max(pct) for pct in all_pct_changes) + 10
        ax.set_ylim(y_min, y_max)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.01)
    
    # Print the values for each peak
    print("\nParameter values for each local maximum:")
    for i, (peak, point) in enumerate(zip(peak_positions, closest_points)):
        print(f"\nPeak {i+1} (PCA coordinates: [{peak[0]:.2f}, {peak[1]:.2f}]):")
        for param, value in zip(param_subset, point):
            pct_change = ((value - prior_means[param_subset.index(param)]) / prior_means[param_subset.index(param)]) * 100
            print(f"{param}: {value:.4e} ({pct_change:+.1f}% from prior mean)")
    
    return fig

# 8. Parallel Coordinates Plot with Percentage Change
def plot_parallel_coordinates_with_change(prior_data, posterior_data, param_subset=None, 
                                        sample_size=1000, alpha=0.1):
    """
    Create a parallel coordinates plot showing percentage change from reference values.
    
    Args:
        prior_data: DataFrame containing prior samples
        posterior_data: DataFrame containing posterior samples
        param_subset: List of parameters to plot (default: first 5 parameters)
        sample_size: Number of samples to plot (default: 1000)
        alpha: Transparency of lines (default: 0.1)
    """
    if param_subset is None:
        param_subset = param_names[:5]
    
    ref_values = prior_data[param_subset].mean()
    
    # Calculate percentage change
    posterior_df = posterior_data[param_subset].copy()
    for param in param_subset:
        posterior_df[param] = ((posterior_df[param] - ref_values[param]) / ref_values[param]) * 100
    
    # Sample the data if it's too large
    if len(posterior_df) > sample_size:
        posterior_df = posterior_df.sample(n=sample_size, random_state=42)
    
    # Create the plot with a larger figure size
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot each line with a single color
    for i, (_, row) in enumerate(posterior_df.iterrows()):
        values = row[param_subset].values
        ax.plot(range(len(param_subset)), values, 
                color='blue', alpha=alpha, linewidth=0.5)
    
    # Add reference line at 0% change
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, 
               label='Reference (0% change)', linewidth=1.5)
    
    # Customize the plot
    ax.set_title(f'Parallel Coordinate Plot: Percentage Change from Prior Mean', 
                 pad=20, fontsize=12)
    ax.set_ylabel('Percentage Change (%)', fontsize=10)
    ax.set_xticks(range(len(param_subset)))
    ax.set_xticklabels(param_subset, rotation=45, ha='right', fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend with only selected entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Create all plots
print("Generating visualization plots...")


param_subset = ["PROLIFERATION_ENERGY_MU", "CONVERSION_FRACTION_MU", "ATP_PRODUCTION_RATE_MU", "GLUCOSE_UPTAKE_RATE_MU"]#, "CELL_VOLUME_MU"]
param_subset = prior_df.columns.tolist()
param_subset = [param for param in param_subset if not param.endswith("_SIGMA")]
param_subset.remove("OXYGEN_CONCENTRATION")
param_subset.remove("CAPILLARY_DENSITY")
param_subset.remove("GLUCOSE_CONCENTRATION")
# 1. Corner plot
"""
corner_fig = create_corner_plot(prior_df, posterior_df, param_subset)
plt.figure(corner_fig.number)
plt.savefig('corner_plot.png', dpi=300, bbox_inches='tight')

# 2. Marginal distributions
marginal_fig = plot_marginal_distributions(prior_df, posterior_df)
plt.figure(marginal_fig.number)
plt.savefig('marginal_distributions.png', dpi=300, bbox_inches='tight')

# 3. PCA analysis
pca_fig = plot_pca_analysis(prior_df, posterior_df)
plt.figure(pca_fig.number)
plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')

# 4. Correlation matrices
corr_fig = plot_correlation_matrix(prior_df, posterior_df)
plt.figure(corr_fig.number)
plt.savefig('correlation_matrices.png', dpi=300, bbox_inches='tight')

# 5. Trace analysis
trace_fig = plot_trace_analysis(prior_df, posterior_df)
plt.figure(trace_fig.number)
plt.savefig('trace_analysis.png', dpi=300, bbox_inches='tight')
#plt.show()

# 6. Summary comparison
summary_fig = plot_summary_comparison(prior_df, posterior_df)
plt.figure(summary_fig.number)
plt.savefig('summary_comparison.png', dpi=300, bbox_inches='tight')

# 7. Posterior pair density plot
posterior_pair_fig = plot_posterior_pair_density(posterior_df, param_subset)
plt.figure(posterior_pair_fig.number)
plt.savefig('posterior_pair_density.png', dpi=300, bbox_inches='tight')

# 8. Parallel coordinates plot with percentage change
parallel_fig = plot_parallel_coordinates_with_change(
    prior_df, 
    posterior_df, 
    param_subset, 
    sample_size=1000,  # Adjust this number based on your data size
    alpha=0.5          # Adjust transparency
)
plt.figure(parallel_fig.number)
plt.savefig('parallel_coordinates_change.png', dpi=300, bbox_inches='tight')

# 9. PCA density plot
pca_density_fig = plot_pca_density(posterior_df, prior_df, param_subset)
plt.figure(pca_density_fig.number)
plt.savefig('pca_density.png', dpi=300, bbox_inches='tight')
"""
# 10. Correlation matrix for each peak
peak_corr_fig = plot_correlation_matrix_for_peaks(posterior_df, param_subset)
plt.figure(peak_corr_fig.number)
plt.savefig('peak_correlation_matrices.png', dpi=300, bbox_inches='tight')

"""
plt.figure(figsize=(20, 18))
sns.heatmap(posterior_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Posterior Parameters')
plt.show()
"""
print("All visualizations have been created and saved!")