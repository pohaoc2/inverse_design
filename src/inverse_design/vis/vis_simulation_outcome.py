import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_and_prepare_data(param_file, metrics_file):
    # Load parameter and metrics data
    params_df = pd.read_csv(param_file)
    metrics_df = pd.read_csv(metrics_file)
    
    # Clean up parameter data
    params_df = params_df.drop('file_name', axis=1)
    metrics_list = ["doub_time", "act_t2", "colony_g_rate"]
    # Select std columns from metrics
    std_cols = [col for col in metrics_df.columns if 'std' in col]
    std_cols = [col for col in metrics_df.columns if col in metrics_list]
    metrics_std_df = metrics_df[std_cols]
    
    # Normalize both dataframes to [0,1] range
    scaler = MinMaxScaler()
    params_normalized = pd.DataFrame(
        scaler.fit_transform(params_df),
        columns=params_df.columns
    )
    metrics_normalized = pd.DataFrame(
        scaler.fit_transform(metrics_std_df),
        columns=metrics_std_df.columns
    )
    
    return params_normalized, metrics_normalized

def plot_pairwise_relationships(metrics_df):
    # Calculate number of metrics
    n_metrics = len(metrics_df.columns)
    
    # Create a grid of subplots
    fig, axes = plt.subplots(n_metrics, n_metrics, figsize=(2*n_metrics, 2*n_metrics))
    
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
                ax.plot(x_range, slope * x_range + intercept, 'r--', 
                       label=f'slope={slope:.2f}\nr={correlation:.2f}')
                
                ax.legend(fontsize='small')
            else:
                # On diagonal, show density plot
                sns.kdeplot(data=metrics_df[metric1], ax=ax)
                ax.set_xlabel("")
                ax.set_ylabel("")
            
            # Only show labels on edge plots
            if i == n_metrics-1:
                ax.set_xlabel(metric2)
            if j == 0:
                ax.set_ylabel(metric1)  # Changed from metric2 to metric1 for correct y-axis labeling
            
            # Remove ticks for cleaner look
            ax.tick_params(labelsize='small')
    
    plt.tight_layout()
    plt.savefig('metric_pairwise_relationships.png')
    #plt.show()

def calculate_chaos_metric(metrics_df):
    # First normalize all std values globally to [0,1] range
    scaler = MinMaxScaler()
    normalized_stds = pd.DataFrame(
        scaler.fit_transform(metrics_df),
        columns=metrics_df.columns,
        index=metrics_df.index
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
    chaos_df = pd.DataFrame({
        'chaos_metric': shannon_entropies,
    }, index=metrics_df.index)
    
    # Plot distribution of chaos metric
    plt.figure(figsize=(10, 6))
    sns.histplot(data=chaos_df, x='chaos_metric', kde=True)
    plt.title('Distribution of Chaos Metric\n(Higher values indicate more chaotic behavior)')
    plt.xlabel('Chaos Metric (Weighted Shannon Entropy)')
    plt.ylabel('Count')
    #plt.show()
    
    return chaos_df

def plot_metric_distributions(posterior_metrics_file, prior_metrics_file, target_metrics, save_file=None):
    """
    Plot density distributions of metrics from posterior and prior, with target values.
    
    Args:
        posterior_metrics_file: Path to simulation_metrics.csv (posterior)
        prior_metrics_file: Path to completed_doubling.csv (prior)
        target_metrics: Dictionary of target values for each metric
    """
    # Load data
    posterior_df = pd.read_csv(posterior_metrics_file)
    prior_df = pd.read_csv(prior_metrics_file)
    
    metrics_list = list(target_metrics.keys())
    
    # Create subplot for each metric
    fig, axes = plt.subplots(1, len(metrics_list), figsize=(15, 5))
    
    for idx, metric in enumerate(metrics_list):
        # Plot posterior distribution
        sns.kdeplot(data=posterior_df[metric], ax=axes[idx], label='Posterior', color='blue')
        
        # Plot prior distribution
        sns.kdeplot(data=prior_df[metric], ax=axes[idx], label='Prior', color='gray')
        
        # Add vertical line for target value
        axes[idx].axvline(x=target_metrics[metric], color='red', linestyle='--', 
                         label=f'Target ({target_metrics[metric]})')
        
        axes[idx].set_title(f'{metric} Distribution')
        axes[idx].set_xlabel(metric)
        axes[idx].set_ylabel('Density')
        axes[idx].legend()
    
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()

def main():

    param_file = 'inputs/kde_sampled_inputs_small_std/kde_sampled_parameters_log.csv'
    param_file = 'inputs/manual_volume_apoptosis/kde_sampled_parameters_log.csv'
    posterior_metrics_file = 'ARCADE_OUTPUT/SMALL_STD_ONLY_VOLUME/simulation_metrics.csv'
    posterior_metrics_file = 'ARCADE_OUTPUT/MANUAL_VOLUME_APOTOSIS/simulation_metrics.csv'
    prior_metrics_file = 'completed_doubling_formatted.csv'
    
    # Define target metrics
    target_metrics = {
        "doub_time": 50,
        "doub_time_std": 0.0,
        "act_t2": 0.5,
        "colony_g_rate": 0.8
    }
    
    # Plot metric distributions
    save_file = 'manual_volume_apoptosis_metric_distributions.png'
    plot_metric_distributions(posterior_metrics_file,
    prior_metrics_file,
    target_metrics,
    save_file=save_file)
    
    # Load and prepare data for other visualizations
    # params_df, metrics_df = load_and_prepare_data(param_file, posterior_metrics_file)
    #plot_pairwise_relationships(metrics_df)

if __name__ == "__main__":
    main()