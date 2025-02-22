import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import scipy.stats as stats


class MetricsVisualizationWindow:
    def __init__(self, parent, metrics_df, accepted_indices, metric_names, target_values):
        self.window = tk.Toplevel(parent)
        self.window.title("Metrics KDE Visualization")
        self.window.geometry("800x600")

        self.metrics_df = metrics_df
        self.accepted_indices = accepted_indices
        self.metric_names = metric_names
        self.target_values = target_values  # Store target values

        self.fig = plt.Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.update_plot()

    def update_plot(self):
        self.fig.clear()
        n_metrics = len(self.metric_names)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        axes = self.fig.subplots(n_rows, n_cols)
        if n_metrics == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, metric in enumerate(self.metric_names):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # All samples KDE
            all_values = self.metrics_df[metric].values
            kde_all = stats.gaussian_kde(all_values)

            # Accepted samples KDE
            accepted_values = self.metrics_df.iloc[self.accepted_indices][metric].values
            if len(accepted_values) > 1:
                kde_accepted = stats.gaussian_kde(accepted_values)

            # Plot histograms with common bins
            bins = np.histogram_bin_edges(all_values, bins=15)
            ax.hist(
                all_values, bins=bins, alpha=0.3, density=True, label="All Samples", color="blue"
            )
            ax.hist(
                accepted_values,
                bins=bins,
                alpha=0.3,
                density=True,
                label="Accepted Samples",
                color="orange",
            )

            # Plot KDE
            x_min = min(all_values)
            x_max = max(all_values)
            x = np.linspace(x_min, x_max, 200)

            ax.plot(x, kde_all(x), "b-", linewidth=2, label="All Samples (KDE)")
            if len(accepted_values) > 1:
                ax.plot(x, kde_accepted(x), "r-", linewidth=2, label="Accepted Samples (KDE)")

            # Draw vertical dashed line for target value
            if metric in self.target_values:
                target_value = self.target_values[metric]
                ax.axvline(
                    target_value, color="red", linestyle="--", label=f"Target: {target_value}"
                )

            ax.set_title(metric)
            ax.set_xlabel("Value")
            if col == 0:
                ax.set_ylabel("Density")
            ax.legend()

        # Hide empty subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        ax.set_xlim(10, 70)
        self.fig.tight_layout()
        self.canvas.draw()
