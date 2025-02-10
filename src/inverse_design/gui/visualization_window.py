import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class VisualizationWindow:
    def __init__(self, parent, parameter_pdfs, abc_config, param_names):
        self.window = tk.Toplevel(parent)
        self.window.title("Parameter KDE Visualization")
        self.window.geometry("800x600")
        
        self.parameter_pdfs = parameter_pdfs
        self.abc_config = abc_config
        self.param_names = param_names
        
        self.fig = plt.Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.update_plot()
        
    def update_plot(self):
        self.fig.clear()
        
        n_params = len(self.param_names)
        axes = self.fig.subplots(2, n_params, height_ratios=[1, 2])
                
        for i, param_name in enumerate(self.param_names):
            kde = self.parameter_pdfs["independent"][param_name]
            samples = kde.resample(1000)
            mean = float(samples.mean())
            std = float(samples.std())
            
            prior_min = self.abc_config.parameter_ranges[param_name].min
            prior_max = self.abc_config.parameter_ranges[param_name].max
            
            prior_height = 1.0 / (prior_max - prior_min)
            axes[0, i].fill_between(
                [prior_min, prior_max], [prior_height, prior_height], alpha=0.5, color="gray"
            )
            axes[0, i].axhline(y=prior_height, color="gray", linestyle=":", alpha=0.5)
            axes[0, i].set_title(f"{param_name} Prior")
            axes[0, i].set_xlabel("Value")
            if i == 0:
                axes[0, i].set_ylabel("Prior Density")
            
            x = np.linspace(prior_min, max(prior_max, mean + 4 * std), 200)
            y = kde(x)
            
            axes[1, i].plot(x, y, "b-", label="Posterior")
            
            axes[1, i].axvline(mean, color="r", linestyle="--", label="Mean")
            axes[1, i].axvline(mean - std, color="g", linestyle=":", alpha=0.7, label="±1σ")
            axes[1, i].axvline(mean + std, color="g", linestyle=":", alpha=0.7)
            
            axes[1, i].set_title(f"μ={mean:.3f}, σ={std:.3f}")
            axes[1, i].set_xlabel("Value")
            if i == 0:
                axes[1, i].set_ylabel("Posterior Density")
                axes[1, i].legend(loc="best")
            
            xlim = (prior_min - 0.1 * (prior_max - prior_min),
                   prior_max + 0.1 * (prior_max - prior_min))
            axes[0, i].set_xlim(xlim)
            axes[1, i].set_xlim(xlim)
        
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0.1)
        self.canvas.draw()
    
    def save_plot(self, filename):
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
