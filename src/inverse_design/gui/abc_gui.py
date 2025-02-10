import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict
import os

from inverse_design.abc.abc_precomputed import ABCPrecomputed
from inverse_design.conf.config import ABCConfig
from inverse_design.models.model_base import ModelRegistry
from inverse_design.common.enum import Target, Metric
from inverse_design.vis.vis import plot_parameter_kde
from inverse_design.analyze import evaluate

class ABCPrecomputedGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ABC Precomputed Interface")
        
        # File paths
        self.param_file_path = tk.StringVar()
        self.metrics_file_path = tk.StringVar()
        
        # Target selections
        self.target_vars = {}
        self.target_entries = {}
        self.target_weights = {}
        
        self.setup_gui()
        
    def setup_gui(self):
        # File selection frame
        file_frame = ttk.LabelFrame(self.root, text="File Selection", padding=10)
        file_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        ttk.Label(file_frame, text="Parameter File:").grid(row=0, column=0, sticky="w")
        ttk.Entry(file_frame, textvariable=self.param_file_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.browse_file("param")).grid(row=0, column=2)
        
        ttk.Label(file_frame, text="Metrics File:").grid(row=1, column=0, sticky="w")
        ttk.Entry(file_frame, textvariable=self.metrics_file_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.browse_file("metrics")).grid(row=1, column=2)
        
        # Targets frame
        targets_frame = ttk.LabelFrame(self.root, text="Select Targets", padding=10)
        targets_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Create checkboxes and entry fields for each metric
        for i, metric in enumerate(Metric):
            self.target_vars[metric] = tk.BooleanVar()
            ttk.Checkbutton(targets_frame, text=metric.value, variable=self.target_vars[metric]).grid(row=i, column=0, sticky="w")
            
            # Value entry
            ttk.Label(targets_frame, text="Value:").grid(row=i, column=1, padx=5)
            value_entry = ttk.Entry(targets_frame, width=10)
            value_entry.grid(row=i, column=2, padx=5)
            self.target_entries[metric] = value_entry
            
            # Weight entry
            ttk.Label(targets_frame, text="Weight:").grid(row=i, column=3, padx=5)
            weight_entry = ttk.Entry(targets_frame, width=10)
            weight_entry.insert(0, "1.0")  # Default weight
            weight_entry.grid(row=i, column=4, padx=5)
            self.target_weights[metric] = weight_entry
        
        # Buttons frame
        button_frame = ttk.Frame(self.root, padding=10)
        button_frame.grid(row=2, column=0, pady=5)
        
        ttk.Button(button_frame, text="Run ABC", command=self.run_abc).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Create Visualization", command=self.create_visualization).grid(row=0, column=1, padx=5)
    
    def browse_file(self, file_type):
        filename = filedialog.askopenfilename(
            title=f"Select {file_type} file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_type == "param":
            self.param_file_path.set(filename)
        else:
            self.metrics_file_path.set(filename)
    
    def get_selected_targets(self) -> List[Target]:
        targets = []
        for metric, var in self.target_vars.items():
            if var.get():
                try:
                    value = float(self.target_entries[metric].get())
                    weight = float(self.target_weights[metric].get())
                    targets.append(Target(metric=metric, value=value, weight=weight))
                except ValueError:
                    messagebox.showerror("Error", f"Invalid value or weight for {metric.value}")
                    return []
        return targets
    
    def run_abc(self):
        if not self.param_file_path.get() or not self.metrics_file_path.get():
            messagebox.showerror("Error", "Please select both parameter and metrics files")
            return
            
        targets = self.get_selected_targets()
        if not targets:
            messagebox.showerror("Error", "Please select at least one target")
            return
            
        try:
            # Initialize with default config (you might want to make this configurable)
            cfg = OmegaConf.create({
                "abc": {
                    "model_type": "YourModelType",  # Replace with actual model type
                    "epsilon": 0.1,
                    "min_samples": 100
                }
            })
            
            model = ModelRegistry.get_model(cfg.abc.model_type)
            abc_config = ABCConfig.from_dictconfig(cfg.abc)
            config_class = model.get_config_class()
            model_config = config_class.from_dictconfig({})  # Add default model config
            
            abc = ABCPrecomputed(
                model_config,
                abc_config,
                targets,
                param_file=self.param_file_path.get(),
                metrics_file=self.metrics_file_path.get()
            )
            
            self.results = abc.run_inference()
            messagebox.showinfo("Success", "ABC inference completed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def create_visualization(self):
        if not hasattr(self, 'results'):
            messagebox.showerror("Error", "Please run ABC inference first")
            return
            
        try:
            param_keys = ["accuracy", "affinity"]  # Update with your actual parameter keys
            accepted_params = [
                {key: sample[key] for key in param_keys}
                for sample in self.results
                if sample["accepted"]
            ]
            
            parameter_pdfs = evaluate.estimate_pdfs(accepted_params)
            save_path = "parameter_pdfs.png"
            plot_parameter_kde(parameter_pdfs, abc_config, save_path)
            messagebox.showinfo("Success", f"Visualization saved as {save_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = ABCPrecomputedGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 