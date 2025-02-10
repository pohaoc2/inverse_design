import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from typing import List
from inverse_design.abc.abc_precomputed import ABCPrecomputed
from inverse_design.conf.config import ABCConfig
from inverse_design.models.model_base import ModelRegistry
from inverse_design.common.enum import Target, Metric
from inverse_design.analyze import evaluate
from .parameter_selection_dialog import ParameterSelectionDialog

class ABCPrecomputedGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ABC Precomputed Interface")
        
        self.param_file_path = tk.StringVar()
        self.metrics_file_path = tk.StringVar()
        self.config_file_path = tk.StringVar()
        
        self.model_type = tk.StringVar(value="BDM")
        
        self.target_vars = {}
        self.target_entries = {}
        self.target_weights = {}
        
        self.status_label = ttk.Label(self.root, text="")
        
        self.setup_gui()
    
    def setup_gui(self):
        file_frame = ttk.LabelFrame(self.root, text="File Selection", padding=10)
        file_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        ttk.Label(file_frame, text="ABC Config File:").grid(row=0, column=0, sticky="w")
        ttk.Entry(file_frame, textvariable=self.config_file_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.browse_file("config")).grid(row=0, column=2)
        
        ttk.Label(file_frame, text="Parameter File:").grid(row=1, column=0, sticky="w")
        ttk.Entry(file_frame, textvariable=self.param_file_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.browse_file("param")).grid(row=1, column=2)
        
        ttk.Label(file_frame, text="Metrics File:").grid(row=2, column=0, sticky="w")
        ttk.Entry(file_frame, textvariable=self.metrics_file_path, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.browse_file("metrics")).grid(row=2, column=2)
        
        model_frame = ttk.LabelFrame(self.root, text="Model Selection", padding=10)
        model_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        ttk.Radiobutton(model_frame, text="BDM", variable=self.model_type, value="BDM", 
                       command=self.update_available_metrics).grid(row=0, column=0, padx=5)
        ttk.Radiobutton(model_frame, text="ARCADE", variable=self.model_type, value="ARCADE", 
                       command=self.update_available_metrics).grid(row=0, column=1, padx=5)
        ttk.Radiobutton(model_frame, text="Custom", variable=self.model_type, value="Custom", 
                       command=self.update_available_metrics).grid(row=0, column=2, padx=5)
        
        self.targets_frame = ttk.LabelFrame(self.root, text="Select Targets", padding=10)
        self.targets_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        
        self.setup_target_options()
        button_frame = ttk.Frame(self.root, padding=10)
        button_frame.grid(row=3, column=0, pady=5)
        
        ttk.Button(button_frame, text="Run ABC", command=self.run_abc).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Create Visualization", command=self.create_visualization).grid(row=0, column=1, padx=5)
        
        self.status_label.grid(row=4, column=0, pady=5)
    
    def browse_file(self, file_type):
        filetypes = [("All files", "*.*")]
        if file_type == "config":
            filetypes = [("YAML files", "*.yaml"), ("YAML files", "*.yml"), ("All files", "*.*")]
        elif file_type in ["param", "metrics"]:
            filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
            
        filename = filedialog.askopenfilename(
            title=f"Select {file_type} file",
            filetypes=filetypes
        )
        
        if filename:
            if file_type == "param":
                self.param_file_path.set(filename)
            elif file_type == "metrics":
                self.metrics_file_path.set(filename)
            elif file_type == "config":
                self.config_file_path.set(filename)
                self.load_config()

    def load_config(self):
        """Load and apply ABC configuration from yaml file"""
        try:
            if not self.config_file_path.get():
                return
                
            cfg = OmegaConf.load(self.config_file_path.get())
            
            if hasattr(cfg.abc, "model_type"):
                self.model_type.set(cfg.abc.model_type)
                self.update_available_metrics()
            
            if hasattr(cfg.abc, "targets"):
                self.update_targets_from_config(cfg.abc.targets)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config file: {str(e)}")

    def update_targets_from_config(self, targets_config):
        """Update target values and weights from config"""
        self.setup_target_options()
        
        for target in targets_config:
            metric = Metric(target.metric)
            if metric in self.target_vars:
                self.target_vars[metric].set(True)
                self.target_entries[metric].delete(0, tk.END)
                self.target_entries[metric].insert(0, str(target.value))
                self.target_weights[metric].delete(0, tk.END)
                self.target_weights[metric].insert(0, str(target.weight))

    def setup_target_options(self):
        for widget in self.targets_frame.winfo_children():
            widget.destroy()
        
        self.target_vars.clear()
        self.target_entries.clear()
        self.target_weights.clear()
        
        available_metrics = self.get_available_metrics()
        
        for i, metric in enumerate(available_metrics):
            self.target_vars[metric] = tk.BooleanVar()
            ttk.Checkbutton(self.targets_frame, text=metric.value, 
                          variable=self.target_vars[metric]).grid(row=i, column=0, sticky="w")
            
            ttk.Label(self.targets_frame, text="Value:").grid(row=i, column=1, padx=5)
            value_entry = ttk.Entry(self.targets_frame, width=10)
            value_entry.grid(row=i, column=2, padx=5)
            self.target_entries[metric] = value_entry
            
            ttk.Label(self.targets_frame, text="Weight:").grid(row=i, column=3, padx=5)
            weight_entry = ttk.Entry(self.targets_frame, width=10)
            weight_entry.insert(0, "1.0")
            weight_entry.grid(row=i, column=4, padx=5)
            self.target_weights[metric] = weight_entry

    def load_metrics_from_file(self):
        """Load metrics from the metrics CSV file"""
        try:
            Metric.load_from_file(self.metrics_file_path.get())
            return Metric.get_all_metrics()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load metrics file: {str(e)}")
            return []

    def get_available_metrics(self):
        if self.model_type.get() == "BDM":
            return [Metric.get("density"), Metric.get("time_to_equilibrium")]
        elif self.model_type.get() == "ARCADE":
            return [Metric.get("growth_rate"), Metric.get("symmetry"), Metric.get("activity")]
        elif self.model_type.get() == "Custom":
            return self.load_metrics_from_file()
        else:
            return []
    
    def update_available_metrics(self):
        self.setup_target_options()

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
            if self.config_file_path.get():
                self.cfg = OmegaConf.load(self.config_file_path.get())
            else:
                self.cfg = OmegaConf.create({
                    "abc": {
                        "model_type": self.model_type.get(),
                        "epsilon": 0.1,
                        "min_samples": 100
                    }
                })
            
            self.cfg.abc.model_type = self.model_type.get()
            self.model = ModelRegistry.get_model(self.cfg.abc.model_type)
            self.abc_config = ABCConfig.from_dictconfig(self.cfg.abc)
            self.config_class = self.model.get_config_class()
            self.model_config_key = self.cfg.abc.model_type.lower()
            self.model_config = self.config_class.from_dictconfig(self.cfg[self.model_config_key])
            
            abc = ABCPrecomputed(
                self.model_config,
                self.abc_config,
                targets,
                param_file=self.param_file_path.get(),
                metrics_file=self.metrics_file_path.get()
            )
            self.results = abc.run_inference()
            self.status_label.config(text="ABC inference completed successfully!")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def create_visualization(self):
        if not hasattr(self, 'results'):
            messagebox.showerror("Error", "Please run ABC inference first")
            return
            
        try:
            param_df = pd.read_csv(self.param_file_path.get())
            param_names = param_df.columns.tolist()
            param_names.remove('file_name')
            
            param_dialog = ParameterSelectionDialog(self, param_names)
            self.root.wait_window(param_dialog.dialog)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = ABCPrecomputedGUI(root)
    root.mainloop()

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    main() 