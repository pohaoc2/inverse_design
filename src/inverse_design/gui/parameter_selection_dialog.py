import tkinter as tk
from tkinter import ttk, messagebox
from .param_vis_window import VisualizationWindow
from inverse_design.analyze import evaluate

class ParameterSelectionDialog:
    def __init__(self, parent, param_names):
        self.dialog = tk.Toplevel(parent.root)
        self.dialog.title("Select Parameters to Plot")
        self.dialog.geometry("500x400")
        
        self.dialog.transient(parent.root)
        self.dialog.grab_set()
        

        self.parent = parent.root
        self.abc_gui = parent
        
        self.setup_gui(param_names)
    
    def setup_gui(self, param_names):
        select_frame = ttk.LabelFrame(self.dialog, text="Available Parameters", padding=10)
        select_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(select_frame)
        scrollbar = ttk.Scrollbar(select_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.create_parameter_checkboxes(param_names)
        self.create_buttons()
        self.setup_scrolling(canvas, scrollbar)
        
        self.center_on_parent()
    
    def create_parameter_checkboxes(self, param_names):
        self.param_vars = {}
        for param in param_names:
            self.param_vars[param] = tk.BooleanVar(value=True)
            ttk.Checkbutton(self.scrollable_frame, text=param, 
                          variable=self.param_vars[param]).pack(anchor="w", padx=5, pady=2)
    
    def create_buttons(self):
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill="x", padx=15, pady=10)
        
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side="left")
        
        ttk.Button(left_buttons, text="Select All", 
                  command=self.select_all).pack(side="left", padx=5)
        ttk.Button(left_buttons, text="Deselect All", 
                  command=self.deselect_all).pack(side="left", padx=5)
        
        ttk.Button(button_frame, text="Create Visualization", 
                  command=self.create_vis).pack(side="right", padx=5)
    
    def setup_scrolling(self, canvas, scrollbar):
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", 
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")
        
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
    def center_on_parent(self):
        self.dialog.update_idletasks()
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"+{x}+{y}")
    
    def select_all(self):
        for var in self.param_vars.values():
            var.set(True)
    
    def deselect_all(self):
        for var in self.param_vars.values():
            var.set(False)
    
    def get_selected_params(self):
        return [param for param, var in self.param_vars.items() if var.get()]

    def create_vis(self):
        selected_params = self.get_selected_params()
        if not selected_params:
            messagebox.showwarning("Warning", "No parameters selected for visualization")
            return
            
        try:
            if not hasattr(self.abc_gui, 'results'):
                messagebox.showerror("Error", "Please run ABC inference first")
                return
                
            accepted_params = [
                {key: sample[key] for key in selected_params}
                for sample in self.abc_gui.results
                if sample["accepted"]
            ]
            
            parameter_pdfs = evaluate.estimate_pdfs(accepted_params)
            
            VisualizationWindow(self.dialog, parameter_pdfs, self.abc_gui.abc_config, selected_params)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
