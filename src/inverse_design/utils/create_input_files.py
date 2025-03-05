import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import scipy.stats.qmc
from typing import Dict
import os
import json

def _format_param_value(value: float, precision: int, is_bounded_by_one: bool) -> str:
    """Format parameter value with bounds checking and precision."""
    if is_bounded_by_one:
        value = min(1, max(0, value))
    else:
        value = max(0, value)
    return f"{value:.{precision}f}"

def _get_param_path(base_name: str, subfolder: str | None) -> str:
    """Get full parameter path including subfolder if any."""
    return f"{subfolder}/{base_name}" if subfolder else base_name

# Define parameter formatting configurations
PARAM_CONFIGS = {
    # Format: (precision, is_bounded_by_one, subfolder)
    "CELL_VOLUME": (1, False, None),
    "APOPTOSIS_AGE": (1, False, None),
    "NECROTIC_FRACTION": (3, True, None),
    "ACCURACY": (3, True, None),
    "AFFINITY": (3, True, None),
    "COMPRESSION_TOLERANCE": (3, False, None),
    "SYNTHESIS_DURATION": (1, False, "proliferation"),
    "BASAL_ENERGY": (6, False, "metabolism"),
    "PROLIFERATION_ENERGY": (6, False, "metabolism"),
    "MIGRATION_ENERGY": (6, False, "metabolism"),
    "METABOLIC_PREFERENCE": (3, False, "metabolism"),
    "CONVERSION_FRACTION": (3, False, "metabolism"),
    "RATIO_GLUCOSE_PYRUVATE": (3, False, "metabolism"),
    "LACTATE_RATE": (3, False, "metabolism"),
    "AUTOPHAGY_RATE": (6, False, "metabolism"),
    "GLUCOSE_UPTAKE_RATE": (3, False, "metabolism"),
    "ATP_PRODUCTION_RATE": (3, False, "metabolism"),
    "MIGRATORY_THRESHOLD": (3, False, "signaling"),
}

def save_parameter_ranges(param_ranges: dict, output_dir: str):
    """Save the parameter ranges used for generating inputs to a CSV file."""
    # Create DataFrame with min and max values
    ranges_df = pd.DataFrame(
        {
            "parameter": param_ranges.keys(),
            "min_value": [r[0] for r in param_ranges.values()],
            "max_value": [r[1] for r in param_ranges.values()],
        }
    )

    # Save to CSV
    ranges_df.to_csv(f"{output_dir}/parameter_ranges.csv", index=False)


def generate_perturbed_parameters(
    sobol_power: int,
    param_ranges: dict,
    template_path: str = "sample_input_v3.xml",
    output_dir: str = "perturbed_inputs",
    seed: int = 42,
):
    """Generate perturbed parameter sets using Sobol sampling

    Args:
        sobol_power: Power of 2 for number of samples (n_samples = 2^sobol_power)
        param_ranges: Dictionary of parameter ranges {param_name: (min, max)}
        template_path: Path to template XML file
        output_dir: Directory to save generated XML files
        seed: Random seed for reproducibility
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save parameter ranges to CSV
    save_parameter_ranges(param_ranges, output_dir)

    # Define parameter ranges
    param_ranges_list = list(param_ranges.values())

    # Initialize Sobol sequence generator with random seed
    sobol_engine = scipy.stats.qmc.Sobol(d=len(param_ranges_list), scramble=True, seed=seed)

    # Generate samples in [0, 1] space
    samples = sobol_engine.random_base2(m=sobol_power)
    n_samples = len(samples)

    print(f"Generating {n_samples} samples...")

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Read template XML
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Prepare DataFrame for parameter logging
    param_log = []

    # Generate XML files for each parameter set
    for i, sample in enumerate(samples):
        # Scale sample to parameter ranges
        params = []
        for value, (min_val, max_val) in zip(sample, param_ranges_list):
            scaled_value = min_val + (max_val - min_val) * value
            params.append(scaled_value)

        (
            CELL_VOLUME_MU,
            CELL_VOLUME_SIGMA,
            APOPTOSIS_AGE_MU,
            APOPTOSIS_AGE_SIGMA,
            NECROTIC_FRACTION,
            ACCURACY,
            AFFINITY,
            COMPRESSION_TOLERANCE,
            SYNTHESIS_DURATION_MU,
            SYNTHESIS_DURATION_SIGMA,
            BASAL_ENERGY_MU,
            BASAL_ENERGY_SIGMA,
            PROLIFERATION_ENERGY_MU,
            PROLIFERATION_ENERGY_SIGMA,
            MIGRATION_ENERGY_MU,
            MIGRATION_ENERGY_SIGMA,
            METABOLIC_PREFERENCE_MU,
            METABOLIC_PREFERENCE_SIGMA,
            CONVERSION_FRACTION_MU,
            CONVERSION_FRACTION_SIGMA,
            RATIO_GLUCOSE_PYRUVATE_MU,
            RATIO_GLUCOSE_PYRUVATE_SIGMA,
            LACTATE_RATE_MU,
            LACTATE_RATE_SIGMA,
            AUTOPHAGY_RATE_MU,
            AUTOPHAGY_RATE_SIGMA,
            GLUCOSE_UPTAKE_RATE_MU,
            GLUCOSE_UPTAKE_RATE_SIGMA,
            ATP_PRODUCTION_RATE_MU,
            ATP_PRODUCTION_RATE_SIGMA,
            MIGRATORY_THRESHOLD_MU,
            MIGRATORY_THRESHOLD_SIGMA,
        ) = params

        # Create a dictionary for the log entry
        log_entry = {"file_name": f"input_{i+1}.xml"}
        
        # Add parameters with bounds checking using PARAM_CONFIGS
        for base_param, (precision, is_bounded, _) in PARAM_CONFIGS.items():
            # Handle direct parameters (those without MU/SIGMA)
            if base_param in locals():
                value = _format_param_value(locals()[base_param], precision, is_bounded)
                log_entry[base_param] = float(value)
            
            # Handle MU/SIGMA parameters
            mu_key = f"{base_param}_MU"
            sigma_key = f"{base_param}_SIGMA"
            
            if mu_key in locals():
                mu_value = _format_param_value(locals()[mu_key], precision, is_bounded)
                log_entry[mu_key] = float(mu_value)
            
            if sigma_key in locals():
                sigma_value = _format_param_value(locals()[sigma_key], precision, False)  # sigma is never bounded by 1
                log_entry[sigma_key] = float(sigma_value)
        
        param_log.append(log_entry)

        # Create a dictionary for easier parameter access
        params = {
            'CELL_VOLUME_MU': CELL_VOLUME_MU,
            'CELL_VOLUME_SIGMA': CELL_VOLUME_SIGMA,
            'APOPTOSIS_AGE_MU': APOPTOSIS_AGE_MU,
            'APOPTOSIS_AGE_SIGMA': APOPTOSIS_AGE_SIGMA,
            'NECROTIC_FRACTION': NECROTIC_FRACTION,
            'ACCURACY': ACCURACY,
            'AFFINITY': AFFINITY,
            'COMPRESSION_TOLERANCE': COMPRESSION_TOLERANCE,
            'SYNTHESIS_DURATION_MU': SYNTHESIS_DURATION_MU,
            'SYNTHESIS_DURATION_SIGMA': SYNTHESIS_DURATION_SIGMA,
            'BASAL_ENERGY_MU': BASAL_ENERGY_MU,
            'BASAL_ENERGY_SIGMA': BASAL_ENERGY_SIGMA,
            'PROLIFERATION_ENERGY_MU': PROLIFERATION_ENERGY_MU,
            'PROLIFERATION_ENERGY_SIGMA': PROLIFERATION_ENERGY_SIGMA,
            'MIGRATION_ENERGY_MU': MIGRATION_ENERGY_MU,
            'MIGRATION_ENERGY_SIGMA': MIGRATION_ENERGY_SIGMA,
            'METABOLIC_PREFERENCE_MU': METABOLIC_PREFERENCE_MU,
            'METABOLIC_PREFERENCE_SIGMA': METABOLIC_PREFERENCE_SIGMA,
            'CONVERSION_FRACTION_MU': CONVERSION_FRACTION_MU,
            'CONVERSION_FRACTION_SIGMA': CONVERSION_FRACTION_SIGMA,
            'RATIO_GLUCOSE_PYRUVATE_MU': RATIO_GLUCOSE_PYRUVATE_MU,
            'RATIO_GLUCOSE_PYRUVATE_SIGMA': RATIO_GLUCOSE_PYRUVATE_SIGMA,
            'LACTATE_RATE_MU': LACTATE_RATE_MU,
            'LACTATE_RATE_SIGMA': LACTATE_RATE_SIGMA,
            'AUTOPHAGY_RATE_MU': AUTOPHAGY_RATE_MU,
            'AUTOPHAGY_RATE_SIGMA': AUTOPHAGY_RATE_SIGMA,
            'GLUCOSE_UPTAKE_RATE_MU': GLUCOSE_UPTAKE_RATE_MU,
            'GLUCOSE_UPTAKE_RATE_SIGMA': GLUCOSE_UPTAKE_RATE_SIGMA,
            'ATP_PRODUCTION_RATE_MU': ATP_PRODUCTION_RATE_MU,
            'ATP_PRODUCTION_RATE_SIGMA': ATP_PRODUCTION_RATE_SIGMA,
            'MIGRATORY_THRESHOLD_MU': MIGRATORY_THRESHOLD_MU,
            'MIGRATORY_THRESHOLD_SIGMA': MIGRATORY_THRESHOLD_SIGMA,
        }

        # Find cancerous population element
        cancerous_pop = root.find(".//population[@id='cancerous']")

        # Update parameters
        for param in cancerous_pop.findall("population.parameter"):
            param_id = param.get("id")
            
            # Handle all parameters
            for base_param, (precision, is_bounded, subfolder) in PARAM_CONFIGS.items():
                param_path = _get_param_path(base_param, subfolder)
                
                if param_id == param_path:
                    mu_key = f"{base_param}_MU"
                    sigma_key = f"{base_param}_SIGMA"
                    
                    # Handle direct parameters (those without MU/SIGMA)
                    if base_param in params:
                        value = _format_param_value(params[base_param], precision, is_bounded)
                        param.set("value", value)
                    
                    # Handle normal distribution parameters
                    elif mu_key in params and sigma_key in params:
                        mu = _format_param_value(params[mu_key], precision, is_bounded)
                        sigma = _format_param_value(params[sigma_key], precision, False)  # sigma is never bounded by 1
                        param.set("value", f"NORMAL(MU={mu},SIGMA={sigma})")

        # Save modified XML
        if not os.path.exists(f"{output_dir}/inputs"):
            os.makedirs(f"{output_dir}/inputs")
        output_file = f"{output_dir}/inputs/input_{i+1}.xml"
        tree.write(output_file, encoding="utf-8", xml_declaration=True)

    # Save parameters to Excel
    df = pd.DataFrame(param_log)
    df.to_csv(f"{output_dir}/parameter_log.csv", index=False)

    print(f"Generated {n_samples} XML files and parameter log in {output_dir}/")


def generate_parameters_from_kde(
    parameter_pdfs: Dict,
    n_samples: int,
    output_dir: str = "kde_sampled_inputs",
    template_path: str = "sample_input_v3.xml",
):
    """Generate parameter sets by sampling from kernel density estimates (KDE)

    Args:
        parameter_pdfs: Dictionary containing KDE objects for each parameter
        n_samples: Number of parameter sets to generate
        output_dir: Directory to save generated XML files
        template_path: Path to template XML file
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read template XML
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Prepare DataFrame for parameter logging
    param_log = []

    # Sample from each parameter's KDE
    kde_dict = parameter_pdfs["independent"]
    param_names = parameter_pdfs["param_names"]

    # Generate samples for each parameter
    sampled_params = {param: kde.resample(n_samples, seed=42)[0] for param, kde in kde_dict.items()}

    # Generate XML files for each parameter set
    for i in range(n_samples):
        # Get raw parameters
        raw_params = {param: sampled_params[param][i] for param in param_names}
        
        # Create bounded parameters dictionary
        bounded_params = {"file_name": f"input_{i+1}.xml"}
        
        # Apply bounds to all parameters
        for base_param, (precision, is_bounded, _) in PARAM_CONFIGS.items():
            # Handle direct parameters (those without MU/SIGMA)
            if base_param in raw_params:
                value = float(_format_param_value(raw_params[base_param], precision, is_bounded))
                bounded_params[base_param] = value
            
            # Handle MU/SIGMA parameters
            mu_key = f"{base_param}_MU"
            sigma_key = f"{base_param}_SIGMA"
            
            if mu_key in raw_params:
                mu_value = float(_format_param_value(raw_params[mu_key], precision, is_bounded))
                bounded_params[mu_key] = mu_value
            
            if sigma_key in raw_params:
                sigma_value = float(_format_param_value(raw_params[sigma_key], precision, False))  # sigma is never bounded by 1
                bounded_params[sigma_key] = sigma_value

        # Find cancerous population element
        cancerous_pop = root.find(".//population[@id='cancerous']")

        # Update parameters in XML
        for param in cancerous_pop.findall("population.parameter"):
            param_id = param.get("id")
            
            # Handle all parameters
            for base_param, (precision, is_bounded, subfolder) in PARAM_CONFIGS.items():
                param_path = _get_param_path(base_param, subfolder)
                
                if param_id == param_path:
                    mu_key = f"{base_param}_MU"
                    sigma_key = f"{base_param}_SIGMA"
                    
                    # Handle direct parameters (those without MU/SIGMA)
                    if base_param in bounded_params:
                        value = f"{bounded_params[base_param]:.{precision}f}"
                        param.set("value", value)
                    
                    # Handle normal distribution parameters
                    elif mu_key in bounded_params and sigma_key in bounded_params:
                        mu = f"{bounded_params[mu_key]:.{precision}f}"
                        sigma = f"{bounded_params[sigma_key]:.{precision}f}"
                        param.set("value", f"NORMAL(MU={mu},SIGMA={sigma})")

        # Save modified XML
        if not os.path.exists(f"{output_dir}/inputs"):
            os.makedirs(f"{output_dir}/inputs")
        output_file = f"{output_dir}/inputs/input_{i+1}.xml"
        tree.write(output_file, encoding="utf-8", xml_declaration=True)

        # Log bounded parameters
        param_log.append(bounded_params)

    # Save parameters to CSV
    df = pd.DataFrame(param_log)
    df.to_csv(f"{output_dir}/kde_sampled_parameters_log.csv", index=False)

    print(f"Generated {n_samples} XML files and parameter log in {output_dir}/")


def generate_2param_perturbation(
    sensitivity_json: str,
    metric: str,
    template_path: str = "sample_input_v3.xml",
    perturbation_range: range = range(-50, 51, 10),
    output_dir: str = "inputs/STEM_CELL/",
):
    """Generate input files by perturbing the top 2 parameters based on MI scores.
    
    Args:
        sensitivity_json: Path to sensitivity analysis JSON file
        metric: Metric name to analyze ('symmetry', 'cycle_length', or 'vol_std')
        template_path: Path to template XML file
        perturbation_range: Range of perturbation percentages
    """
    # Load sensitivity analysis results
    with open(sensitivity_json, 'r') as f:
        sensitivity_data = json.load(f)
    
    if metric not in sensitivity_data:
        raise ValueError(f"Metric {metric} not found in sensitivity analysis data")
    
    # Get parameter names and MI scores
    param_names = sensitivity_data[metric]["names"]
    mi_scores = sensitivity_data[metric]["MI"]
    
    # Find top 2 parameters
    top_2_indices = np.argsort(mi_scores)[-2:][::-1]  # Sort descending
    top_2_params = [param_names[i] for i in top_2_indices]
    
    print(f"Top 2 parameters for {metric}:")
    print(f"1. {top_2_params[0]} (MI = {mi_scores[top_2_indices[0]]:.4f})")
    print(f"2. {top_2_params[1]} (MI = {mi_scores[top_2_indices[1]]:.4f})")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read template XML to get default values
    tree = ET.parse(template_path)
    root = tree.getroot()
    cancerous_pop = root.find(".//population[@id='cancerous']")
    
    # Get default values for the top 2 parameters
    default_values = {}
    for param in cancerous_pop.findall("population.parameter"):
        param_id = param.get("id")
        full_param_id = param_id
        
        # Handle parameters in subfolders
        if param_id.startswith("metabolism/") or param_id.startswith("signaling/") or param_id.startswith("proliferation/"):
            full_param_id = param_id
            param_id = param_id.split("/")[1]
        
        # Case 1: Direct parameter (like CELL_VOLUME)
        base_param = param_id.split("/")[-1]
        if base_param in [p.replace("_MU", "").replace("_SIGMA", "") for p in top_2_params]:
            value_str = param.get("value")
            if "NORMAL" not in value_str:
                default_values[base_param] = {
                    "type": "direct",
                    "value": float(value_str)
                }
            else:
                # Store both MU and SIGMA values
                mu = float(value_str.split("MU=")[1].split(",")[0])
                sigma = float(value_str.split("SIGMA=")[1].split(")")[0])
                default_values[base_param] = {
                    "type": "normal",
                    "mu": mu,
                    "sigma": sigma
                }
    
    # Generate perturbed parameter combinations
    param_log = []
    file_counter = 1
    for p1_pct in perturbation_range:
        for p2_pct in perturbation_range:
            # Calculate perturbed values
            p1_base = top_2_params[0].replace("_MU", "").replace("_SIGMA", "")
            p2_base = top_2_params[1].replace("_MU", "").replace("_SIGMA", "")
            p1_property = top_2_params[0].split("_")[-1]  # MU or SIGMA
            p2_property = top_2_params[1].split("_")[-1]  # MU or SIGMA
            
            # Calculate perturbed values based on parameter type
            if default_values[p1_base]["type"] == "direct":
                p1_value = default_values[p1_base]["value"] * (1 + p1_pct/100)
            else:
                if p1_property == "MU":
                    p1_value = default_values[p1_base]["mu"] * (1 + p1_pct/100)
                else:  # SIGMA
                    p1_value = default_values[p1_base]["sigma"] * (1 + p1_pct/100)
                    
            if default_values[p2_base]["type"] == "direct":
                p2_value = default_values[p2_base]["value"] * (1 + p2_pct/100)
            else:
                if p2_property == "MU":
                    p2_value = default_values[p2_base]["mu"] * (1 + p2_pct/100)
                else:  # SIGMA
                    p2_value = default_values[p2_base]["sigma"] * (1 + p2_pct/100)
            
            # Create new XML tree for this combination
            tree = ET.parse(template_path)
            root = tree.getroot()
            cancerous_pop = root.find(".//population[@id='cancerous']")
            
            # Update parameters
            for param in cancerous_pop.findall("population.parameter"):
                param_id = param.get("id")
                full_param_id = param_id
                
                # Handle parameters in subfolders
                if param_id.startswith("metabolism/") or param_id.startswith("signaling/") or param_id.startswith("proliferation/"):
                    full_param_id = param_id
                    param_id = param_id.split("/")[1]
                
                base_param = full_param_id.split("/")[-1]
                if base_param == p1_base:
                    param_info = default_values[base_param]
                    if param_info["type"] == "direct":
                        # Case 1: Direct parameter
                        new_value = param_info["value"] * (1 + p1_pct/100)
                        param.set("value", f"{new_value:.6f}")
                    else:
                        # Case 2 & 3: NORMAL distribution
                        mu = param_info["mu"]
                        sigma = param_info["sigma"]
                        if p1_property == "MU":
                            mu = mu * (1 + p1_pct/100)
                        else:  # SIGMA
                            sigma = sigma * (1 + p1_pct/100)
                        param.set("value", f"NORMAL(MU={mu:.6f},SIGMA={sigma:.6f})")
                
                elif base_param == p2_base:
                    param_info = default_values[base_param]
                    if param_info["type"] == "direct":
                        # Case 1: Direct parameter
                        new_value = param_info["value"] * (1 + p2_pct/100)
                        param.set("value", f"{new_value:.6f}")
                    else:
                        # Case 2 & 3: NORMAL distribution
                        mu = param_info["mu"]
                        sigma = param_info["sigma"]
                        if p2_property == "MU":
                            mu = mu * (1 + p2_pct/100)
                        else:  # SIGMA
                            sigma = sigma * (1 + p2_pct/100)
                        param.set("value", f"NORMAL(MU={mu:.6f},SIGMA={sigma:.6f})")
            
            # Save modified XML
            if not os.path.exists(f"{output_dir}/inputs"):
                os.makedirs(f"{output_dir}/inputs")
            output_file = f"{output_dir}/inputs/input_{file_counter}.xml"
            tree.write(output_file, encoding="utf-8", xml_declaration=True)
            
            # Log parameters
            param_log.append({
                "file_name": f"input_{file_counter}.xml",
                f"{top_2_params[0]}_perturbation": p1_pct,
                f"{top_2_params[1]}_perturbation": p2_pct,
                top_2_params[0]: p1_value,
                top_2_params[1]: p2_value
            })
            
            file_counter += 1
    
    # Save parameter log
    df = pd.DataFrame(param_log)
    df.to_csv(f"{output_dir}/parameter_log.csv", index=False)
    
    # Calculate parameter ranges based on parameter type and property
    parameter_ranges = {}
    for param, base_param, property_type in [(top_2_params[0], p1_base, p1_property), 
                                           (top_2_params[1], p2_base, p2_property)]:
        param_info = default_values[base_param]
        if param_info["type"] == "direct":
            base_value = param_info["value"]
            parameter_ranges[param] = (
                base_value * (1 + min(perturbation_range)/100),
                base_value * (1 + max(perturbation_range)/100)
            )
        else:  # normal distribution
            if property_type == "MU":
                base_value = param_info["mu"]
            else:  # SIGMA
                base_value = param_info["sigma"]
            parameter_ranges[param] = (
                base_value * (1 + min(perturbation_range)/100),
                base_value * (1 + max(perturbation_range)/100)
            )
    
    save_parameter_ranges(parameter_ranges, output_dir)
    
    print(f"Generated {len(param_log)} XML files and parameter log in {output_dir}/")


def main():
    output_dir = "inputs/STEM_CELL/TEST"
    param_ranges = {
        "CELL_VOLUME_MU": (-200, 500),
        "CELL_VOLUME_SIGMA": (50, 250),
        "APOPTOSIS_AGE_MU": (120960, 120960),
        "APOPTOSIS_AGE_SIGMA": (6000, 6000),
        "NECROTIC_FRACTION": (1.0, 1.0),
        "ACCURACY": (0.0, 1.0),
        "AFFINITY": (0.0, 1.0),
        "COMPRESSION_TOLERANCE": (3, 10),
        "SYNTHESIS_DURATION_MU": (580, 680),
        "SYNTHESIS_DURATION_SIGMA": (20, 70),
        "BASAL_ENERGY_MU": (0.0008, 0.0012),
        "BASAL_ENERGY_SIGMA": (0.00006, 0.0001),
        "PROLIFERATION_ENERGY_MU": (0.0008, 0.0012),
        "PROLIFERATION_ENERGY_SIGMA": (0.00006, 0.0001),
        "MIGRATION_ENERGY_MU": (0.00016, 0.00024),
        "MIGRATION_ENERGY_SIGMA": (0.000012, 0.00002),
        "METABOLIC_PREFERENCE_MU": (0.24, 0.36),
        "METABOLIC_PREFERENCE_SIGMA": (0.019, 0.029),
        "CONVERSION_FRACTION_MU": (0.2, 0.3),
        "CONVERSION_FRACTION_SIGMA": (0.016, 0.024),
        "RATIO_GLUCOSE_PYRUVATE_MU": (0.4, 0.6),
        "RATIO_GLUCOSE_PYRUVATE_SIGMA": (0.032, 0.048),
        "LACTATE_RATE_MU": (0.08, 0.12),
        "LACTATE_RATE_SIGMA": (0.006, 0.01),
        "AUTOPHAGY_RATE_MU": (0.00008, 0.00012),
        "AUTOPHAGY_RATE_SIGMA": (0.000006, 0.00001),
        "GLUCOSE_UPTAKE_RATE_MU": (0.9, 1.34),
        "GLUCOSE_UPTAKE_RATE_SIGMA": (0.072, 0.107),
        "ATP_PRODUCTION_RATE_MU": (7.14, 10.71),
        "ATP_PRODUCTION_RATE_SIGMA": (0.57, 0.86),
        "MIGRATORY_THRESHOLD_MU": (8, 12),
        "MIGRATORY_THRESHOLD_SIGMA": (0.64, 0.96),
    }
    if 1:
        generate_perturbed_parameters(
            sobol_power=8,
            param_ranges=param_ranges,
            output_dir=output_dir,
        )
    metric = "symmetry"
    output_dir = f"inputs/sensitivity_analysis/{metric}"
    if 0:
        generate_2param_perturbation(
            sensitivity_json="ARCADE_OUTPUT/STEM_CELL_META_SIGNAL_HETEROGENEITY/sensitivity_analysis.json",
            metric=metric,
            perturbation_range=range(-100, 101, 10),
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
