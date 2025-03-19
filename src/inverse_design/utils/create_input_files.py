import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import scipy.stats.qmc
from typing import Dict
import os
import json
from inverse_design.analyze.parameter_config import PARAM_RANGES, SOURCE_PARAM_RANGES
from inverse_design.analyze.source_metrics import calculate_capillary_density, calculate_distance_between_points
from scipy.stats import qmc


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
    "METABOLIC_PREFERENCE": (6, False, "metabolism"),
    "CONVERSION_FRACTION": (6, False, "metabolism"),
    "RATIO_GLUCOSE_PYRUVATE": (6, False, "metabolism"),
    "LACTATE_RATE": (6, False, "metabolism"),
    "AUTOPHAGY_RATE": (6, False, "metabolism"),
    "GLUCOSE_UPTAKE_RATE": (6, False, "metabolism"),
    "ATP_PRODUCTION_RATE": (6, False, "metabolism"),
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


def _create_parameter_log_entry(params: dict, file_number: int) -> dict:
    """Create a parameter log entry with bounds checking.

    Args:
        params: Dictionary of parameter values
        file_number: Number to use in the file name

    Returns:
        Dictionary containing the file name and formatted parameter values
    """
    # Create a dictionary for the log entry
    log_entry = {"file_name": f"input_{file_number}.xml"}

    # Add parameters with bounds checking using PARAM_CONFIGS
    for base_param, (precision, is_bounded, _) in PARAM_CONFIGS.items():
        # Handle direct parameters (those without MU/SIGMA)
        if base_param in params:
            value = _format_param_value(params[base_param], precision, is_bounded)
            log_entry[base_param] = float(value)

        # Handle MU/SIGMA parameters
        mu_key = f"{base_param}_MU"
        sigma_key = f"{base_param}_SIGMA"

        if mu_key in params:
            mu_value = _format_param_value(params[mu_key], precision, is_bounded)
            log_entry[mu_key] = float(mu_value)

        if sigma_key in params:
            sigma_value = _format_param_value(
                params[sigma_key], precision, False
            )  # sigma is never bounded by 1
            log_entry[sigma_key] = float(sigma_value)

    return log_entry


def generate_perturbed_parameters(
    sobol_power: int,
    param_ranges: dict,
    config_params: Dict[str, str],
    output_dir: str = "perturbed_inputs",
    seed: int = 42,
):
    """Generate perturbed parameter sets using Sobol sampling

    Args:
        sobol_power: Power of 2 for number of samples (n_samples = 2^sobol_power)
        param_ranges: Dictionary of parameter ranges {param_name: (min, max)}
        config_params: Dictionary of input configurations {config_type: {param_name: value}}
        output_dir: Directory to save generated XML files
        seed: Random seed for reproducibility
    """
    config_type = config_params["perturbed_config"]
    if config_type not in ["cellular", "source"]:
        raise ValueError('config_type must be either "cellular" or "source"')
    template_path = config_params["template_path"]
    point_based = config_params["point_based"]
    y_interval = config_params["y_interval"]
    radius_bound = config_params["radius_bound"]
    side_length = config_params["side_length"]

    # Check if the input directory exists
    if os.path.exists(output_dir):
        num_files = len([f for f in os.listdir(output_dir + "/inputs") if f.startswith("input_")])
        if num_files >= 2**sobol_power:
            print(f"Input directory exists with sufficient files ({num_files} ≥ {2**sobol_power}), skipping generation")
            return
        else:
            print(f"Input directory exists but needs more files ({num_files} < {2**sobol_power}), generating new files")
    else:
        os.makedirs(f"{output_dir}/inputs")

    # Save parameter ranges to CSV
    save_parameter_ranges(param_ranges, output_dir)

    if config_type == "cellular":
        # Generate cellular parameter samples
        sampler = qmc.Sobol(d=len(param_ranges), scramble=True, seed=seed)
        samples = sampler.random_base2(m=sobol_power)

        # Scale samples and convert to parameter dictionary format
        params = {name: [] for name in param_ranges.keys()}
        for sample in samples:
            for j, ((name, (min_val, max_val))) in enumerate(param_ranges.items()):
                scaled_value = min_val + (max_val - min_val) * sample[j]
                params[name].append(scaled_value)

        # Generate XML files using the samples
        param_log = generate_cellular_perturbations(
            params=params,
            template_path=template_path,
            output_dir=output_dir,
        )

    else: # source configuration
        samples = generate_source_site_samples(
            param_ranges=param_ranges,
            sobol_power=sobol_power,
            point_based=point_based,
            y_interval=y_interval,
            radius_bound=radius_bound,
            side_length=side_length,
        )
        param_log = generate_source_site_perturbations(
            params=samples,
            template_path=template_path,
            output_dir=output_dir,
        )

    _save_param_log(param_log, output_dir)

def _save_param_log(param_log: list, output_dir: str):
    df = pd.DataFrame(param_log)
    df.to_csv(f"{output_dir}/parameter_log.csv", index=False)
    print(f"Saved parameter log to {output_dir}/parameter_log.csv")

def generate_cellular_perturbations(
    params: dict,
    template_path: str,
    output_dir: str,
) -> None:
    """Generate input XML files for cellular parameter perturbations.

    Args:
        params: Dictionary of parameter values {param_name: [values]}
        template_path: Path to template XML file
        output_dir: Directory to save generated XML files
    """
    # Create output directories if they don't exist
    if not os.path.exists(f"{output_dir}/inputs"):
        os.makedirs(f"{output_dir}/inputs")

    # Read template XML
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Prepare parameter logging
    param_log = []
    n_samples = len(next(iter(params.values())))

    # Generate files for each parameter set
    for i in range(n_samples):
        # Create parameter dictionary for this sample
        sample_params = {
            name: values[i] 
            for name, values in params.items()
        }

        # Create log entry
        log_entry = _create_parameter_log_entry(sample_params, i + 1)
        param_log.append(log_entry)

        # Update XML parameters
        update_xml_parameters(root, sample_params)

        # Save modified XML
        output_file = f"{output_dir}/inputs/input_{i+1}.xml"
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
    return param_log

def generate_parameters_from_kde(
    parameter_pdfs: Dict,
    n_samples: int,
    output_dir: str = "kde_sampled_inputs",
    template_path: str = "sample_input_v3.xml",
    const_params_values: Dict[str, float] = None,
    seed: int = 42,
):
    """Generate parameter sets by sampling from kernel density estimates (KDE)

    Args:
        parameter_pdfs: Dictionary containing KDE objects for each parameter
        n_samples: Number of parameter sets to generate
        output_dir: Directory to save generated XML files
        template_path: Path to template XML file
        const_params_values: Dictionary of parameters to keep constant {param_name: value}
    """
    # Check if the input directory exists, if exists, skip the generation
    if os.path.exists(output_dir):
        num_files = len([f for f in os.listdir(output_dir + "/inputs") if f.startswith("input_")])
        if num_files >= n_samples:
            print(
                f"Input directory {output_dir} already exists, and number of files ({num_files}) is greater than or equal to the number of target number of files ({n_samples}), skipping generation"
            )
            return
        else:
            print(
                f"Input directory {output_dir} already exists, but number of files ({num_files}) is less than the number of target number of files ({n_samples}), generating new files"
            )
    else:
        os.makedirs(f"{output_dir}/inputs")

    # Read template XML
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Prepare DataFrame for parameter logging
    param_log = []

    # Sample from each parameter's KDE
    kde_dict = {key: value["kde"] for key, value in parameter_pdfs.items()}
    param_names = parameter_pdfs.keys()

    # Generate samples for each parameter
    sampled_params = {
        param: kde.resample(n_samples, seed=seed)[0] for param, kde in kde_dict.items()
    }

    # Generate XML files for each parameter set
    for i in range(n_samples):
        # Get raw parameters
        raw_params = {param: sampled_params[param][i] for param in param_names}

        # Add constant parameters to raw_params
        if const_params_values:
            raw_params.update(const_params_values)

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
                sigma_value = float(
                    _format_param_value(raw_params[sigma_key], precision, False)
                )  # sigma is never bounded by 1
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
    with open(sensitivity_json, "r") as f:
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
        if (
            param_id.startswith("metabolism/")
            or param_id.startswith("signaling/")
            or param_id.startswith("proliferation/")
        ):
            full_param_id = param_id
            param_id = param_id.split("/")[1]

        # Case 1: Direct parameter (like CELL_VOLUME)
        base_param = param_id.split("/")[-1]
        if base_param in [p.replace("_MU", "").replace("_SIGMA", "") for p in top_2_params]:
            value_str = param.get("value")
            if "NORMAL" not in value_str:
                default_values[base_param] = {"type": "direct", "value": float(value_str)}
            else:
                # Store both MU and SIGMA values
                mu = float(value_str.split("MU=")[1].split(",")[0])
                sigma = float(value_str.split("SIGMA=")[1].split(")")[0])
                default_values[base_param] = {"type": "normal", "mu": mu, "sigma": sigma}

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
                p1_value = default_values[p1_base]["value"] * (1 + p1_pct / 100)
            else:
                if p1_property == "MU":
                    p1_value = default_values[p1_base]["mu"] * (1 + p1_pct / 100)
                else:  # SIGMA
                    p1_value = default_values[p1_base]["sigma"] * (1 + p1_pct / 100)

            if default_values[p2_base]["type"] == "direct":
                p2_value = default_values[p2_base]["value"] * (1 + p2_pct / 100)
            else:
                if p2_property == "MU":
                    p2_value = default_values[p2_base]["mu"] * (1 + p2_pct / 100)
                else:  # SIGMA
                    p2_value = default_values[p2_base]["sigma"] * (1 + p2_pct / 100)

            # Create new XML tree for this combination
            tree = ET.parse(template_path)
            root = tree.getroot()
            cancerous_pop = root.find(".//population[@id='cancerous']")

            # Update parameters
            for param in cancerous_pop.findall("population.parameter"):
                param_id = param.get("id")
                full_param_id = param_id

                # Handle parameters in subfolders
                if (
                    param_id.startswith("metabolism/")
                    or param_id.startswith("signaling/")
                    or param_id.startswith("proliferation/")
                ):
                    full_param_id = param_id
                    param_id = param_id.split("/")[1]

                base_param = full_param_id.split("/")[-1]
                if base_param == p1_base:
                    param_info = default_values[base_param]
                    if param_info["type"] == "direct":
                        # Case 1: Direct parameter
                        new_value = param_info["value"] * (1 + p1_pct / 100)
                        param.set("value", f"{new_value:.6f}")
                    else:
                        # Case 2 & 3: NORMAL distribution
                        mu = param_info["mu"]
                        sigma = param_info["sigma"]
                        if p1_property == "MU":
                            mu = mu * (1 + p1_pct / 100)
                        else:  # SIGMA
                            sigma = sigma * (1 + p1_pct / 100)
                        param.set("value", f"NORMAL(MU={mu:.6f},SIGMA={sigma:.6f})")

                elif base_param == p2_base:
                    param_info = default_values[base_param]
                    if param_info["type"] == "direct":
                        # Case 1: Direct parameter
                        new_value = param_info["value"] * (1 + p2_pct / 100)
                        param.set("value", f"{new_value:.6f}")
                    else:
                        # Case 2 & 3: NORMAL distribution
                        mu = param_info["mu"]
                        sigma = param_info["sigma"]
                        if p2_property == "MU":
                            mu = mu * (1 + p2_pct / 100)
                        else:  # SIGMA
                            sigma = sigma * (1 + p2_pct / 100)
                        param.set("value", f"NORMAL(MU={mu:.6f},SIGMA={sigma:.6f})")

            # Save modified XML
            if not os.path.exists(f"{output_dir}/inputs"):
                os.makedirs(f"{output_dir}/inputs")
            output_file = f"{output_dir}/inputs/input_{file_counter}.xml"
            tree.write(output_file, encoding="utf-8", xml_declaration=True)

            # Log parameters
            param_log.append(
                {
                    "file_name": f"input_{file_counter}.xml",
                    f"{top_2_params[0]}_perturbation": p1_pct,
                    f"{top_2_params[1]}_perturbation": p2_pct,
                    top_2_params[0]: p1_value,
                    top_2_params[1]: p2_value,
                }
            )

            file_counter += 1

    # Save parameter log
    df = pd.DataFrame(param_log)
    df.to_csv(f"{output_dir}/parameter_log.csv", index=False)

    # Calculate parameter ranges based on parameter type and property
    parameter_ranges = {}
    for param, base_param, property_type in [
        (top_2_params[0], p1_base, p1_property),
        (top_2_params[1], p2_base, p2_property),
    ]:
        param_info = default_values[base_param]
        if param_info["type"] == "direct":
            base_value = param_info["value"]
            parameter_ranges[param] = (
                base_value * (1 + min(perturbation_range) / 100),
                base_value * (1 + max(perturbation_range) / 100),
            )
        else:  # normal distribution
            if property_type == "MU":
                base_value = param_info["mu"]
            else:  # SIGMA
                base_value = param_info["sigma"]
            parameter_ranges[param] = (
                base_value * (1 + min(perturbation_range) / 100),
                base_value * (1 + max(perturbation_range) / 100),
            )

    save_parameter_ranges(parameter_ranges, output_dir)

    print(f"Generated {len(param_log)} XML files and parameter log in {output_dir}/")


def generate_input_files(
    param_names: list[str],
    param_values: list[list[float]],
    config_params: Dict[str, str],
    output_dir: str = "inputs",
) -> None:
    """Generate input XML files from parameter values.

    Args:
        param_names: List of parameter names
        param_values: List of parameter value lists, where each inner list contains values
                     corresponding to param_names
        config_params: Dictionary of input configurations {config_type: {param_name: value}}
        template_path: Path to template XML file
        output_dir: Directory to save generated XML files
    """
    config_type = config_params["perturbed_config"]
    if config_type not in ["cellular", "source"]:
        raise ValueError('config_type must be either "cellular" or "source"')
    template_path = config_params["template_path"]

    # Check if the input directory exists
    if os.path.exists(output_dir):
        num_files = len([f for f in os.listdir(output_dir + "/inputs") if f.startswith("input_")])
        if num_files >= len(param_values):
            print(f"Input directory exists with sufficient files ({num_files} ≥ {len(param_values)}), skipping generation")
            return
        else:
            print(f"Input directory exists but needs more files ({num_files} < {len(param_values)}), generating new files")
    else:
        os.makedirs(f"{output_dir}/inputs")

    # Save parameter ranges
    param_ranges = {
        name: (min(values), max(values)) 
        for name, values in zip(param_names, zip(*param_values))
    }
    save_parameter_ranges(param_ranges, output_dir)

    if config_type == "cellular":
        # Convert to dictionary format for cellular parameters
        params_dict = {
            name: [values[i] for values in param_values]
            for i, name in enumerate(param_names)
        }
        param_log = generate_cellular_perturbations(
            params=params_dict,
            template_path=template_path,
            output_dir=output_dir,
        )

    else:  # source configuration
        # Convert to dictionary format for source parameters
        params = {
            "X_SPACING": [],
            "Y_SPACING": [],
            "GLUCOSE_CONCENTRATION": [],
            "OXYGEN_CONCENTRATION": [],
        }
        
        for values in param_values:
            params_dict = dict(zip(param_names, values))
            params["X_SPACING"].append(params_dict.get("X_SPACING", ""))
            params["Y_SPACING"].append(params_dict.get("Y_SPACING", ""))
            params["GLUCOSE_CONCENTRATION"].append(params_dict.get("GLUCOSE_CONCENTRATION", 0))
            params["OXYGEN_CONCENTRATION"].append(params_dict.get("OXYGEN_CONCENTRATION", 0))

        param_log = generate_source_site_perturbations(
            params=params,
            template_path=template_path,
            output_dir=output_dir,
        )
    _save_param_log(param_log, output_dir)

def generate_source_site_perturbations(
    params: dict,
    template_path: str,
    output_dir: str,
) -> None:
    """Generate input XML files for source site perturbations.

    Args:
        params: Dictionary containing:
            - x_spacing: List of X_SPACING values
            - y_spacing: List of Y_SPACING values
            - glucose: List of glucose concentrations
            - oxygen: List of oxygen concentrations
            - capillary_density: List of capillary densities
            - distance_to_center: List of distances
        template_path: Path to template XML file
        output_dir: Directory to save generated XML files
    """
    # Generate XML files
    tree = ET.parse(template_path)
    root = tree.getroot()
    param_log = []
    n_samples = len(params["X_SPACING"])

    def get_param_value(param_name, index):
        """Get parameter value at index or return 0 if parameter doesn't exist"""
        if param_name not in params:
            return 0
        param_list = params[param_name]
        if index >= len(param_list):
            return 0
        return param_list[index]

    for i in range(n_samples):
        # Find and update source sites component
        sites_component = root.find(".//component[@id='SITES']")
        if sites_component is None:
            raise ValueError("Could not find component with id='SITES' in XML")

        # Update parameters
        sites_component.find("component.parameter[@id='X_SPACING']").set("value", str(get_param_value("X_SPACING", i)))
        sites_component.find("component.parameter[@id='Y_SPACING']").set("value", str(get_param_value("Y_SPACING", i)))

        # Update concentrations
        for layer_id, param_name in [("GLUCOSE", "GLUCOSE_CONCENTRATION"), ("OXYGEN", "OXYGEN_CONCENTRATION")]:
            layer = root.find(f".//layer[@id='{layer_id}']")
            for param in layer.findall("layer.parameter"):
                if param.get("operation") == "generator" or param.get("id") == "INITIAL_CONCENTRATION":
                    param.set("value", str(get_param_value(param_name, i)))

        # Save modified XML
        output_file = f"{output_dir}/inputs/input_{i+1}.xml"
        tree.write(output_file, encoding="utf-8", xml_declaration=True)

        # Log parameters
        param_log.append({
            "file_name": f"input_{i+1}.xml",
            "X_SPACING": get_param_value("X_SPACING", i),
            "Y_SPACING": get_param_value("Y_SPACING", i),
            "GLUCOSE_CONCENTRATION": get_param_value("GLUCOSE_CONCENTRATION", i),
            "OXYGEN_CONCENTRATION": get_param_value("OXYGEN_CONCENTRATION", i),
            "CAPILLARY_DENSITY": get_param_value("CAPILLARY_DENSITY", i),
            "DISTANCE_TO_CENTER": get_param_value("DISTANCE_TO_CENTER", i),
        })

    return param_log

def generate_source_site_samples(
    param_ranges: dict,
    sobol_power=10,
    point_based=True,
    y_interval=4,
    radius_bound=10,
    side_length=1,
):
    """
    Generate Sobol samples for x_spacings, y_spacings, glucose, and oxygen concentrations.

    Parameters:
    -----------
    sobol_power : int
        Power of 2 for number of samples (n_samples = 2^sobol_power)
    y_spacing_interval : bool
        If True, sample y_spacing from specific values
        If False, use continuous Sobol sampling
    y_spacing_values : list or None
        List of specific values to sample from when y_spacing_interval is True
        Default values are [2, 5, 8, 12, 15] if None

    Returns:
    --------
    dict : Dictionary containing the sampled parameters
    """
    n_samples = 2**sobol_power
    sampler = qmc.Sobol(d=len(param_ranges), seed=42)
    samples = sampler.random(n=n_samples)

    length = int(6 * radius_bound - 3)
    width = int(4 * radius_bound - 2)
    scaled_samples = {}
    x_values = None
    for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
        if point_based:
            if param_name == "X_SPACING":
                x_center = int(length/2)
                scaled_samples["X_SPACING"] = [f"{x_center-1}:{x_center+1}" for _ in samples]
            elif param_name == "Y_SPACING":
                sample_2d = samples[:, i : i + 1]
                sample_2d = samples[:, i : i + 1]
                if 1 + (max_val - 1) * y_interval > width:
                    max_val = (width - 1) / y_interval + 1
                scaled_values = (
                    np.round(qmc.scale(sample_2d, l_bounds=min_val, u_bounds=max_val))
                    .flatten()
                    .astype(int)
                )
                scaled_values = 1 + (scaled_values - 1) * y_interval
                scaled_samples["Y_SPACING"] = [f"{value}:{value+1}" for value in scaled_values]

        else: #grid based source
            if param_name in ["X_SPACING", "Y_SPACING"]:
                sample_2d = samples[:, i : i + 1]
                scaled_values = (
                    np.round(qmc.scale(sample_2d, l_bounds=min_val, u_bounds=max_val))
                    .flatten()
                    .astype(int)
                )
                
                if param_name == "X_SPACING":
                    x_values = scaled_values
                else:
                    if x_values is None:
                        raise ValueError("x_spacing must come before y_spacing in param_ranges")
                    # Ensure x_spacing ≤ y_spacing
                    x_final = np.minimum(x_values, scaled_values)
                    y_final = np.maximum(x_values, scaled_values)
                    scaled_samples["X_SPACING"] = [f"*:{i}" for i in x_final]
                    scaled_samples["Y_SPACING"] = [f"*:{i}" for i in y_final]
        if param_name not in ["X_SPACING", "Y_SPACING"]:
            sample_2d = samples[:, i : i + 1]
            scaled_values = qmc.scale(sample_2d, l_bounds=min_val, u_bounds=max_val).flatten()
            scaled_samples[param_name] = scaled_values
    if point_based:
        capillary_density = [
            calculate_capillary_density(
                radius_bound,
                length,
                width,
                side_length,
            )
            for i in range(n_samples)
        ]
        point_center = np.array([length//2, width//2, 0]).astype(int)
        distance_to_center = []
        for i in range(n_samples):
            source_site = np.array([length//2, scaled_samples["Y_SPACING"][i].split(":")[1], 0]).astype(int)
            distance_to_center.append(
                calculate_distance_between_points(
                    point_center,
                    source_site,
                    side_length,
                    radius_bound,
                )
            )
        
    else:
        capillary_density = [
            calculate_capillary_density(
                radius_bound,
                int(scaled_samples["X_SPACING"][i].split(":")[1]),
                int(scaled_samples["Y_SPACING"][i].split(":")[1]),
                side_length,
            )
            for i in range(n_samples)
        ]
        distance_to_center = [np.nan] * n_samples
    scaled_samples["CAPILLARY_DENSITY"] = capillary_density
    scaled_samples["DISTANCE_TO_CENTER"] = distance_to_center


    return scaled_samples

def update_xml_parameters(root: ET.Element, params: dict) -> None:
    """Update XML parameters with provided values.

    Args:
        root: XML root element
        params: Dictionary of parameter values to update
    """
    # Find cancerous population element
    cancerous_pop = root.find(".//population[@id='cancerous']")
    if cancerous_pop is None:
        raise ValueError("Could not find population with id='cancerous' in XML")

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
                    sigma = _format_param_value(
                        params[sigma_key], precision, False
                    )  # sigma is never bounded by 1
                    param.set("value", f"NORMAL(MU={mu},SIGMA={sigma})")



def main():
    
    
    radius = 10
    margin = 2
    hex_size = 30
    side_length = hex_size / np.sqrt(3)
    configs = [
            {
                "perturbed_config": "cellular",
                "template_path": "test_v3.xml",
                "point_based": None,
                "y_interval": None,
                "radius_bound": None,
                "side_length": None
            },
            {
                "perturbed_config": "source",
                "template_path": "sample_source_v3.xml",
                "point_based": True,
                "y_interval": 4,
                "radius_bound": radius+margin,
                "side_length": side_length
            }
        ]
    if 0:
        output_dir = "inputs/STEM_CELL/density_source/cellular_test"
        generate_perturbed_parameters(
            sobol_power=8,
            param_ranges=PARAM_RANGES,
            output_dir=output_dir,
            config_params=configs[0],
        )

    if 1:
        source_type = "point" if configs[1]["point_based"] else "grid"
        output_dir = f"inputs/STEM_CELL/density_source/low_oxygen/{source_type}"
        generate_perturbed_parameters(
            sobol_power=10,
            param_ranges=SOURCE_PARAM_RANGES,
            output_dir=output_dir,
            config_params=configs[1],
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
