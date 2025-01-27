import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from .metrics import MetricsFactory


def estimate_pdfs(accepted_params: List[Dict]) -> Dict:
    """
    Estimate PDFs for each parameter independently and jointly using multivariate normal
    Args:
        accepted_params: List of accepted parameter dictionaries
    Returns:
        Dictionary containing:
            'independent': Dict mapping parameter names to their KDE objects
            'joint_mean': Mean vector of parameters
            'joint_cov': Covariance matrix of parameters
            'param_names': List of parameter names
    """
    if not accepted_params:
        return {}

    # Extract parameter values into arrays
    param_names = list(accepted_params[0].keys())
    param_arrays = {param: np.array([p[param] for p in accepted_params]) for param in param_names}

    # Compute independent KDEs for each parameter
    param_pdfs = {}
    for param_name, values in param_arrays.items():
        kde = stats.gaussian_kde(values)
        param_pdfs[param_name] = kde

    # Compute joint distribution parameters
    X = np.array([list(p.values()) for p in accepted_params])
    joint_mean = np.mean(X, axis=0)
    joint_cov = np.cov(X, rowvar=False)

    return {
        "independent": param_pdfs,
        "joint_mean": joint_mean,
        "joint_cov": joint_cov,
        "param_names": param_names,
    }


def evaluate_parameters(abc_instance, parameters: Dict[str, float]) -> Tuple[Dict, float]:
    """Helper function to evaluate a single parameter set
    Args:
        abc_instance: ABC instance for accessing config and metric calculations
        parameters: Dictionary of parameter names and values
    Returns:
        Tuple of (metrics, distance)
    """
    # Import BDM here to avoid circular import
    from .models.bdm import BDM

    config = abc_instance.base_config.copy()
    for param_name, value in parameters.items():
        setattr(config.rates, param_name, value)
    model = BDM(config)
    time_points, _, grid_states = model.step()
    metrics_calculator = MetricsFactory.create_metrics(
        abc_instance.model_type, grid_states, time_points, abc_instance.max_time
    )
    metrics = abc_instance.calculate_metrics(metrics_calculator=metrics_calculator)
    distance = abc_instance.calculate_distance(metrics, metrics_calculator.normalization_factors)
    return {"metrics": metrics, "distance": distance}


def compare_sampling_methods(
    abc_instance, pdf_results: Dict, num_samples: int = 100
) -> Dict[str, Dict]:
    """Compare different sampling methods by evaluating their average distance to targets
    Args:
        abc_instance: ABC instance for accessing config and metric calculations
        pdf_results: Results from ABC._estimate_pdfs
        num_samples: Number of samples to generate for comparison
    Returns:
        Dictionary containing performance metrics and calculated metrics for each method
    """
    if not pdf_results:
        return {}

    results = {}

    independent_means = {}
    for param_name, kde in pdf_results["independent"].items():
        independent_means[param_name] = float(kde.resample(1000).mean())

    eval_result = evaluate_parameters(abc_instance, independent_means)
    results["independent_means"] = {
        "distance": float(eval_result["distance"]),
        "metrics": eval_result["metrics"],
    }

    joint_samples = np.random.multivariate_normal(
        pdf_results["joint_mean"], pdf_results["joint_cov"], size=num_samples
    )

    # Evaluate joint samples
    joint_distances = []
    joint_metrics = []
    for sample in joint_samples:
        parameters = dict(zip(pdf_results["param_names"], sample))
        eval_result = evaluate_parameters(abc_instance, parameters)
        joint_distances.append(float(eval_result["distance"]))
        joint_metrics.append(eval_result["metrics"])

    # Convert to numpy array for calculations
    joint_distances = np.array(joint_distances, dtype=np.float64)

    # Calculate mean metrics from joint sampling
    mean_joint_metrics = {}
    for metric in joint_metrics[0].keys():
        metric_values = [m[metric] for m in joint_metrics]
        mean_joint_metrics[metric] = float(np.mean(metric_values))

    results["joint_sampling"] = {
        "distance_mean": float(np.mean(joint_distances)),
        "distance_std": float(np.std(joint_distances)),
        "metrics": mean_joint_metrics,
    }

    return results
