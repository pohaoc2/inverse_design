# Metrics configuration organized by cellular and population levels

CELLULAR_METRICS = {
    "vol": {
        "name": "Cell Volume",
        "keys": ["vol"],
        "stats": ["mean", "std"],
        "description": "Average cell volume",
        "spatial": False,
        "unit": "um^3"
    },
    "states": {
        "name": "Cell States",
        "keys": ["states"],
        "stats": ["distribution"],
        "description": "Distribution of cell states",
        "spatial": False,
        "unit": "none"
    },
    "age": {
        "name": "Cell Age",
        "keys": ["age"],
        "stats": ["median", "std"],
        "description": "Age distribution of cells",
        "spatial": False,
        "unit": "minutes"
    },
    "cycle_length": {
        "name": "Cell Cycle Length",
        "keys": ["cycle_length"],
        "stats": ["mean", "std"],
        "description": "Average cell cycle length",
        "spatial": False,
        "unit": "hours"
    }
}

POPULATION_METRICS = {
    "n_cells": {
        "name": "Population Size",
        "keys": ["n_cells"],
        "stats": ["median", "std"],
        "description": "Total number of cells",
        "spatial": False,
        "unit": "none"
    },
    "symmetry": {
        "name": "Colony Symmetry",
        "keys": ["sym"],
        "stats": ["median", "std"],
        "description": "Colony symmetry measure",
        "spatial": True,
        "unit": "none"
    },
    "shannon": {
        "name": "Cell state diversity",
        "keys": ["shannon"],
        "stats": ["value"],
        "description": "Shannon diversity index",
        "spatial": False,
        "unit": "none"
    },
    "colony_diameter": {
        "name": "Colony Diameter",
        "keys": ["col_dia"],
        "stats": ["median", "std"],
        "description": "Colony diameter measurement",
        "spatial": True,
        "unit": "um"
    },
    "act": {
        "name": "Cell Activity",
        "keys": ["act"],
        "stats": ["median", "std"],
        "description": "number of active cells over total number of cells",
        "spatial": False,
        "unit": "none"
    },
}

SUMMARY_METRICS = {
    "doub_time": {
        "name": "Population Doubling Time",
        "keys": ["doub_time"],
        "stats": ["median", "std"],
        "description": "Population doubling time",
        "spatial": False,
        "unit": "hours"
    },
    "colony_growth": {
        "name": "Colony Growth Properties",
        "keys": ["col_g_rate", "col_g_r_squared"],
        "stats": ["median", "std"],
        "description": "Colony growth measurements",
        "spatial": True,
        "unit": "um^2/hour"
    }
}

DEFAULT_METRICS = {
    "vol": {
        "mean": 2250,
        "std": 100
    },
    "cycle_length": {
        "mean": 26.54,
        "std": 20/24
    },
    "n_cells": {
        "mean": 1000,
        "std": 100
    },
    "symmetry": {
        "mean": 0.5,
        "std": 0.1
    },
    "shannon": {
        "mean": 0.5,
        "std": 0.1
    },
    "colony_diameter": {
        "mean": 100,
        "std": 10
    },
    "act": {
        "mean": 0.5,
        "std": 0.1
    },

    "col_dia": {
        "mean": 0.5,
        "std": 0.1
    },
    "doub_time": {
        "mean": 24,
        "std": 2
    },
    "colony_growth": {
        "mean": 100,
        "std": 10
    }
}
