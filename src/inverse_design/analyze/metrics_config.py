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
    "act_ratio": {
        "name": "Cell Activity Ratio",
        "keys": ["act_ratio"],
        "stats": ["median", "std"],
        "description": "number of active cells over total number of cells",
        "spatial": False,
        "unit": "none"
    },
    "activity": {
        "name": "Cell Activity",
        "keys": ["activity"],
        "stats": ["median", "std"],
        "description": "number of active (PROLIFERATIVE, MIGRATORY) over inactive (NECROTIC, APOPTOTIC)",
        "spatial": False,
        "unit": "none"
    }
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
        "mean": 3680.141,
        "std": 161.766
    },
    "cycle_length": {
        "mean": 24.063,
        "std": 0.881
    },
    "n_cells": {
        "mean": 122.0,
        "std": 17.131
    },
    "symmetry": {
        "mean": 0.913,
        "std": 0.03
    },
    "shannon": {
        "mean": 0.781,
        "std": 0.058
    },
    "colony_diameter": {
        "mean": 300.0,
        "std": 24.661
    },
    "act_ratio": {
        "mean": 0.486,
        "std": 0.038
    },
    "activity": {
        "mean": 1.000,
        "std": 0.000
    },
    "doub_time": {
        "mean": 40.743,
        "std": 2.298
    },
    "colony_growth": {
        "mean": 31.402,
        "std": 3.137
    }
}
