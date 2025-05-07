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

EXP_METRICS = {
    "Breast Cancer": {
        "doubling_time": {
            "min": 20,
            "25%": 25,
            "median": 32,
            "75%": 44,
            "max": 50,
            "std": 4.32
        },
        "symmetry": {
            "min": 0.7,
            "25%": 0.75,
            "median": 0.82,
            "75%": 0.9,
            "max": 1.0,
            "std": 0.05
        },
        "growth_rate": {
            "min": 0.015,
            "25%": 0.02,
            "median": 0.025,
            "75%": 0.03,
            "max": 0.035,
            "std": 0.005
        }
    },
    "Glioblastoma": {
        "doubling_time": {
            "min": 15,
            "25%": 18,
            "median": 22,
            "75%": 27,
            "max": 33,
            "std": 3.1
        },
        "symmetry": {
            "min": 0.6,
            "25%": 0.68,
            "median": 0.75,
            "75%": 0.82,
            "max": 0.9,
            "std": 0.04
        },
        "growth_rate": {
            "min": 0.02,
            "25%": 0.03,
            "median": 0.04,
            "75%": 0.05,
            "max": 0.06,
            "std": 0.007
        }
    },
    "Melanoma": {
        "doubling_time": {
            "min": 10,
            "25%": 15,
            "median": 18,
            "75%": 22,
            "max": 28,
            "std": 2.7
        },
        "symmetry": {
            "min": 0.65,
            "25%": 0.7,
            "median": 0.78,
            "75%": 0.85,
            "max": 0.93,
            "std": 0.045
        },
        "growth_rate": {
            "min": 0.04,
            "25%": 0.05,
            "median": 0.06,
            "75%": 0.08,
            "max": 0.1,
            "std": 0.012
        }
    }
}

