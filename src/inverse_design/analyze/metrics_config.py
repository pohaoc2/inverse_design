# Metrics configuration organized by cellular and population levels

CELLULAR_METRICS = {
    "vol": {
        "name": "Cell Volume",
        "keys": ["vol"],
        "stats": ["mean", "std"],
        "description": "Average cell volume",
        "spatial": False
    },
    "states": {
        "name": "Cell States",
        "keys": ["states"],
        "stats": ["distribution"],
        "description": "Distribution of cell states",
        "spatial": False
    },
    "age": {
        "name": "Cell Age",
        "keys": ["age"],
        "stats": ["median", "std"],
        "description": "Age distribution of cells",
        "spatial": False
    }
}

POPULATION_METRICS = {
    "n_cells": {
        "name": "Population Size",
        "keys": ["n_cells"],
        "stats": ["median", "std"],
        "description": "Total number of cells",
        "spatial": False
    },
    "symmetry": {
        "name": "Colony Symmetry",
        "keys": ["sym"],
        "stats": ["median", "std"],
        "description": "Colony symmetry measure",
        "spatial": True
    },
    "shannon": {
        "name": "Cell state diversity",
        "keys": ["shannon"],
        "stats": ["value"],
        "description": "Shannon diversity index",
        "spatial": False
    },
    "colony_diameter": {
        "name": "Colony Diameter",
        "keys": ["col_dia"],
        "stats": ["median", "std"],
        "description": "Colony diameter measurement",
        "spatial": True
    },
    "act": {
        "name": "Cell Activity",
        "keys": ["act"],
        "stats": ["median", "std"],
        "description": "number of active cells over total number of cells",
        "spatial": False
    },
}

SUMMARY_METRICS = {
    "doub_time": {
        "name": "Population Doubling Time",
        "keys": ["doub_time"],
        "stats": ["median", "std"],
        "description": "Population doubling time",
        "spatial": False
    },
    "colony_growth": {
        "name": "Colony Growth Properties",
        "keys": ["col_g_rate", "col_g_r_squared"],
        "stats": ["median", "std"],
        "description": "Colony growth measurements",
        "spatial": True
    }
} 