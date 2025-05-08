from analyze.population_metrics import PopulationMetrics
from pathlib import Path
import pandas as pd
import numpy as np
def main():
    population_metrics = PopulationMetrics()
    glioblastoma_data = pd.read_csv("../../data/glioblastoma.csv")
    u87mg_data = glioblastoma_data[glioblastoma_data["type"] == "U87MG"]
    gbp03_data = glioblastoma_data[glioblastoma_data["type"] == "GBP03"]
    data = gbp03_data #u87mg_data
    diameters = list(map(lambda x: 2 * (x / np.pi) ** 0.5 * 1000, data["area"].tolist())) # convert mm^2 to um
    diameters = {f"seed_0": diameters}
    timestamps = data["time"].tolist()
    results = population_metrics.calculate_colony_growth(diameters, timestamps)
    print(results)

if __name__ == "__main__":
    main()
