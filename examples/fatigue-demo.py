import pandas as pd
from pycreep import fatigue
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pd.read_csv("/Users/messner/Work/Projects/A617-fatigue/A617-fatigue.csv")

    fatigue_analysis = fatigue.LumpedTemperatureFatigueAnalysis(
        fatigue.DiercksEquation(4), [850, 950], data
    ).analyze()

    for T, inds in fatigue_analysis.temperature_groups.items():
        (l,) = plt.loglog(
            fatigue_analysis.cycles[inds],
            fatigue_analysis.strain_range[inds],
            "o",
            label=f"T={T}C",
        )
        erange = np.logspace(
            np.log10(fatigue_analysis.strain_range[inds].min()),
            np.log10(fatigue_analysis.strain_range[inds].max()),
            100,
        )
        pred = fatigue_analysis.predict(np.full_like(erange, T), erange)
        plt.loglog(pred, erange, ls="--", color=l.get_color(), label="Prediction")

    plt.xlabel("Cycles")
    plt.ylabel("Strain Range")
    plt.legend(loc="best")
    plt.show()
