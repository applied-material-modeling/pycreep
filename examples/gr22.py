#!/usr/bin/env python3

import sys
sys.path.append('../')

from pycreep import data, ttp, time_independent

import numpy as np

import pickle

if __name__ == "__main__":
    df_rupture = data.load_data_from_file("Gr22-rupture.csv")
    df_yield = data.load_data_from_file("Gr22-yield.csv")

    param = ttp.LarsonMillerParameter()
    rupture_order = 1

    fraction = 0.5

    yield_model = time_independent.TabulatedTimeIndependentCorrelation(
            np.array([27.0,77,130,180,230,280,330,380,430,480,530,580,630,680,730]) + 273.15,
            np.array([301.33,291.03,285.8,284.12,283.73,282.87,279.81,272.93,260.89,242.87,218.85,189.69,157.19,123.75,91.9999]),
            df_yield,
            stress_field = "Yield Strength (MPa)", analysis_temp_units="K")
    lower_model = ttp.LotCenteredAnalysis(param, rupture_order, df_rupture)
    upper_model = ttp.LotCenteredAnalysis(param, rupture_order, df_rupture)

    model = ttp.SplitAnalysis(yield_model, fraction, lower_model, upper_model,
            df_rupture).analyze()
    
    print("Upper stress range")
    print(model.upper_model.report())

    print("")
    
    print("Lower stress range")
    print(model.lower_model.report())
