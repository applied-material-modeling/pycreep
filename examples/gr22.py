#!/usr/bin/env python3

import sys
sys.path.append('../')

from pycreep import data, ttp, time_independent

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df_rupture = data.load_data_from_file("Gr22-rupture.csv")
    df_yield = data.load_data_from_file("Gr22-yield.csv")

    yield_model = time_independent.TimeIndependentCorrelation(df_yield,
            stress_field = "Yield Strength (MPa)", analysis_temp_units="degC")

    res = yield_model.polynomial_analysis(5)
    print(res)

    plt.plot(yield_model.temperature, yield_model.stress, 'kx')
    Ts = np.linspace(400, 700)
    plt.plot(Ts, np.polyval(res['polyavg'], Ts))
    plt.plot(Ts, np.polyval([-4.533333e-10, 1.1533333e-6, -1.1723333e-3, 5.949166666e-1, -1.50963333e+2, 1.5581e4], Ts))

    plt.show()
