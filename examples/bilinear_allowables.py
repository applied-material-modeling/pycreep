#!/usr/bin/env python3

import sys

sys.path.append("..")

from pycreep import data, ttp, time_independent, allowables, reports

import numpy as np

if __name__ == "__main__":
    # Load the data
    df_data = data.load_data_from_file("bilinear.csv")

    # Setup what TTP and polynomial order you want
    param = ttp.LarsonMillerParameter()
    rupture_order = 1
    rate_order = 1

    # Setup the breakpoint for the region split
    fraction = 0.5

    # Setup the model for the yield strength
    Trange = np.array(
        [
            773.15,
            791.00714286,
            808.86428571,
            826.72142857,
            844.57857143,
            862.43571429,
            880.29285714,
            898.15,
            916.00714286,
            933.86428571,
            951.72142857,
            969.57857143,
            987.43571429,
            1005.29285714,
            1023.15,
        ]
    )
    ydata = np.array(
        [
            332.5,
            325.71428571,
            318.92857143,
            312.14285714,
            305.35714286,
            298.57142857,
            291.78571429,
            285.0,
            278.21428571,
            271.42857143,
            264.64285714,
            257.85714286,
            251.07142857,
            244.28571429,
            237.5,
        ]
    )

    yield_model = time_independent.TabulatedTimeIndependentCorrelation(
        Trange, ydata, None, input_temp_units="K"
    )

    # Rupture analysis
    rupture_lower_model = ttp.LotCenteredAnalysis(param, rupture_order, df_data)
    rupture_upper_model = ttp.LotCenteredAnalysis(param, rupture_order, df_data)
    rupture_model = ttp.SplitAnalysis(
        yield_model, fraction, rupture_lower_model, rupture_upper_model, df_data
    ).analyze()

    # Creep rate analysis
    rate_lower_model = ttp.UncenteredAnalysis(
        param, rate_order, df_data, time_field="Creep rate (%/hr)"
    )
    rate_upper_model = ttp.UncenteredAnalysis(
        param, rate_order, df_data, time_field="Creep rate (%/hr)"
    )
    rate_model = ttp.SplitAnalysis(
        yield_model,
        fraction,
        rate_lower_model,
        rate_upper_model,
        df_data,
        time_field="Creep rate (%/hr)",
    ).analyze()

    Ts = np.array([500, 525, 550, 575, 600, 625, 650, 700, 750.0]) + 273.15
    res = allowables.Sc_SectionII_1A_1B(
        rupture_model, rate_model, Ts, full_results=True
    )
    times = np.array(
        [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000.0]
    )

    reports.write_complete_report(
        "bilinear_report.xlsx", rupture_model, rate_model, yield_model, Ts, times
    )
