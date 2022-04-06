#!/usr/bin/env python3

import sys
sys.path.append('../')

from pycreep import data, ttp, time_independent, allowables

import numpy as np

if __name__ == "__main__":
    # Load the data
    df_rupture = data.load_data_from_file("Gr22-rupture.csv")
    df_yield = data.load_data_from_file("Gr22-yield.csv")
    df_rate = data.load_data_from_file("Gr22-creeprate.csv")
    
    # Setup what TTP and polynomial order you want
    param = ttp.LarsonMillerParameter()
    rupture_order = 1
    rate_order = 1
    
    # Setup the breakpoint for the region split
    fraction = 0.5
    
    # Setup the model for the yield strength
    yield_model = time_independent.TabulatedTimeIndependentCorrelation(
            np.array([27.0,77,130,180,230,280,330,380,430,480,530,580,630,680,730]) + 273.15,
            np.array([301.33,291.03,285.8,284.12,283.73,282.87,279.81,272.93,260.89,242.87,218.85,189.69,157.19,123.75,91.9999]),
            df_yield,
            stress_field = "Yield Strength (MPa)", analysis_temp_units="K")
    
    # Rupture analysis
    rupture_lower_model = ttp.UncenteredAnalysis(param, rupture_order, 
            df_rupture)
    rupture_upper_model = ttp.UncenteredAnalysis(param, rupture_order, 
            df_rupture)
    rupture_model = ttp.SplitAnalysis(yield_model, fraction, 
            rupture_lower_model, rupture_upper_model,df_rupture).analyze()
    
    print("Creep rupture")
    print("")

    print("Upper stress range")
    print(rupture_model.upper_model.report())

    print("")
    
    print("Lower stress range")
    print(rupture_model.lower_model.report())

    # Creep rate analysis
    rate_lower_model = ttp.UncenteredAnalysis(param, rate_order, 
            df_rate, time_field = 'Creep rate (%/hr)')
    rate_upper_model = ttp.UncenteredAnalysis(param, rate_order, 
            df_rate, time_field = 'Creep rate (%/hr)')
    rate_model = ttp.SplitAnalysis(yield_model, fraction, 
            rate_lower_model, rate_upper_model,df_rate,
            time_field = "Creep rate (%/hr)").analyze()
    
    print("")
    print("----------------------------------------------------------")
    print("")

    print("Creep rate")
    print("")

    print("Upper stress range")
    print(rate_model.upper_model.report())

    print("")
    
    print("Lower stress range")
    print(rate_model.lower_model.report())

    print("")
    print("----------------------------------------------------------")
    print("")

    print("Allowable stress values")
    Ts = np.array([400,425,450,475,500,525,550,575,600,625,650]) + 273.15
    res = allowables.Sc_SectionII_1A_1B(rupture_model, rate_model, Ts, 
            full_results = True)

    print(res)
