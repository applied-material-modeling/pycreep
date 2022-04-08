#!/usr/bin/env python3

import sys
sys.path.append('..')

from pycreep import data, ttp, time_independent, allowables

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
    Trange = np.linspace(500,750.0,15) + 273.15
    ydata = np.linspace(125.0,175.0,15)[::-1]*1.9

    print(ydata)

    yield_model = time_independent.TabulatedTimeIndependentCorrelation(
            Trange, ydata, None) 
    
    # Rupture analysis
    rupture_lower_model = ttp.LotCenteredAnalysis(param, rupture_order, 
            df_data)
    rupture_upper_model = ttp.LotCenteredAnalysis(param, rupture_order, 
            df_data)
    rupture_model = ttp.SplitAnalysis(yield_model, fraction, 
            rupture_lower_model, rupture_upper_model,df_data).analyze()
    
    print("Creep rupture")
    print("")

    print("Upper stress range")
    print(rupture_model.upper_model.report())

    print("")
    
    print("Lower stress range")
    print(rupture_model.lower_model.report())

    # Creep rate analysis
    rate_lower_model = ttp.UncenteredAnalysis(param, rate_order, 
            df_data, time_field = 'Creep rate (%/hr)')
    rate_upper_model = ttp.UncenteredAnalysis(param, rate_order, 
            df_data, time_field = 'Creep rate (%/hr)')
    rate_model = ttp.SplitAnalysis(yield_model, fraction, 
            rate_lower_model, rate_upper_model,df_data,
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
