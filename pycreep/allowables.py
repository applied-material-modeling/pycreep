"""Calculates ASME-style allowable stresses from underlying correlations
"""

import numpy as np

from pycreep import units


def Sc_SectionII_1A_1B(
    rupture_model, rate_model, temperatures, conf=0.90, dt=1e-6, full_results=False
):
    """
    Calculate the creep-controlled allowable stress for Tables 1A and
    1B in Section II.  This stress is the lesser of

    1. 100% the average stress to  produce a creep rate of 0.01%/1000 hours,
       implemented here as the stress to cause a minimum creep rate of
       1e-5%/hour.
    2. 100Favg% of the average stress to cause rupture at 100,000 hours
    3. 80% of the minimum stress to cause rupture at 100,000 hours, where
       minimum here is interpreted to mean the predicted rupture stress
       at 100,000 hours for a given conference interval on the model

    Favg is defined as:

    1. 0.67 below 815 C
    2. log Favg = 1/n where n is the slope of the log time-to-rupture
       versus stress plot at 100,000 hours, but not greater than 0.67

    Args:
        rupture_model (pycreep.ttp.TTPAnalysis):    rupture correlation
        rate_model (pycreep.ttp.TTPAnalysis):       rate correlation
        temperatures (np.array):                    temperature values

    Keyword Args:
        conf:           desired confidence interval
        dt:             finite difference value to calculate n
        full_results:   if True, return a detailed results dictionary
    """
    # Calculate the values of n and Favg
    times = np.ones_like(temperatures) * 100000.0
    n = (
        np.log10(rupture_model.predict_stress((1.0 + dt) * times, temperatures))
        - np.log10(rupture_model.predict_stress(times, temperatures))
    ) / (np.log10((1.0 + dt) * times) - np.log10(times))
    Favg = 10.0**n
    thresh = units.convert(
        np.array([815.0]), "degC", rupture_model.analysis_temp_units
    )[0]
    Favg[temperatures < thresh] = 0.67
    Favg = np.minimum(Favg, 0.67)

    # Calculate Savg and Favg*Savg
    Savg = rupture_model.predict_stress(times, temperatures)
    FSavg = Favg * Savg

    # Calculate Smin and 80%*Smin
    Smin = rupture_model.predict_stress(times, temperatures, confidence=conf)
    FSmin = 0.80 * Smin

    # Calculate the creep rate stresses
    rates = np.ones_like(temperatures) * 1e-5
    Sc = rate_model.predict_stress(rates, temperatures)

    # Calculate the allowable
    S = np.minimum(FSavg, np.minimum(FSmin, Sc))

    if full_results:
        return {
            "Savg": Savg,
            "n": -1.0 / n,
            "Favg": Favg,
            "FSavg": FSavg,
            "Smin": Smin,
            "FSmin": FSmin,
            "Sc": Sc,
            "S": S,
        }
    return S
