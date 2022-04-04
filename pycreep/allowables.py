def Sc_SectionII_1A_1B(rupture_model, rate_model, temperatures, conf = 0.90):
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
            conf:       desired confidence interval
    """
    # Calculate the values of n
    pass
