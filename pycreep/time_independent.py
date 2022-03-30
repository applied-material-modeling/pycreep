from pycreep import dataset, methods

import numpy as np

class TimeIndependentCorrelation(dataset.DataSet):
    """
        Class used to correlate time independent/temperature dependent
        data as a function of temperature.

        Args:
            data:                       dataset as a pandas dataframe

        Keyword Args:
            temp_field (str):           field in array giving temperature, default
                                        is "Temp (C)"
            stress_field (str):         field in array giving stress, default is
                                        "Stress (MPa)"
            heat_field (str):           field in array giving heat ID, default is
                                        "Heat/Lot ID"
            input_temp_units (str):     temperature units, default is "C"
            input_stress_units (str):   stress units, default is "MPa"
            analysis_temp_units (str):  temperature units for analysis, 
                                        default is "K"
            analysis_stress_units (str):    analysis stress units, default is 
                                            "MPa"

    """
    def __init__(self, data, temp_field = "Temp (C)",
            stress_field = "Stress (MPa)", heat_field = "Heat/Lot ID",
            input_temp_units = "degC", input_stress_units = "MPa", 
            analysis_temp_units = "K",
            analysis_stress_units = "MPa"):
        super().__init__(data)

        self.add_field_units("temperature", temp_field, input_temp_units, 
                analysis_temp_units)
        self.add_field_units("stress", stress_field, input_stress_units,
                analysis_stress_units)

        self.add_heat_field(heat_field)

    def polynomial_analysis(self, deg):
        """
            For now, until I have a better idea of what to do, this just
            returns a polynomial relating strength to temperature
            using all the data

            Parameters:
                deg:        degree of polynomial correlation
        """
        X = np.vander(self.temperature, deg + 1)
        b, p, SSE, R2, SEE = methods.least_squares(X, self.stress)

        return {
                "preds": p,
                "polyavg": b,
                "R2": R2,
                "SSE": SSE,
                "SEE": SEE
                }

