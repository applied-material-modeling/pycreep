from pycreep import dataset, methods

import numpy as np
import scipy.interpolate as inter

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

    def __call__(self, T):
        """
            Alias for self.predict(T)

            Args:
                T:      temperature data
        """
        return self.predict(T)

class PolynomialTimeIndependentCorrelation(TimeIndependentCorrelation):
    """
        Class used to correlate time independent/temperature dependent
        data as a function of temperature using polynomial regression.

        Args:
            deg:                        polynomial degree
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
    def __init__(self, deg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deg = deg
        self.heat_correlations = {}
        self.heat_correlation_data = {}

    def analyze(self):
        """
            Run the stress analysis and store results
        """
        # Overall correlation 
        self.polyavg, self.preds, self.SSE, self.R2, self.SEE = methods.polynomial_fit(
                self.temperature, self.stress, self.deg)

        # Heat-specific correlations 
        for heat in self.unique_heats:
            polyavg, preds, SSE, R2, SEE = methods.polynomial_fit(
                    self.temperature[self.heat_indices[heat]],
                    self.stress[self.heat_indices[heat]], self.deg)
            self.heat_correlations[heat] = polyavg 
            self.heat_correlation_data[heat] = (preds, SSE, R2, SEE)

        return self

    def predict(self, T):
        """
            Predict some new values as a function of temperature

            Args:
                T:      temperature data
        """
        return np.polyval(self.polyavg, T)

    def predict_heat(self, heat, T):
        """
            Predict heat-specific values as a function of temperature 

            Args:
                heat:   heat ID
                T:      temperature
        """
        return np.polyval(self.heat_correlations[heat], T)

class TabulatedTimeIndependentCorrelation(TimeIndependentCorrelation):
    """
        Class used to correlate time independent/temperature dependent
        data as a function of temperature using a user-provided table
        of values

        Args:
            temp_table:                 temperature table values
            stress_table:               stress table values
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
    def __init__(self, temp_table, stress_table, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.temp_table = temp_table
        self.stress_table = stress_table

    def analyze(self):
        """
            Run the stress analysis and store results
        """
        self.fn = inter.interp1d(self.temp_table, self.stress_table)

        return self

    def predict(self, T):
        """
            Predict some new values as a function of temperature

            Args:
                T:      temperature data
        """
        return self.fn(T)
