from pycreep import units, methods

import numpy as np
import numpy.linalg as la

class TTPAnalysis:
    """
        Superclass for time-temperature parameter (TTP) analysis of a
        dataset

        Args:
            data:                       dataset as a pandas dataframe

        Keyword Args:
            time_field (str):           field in array giving time, default is
                                        "Life (h)"
            temp_field (str):           field in array giving temperature, default
                                        is "Temp (C)"
            stress_field (str):         field in array giving stress, default is
                                        "Stress (MPa)"
            heat_field (str):           filed in array giving heat ID, default is
                                        "Heat/Lot ID"
            input_temp_units (str):     temperature units, default is "C"
            input_stress_units (str):   stress units, default is "MPa"
            input_time_units (str):     time units, default is "hr"
            analysis_temp_units (str):  temperature units for analysis, 
                                        default is "K"
            analysis_stress_units (str):    analysis stress units, default is 
                                            "MPa"
            analysis_time_units (str):  analysis time units, default is "hr"

    """
    def __init__(self, data, time_field = "Life (h)", temp_field = "Temp (C)",
            stress_field = "Stress (MPa)", heat_field = "Heat/Lot ID",
            input_temp_units = "degC", input_stress_units = "MPa", 
            input_time_units = "hrs", analysis_temp_units = "K",
            analysis_stress_units = "MPa", analysis_time_units = "hrs"):
        self.data = data
        self.time_field = time_field
        self.temp_field = temp_field
        self.stress_field = stress_field
        self.heat_field = heat_field

        self.input_temp_units = input_temp_units
        self.input_stress_units = input_stress_units
        self.input_time_units = input_time_units
        
        self.analysis_temp_units = analysis_temp_units
        self.analysis_stress_units = analysis_stress_units
        self.analysis_time_units = analysis_time_units
        
        self.setup_data()

    def setup_data(self):
        """
            Read data into required arrays from dataframe, translate
            units if requested
        """
        self.temperature = units.convert(np.array(self.data[self.temp_field]),
                self.input_temp_units, self.analysis_temp_units)
        self.stress = units.convert(np.array(self.data[self.stress_field]),
                self.input_stress_units, self.analysis_stress_units)
        self.time = units.convert(np.array(self.data[self.time_field]),
                self.input_time_units, self.analysis_time_units)

        self.heats = self.data[self.heat_field]
        self.heat_indices = {hi: self.heats.index[self.heats == hi] 
                for hi in set(self.heats)}

    @property
    def nheats(self):
        return len(self.heat_indices.keys())

    def polynomial_analysis(self, order, lot_centering = False):
        """
            Completes a polynomial correlation between log stress
            and the TTP.

            Args:
                order (int):            polynomial order to use

            Keyword Args:
                lot_centering (bool):   if True lot center the TTP parameter

            Returns:
                dict of results, which includes:
                    * "preds":      predictions for each point
                    * "C_avg":      overall TTP parameter
                    * "C_heat":     dictionary mapping each heat to the 
                                    lot-specific TTP
                    * "poly_avg":   polynomial coefficients for the average
                                    model
                    * "R2":         coefficient of determination
                    * "SSE":        standard squared error
                    * "SEE":        standard error estimate
                    * "SEE_heat":   SEE without lot centering, i.e. if you have a random heat
                    * "R2_heat":    R2 without lot centering, i.e. if you have a random heat
        """
        if lot_centering:
            return self.lot_centered_polynomial_analysis(order)
        else:
            return self.all_lot_polynomial_analysis(order)

    def all_lot_polynomial_analysis(self, order):
        """
            Uncentered polynomial analysis

            Args:
                order:      polynomial order

            Returns:
                standard results dictionary described in polynomial_analysis
        """
        X = np.concatenate((
            np.vander(np.log10(self.stress), N = order + 1) * self.stress_transform()[:,None],
            -np.ones((len(self.stress),1))), axis = 1)
        y = np.log10(self.time)

        b, p, SSE, R2, SEE = methods.least_squares(X, y)

        return {
                "preds": p,
                "C_avg": b[-1],
                "C_heat": {h: b[-1] for h in self.heat_indices.keys()},
                "polyavg": b[:-1],
                "R2": R2,
                "SSE": SSE,
                "SEE": SEE,
                "SEE_heat": SEE,
                "R2_heat": R2
                }
    
    def lot_centered_polynomial_analysis(self, order):
        """
            Lot centered polynomial analysis

            Args:
                order:      polynomial order

            Returns:
                standard results dictionary described in polynomial_analysis
        """
        # Setup the lot matrix
        C = np.zeros((len(self.stress), self.nheats+1))
        C[:,0] = -1.0
        for i, inds in enumerate(self.heat_indices.values()):
            C[inds,i+1] = -1.0

        # Setup the correlation matrix
        X = np.concatenate((
            np.vander(np.log10(self.stress), N = order + 1) * self.stress_transform()[:,None],
            C), axis = 1)
        y = np.log10(self.time)
        
        b, p, SSE, R2, SEE = methods.least_squares(X, y)

        C_avg = sum((b[order+1]+b[order+1+i+1]) * len(inds) for i,(h,inds) in enumerate(self.heat_indices.items())
                ) / len(self.stress)

        # Now go back and calculate the SEE and the R2 values as if you have a random heat
        poly = b[:order+1]
        p_prime = self.predict_polynomial(poly, C_avg)
        e_prime = y - p_prime
        SEE_prime = np.sqrt(np.sum(e_prime**2.0) / (X.shape[0] - order - 2))
        ybar = np.mean(y)
        SST = np.sum((y - ybar)**2.0)
        R2_heat = 1.0 - np.sum(e_prime**2.0) / SST

        return {
                "preds": p,
                "C_avg": C_avg,
                "C_heat": {h: b[order+1]+b[order+1+i+1] for i,h in enumerate(self.heat_indices.keys())},
                "polyavg": poly,
                "R2": R2,
                "SSE": SSE,
                "SEE": SEE,
                "SEE_heat": SEE_prime,
                "R2_heat": R2_heat
                }

class LMPAnalysis(TTPAnalysis):
    """
        Larson Miller analysis of the data

        Args:
            data:                       dataset as a pandas dataframe

        Keyword Args:
            time_field (str):           field in array giving time, default is
                                        "Life (h)"
            temp_field (str):           field in array giving temperature, default
                                        is "Temp (C)"
            stress_field (str):         field in array giving stress, default is
                                        "Stress (MPa)"
            heat_field (str):           filed in array giving heat ID, default is
                                        "Heat/Lot ID"
            input_temp_units (str):     temperature units, default is "C"
            input_stress_units (str):   stress units, default is "MPa"
            input_time_units (str):     time units, default is "hr"
            analysis_temp_units (str):  temperature units for analysis, 
                                        default is "K"
            analysis_stress_units (str):    analysis stress units, default is 
                                            "MPa"
            analysis_time_units (str):  analysis time units, default is "hr"
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def stress_transform(self):
        """
            Multiply stress terms by this to transform
        """
        return 1.0 / self.temperature

    def predict_polynomial(self, poly, C):
        """
            Predict log time for a given value of the polynomial and the TPP parameter

            Parameters:
                poly:   regression polynomial
                C:      TTP parameter

            Returns:
                prediction of log time for each stress
        """
        return np.polyval(poly, np.log10(self.stress)) / self.temperature - C
