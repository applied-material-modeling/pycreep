from pycreep import ttp, units, methods

import numpy as np
import scipy.stats

# Universal gas constant
R = 8.3145
R_units = "J/(mol*K)"

class WilshireAnalysis(ttp.TTPAnalysis):
    """
        Lot-centered Wilshire analysis of a creep data set.

        Args:
            norm_data:                  time-independent data to normalize on, often tensile strength
            creep_data:                 creep rupture, deformation time, or creep strain rate data

        Keyword Args:
            sign_Q (str):               what sign to use in the Arrhenius term, default "-"
            allow_avg_norm (str):       if True, fall back on the all-heat average correlation for
                                        normalization
            energy_units (str):         units for activation energy, default "kJ/mol"
            Q_guess (float):            guess at average activation energy
            Q_mult (float):             upper bound on Q = Q_guess * Q_mult
            Q_dev (float):              bounds for heat-specific Q, [-Q_dev * Q_guess, Q_dev * Q_guess]
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
            predict_norm:               strength object to use for predictions, defaults to norm_data

        The setup and analyzed objects are suppose to maintain the following properties:
            * "preds":      predictions for each point
            * "Q_avg":      overall activation energy
            * "Q_heat":     dictionary mapping each heat to the 
                            lot-specific activation energy
            * "k"           average intercept
            * "u"           average slope
            * "R2":         coefficient of determination
            * "SSE":        standard squared error
            * "SEE":        standard error estimate
    """
    def __init__(self, norm_data, *args, sign_Q = "-",  allow_avg_norm = True,
            energy_units = "kJ/mol", Q_guess = 250.0, Q_mult = 10.0, Q_dev = 0.25,
            predict_norm = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.norm_data = norm_data 
        if sign_Q == "-" or sign_Q == -1:
            self.sign_Q = -1.0
        elif sign_Q == "+" or sign_Q == 1:
            self.sign_Q = 1.0
        else:
            raise ValueError("Unknown sign_Q value of %s" % sign_Q)

        self.allow_avg_norm = allow_avg_norm

        self.R = units.convert(R, R_units, energy_units+"/"+self.analysis_temp_units)

        self.Q_guess = Q_guess
        self.Q_mult = Q_mult
        self.Q_dev = Q_dev

        if predict_norm is None:
            self.predict_norm = norm_data
        else:
            self.predict_norm = predict_norm

    def _write_excel_report(self, tab):
        """
            Write an excel report to a given tab 

            Args:
                tab (openpyxl tab):     tab handle to write to
        """
        tab['A1'] = "Regression results:"
        tab['A2'] = "k:"
        tab['B2'] = self.k
        tab['A3'] = "u:"
        tab['B3'] = self.u
        of = 4
        tab.cell(row=of, column=1, value = "Overall Q:")
        tab.cell(row=of, column=2, value = self.Q_avg)
        of += 2

        tab.cell(row=of, column = 1, value = "Statistics:")
        of += 1
        tab.cell(row=of, column = 1, value = "R2")
        tab.cell(row=of, column = 2, value = self.R2)
        of += 1
        tab.cell(row=of, column = 1, value = "SEE")
        tab.cell(row=of, column = 2, value = self.SEE)
        of += 2

        tab.cell(row=of, column = 1, value = "Heat summary:")
        of += 1
        tab.cell(row=of, column = 1, value = "Heat")
        tab.cell(row=of, column = 2, value = "Count")
        tab.cell(row=of, column = 3, value = "Lot Q")
        tab.cell(row=of, column = 4, value = "Lot RMS error")
        of +=1 
        
        heat_count = {h: len(i) for h,i in self.heat_indices.items()}

        for heat in sorted(self.Q_heat.keys()):
            tab.cell(row=of, column = 1, value = heat)
            tab.cell(row=of, column = 2, value = heat_count[heat])
            tab.cell(row=of, column = 3, value = self.Q_avg + self.Q_heat[heat])
            tab.cell(row=of, column = 4, value = self.heat_rms[heat])
            of += 1

    def analyze(self):
        """
            Run or rerun analysis
        """
        # Make sure the normalization model is current
        self.norm_data.analyze()
        self.predict_norm.analyze()

        # Form the normalized stresses 
        y = np.copy(self.stress)
        for heat in self.unique_heats:
            inds = self.heat_indices[heat]
            if (heat in self.norm_data.unique_heats):
                y[inds] /= self.norm_data.predict_heat(heat, self.temperature[inds]) 
            elif (heat not in self.norm_data.unique_heats) and self.allow_avg_norm:
                y[inds] /= self.norm_data.predict(self.temperature[inds])
            else:
                raise ValueError("Heat %s not in time independent data" % heat)
        
        # Wilshire correlates on the log of the log
        y = np.log(-np.log(y)) 

        # Form the (unmodified) x values 
        x = np.log(self.time)

        # Function which maps the unmodified x values to the actual x values for regression
        def map_fn(xp, X):
            Q_mean = X[0]
            Q_vals = np.full((len(xp),), Q_mean)
            for i,heat in enumerate(self.unique_heats):
                inds = self.heat_indices[heat]
                Q_vals[inds] += X[1+i]

            return xp + self.sign_Q* Q_vals / (self.temperature * self.R)

        # A guess at Q, this could be difficult
        X0 = np.zeros((self.nheats+1))
        X0[0] = self.Q_guess

        bounds = [(0,self.Q_mult*self.Q_guess)] + [(-self.Q_guess*self.Q_dev,self.Q_guess*self.Q_dev)] * self.nheats
        
        # Solve for the optimal heat-specific Q values and coefficients 
        self.X, self.coefs, preds, self.SSE, self.R2, self.SEE = methods.optimize_polynomial_fit(x, y, 1, X0, bounds, map_fn)

        # Save the x and y coordinates of each point 
        self.x_points = map_fn(x, self.X)
        self.y_points = y

        # Extract the heat specific Q values and the "actual" coefficients
        self.Q_avg = self.X[0]
        self.Q_heat = {heat: self.X[i+1] for i,heat in enumerate(self.unique_heats)}

        self.k = np.exp(self.coefs[1])
        self.u = self.coefs[0]

        self.heat_rms = {h: np.sqrt(np.mean((preds[inds] - y[inds])**2.0)) for 
                h,inds, in self.heat_indices.items()}

        return self

    def make_x(self, time, temperature):
        """
            Transform time and temperature to the ordinate
        """
        return np.log(time) + self.sign_Q * self.Q_avg / (temperature * self.R)

    def time_from_x(self, x, temperature):
        """
            Recover times from x values and temperature
        """
        return np.exp(x - self.sign_Q * self.Q_avg / (temperature * self.R))

    def predict_stress(self, time, temperature, confidence = None):
        """
            Predict new stress given time and temperature

            Args:
                time:           input time values
                temperature:    input temperature values

            Keyword Args:
                confidence:     confidence interval, if None provide
                                average values
        """
        if confidence is None:
            h = 0.0
        else:
            h = scipy.stats.norm.interval(confidence)[1]
        
        delta = self.SEE * h

        y = np.log(self.k) + self.u * self.make_x(time, temperature) + delta 

        sr = np.exp(-np.exp(y))

        tensile_strength = self.predict_norm.predict(temperature)

        return sr * tensile_strength

    def predict_time(self, stress, temperature, confidence = None):
        """
            Predict new times given stress and temperature 

            Args:
                stress:         input stress values 
                temperature:    input temperature values 

            Keyword Args:
                confidence:     confidence interval, if None 
                                provide average values
        """
        if confidence is None:
            h = 0.0
        else:
            h = scipy.stats.norm.interval(confidence)[1]
        
        delta = self.SEE * h

        tensile_strength = self.predict_norm.predict(temperature)
        y = np.log(-np.log(stress/tensile_strength))
        
        x = (y - delta - np.log(self.k)) / self.u
        
        return self.time_from_x(x, temperature)
