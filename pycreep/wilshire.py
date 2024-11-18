"""Correlate time dependent data using the Wilshire model
"""

import numpy as np
import scipy.stats

from pycreep import ttp, units, methods

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
        ls_ratio_max (float):       max log stress ratio to allow
        override_Q (None or dict):  if provided, dictionary of heat-specific Q values to use
                                    instead of regressing

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

    def __init__(
        self,
        norm_data,
        *args,
        sign_Q="-",
        allow_avg_norm=True,
        energy_units="kJ/mol",
        predict_norm=None,
        ls_ratio_max=0.99,
        override_Q=None,
        min_time=1e-20,
        **kwargs,
    ):
        # pylint: disable=consider-using-in
        super().__init__(*args, **kwargs)

        self.norm_data = norm_data
        if sign_Q == "-" or sign_Q == -1:
            self.sign_Q = -1.0
        elif sign_Q == "+" or sign_Q == 1:
            self.sign_Q = 1.0
        else:
            raise ValueError(f"Unknown sign_Q value of {sign_Q}")

        self.allow_avg_norm = allow_avg_norm

        self.R = units.convert(
            R, R_units, energy_units + "/" + self.analysis_temp_units
        )

        self.override_Q = override_Q

        self.ls_ratio_max = ls_ratio_max

        if predict_norm is None:
            self.predict_norm = norm_data
        else:
            self.predict_norm = predict_norm

        self.min_time = min_time

    def write_excel_report_to_tab(self, tab):
        """
        Write an excel report to a given tab

        Args:
            tab (openpyxl tab):     tab handle to write to
        """
        tab["A1"] = "Regression results:"
        tab["A2"] = "k:"
        tab["B2"] = self.k
        tab["A3"] = "u:"
        tab["B3"] = self.u
        of = 4
        tab.cell(row=of, column=1, value="Overall Q:")
        tab.cell(row=of, column=2, value=self.Q_avg)
        of += 2

        tab.cell(row=of, column=1, value="Statistics:")
        of += 1
        tab.cell(row=of, column=1, value="R2")
        tab.cell(row=of, column=2, value=self.R2)
        of += 1
        tab.cell(row=of, column=1, value="SEE")
        tab.cell(row=of, column=2, value=self.SEE_avg)
        of += 1
        tab.cell(row=of, column=1, value="SEE (log10 time)")
        tab.cell(row=of, column=2, value=self.SEE_avg_log_time)
        of += 2

        tab.cell(row=of, column=1, value="Heat summary:")
        of += 1
        tab.cell(row=of, column=1, value="Heat")
        tab.cell(row=of, column=2, value="Count")
        tab.cell(row=of, column=3, value="Lot Q")
        tab.cell(row=of, column=4, value="Lot RMS error")
        of += 1

        heat_count = {h: len(i) for h, i in self.heat_indices.items()}

        for heat in sorted(self.Q_heat.keys()):
            tab.cell(row=of, column=1, value=heat)
            tab.cell(row=of, column=2, value=heat_count[heat])
            tab.cell(row=of, column=3, value=self.Q_heat[heat])
            tab.cell(row=of, column=4, value=self.heat_rms[heat])
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
            if heat in self.norm_data.unique_heats:
                y[inds] /= self.norm_data.predict_heat(heat, self.temperature[inds])
            elif (heat not in self.norm_data.unique_heats) and self.allow_avg_norm:
                y[inds] /= self.norm_data.predict(self.temperature[inds])
            else:
                raise ValueError(f"Heat {heat} not in time independent data")

        # In some cases this can happen
        y = np.minimum(self.ls_ratio_max, y)

        # Wilshire correlates on the log of the log
        y = np.log(-np.log(y))

        # Form the (unmodified) x values
        x = np.log(self.time)

        if self.override_Q:
            Qs = np.zeros_like(x)
            for heat in self.unique_heats:
                Qs[self.heat_indices[heat]] = self.override_Q[heat]
            X = np.zeros((len(x), 2))
            X[:, 0] = 1.0
            X[:, 1] = x + self.sign_Q * Qs / (self.R * self.temperature)
            params, preds, self.SSE, self.R2, self.SEE = methods.least_squares(X, y)
            self.Q_heat = self.override_Q
        else:
            # Setup the regression matrix
            X = np.zeros((len(x), 2 + len(self.unique_heats)))
            X[:, 0] = 1.0
            X[:, 1] = x
            for i, heat in enumerate(self.unique_heats):
                inds = self.heat_indices[heat]
                X[inds, 2 + i] = self.sign_Q / (self.R * self.temperature[inds])
            # Solve for the optimal heat-specific Q values and coefficients
            params, preds, self.SSE, self.R2, self.SEE = methods.least_squares(X, y)
            self.Q_heat = {
                heat: params[i + 2] / params[1]
                for i, heat in enumerate(self.unique_heats)
            }

        # Extract the parameter values
        self.k = np.exp(params[0])
        self.u = params[1]
        self.Q_avg = np.sum(
            [
                self.Q_heat[heat] * len(self.heat_indices[heat])
                for i, heat in enumerate(self.unique_heats)
            ]
        ) / len(y)

        # Save the x and y points (with the heat-specific values) for plotting
        Q_full = np.zeros_like(y)
        for heat in self.unique_heats:
            Q_full[self.heat_indices[heat]] = self.Q_heat[heat]
        self.x_points = np.log(self.time) + self.sign_Q * Q_full / (
            self.temperature * self.R
        )
        self.y_points = y

        # Extract heat specific RMS
        self.heat_rms = {
            h: np.sqrt(np.mean((preds[inds] - y[inds]) ** 2.0))
            for h, inds, in self.heat_indices.items()
        }

        # Calculate the all heat/average SEE
        tensile_strength = self.predict_norm.predict(self.temperature)
        y_pred_avg = np.log(
            -np.log(self.predict_stress(self.time, self.temperature) / tensile_strength)
        )

        self.SEE_avg = np.sqrt(np.sum((y - y_pred_avg) ** 2.0) / (len(x) - 3.0))

        # Calculate Sam's SEE metrics
        times_prime = np.nan_to_num(self.predict_time(self.stress, self.temperature))
        self.SEE_avg_log_time = np.sqrt(
            np.sum(
                (
                    np.log10(np.maximum(self.time, self.min_time))
                    - np.log10(np.maximum(times_prime, self.min_time))
                )
                ** 2.0
            )
            / (len(x) - 3.0)
        )

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

    def predict_stress(self, time, temperature, confidence=None):
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

    def predict_time(self, stress, temperature, confidence=None):
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

        delta = self.SEE_avg * h

        tensile_strength = self.predict_norm.predict(temperature)
        y = np.log(-np.log(stress / tensile_strength))

        x = (y - delta - np.log(self.k)) / self.u

        return self.time_from_x(x, temperature)
