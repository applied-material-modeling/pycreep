# pylint: disable=too-few-public-methods
"""Correlate fatigue data into fatigue curves"""

import numpy as np

from pycreep import methods, dataset


class FatigueAnalysis(dataset.DataSet):
    """
    Superclass for analysis of fatigue data

    Args:
        data:                       dataset as a pandas dataframe

    Keyword Args:
        cycles_field (str):         field in array giving cycles to use, default is
                                    "Cycles"
        temp_field (str):           field in array giving temperature, default
                                    is "Temp (C)"
        strain_range_field (str):   field in array giving strain range, default is
                                    "Strain range"
        r_ratio_field (str):        field in array giving the R ratio, default is "R"
        heat_field (str):           field in array giving heat ID, default is
                                    "Heat/Lot ID"
        input_temp_units (str):     temperature units, default is "C"
        analysis_temp_units (str):  temperature units for analysis,
                                    default is "C"
    """

    def __init__(
        self,
        data,
        cycles_field="Cycles",
        temp_field="Temp (C)",
        strain_range_field="Strain range",
        r_ratio_field="R",
        heat_field="Heat/Lot ID",
        input_temp_units="degC",
        analysis_temp_units="degC",
    ):
        super().__init__(data)

        self.add_field_units("cycles", cycles_field, "", "")
        self.add_field_units(
            "temperature", temp_field, input_temp_units, analysis_temp_units
        )
        self.add_field_units("strain_range", strain_range_field, "", "")
        self.add_field_units("r", r_ratio_field, "", "")
        self.add_heat_field(heat_field)

        self.analysis_temp_units = analysis_temp_units


class LumpedTemperatureFatigueAnalysis(FatigueAnalysis):
    """
    Fatigue analysis binning data by temperature

    Args:
        method:                     method to use to correlate strain range to cycles
        temperature_bins:           list of temperature bins to use for analysis
        data:                       dataset as a pandas dataframe

    Keyword Args:
        temperature_range (float):  range of temperatures on either side of the bins
                                    to collect, default 50
        cycles_field (str):         field in array giving cycles to use, default is
                                    "Cycles"
        temp_field (str):           field in array giving temperature, default
                                    is "Temp (C)"
        strain_range_field (str):   field in array giving strain range, default is
                                    "Strain range"
        r_ratio_field (str):        field in array giving the R ratio, default is "R"
        heat_field (str):           filed in array giving heat ID, default is
                                    "Heat/Lot ID"
        input_temp_units (str):     temperature units, default is "C"
        analysis_temp_units (str):  temperature units for analysis,
                                    default is "C"
    """

    def __init__(
        self, method, temperature_bins, *args, temperature_range=50.0, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.method = method
        self.temperature_bins = sorted(temperature_bins)
        self.temperature_range = temperature_range
        self.fields["temperature_groups"] = lambda self: {
            T: np.where(
                np.logical_and(
                    self.temperature < T + self.temperature_range,
                    self.temperature > T - self.temperature_range,
                )
            )[0]
            for T in self.temperature_bins
        }

        for T, inds in self.temperature_groups.items():
            if len(inds) == 0:
                raise ValueError(f"No data found for temperature {T} C!")

    def analyze(self):
        """
        Analyze by fitting the methods to the data
        """
        self.submodels = {
            T: self.method(self.strain_range[inds], self.cycles[inds])
            for T, inds in self.temperature_groups.items()
        }
        return self

    def predict(self, temperature, erange):
        """
        Predict the number of cycles to failure given the temperature and strain range

        Args:
            temperature (array like):   temperature values to predict for
            erange (array like):        strain range values to predict for
        """
        preds = np.zeros_like(erange)

        for i, (T, de) in enumerate(zip(temperature, erange)):
            mi = methods.find_nearest_index(self.temperature_bins, T)
            preds[i] = self.submodels[self.temperature_bins[mi]].predict(de)

        return preds


class DiercksEquation:
    """
    Diercks method:

    1/sqrt(log10(Nf)) = p(log10(strain_range))

    Args:
        order (int):           polynomial order to use for the regression
    """

    def __init__(self, order):
        self.order = order

    def __call__(self, strain_range, cycles):
        lr = np.log10(strain_range)
        lc = 1.0 / np.sqrt(np.log10(cycles))
        return DiercksFit(np.polyfit(lr, lc, self.order))


class DiercksFit:
    """
    Actual method to predict fatigue with a Diercks equation.

    Args:
        p (np.array):    polynomial coefficients for the Diercks equation
    """

    def __init__(self, p):
        self.p = p

    def predict(self, strain_range):
        """
        Predict the number of cycles to failure given the strain range
        Args:
            strain_range (array like):   strain range values to predict for
        """
        A = np.polyval(self.p, np.log10(strain_range))
        return 10 ** ((1 / A) ** 2)
