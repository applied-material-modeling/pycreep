import unittest
import os.path
import pickle

import numpy as np

from pycreep import data, ttp, time_independent

class PolynomialComparison:
    def _compare(self, a, b):
        """
            Compare key features of two results dictionaries
        """
        scalars = ["C_avg", "R2", "SSE", "SEE", "SEE_heat", "R2_heat"]
        for s in scalars:
            self.assertAlmostEqual(a[s], b[s])

        arrays = ["polyavg", "preds"]
        for name in arrays:
            self.assertTrue(np.allclose(a[name], b[name]))

class Regression304Rupture(unittest.TestCase, PolynomialComparison):
    """
        Do a regression test on a set of 304H data checked against
        an alternative implementation.  This check covers
        batch averaged and non-averaged Larson Miller correlations
    """
    def setUp(self):
        self.data = data.load_data_from_file(os.path.join(
            os.path.dirname(__file__), "304H-rupture.csv"))

        self.centered = pickle.load(open(os.path.join(
            os.path.dirname(__file__), "304H-rupture-centered.pickle"), 'rb'))
        self.uncentered = pickle.load(open(os.path.join(
            os.path.dirname(__file__), "304H-rupture-uncentered.pickle"), 'rb'))

        self.order = 2
        self.TTP = ttp.LarsonMillerParameter()

    def test_lot_centered(self):
        """
            Regression test for lot centered Larson-Miller analysis
        """
        model = ttp.LotCenteredAnalysis(self.TTP, self.order, self.data
                ).analyze()
        
        self._compare(model.report(), self.centered)

    def test_not_centered(self):
        """
            Regression test for uncentered Larson-Miller analysis
        """
        model = ttp.UncenteredAnalysis(self.TTP, self.order, self.data
                ).analyze()
        
        self._compare(model.report(), self.uncentered)

class RegressionGr22Rupture(unittest.TestCase, PolynomialComparison):
    """
        Do a regression test on a set of Gr 22 data checked against an
        alternative implementation.  This check covers region split
        models using batch averaged Larson Miller correlations.
    """
    def setUp(self):
        self.rupture_data = data.load_data_from_file(os.path.join(
            os.path.dirname(__file__), "Gr22-rupture.csv"))
        self.yield_data = data.load_data_from_file(os.path.join(
            os.path.dirname(__file__), "Gr22-yield.csv"))

        self.lower = pickle.load(open(os.path.join(
            os.path.dirname(__file__), "gr22-centered-lower.pickle"), 'rb'))
        self.upper = pickle.load(open(os.path.join(
            os.path.dirname(__file__), "gr22-centered-upper.pickle"), 'rb'))

        self.order = 1
        self.TTP = ttp.LarsonMillerParameter()
        self.fraction = 0.5

    def test_regression(self):
        yield_model = time_independent.TabulatedTimeIndependentCorrelation(
                np.array([27.0,77,130,180,230,280,330,380,430,480,530,580,630,680,730]) + 273.15,
                np.array([301.33,291.03,285.8,284.12,283.73,282.87,279.81,272.93,260.89,242.87,218.85,189.69,157.19,123.75,91.9999]),
                self.yield_data,
                stress_field = "Yield Strength (MPa)", analysis_temp_units="K")
        lower_model = ttp.LotCenteredAnalysis(self.TTP, self.order, self.rupture_data)
        upper_model = ttp.LotCenteredAnalysis(self.TTP, self.order, self.rupture_data)

        model = ttp.SplitAnalysis(yield_model, self.fraction, lower_model, upper_model,
                self.rupture_data).analyze()

        self._compare(model.upper_model.report(), self.upper)
        self._compare(model.lower_model.report(), self.lower)
