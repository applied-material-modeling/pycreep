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

class RegressionQuadraticRupture(unittest.TestCase, PolynomialComparison):
    """
        Do a regression test on a set of synthetic data checked against
        an alternative implementation.  This check covers
        batch averaged and non-averaged Larson Miller correlations
    """
    def setUp(self):
        self.data = data.load_data_from_file(os.path.join(
            os.path.dirname(__file__), "quadratic.csv"))

        self.centered = pickle.load(open(os.path.join(
            os.path.dirname(__file__), "quadratic-centered.pickle"), 'rb'))
        self.uncentered = pickle.load(open(os.path.join(
            os.path.dirname(__file__), "quadratic-uncentered.pickle"), 'rb'))

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
