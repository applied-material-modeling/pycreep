import unittest
import os.path
import pickle

import numpy as np

from pycreep import data, ttp, time_independent, allowables


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
        self.data = data.load_data_from_file(
            os.path.join(os.path.dirname(__file__), "quadratic.csv")
        )

        self.centered = pickle.load(
            open(
                os.path.join(os.path.dirname(__file__), "quadratic-centered.pickle"),
                "rb",
            )
        )
        self.uncentered = pickle.load(
            open(
                os.path.join(os.path.dirname(__file__), "quadratic-uncentered.pickle"),
                "rb",
            )
        )

        self.order = 2
        self.TTP = ttp.LarsonMillerParameter()

    def test_lot_centered(self):
        """
        Regression test for lot centered Larson-Miller analysis
        """
        model = ttp.LotCenteredAnalysis(self.TTP, self.order, self.data).analyze()

        self._compare(model.report(), self.centered)

    def test_not_centered(self):
        """
        Regression test for uncentered Larson-Miller analysis
        """
        model = ttp.UncenteredAnalysis(self.TTP, self.order, self.data).analyze()

        self._compare(model.report(), self.uncentered)


class RegressionBilinearRupture(unittest.TestCase, PolynomialComparison):
    """
    Do a regression test on a set of synthetic data checked against
    an alternative implementation.  This check covers region split
    models.
    """

    def setUp(self):
        self.data = data.load_data_from_file(
            os.path.join(os.path.dirname(__file__), "bilinear.csv")
        )

        self.upper = pickle.load(
            open(os.path.join(os.path.dirname(__file__), "bilinear-upper.pickle"), "rb")
        )
        self.lower = pickle.load(
            open(os.path.join(os.path.dirname(__file__), "bilinear-lower.pickle"), "rb")
        )

        self.order = 1
        self.TTP = ttp.LarsonMillerParameter()
        Trange = np.array(
            [
                773.15,
                791.00714286,
                808.86428571,
                826.72142857,
                844.57857143,
                862.43571429,
                880.29285714,
                898.15,
                916.00714286,
                933.86428571,
                951.72142857,
                969.57857143,
                987.43571429,
                1005.29285714,
                1023.15,
            ]
        )
        ydata = np.array(
            [
                332.5,
                325.71428571,
                318.92857143,
                312.14285714,
                305.35714286,
                298.57142857,
                291.78571429,
                285.0,
                278.21428571,
                271.42857143,
                264.64285714,
                257.85714286,
                251.07142857,
                244.28571429,
                237.5,
            ]
        )

        self.yield_model = time_independent.TabulatedTimeIndependentCorrelation(
            Trange, ydata, None, input_temp_units="K"
        )
        self.frac = 0.5

    def test_upper(self):
        """
        Regression test for the upper stress regime
        """
        rupture_lower_model = ttp.LotCenteredAnalysis(self.TTP, self.order, self.data)
        rupture_upper_model = ttp.LotCenteredAnalysis(self.TTP, self.order, self.data)
        rupture_model = ttp.SplitAnalysis(
            self.yield_model,
            self.frac,
            rupture_lower_model,
            rupture_upper_model,
            self.data,
        ).analyze()

        self._compare(rupture_model.upper_model.report(), self.upper)

    def test_lower(self):
        """
        Regression test for the lower stress regime
        """
        rupture_lower_model = ttp.LotCenteredAnalysis(self.TTP, self.order, self.data)
        rupture_upper_model = ttp.LotCenteredAnalysis(self.TTP, self.order, self.data)
        rupture_model = ttp.SplitAnalysis(
            self.yield_model,
            self.frac,
            rupture_lower_model,
            rupture_upper_model,
            self.data,
        ).analyze()

        self._compare(rupture_model.lower_model.report(), self.lower)


class RegressionBilinearAllowables(unittest.TestCase):
    """
    Regression check on the complete process of generating Section II
    allowable stresses
    """

    def setUp(self):
        self.data = data.load_data_from_file(
            os.path.join(os.path.dirname(__file__), "bilinear.csv")
        )

        self.order = 1
        self.TTP = ttp.LarsonMillerParameter()
        Trange = np.array(
            [
                773.15,
                791.00714286,
                808.86428571,
                826.72142857,
                844.57857143,
                862.43571429,
                880.29285714,
                898.15,
                916.00714286,
                933.86428571,
                951.72142857,
                969.57857143,
                987.43571429,
                1005.29285714,
                1023.15,
            ]
        )
        ydata = np.array(
            [
                332.5,
                325.71428571,
                318.92857143,
                312.14285714,
                305.35714286,
                298.57142857,
                291.78571429,
                285.0,
                278.21428571,
                271.42857143,
                264.64285714,
                257.85714286,
                251.07142857,
                244.28571429,
                237.5,
            ]
        )

        self.yield_model = time_independent.TabulatedTimeIndependentCorrelation(
            Trange, ydata, None, input_temp_units="K"
        )
        self.frac = 0.5

        rupture_lower_model = ttp.LotCenteredAnalysis(self.TTP, self.order, self.data)
        rupture_upper_model = ttp.LotCenteredAnalysis(self.TTP, self.order, self.data)
        self.rupture_model = ttp.SplitAnalysis(
            self.yield_model,
            self.frac,
            rupture_lower_model,
            rupture_upper_model,
            self.data,
        ).analyze()

        rate_lower_model = ttp.UncenteredAnalysis(
            self.TTP, self.order, self.data, time_field="Creep rate (%/hr)"
        )
        rate_upper_model = ttp.UncenteredAnalysis(
            self.TTP, self.order, self.data, time_field="Creep rate (%/hr)"
        )
        self.rate_model = ttp.SplitAnalysis(
            self.yield_model,
            self.frac,
            rate_lower_model,
            rate_upper_model,
            self.data,
            time_field="Creep rate (%/hr)",
        ).analyze()

    def test_regression(self):
        """
        Regression test on allowable stresses
        """
        Ts = np.array([400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650]) + 273.15
        res = allowables.Sc_SectionII_1A_1B(self.rupture_model, self.rate_model, Ts)

        regression = np.array(
            [
                6.56602502e02,
                5.04290857e02,
                3.87310843e02,
                2.97466605e02,
                2.28463475e02,
                9.44485410e01,
                2.59208936e01,
                7.11384971e00,
                1.95235775e00,
                5.35814075e-01,
                1.47051288e-01,
            ]
        )

        self.assertTrue(np.allclose(res, regression))
