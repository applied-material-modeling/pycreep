import unittest
import os.path

import numpy as np

from pycreep import data, ttp, gpr


class TestGPR(unittest.TestCase):
    """
    Check basics for GPR model
    """

    def setUp(self):
        self.data = data.load_data_from_file(
            os.path.join(os.path.dirname(__file__), "quadratic_perfect.csv")
        )

        self.gpr_model = gpr.GPRLMPModel(self.data).analyze()

        self.temps = self.data["Temp (C)"].values + 273.15
        self.stresses = self.data["Stress (MPa)"].values

    def test_match_mean(self):
        base_model = ttp.LotCenteredAnalysis(
            ttp.LarsonMillerParameter(),
            2,
            self.data,
        ).analyze()

        base_predictions = base_model.predict_time(self.stresses, self.temps)
        gpr_predictions = self.gpr_model.predict_time(self.stresses, self.temps)

        diff = np.log10(base_predictions) - np.log10(gpr_predictions)
        rerr = np.abs(diff) / np.abs(np.log10(base_predictions))

        self.assertTrue(np.all(rerr < 0.35))
