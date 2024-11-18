import unittest
import os.path

import numpy as np

from pycreep import data, omega, ttp


class TestOmega(unittest.TestCase):
    """
    Check basics for Omega model
    """

    def setUp(self):
        self.data = data.load_data_from_file(
            os.path.join(os.path.dirname(__file__), "bilinear.csv")
        )

        self.omega_model = omega.MPCOmega(self.data, self.data)

        self.Trange = np.linspace(774, 1022, 10)
        self.srange = np.linspace(250.0, 120, 10)

    def test_match_rate(self):
        base_rate = ttp.LotCenteredAnalysis(
            ttp.LarsonMillerParameter(),
            4,
            self.data,
            time_sign=-1.0,
            time_field="Creep rate (%/hr)",
        ).analyze()

        base_predictions = base_rate.predict_time(self.srange, self.Trange)
        omega_predictions = self.omega_model.eps0(self.srange, self.Trange)

        self.assertTrue(np.allclose(base_predictions, omega_predictions))

    def test_match_time(self):
        base_time = ttp.LotCenteredAnalysis(
            ttp.LarsonMillerParameter(), 4, self.data
        ).analyze()

        base_predictions = base_time.predict_time(self.srange, self.Trange)
        omega_predictions = self.omega_model.rupture(self.srange, self.Trange)

        self.assertTrue(np.allclose(base_predictions, omega_predictions))

    def test_match_rate_lb(self):
        base_rate = ttp.LotCenteredAnalysis(
            ttp.LarsonMillerParameter(),
            4,
            self.data,
            time_sign=-1.0,
            time_field="Creep rate (%/hr)",
        ).analyze()

        base_predictions = base_rate.predict_time(
            self.srange, self.Trange, confidence=0.90
        )
        omega_predictions = self.omega_model.eps0(self.srange, self.Trange, lb=True)

        self.assertTrue(np.allclose(base_predictions, omega_predictions))

    def test_match_time_lb(self):
        base_time = ttp.LotCenteredAnalysis(
            ttp.LarsonMillerParameter(), 4, self.data
        ).analyze()

        base_predictions = base_time.predict_time(
            self.srange, self.Trange, confidence=0.9
        )
        omega_predictions = self.omega_model.rupture(self.srange, self.Trange, lb=True)

        self.assertTrue(np.allclose(base_predictions, omega_predictions))
