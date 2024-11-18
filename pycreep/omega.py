"""Fits MPC Omega type models
"""

import numpy as np
import scipy.stats

from pycreep import ttp


class MPCOmega:
    """MPC Omega model for remaining life

    Args:
        rupture_data (dataset.Dataset): creep rupture data
        rate_data (dataset.Dataset): creep rate data set

    Keyword Args:
        conf (float): confidence level for determining delta, default 0.9
        order (int): polynomial order, default 4
    """

    def __init__(
        self,
        rupture_data,
        rate_data,
        conf=0.9,
        order=4,
        rupture_time_field="Life (h)",
        creep_rate_field="Creep rate (%/hr)",
        temp_field="Temp (C)",
        stress_field="Stress (MPa)",
        heat_field="Heat/Lot ID",
        input_temp_units="degC",
        input_stress_units="MPa",
        input_time_units="hrs",
        analysis_temp_units="K",
        analysis_stress_units="MPa",
        analysis_time_units="hrs",
    ):
        self.rupture_model = ttp.LotCenteredAnalysis(
            ttp.LarsonMillerParameter(),
            order,
            rupture_data,
            time_field=rupture_time_field,
            temp_field=temp_field,
            stress_field=stress_field,
            heat_field=heat_field,
            input_temp_units=input_temp_units,
            input_stress_units=input_stress_units,
            input_time_units=input_time_units,
            analysis_temp_units=analysis_temp_units,
            analysis_stress_units=analysis_stress_units,
            analysis_time_units=analysis_time_units,
        ).analyze()
        self.rate_model = ttp.LotCenteredAnalysis(
            ttp.LarsonMillerParameter(),
            order,
            rate_data,
            time_field=creep_rate_field,
            temp_field=temp_field,
            stress_field=stress_field,
            heat_field=heat_field,
            input_temp_units=input_temp_units,
            input_stress_units=input_stress_units,
            input_time_units=input_time_units,
            analysis_temp_units=analysis_temp_units,
            analysis_stress_units=analysis_stress_units,
            analysis_time_units=analysis_time_units,
            time_sign=-1.0,
        ).analyze()

        # Factor on the SEE
        f = scipy.stats.norm.interval(conf)[1]

        # Extract the coefficients
        self.A0 = -self.rate_model.C_avg
        self.D1 = -f * self.rate_model.SEE_heat
        self.A = self.rate_model.polyavg

        self.B0 = -self.rate_model.C_avg + self.rupture_model.C_avg
        self.D2 = f * (-self.rate_model.SEE_heat + self.rupture_model.SEE_heat)
        self.B = self.rate_model.polyavg - self.rupture_model.polyavg

    def eps0(self, stress, T, lb=False):
        """Calculate the reference strain rate as a function of stress and temperature

        Args:
            stress (np.array): input stress
            T (np.array): input absolute temperature
        """
        R = self.A0
        if lb:
            R += self.D1
        return 10.0 ** (-(R + np.polyval(self.A, np.log10(stress)) / T))

    def omega(self, stress, T, lb=False):
        """Calculate Omega as a function of stress and temperature

        Args:
            stress (np.array): input stress
            T (np.array): input absolute temperature

        Keyword Args:
            lb (bool): if true use lower bound properties
        """
        R = self.B0
        if lb:
            R += self.D2
        return 10.0 ** (R + np.polyval(self.B, np.log10(stress)) / T)

    def rupture(self, stress, T, lb=False):
        """Calculate rupture time starting from zero life

        Args:
            stress (np.array): input stress
            T (np.array): absolute temperature

        Keyword Args:
            lb (bool): if true use lower bound properties
        """
        e0 = self.eps0(stress, T, lb=lb)
        o = self.omega(stress, T, lb=lb)

        return 1.0 / (e0 * o)

    def remaining_life(self, stress, T, rate, lb=False):
        """Calculate remaining life

        Args:
            stress (np.array): input stress
            T (np.array): absolute temperature
            rate (np.array): current strain rate

        Keyword Args:
            lb (bool): if true use lower boundary properties
        """
        o = self.omega(stress, T, lb=lb)
        return 1.0 / (rate * o)
