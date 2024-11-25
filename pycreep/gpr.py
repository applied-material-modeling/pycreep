"""Implements Larson-Miller Gaussian process regression
"""

import numpy as np

import scipy.stats as ss

import torch
import pyro
import pyro.contrib.gp as gp
from torch.distributions import constraints

from pyro.contrib import gp
from pyro.contrib.gp.kernels.kernel import Kernel
from pyro.nn.module import PyroParam

from pycreep import ttp

from tqdm import trange


class LMKernel(Kernel):
    """Kernel used for Gaussian process LM regression

    Args:
        heat_counts (list): number of samples in each heat

    Keyword Args:
        lengthscale (torch.tensor): kernel length scale
        variance (torch.tensor): kernel variance
        C_mean (torch.tensor): initial guess at the mean of the Larson-Miller parameter
        C_variance (torch.tensor): initial guess at the variance of the Larson-Miller parameter

    Input order for parameters in X is [temperature, time, heat_index]

    -1 indicates we want to use the all heat average, rather than heat-specific information
    """

    def __init__(
        self,
        heat_counts,
        lengthscale=torch.tensor(1.0),
        variance=torch.tensor(0.01),
        C_mean=torch.tensor(20.0),
        C_variance=torch.tensor(0.01),
    ):
        super().__init__(3, None)

        self.heat_counts = torch.tensor(heat_counts)
        self.lengthscale = PyroParam(lengthscale, constraints.positive)
        self.variance = PyroParam(variance, constraints.positive)

        self.nheats = len(heat_counts)
        self.heat_weight = self.heat_counts / torch.sum(self.heat_counts)
        self.C_mean = PyroParam(C_mean, constraints.positive)
        self.C_variance = PyroParam(C_variance, constraints.positive)
        self.delta_C = PyroParam(torch.full((self.nheats,), 0.0))

    def _d2(self, X, Z):
        """
        Squared distance calculator, cribbed from the base pyro source
        """
        X = self._slice_input(X)
        Z = self._slice_input(Z)

        X2 = (X**2.0).sum(1, keepdim=True)
        Z2 = (Z**2.0).sum(1, keepdim=True)
        XZ = X.matmul(Z.t())
        r2 = X2 - 2 * XZ + Z2.t()
        return r2.clamp(min=0)

    def heat_variation(self):
        """
        Add zero to the start of the lot constants to include the all-heat average
        """
        return torch.cat([torch.zeros((1,)), self.delta_C])

    def calculate_mean_var(self, X):
        """
        Calculate the mean and variance of the lot constant

        Args:
            X (torch.tensor): input data, shape (N,3)
        """
        heat_id = X[:, 2].long()
        C_mean = self.C_mean + self.heat_variation()[heat_id + 1]

        mean_X = X[:, 0] * (C_mean + torch.log10(X[:, 1]))
        var_X = X[:, 0] ** 2.0 * self.C_variance

        return mean_X, var_X

    def forward(self, X, Z=None, diag=False):
        """
        Forward method for the kernel
        """
        if Z is None:
            Z = X

        mean_X, var_X = self.calculate_mean_var(X)
        mean_Z, var_Z = self.calculate_mean_var(Z)

        scov = torch.outer(var_X, var_Z)

        scaled_r2 = torch.exp(
            -0.5 * self._d2(mean_X, mean_Z) / (scov + self.lengthscale)
        )
        sf = self.variance / torch.sqrt(torch.abs(1.0 + scov / self.lengthscale))

        mat = sf * scaled_r2

        if diag:
            return mat.diag()
        return mat


class GPRLMPModel(ttp.TTPAnalysis):
    """
    Parent class for Gaussian process regression Larson-Miller models

    Args:
        data:                       dataset as a pandas dataframe

    Keyword Args:
        uncentered (bool):          do not lot center, default is False
        noise (float):              noise level for the GP, default is 0.01
        niter (int):                number of iterations for training the GP, default is 5
        temperature_scale (float):  scale factor for temperature, default is 1000.0
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
        time_sign (float):          sign to apply to time units, typically 1.0
                                    but for some analysis -1 makes sense

    """

    def __init__(
        self,
        *args,
        uncentered=False,
        noise=0.01,
        niter=5,
        temperature_scale=1000.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.noise = noise
        self.niter = niter
        self.temperature_scale = temperature_scale

        self.uncentered = uncentered

        if self.uncentered:
            n = len(self.time)
            self.heat_counts = [n]
            self.heat_id = {0: 0}
            self.flat_indices = np.zeros(n, dtype=int)
        else:
            self.heat_counts = [len(h) for h in self.heat_indices.values()]
            self.heat_id = {h: i for i, h in enumerate(self.heat_indices.keys())}
            self.flat_indices = np.zeros(len(self.temperature), dtype=int)

            for i, h in enumerate(self.heat_indices.keys()):
                self.flat_indices[self.heat_indices[h]] = i

    def analyze(self, verbose=False):
        """
        Train the GP model

        Keyword Args:
            verbose (bool):     print out the training progress
        """
        # Set up the input data in order
        X = torch.tensor(
            np.stack(
                [
                    self.temperature / self.temperature_scale,
                    self.time,
                    self.flat_indices,
                ],
                axis=1,
            )
        )
        y = torch.log10(torch.tensor(self.stress))

        # Setup optimizer
        pyro.clear_param_store()
        self.kernel = LMKernel(self.heat_counts)

        self.gp = gp.models.GPRegression(X, y, self.kernel, torch.tensor(self.noise))
        optimizer = torch.optim.LBFGS(self.gp.parameters())
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

        # Closure for optimizer
        def closure():
            optimizer.zero_grad()
            loss = loss_fn(self.gp.model, self.gp.guide)
            loss.backward()
            return loss

        # Optimize
        if verbose:
            iter = trange(self.niter)
            iter.set_description("loss= ")
        else:
            iter = range(self.niter)

        for _ in iter:
            loss = optimizer.step(closure)

            if verbose:
                iter.set_description("loss=%e" % loss)

        self.C_avg = self.gp.kernel.C_mean.item()

        return self

    def predict_log_stress(self, time, temperature):
        """
        Predict the log stress at a given time and temperature

        Args:
            time (np.array): time
            temperature (np.array): temperature
        """
        if np.isscalar(temperature):
            temperature = np.full_like(time, temperature)

        X = torch.tensor(
            np.stack(
                [
                    temperature / self.temperature_scale,
                    time,
                    np.full_like(time, -1),
                ],
                axis=1,
            )
        )

        with torch.no_grad():
            mean, var = self.gp(X, full_cov=False, noiseless=False)

        return mean.numpy(), var.numpy()

    def predict_stress(self, time, temperature, confidence=None):
        """
        Predict the stress at a given time and temperature

        Args:
            time (np.array): time
            temperature (np.array): temperature

        Keyword Args:
            confidence (float): confidence level for the prediction, default is None
        """
        log_mean, log_var = self.predict_log_stress(time, temperature)

        if confidence is None:
            return 10.0**log_mean

        z = ss.norm.interval(np.abs(confidence))[1]

        return 10.0 ** (log_mean - np.sign(confidence) * z * np.sqrt(log_var))
