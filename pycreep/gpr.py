"""Implements Larson-Miller Gaussian process regression
"""

import numpy as np

import scipy.stats as ss

from math import log

import torch
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDelta
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO, Predictive

from pyro.nn import PyroModule, PyroSample

from pycreep import ttp

from tqdm import trange


class UncenteredLMP(PyroModule):
    """
    Larson-Miller transformation of the mean and variance of the time and temperature

    Keyword Args:
        C_mean (float):     mean of the Larson-Miller parameter, default is 20.0
        C_variance (float): variance of the Larson-Miller parameter, default is 0.01
    """

    def __init__(self, C_mean=20.0, C_variance=0.01):
        super().__init__()
        self.C_mean = PyroSample(dist.Normal(C_mean, 1.0))
        self.C_variance = PyroSample(dist.HalfNormal(C_variance))

    def forward(self, X):
        """
        Given the model input provide the mean and variance of the Larson-Miller parameter
        """
        return (
            X[:, 0] * (self.C_mean + torch.log10(X[:, 1])),
            X[:, 0] ** 2.0 * self.C_variance,
        )


class Kernel(PyroModule):
    """
    Recreate the squared exponential kernel for better control of the calibration

    Keyword Args:
        var (float):    variance of the kernel, default is 1.0
        length (float): length scale of the kernel, default is 2.0
    """

    def __init__(self, var=0.01, length=1.0):
        super().__init__()
        self.var = PyroSample(dist.HalfNormal(var))
        self.length = PyroSample(dist.Normal(length, 0.1))

    def distance(self, X, Z):
        """
        Calculate the squared distance between two sets of points

        Args:
            X (torch.tensor): first set of points
            Z (torch.tensor): second set of points
        """
        X = X.unsqueeze(1)
        Z = Z.unsqueeze(1)
        X2 = (X**2.0).sum(1, keepdim=True)
        Z2 = (Z**2.0).sum(1, keepdim=True)
        XZ = X.matmul(Z.t())
        r2 = X2 - 2 * XZ + Z2.t()
        return r2.clamp(min=0)

    def forward(self, X_mean, X_variance, Y_mean, Y_variance):
        """
        Evaluate the kernel

        Args:
            X_mean (torch.tensor): mean of the first set of points
            X_variance (torch.tensor): variance of the first set of points
            Y_mean (torch.tensor): mean of the second set of points
            Y_variance (torch.tensor): variance of the second set of points
        """
        scov = X_variance[:, None] + Y_variance
        scaled_r2 = torch.exp(
            -0.5 * self.distance(X_mean, Y_mean) / (scov + self.length)
        )
        sf = self.var / torch.sqrt(torch.abs(1.0 + scov / self.length))

        return sf * scaled_r2


class GPModel(PyroModule):
    """
    Gaussian  process model

    Args:
        kernel (gpr.Kernel): kernel for the GP
        ttp_model (gpr.UncenteredLMP): Larson-Miller model

    Keyword Args:
        noise (float):  noise level for the model, default is 0.01
        jitter (float): jitter for numerical stability, default is 1.0e-6
    """

    def __init__(self, kernel, ttp_model, noise=0.01, jitter=1.0e-6):
        super().__init__()
        self.kernel = kernel
        self.ttp_model = ttp_model
        self.noise = PyroSample(dist.HalfNormal(noise))
        self.jitter = jitter

    def covariance(self, X1, X2, noise=True):
        """
        Calculate the covariance matrix

        Args:
            X1 (torch.tensor): first set of points
            X2 (torch.tensor): second set of points

        Keyword Args:
            noise (bool): include noise in the covariance, default is True
        """
        C1_mean, C1_variance = self.ttp_model(X1)
        C2_mean, C2_variance = self.ttp_model(X2)
        k = self.kernel(C1_mean, C1_variance, C2_mean, C2_variance)
        if noise:
            return k + (self.noise + self.jitter) * torch.eye(X1.shape[0])
        return k

    def forward(self, X1, y, X2=None, noise=True):
        """
        Evaluate the model

        Args:
            X1 (torch.tensor): first set of points
            y (torch.tensor): observed values

        Keyword Args:
            X2 (torch.tensor): second set of points, default is None
            noise (bool): include noise in the covariance, default is True
        """
        if X2 is None:
            X2 = X1

        k = pyro.deterministic(
            "cross_correlation", self.covariance(X1, X2, noise=noise)
        )
        if X2 is not None:
            return k

        Lff = torch.linalg.cholesky(k)

        with pyro.plate(y.shape[0]):
            return pyro.sample(
                "obs",
                dist.MultivariateNormal(torch.zeros(X1.shape[0]), scale_tril=Lff),
                obs=y,
            )


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
        noise=0.001,
        niter=500,
        lr=1.0e-2,
        temperature_scale=1000.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.noise = noise
        self.niter = niter
        self.lr = lr
        self.temperature_scale = temperature_scale

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
        self.X = torch.tensor(
            np.stack(
                [
                    self.temperature / self.temperature_scale,
                    self.time,
                    self.flat_indices,
                ],
                axis=1,
            )
        )
        self.y = torch.log10(torch.tensor(self.stress))

        # Setup optimizer
        pyro.clear_param_store()

        kernel = Kernel()
        ttp_model = UncenteredLMP()
        self.model = GPModel(kernel, ttp_model, noise=self.noise)

        optimizer = optim.Adam({"lr": self.lr})
        self.guide = AutoDelta(self.model)
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

        # Optimize
        if verbose:
            iter = trange(self.niter)
            iter.set_description("loss= ")
        else:
            iter = range(self.niter)

        for _ in iter:
            loss = svi.step(self.X, self.y)

            if verbose:
                iter.set_description("loss=%e" % loss)

        self.C_avg = pyro.param("AutoDelta.ttp_model.C_mean").item()

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

        Xp = torch.tensor(
            np.stack(
                [
                    temperature / self.temperature_scale,
                    time,
                    np.full_like(time, -1),
                ],
                axis=1,
            )
        )

        predict = Predictive(
            self.model,
            guide=self.guide,
            num_samples=1,
        )
        k11 = predict(self.X, self.y, X2=self.X)["cross_correlation"][0]
        k21 = predict(Xp, self.y, X2=self.X, noise=False)["cross_correlation"][0]
        k22 = predict(Xp, self.y, X2=Xp)["cross_correlation"][0]

        (
            LU,
            pivots,
        ) = torch.linalg.lu_factor(k11)

        mean = k21.matmul(
            torch.linalg.lu_solve(LU, pivots, self.y.unsqueeze(-1))
        ).squeeze(-1)
        variance = k22 - k21.matmul(torch.linalg.lu_solve(LU, pivots, k21.t()))

        return mean.numpy(), np.diag(variance.numpy())

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
