"""Implements Larson-Miller Gaussian process regression
"""

import numpy as np

import scipy.stats as ss
import scipy.optimize as opt
import scipy.integrate as inte

import torch
import torch.distributions as dist
import pyro
import pyro.contrib.gp as gp
from torch.distributions import constraints

import torchquad
from torchquad import Simpson

from pyro.contrib import gp
from pyro.contrib.gp.kernels.kernel import Kernel
from pyro.nn.module import PyroParam

from pycreep import ttp

from tqdm import trange

torch.set_default_dtype(torch.float64)
torchquad.set_up_backend("torch", data_type="float64", torch_enable_cuda=False)


class LMKernel(Kernel):
    """Kernel used for Gaussian process LM regression

    For the moment this is not lot-centered

    Keyword Args:
        lengthscale (torch.tensor): kernel length scale
        variance (torch.tensor): kernel variance
        C_mean (torch.tensor): initial guess at the mean of the Larson-Miller parameter
        C_variance (torch.tensor): initial guess at the variance of the Larson-Miller parameter

    Input order for parameters in X is [temperature, time]
    """

    def __init__(
        self,
        lengthscale=torch.tensor(1.0),
        variance=torch.tensor(1.0),
        C_mean=torch.tensor(20.0),
        C_variance=torch.tensor(0.1),
    ):
        super().__init__(3, None)

        self.lengthscale = PyroParam(lengthscale, constraints.positive)
        self.variance = PyroParam(variance, constraints.positive)
        self.C_mean = PyroParam(C_mean, constraints.positive)
        self.C_variance = PyroParam(C_variance, constraints.positive)

    def _d2(self, X, Z):
        """
        Squared distance calculator

        Args:
            X (torch.tensor): first set of points
            Y (torch.tensor): second set of points
        """
        return (X[:, None] - Z) ** 2.0

    def calculate_mean_var(self, X):
        """
        Calculate the mean and variance of the lot constant

        Args:
            X (torch.tensor): input data, shape (N,3)
        """
        mean_X = X[:, 0] * (self.C_mean + X[:, 1])
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

        scov = var_X[:, None] + var_Z

        scaled_r2 = torch.exp(
            -0.5 * self._d2(mean_X, mean_Z) / (scov + self.lengthscale)
        )
        sf = self.variance / torch.sqrt(torch.abs(1.0 + scov / self.lengthscale))

        mat = sf * scaled_r2

        if diag:
            return mat.diag()
        return mat


class InverseGPRLPModel:
    """
    Inverse model mapping (average) stress and temperature to time

    Keyword Args:
        kernel (gp.kernels.Kernel): kernel for the GP, default is RBF
        noise (float): noise level for the GP, default is 0.01
        niter (int): number of iterations for training the GP, default is 200
        lr (float): learning rate, default is 1.0e-2
        verbose (bool): print out the training progress, default is False
    """

    def __init__(
        self,
        kernel=gp.kernels.RBF(input_dim=2),
        noise=0.01,
        niter=200,
        lr=1.0e-2,
        verbose=False,
    ):
        self.kernel = kernel
        self.noise = noise
        self.niter = niter
        self.lr = lr
        self.verbose = verbose

    def train(self, time, temperature, stress):
        # Setup optimizer
        X = self._assemble_X(stress, temperature)
        y = time

        with pyro.get_param_store().scope() as self.backward_model:
            self.gp = gp.models.GPRegression(
                X, y, self.kernel, torch.tensor(self.noise)
            )
            optimizer = torch.optim.Adam(self.gp.parameters(), lr=self.lr)
            loss_fn = pyro.infer.Trace_ELBO().differentiable_loss

            # Closure for optimizer
            def closure():
                optimizer.zero_grad()
                loss = loss_fn(self.gp.model, self.gp.guide)
                loss.backward()
                return loss

            # Optimize
            if self.verbose:
                iter = trange(self.niter)
                iter.set_description("loss= ")
            else:
                iter = range(self.niter)

            for _ in iter:
                loss = optimizer.step(closure)

                if self.verbose:
                    iter.set_description("loss=%e" % loss)

    def __call__(self, stress, temperature):
        return self.predict_log_time(stress, temperature)

    def predict_log_time(self, stress, temperature):
        """
        Predict the log time at a given stress and temperature

        Args:
            stress (torch.tensor): log stress
            temperature (torch.tensor): scaled temperature
        """
        with pyro.get_param_store().scope(self.backward_model):
            with torch.no_grad():
                mean, _ = self.gp(
                    self._assemble_X(stress, temperature),
                    full_cov=False,
                    noiseless=False,
                )

        return mean

    def _assemble_X(self, stress, temperature):
        """
        Assemble the input data for the GP

        Args:
            stress (torch.tensor): stress
            temperature (torch.tensor): temperature
        """
        return torch.stack(
            [
                temperature,
                stress,
            ],
            dim=1,
        )


class GPRLMPModel(ttp.TTPAnalysis):
    """
    Parent class for Gaussian process regression Larson-Miller models

    Args:
        data:                       dataset as a pandas dataframe

    Keyword Args:
        noise (float):              noise level for the GP, default is 0.01
        niter (int):                number of iterations for training the GP, default is 200
        lr (float):                 learning rate, default is 1.0e-2
        temperature_scale (float):  scale factor for temperature, default is 1000.0
        npoints_backward (int):     number of points in each dimension to train the backward model, default is 50
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
        noise=0.01,
        niter=200,
        lr=1.0e-2,
        temperature_scale=1000.0,
        npoints_backward=25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.noise = noise
        self.niter = niter
        self.lr = lr
        self.temperature_scale = temperature_scale

        self.npoints_backward = npoints_backward

    def analyze(self, verbose=False):
        """
        Train the GP model

        Keyword Args:
            verbose (bool):     print out the training progress
        """
        # Set up the input data in order
        X = self._assemble_X(torch.tensor(self.time), torch.tensor(self.temperature))
        y = torch.log10(torch.tensor(self.stress))

        # Setup optimizer
        pyro.clear_param_store()
        with pyro.get_param_store().scope() as self.forward_model:
            self.kernel = LMKernel()

            self.gp = gp.models.GPRegression(
                X, y, self.kernel, torch.tensor(self.noise)
            )
            optimizer = torch.optim.Adam(self.gp.parameters(), lr=self.lr)
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

        self._train_backward_model(verbose)

        return self

    def _train_backward_model(self, verbose=False):
        """
        Train the inverse model
        """
        time, temperature = torch.meshgrid(
            torch.linspace(
                np.log10(np.min(self.time)),
                np.log10(np.max(self.time)),
                self.npoints_backward,
            ),
            torch.linspace(
                np.min(self.temperature),
                np.max(self.temperature),
                self.npoints_backward,
            ),
        )
        stress, _ = self.predict_log_stress(
            10.0 ** time.flatten().cpu().numpy(), temperature.flatten().cpu().numpy()
        )

        self.backward_model = InverseGPRLPModel(verbose=verbose)
        self.backward_model.train(
            time.flatten(),
            temperature.flatten() / self.temperature_scale,
            torch.tensor(stress),
        )

    def _assemble_X(self, time, temperature):
        """
        Assemble the input data for the GP

        Args:
            time (torch.tensor): time
            temperature (torch.tensor): temperature
        """
        return torch.stack(
            [
                temperature / self.temperature_scale,
                torch.log10(time),
            ],
            dim=1,
        )

    def prob_log_stress(self, stress, time, temperature):
        """
        Calculate the probability of a given stress at a given time and temperature

        Args:
            stress (np.array): stress
            time (np.array): time
            temperature (np.array): temperature
        """
        return (
            self.prob_log_stress_torch(
                torch.tensor(stress), torch.tensor(time), torch.tensor(temperature)
            )
            .cpu()
            .numpy()
        )

    def prob_log_stress_torch(self, stress, time, temperature):
        """
        Calculate the probability of a given log stress at a given time and temperature

        This version takes torch inputs

        Args:
            stress (torch.tensor): stress
            time (torch.tensor): time
            temperature (torch.tensor): temperature
        """
        X = self._assemble_X(time.flatten(), temperature.flatten())

        with pyro.get_param_store().scope(self.forward_model):
            with torch.no_grad():
                mean, var = self.gp(X, full_cov=False, noiseless=False)

        return torch.exp(
            dist.Normal(mean, torch.sqrt(var)).log_prob(torch.log10(stress.flatten()))
        ).reshape(stress.shape)

    def prob_log_time(self, time, stress, temperature):
        """
        Calculate the probability of a given log time at a given stress and temperature

        Args:
            time (np.array): time
            stress (np.array): stress
            temperature (np.array): temperature
        """
        return (
            self.prob_log_time_torch(
                torch.tensor(time), torch.tensor(stress), torch.tensor(temperature)
            )
            .cpu()
            .numpy()
        )

    def prob_log_time_torch(self, time, stress, temperature, N=501, dt=6.0):
        """
        Calculate the probability of a given log time at a given stress and temperature

        This version takes torch inputs

        Args:
            time (torch.tensor): time
            stress (torch.tensor): stress
            temperature (torch.tensor): temperature

        Keyword Args:
            N (int): number of points for integration
            dt (float): log time range for integratio
        """
        p_stress = self.prob_log_stress_torch(stress, time, temperature)
        domain = torch.tensor([[0, 1.0]])
        actual_domains = torch.stack(
            [torch.log10(time) - dt, torch.log10(time) + dt], dim=0
        )
        simp = Simpson()

        def f(x):
            xp = x.expand((x.shape[0],) + stress.shape)
            actual_x = (actual_domains[1] - actual_domains[0]) * xp + actual_domains[0]
            dj = actual_domains[1] - actual_domains[0]
            return (
                self.prob_log_stress_torch(
                    stress.unsqueeze(0).expand(xp.shape),
                    10.0**actual_x,
                    temperature.unsqueeze(0).expand(xp.shape),
                )
                * dj
            )

        nf = simp.integrate(
            f,
            dim=1,
            integration_domain=domain,
            N=N,
        )

        return p_stress / nf

    def predict_log_stress(self, time, temperature):
        """
        Predict the log stress at a given time and temperature

        Args:
            time (np.array): time
            temperature (np.array): temperature
        """
        if np.isscalar(temperature):
            temperature = np.full_like(time, temperature)

        X = self._assemble_X(torch.tensor(time), torch.tensor(temperature))

        with pyro.get_param_store().scope(self.forward_model):
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

    def predict_log_time(self, stress, temperature, dt=3.0, N=25):
        """
        Predict the log time at a given stress and temperature

        This assumes the distribution will be normal

        Args:
            stress (np.array): stress
            temperature (np.array): temperature

        Keyword Args:
            dt (float): log range for getting probabilities to fit a normal, default is 3.0
            N (int): number of points for calculating the probability, default is 25
        """
        # Solve for the means
        mean_log_time = self.backward_model(
            torch.log10(torch.tensor(stress)),
            torch.tensor(temperature) / self.temperature_scale,
        )

        # Expand a n x N array for the times to consider
        rng = torch.linspace(-dt, dt, N)
        log_time_range = mean_log_time.unsqueeze(1) + rng

        # Calculate the probability distribution
        p = self.prob_log_time_torch(
            10.0 ** log_time_range.flatten(),
            torch.tensor(stress).unsqueeze(1).expand(log_time_range.shape).flatten(),
            torch.tensor(temperature)
            .unsqueeze(1)
            .expand(log_time_range.shape)
            .flatten(),
        )
        p = p.reshape(log_time_range.shape)

        # This will be fast...
        # If it's not obvious this is the midpoint rule
        dt = torch.diff(log_time_range, dim=1)
        pX = p * log_time_range
        means = torch.sum((pX[:, :1] + pX[:, :-1]) / 2.0 * dt, dim=1)
        pXX = p * log_time_range**2.0
        variances = torch.sum((pXX[:, :1] + pXX[:, :-1]) / 2.0 * dt, dim=1) - means**2.0

        return means.cpu().numpy(), variances.cpu().numpy()

    def predict_time(self, stress, temperature, confidence=None):
        """
        Predict the time a given stress and temperature

        Args:
            stress (np.array): stress
            temperature (np.array): temperature

        Keyword Args:
            confidence (float): confidence level for the prediction, default is None
        """
        log_mean, log_var = self.predict_log_time(stress, temperature)

        if confidence is None:
            return 10.0**log_mean

        z = ss.norm.interval(np.abs(confidence))[1]

        return 10.0 ** (log_mean - np.sign(confidence) * z * np.sqrt(log_var))
