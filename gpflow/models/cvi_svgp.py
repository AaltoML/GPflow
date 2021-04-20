# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import numpy as np
import tensorflow as tf

from .. import kullback_leiblers
from ..base import Parameter
from ..conditionals import conditional
from ..covariances import Kuu, Kuf
from ..config import default_float, default_jitter
from ..utilities import positive, triangular
from .model import GPModel, InputData, MeanAndVariance, RegressionData
from .training_mixins import ExternalDataTrainingLossMixin
from .util import inducingpoint_wrapper


class SVGP(GPModel, ExternalDataTrainingLossMixin):
    """
    This is the Sparse Variational GP (SVGP). The key reference is

    ::

      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews, Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

    """

    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        *,
        mean_function=None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        num_data=None,
    ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_data = num_data
        self.q_diag = q_diag
        self.whiten = whiten
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        # init variational parameters
        num_inducing = self.inducing_variable.num_inducing
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.

        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.

        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent_gps)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        if q_sqrt is None:
            if self.q_diag:
                ones = np.ones((num_inducing, self.num_latent_gps), dtype=default_float())
                self.q_sqrt = Parameter(ones, transform=positive())  # [M, P]
            else:
                q_sqrt = [
                    np.eye(num_inducing, dtype=default_float()) for _ in range(self.num_latent_gps)
                ]
                q_sqrt = np.array(q_sqrt)
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent_gps = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=positive())  # [M, L|P]
            else:
                assert q_sqrt.ndim == 3
                self.num_latent_gps = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]

    def get_mean_chol_cov_inducing_posterior(self):
        return self.q_mu, self.q_sqrt

    def prior_kl(self) -> tf.Tensor:
        q_mu, q_sqrt = self.get_mean_chol_cov_inducing_posterior()
        return kullback_leiblers.prior_kl(
            self.inducing_variable, self.kernel, q_mu, q_sqrt, whiten=self.whiten
        )

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        q_mu, q_sqrt = self.get_mean_chol_cov_inducing_posterior()
        mu, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        return mu + self.mean_function(Xnew), var


class SVGP_CVI(SVGP):

    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        *,
        mean_function=None,
        num_latent_gps: int = 1,
        lambda_1=None,
        lambda_2_sqrt=None,
        num_data=None,
    ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        GPModel.__init__(self, kernel, likelihood, mean_function, num_latent_gps)

        self.num_data = num_data
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        # init variational parameters
        self.num_inducing = self.inducing_variable.num_inducing

        self._init_variational_parameters(self.num_inducing, lambda_1, lambda_2_sqrt)
        self.whiten = False

    def _init_variational_parameters(self, num_inducing, lambda_1, lambda_2_sqrt):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.

        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.

        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """


        lambda_1 = np.zeros((num_inducing, self.num_latent_gps)) if lambda_1 is None else lambda_1
        self.lambda_1 = Parameter(lambda_1, dtype=default_float())  # [M, P]

        if lambda_2_sqrt is None:
            lambda_2_sqrt = [
                tf.eye(num_inducing, dtype=default_float()) * 1e-3 for _ in range(self.num_latent_gps)
            ]
            lambda_2_sqrt = np.array(lambda_2_sqrt)
            self.lambda_2_sqrt = Parameter(lambda_2_sqrt, transform=triangular())  # [P, M, M]
        else:
            assert lambda_2_sqrt.ndim == 3
            self.num_latent_gps = lambda_2_sqrt.shape[0]
            self.lambda_2_sqrt = Parameter(lambda_2_sqrt, transform=triangular())  # [L|P, M, M]

    def get_mean_chol_cov_inducing_posterior_old(self):

        # L1, sqrt(L2)
        lambda_1 = self.lambda_1
        lambda_2_sqrt = self.lambda_2_sqrt
        # Kuu
        K_uu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())  # [P, M, M] or [M, M]
        # chol(Kuu)
        Luu = tf.linalg.cholesky(K_uu)
        # chol(Kuu)^T chol(L2)
        W = tf.matmul(Luu, lambda_2_sqrt, transpose_a=True)
        # chol(L2)^T Kuu chol(L2) + I
        A = tf.matmul(W, W, transpose_a=True) + tf.eye(self.num_inducing, dtype=default_float())
        # chol(A)
        cholA = tf.linalg.cholesky(A)
        # V = Kuu chol(L2) chol(A)
        # V = K_uu @ lambda_2_sqrt @ cholA
        VT = tf.linalg.triangular_solve(cholA, tf.matmul(lambda_2_sqrt,  K_uu, transpose_a=True))

        # Kuu - Kuu chol(L2) A chol(L2)^T Kuu
        # Suu = K_uu - tf.matmul(V, V, transpose_b=True)
        Suu = K_uu - tf.matmul(VT, VT, transpose_a=True)
        chol_Suu = tf.linalg.cholesky(Suu)
        # lambda2 * lambda1
        l2l1 = tf.matmul(
            lambda_2_sqrt,
            tf.matmul(lambda_2_sqrt, lambda_1, transpose_a=True)
        )
        # mu = Suu * (lambda2 * lambda1)
        mu = tf.matmul(Suu, l2l1)[0]
        return mu, chol_Suu

    def get_mean_chol_cov_inducing_posterior(self):

        # L1, sqrt(L2)
        lambda_1 = self.lambda_1
        lambda_2_sqrt = self.lambda_2_sqrt
        # Kuu
        K_uu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())  # [P, M, M] or [M, M]
        # chol(Kuu)
        Luu = tf.linalg.cholesky(K_uu)

        LpTLt = tf.matmul(Luu, lambda_2_sqrt, transpose_a=True)
        W = tf.eye(self.num_inducing, dtype=default_float()) \
            + tf.matmul(LpTLt, LpTLt, transpose_a=True)

        # Lw = chol(W)
        chol_W = tf.linalg.cholesky(W)
        # W = Lw Lw^T
        # W^-1 = Lw^{-T} Lw^{-1}
        # K W^-1 K= [K Lw^{-T}] [Lw^{-1} K] = [Lw^{-1} K]^T [Lw^{-1} K]
        LtTK = tf.matmul(lambda_2_sqrt, K_uu, transpose_a=True)
        iLwLtTK = tf.linalg.triangular_solve(chol_W, LtTK, lower=True, adjoint=False)
        S_q = K_uu - tf.matmul(iLwLtTK, iLwLtTK, transpose_a=True)
        m_q = tf.matmul(S_q, lambda_1)[0]

        chol_S_q = tf.linalg.cholesky(S_q + tf.eye(self.num_inducing, dtype=default_float()) *1e-2 )
        return m_q, chol_S_q


    @property
    def lambda_2(self):
        return tf.matmul(self.lambda_2_sqrt, self.lambda_2_sqrt, transpose_b=True)

    def natgrad_step(self, X, Y, lr=0.1):

        mean, var = self.predict_f(X)
        with tf.GradientTape() as g:
            g.watch([mean, var])
            ve = self.likelihood.variational_expectations(mean, var, Y)
            grads = g.gradient(ve, [mean, var])

        lambda_2 = self.lambda_2
        lambda_1 = self.lambda_1

        # chain rule at f
        grads = gradient_transformation_mean_var_to_expectation([mean, var], grads)

        K_uu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())  # [P, M, M] or [M, M]
        K_uf = Kuf(self.inducing_variable, self.kernel, X)  # [P, M, M] or [M, M]
        chol_Kuu = tf.linalg.cholesky(K_uu)
        P = tf.linalg.cholesky_solve(chol_Kuu, K_uf)

        grads = [
            tf.transpose(grads[0]) * P,
            tf.reshape(grads[1], [1,1,-1]) * P[None, ...] * P[:, None, ...]
        ]

        # compute update in natural form
        lambda_1 = (1-lr) * lambda_1 + lr * tf.reduce_sum(grads[0], axis=-1, keepdims=True)
        lambda_2 = -(1-lr) * lambda_2 + lr * tf.reduce_sum(grads[1], axis=-1)

        # transform and perform update
        lambda_2_sqrt = -tf.linalg.cholesky(-lambda_2)
        self.lambda_1.assign(lambda_1)
        self.lambda_2_sqrt.assign(lambda_2_sqrt)

def gradient_transformation_mean_var_to_expectation(inputs, grads):
    """
    Transforms gradient 𝐠 of a function wrt [μ, σ²]
    into its gradients wrt to [μ, σ² + μ²]
    :param inputs: [μ, σ²]
    :param grads: 𝐠
    """
    return grads[0] - 2.0 * grads[1] * inputs[0], grads[1]