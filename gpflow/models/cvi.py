# Copyright 2020 Paul Chang, Arno Solin, Emtiyaz Khan
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

from typing import Optional

import numpy as np
import tensorflow as tf

import gpflow

from gpflow.base import Parameter
from gpflow.conditionals import conditional
from gpflow.config import default_float, default_jitter
from gpflow.kernels import Kernel
from gpflow.kullback_leiblers import gauss_kl
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction, Zero
from gpflow.models.model import RegressionData, InputData, MeanAndVariance, GPModel
from gpflow.utilities import triangular, positive

class CVI(GPModel):
    r"""
    This method approximates the Gaussian process posterior using a multivariate Gaussian.

    The idea is that the posterior over the function-value vector F is
    approximated by a Gaussian, and the KL divergence is minimised between
    the approximation and the posterior.

    The key reference is:
      Khan, M., & Lin, W. (2017). Conjugate-Computation Variational Inference: Converting Variational Inference in Non-Conjugate Models to Inferences in Conjugate Models. In Artificial Intelligence and Statistics (pp. 878-887).
    
    """
    def __init__(self,
                 data: RegressionData,
                 kernel: Kernel,
                 likelihood: Likelihood,
                 mean_function: Optional[MeanFunction] = None,
                 num_latent: Optional[int] = 1):
        """
        X is a data matrix, size [N, D]
        Y is a data matrix, size [N, R]
        kernel, likelihood, mean_function are appropriate GPflow objects

        """
        super().__init__(kernel, likelihood, mean_function, num_latent)

        x_data, y_data = data
        num_data = x_data.shape[0]
        self.num_data = num_data
        self.num_latent = num_latent or y_data.shape[1]
        self.data = data

        self.lambda_1 = np.zeros((num_data, self.num_latent))
        self.lambda_2 = 1e-6*np.ones((num_data, self.num_latent))

    def maximum_log_likelihood_objective(self, *args, **kwargs) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """

        x_data, y_data = self.data
        pseudo_y = self.lambda_1 / self.lambda_2
        sW = tf.sqrt(tf.abs(self.lambda_2))

        # Computes conversion λ₁, λ₂ → m, V by using q(f) ≃ t(f)p(f)
        K = self.kernel(x_data) + tf.eye(self.num_data, dtype=default_float()) * default_jitter()
        L = tf.linalg.cholesky(tf.eye(self.num_data,dtype=tf.float64) + (sW @ tf.transpose(sW)) * K)
        T = tf.linalg.solve(L,tf.tile(sW,(1,self.num_data)) * K)
        post_v = tf.reshape(tf.linalg.diag_part(K) - tf.reduce_sum(T*T,axis=0),(self.num_data,1))
        alpha = sW*tf.linalg.solve(tf.transpose(L),tf.linalg.solve(L,sW*pseudo_y))
        post_m = K @ alpha
        
        # Store alpha for prediction
        self.q_alpha = alpha

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(post_m, post_v, y_data)

        # Compute the ELBO where KL is first 3 terms.
        elbo = -tf.transpose(pseudo_y) @ alpha/2. - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L))) + \
            tf.reduce_sum(0.5*(self.lambda_2)*((pseudo_y - post_m)**2 + post_v)) + tf.reduce_sum(var_exp)

        return tf.reduce_sum(elbo)

    def neg_elbo(self):
        return -self.elbo()

    def update_variational_parameters(self, beta=0.05) -> tf.Tensor:
        """ Takes natural gradient step in Variational parameters in the local parameters
        λₜ = rₜ▽[Var_exp] + (1-rₜ)λₜ₋₁
        Input:
        :param: X : N x D
        :param: Y:  N x 1
        :param: lr: Scalar

        Output:
        Updates the params
        """

        x_data, y_data = self.data
        pseudo_y = self.lambda_1 / self.lambda_2
        sW = tf.sqrt(tf.abs(self.lambda_2))

        # Computes conversion λ₁, λ₂ → m, V by using q(f) ≃ t(f)p(f)
        K = self.kernel(x_data) + tf.eye(self.num_data, dtype=default_float()) * default_jitter()
        L = tf.linalg.cholesky(tf.eye(self.num_data, dtype=tf.float64) + (sW @ tf.transpose(sW)) * K)
        T = tf.linalg.solve(L, tf.tile(sW, (1, self.num_data)) * K)
        post_v = tf.reshape(tf.linalg.diag_part(K) - tf.reduce_sum(T * T, axis=0), (self.num_data, 1))
        alpha = sW * tf.linalg.solve(tf.transpose(L), tf.linalg.solve(L, sW * pseudo_y))
        post_m = K @ alpha

        # Keep alphas updated
        self.q_alpha = alpha

        # Get variational expectations derivatives.
        with tf.GradientTape(persistent=True) as g:
            g.watch(post_m)
            g.watch(post_v)
            var_exp = self.likelihood.variational_expectations(post_m, post_v, y_data)

        d_exp_dm = g.gradient(var_exp, post_m)
        d_exp_dv = g.gradient(var_exp, post_v)
        del g

        # Take the CVI step and transform to be ▽μ[Var_exp]
        self.lambda_1 = (1. - beta) * self.lambda_1 + beta * (d_exp_dm - 2. * (d_exp_dv * post_m))
        self.lambda_2 = (1. - beta) * self.lambda_2 + beta * (-2. * d_exp_dv)

    def predict_f(self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False) -> MeanAndVariance:
        r"""
        The posterior variance of F is given by
            q(f) = N(f | K alpha + mean, [K^-1 + diag(lambda**2)]^-1)
        Here we project this to F*, the values of the GP at Xnew which is given
        by
           q(F*) = N ( F* | K_{*F} alpha + mean, K_{**} - K_{*f}[K_{ff} +
                                           diag(lambda**-2)]^-1 K_{f*} )

        """
        assert full_output_cov == False
        x_data, _y_data = self.data

        # Evaluate the kernel
        Kx = self.kernel(x_data, Xnew)
        K = self.kernel(x_data)

        # Predictive mean
        f_mean = tf.linalg.matmul(Kx, self.q_alpha, transpose_a=True) + self.mean_function(Xnew)

        # Predictive var
        A = K + tf.linalg.diag(tf.transpose(1. / self.lambda_2))
        L = tf.linalg.cholesky(A)
        Kx_tiled = tf.tile(Kx[None, ...], [self.num_latent, 1, 1])
        LiKx = tf.linalg.solve(L, Kx_tiled)
        if full_cov:
            f_var = self.kernel(Xnew) - tf.linalg.matmul(LiKx, LiKx, transpose_a=True)
        else:
            f_var = self.kernel(Xnew, full_cov=False) - tf.reduce_sum(tf.square(LiKx), 1)
        return f_mean, tf.transpose(f_var)
