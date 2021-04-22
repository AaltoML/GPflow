import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.models.cvi import CVI
from gpflow.optimizers import NaturalGradient
from gpflow import set_trainable


plt.style.use("ggplot")

# for reproducibility of this notebook:
rng = np.random.RandomState(123)
tf.random.set_seed(42)


# %%
def func(x):
    return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)


def my_tf_round(x, decimals = 2):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier

N = 20  # Number of training observations
X = rng.rand(N, 1) * 2 - 1  # X values
Y = func(X) + 0.2 * rng.randn(N, 1)  # Noisy Y values
data = (X, Y)


var_gp = 1.5
len_gp = .1
var_noise = .1
lr_natgrad = .1


m_gpr = gpflow.models.GPR(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    noise_variance=var_noise
)

m_cvi = CVI(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Gaussian(variance=var_noise))

m_vgp = gpflow.models.VGP(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Gaussian(variance=var_noise))


# Testing both elbos are the same
assert my_tf_round(m_cvi.elbo()) == my_tf_round(m_vgp.elbo()), "ELBOs do not match"

set_trainable(m_cvi.kernel.lengthscales, False)
set_trainable(m_cvi.kernel.variance, False)
set_trainable(m_cvi.likelihood.variance, False)

m_cvi.update_variational_parameters(beta=1)

# Testing both CVI lr=1 gives GPR likelihood
assert my_tf_round(m_cvi.elbo()) == my_tf_round(m_gpr.maximum_log_likelihood_objective()), "GPR and CVI do not match"

#Checking ELBO evaluations Natgrad and CVI

m_cvi = CVI(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp, variance=var_gp),
    gpflow.likelihoods.Gaussian(variance=var_noise))

m_cvi.update_variational_parameters(beta=lr_natgrad)


natgrad_opt = NaturalGradient(gamma=lr_natgrad)
variational_params = [(m_vgp.q_mu, m_vgp.q_sqrt)]
natgrad_opt.minimize(m_vgp.training_loss, var_list=variational_params)


# Testing one step in natgrad is same as CVI
assert my_tf_round(m_cvi.elbo()) == my_tf_round(m_vgp.elbo()), "GPR and CVI after one step do not match"