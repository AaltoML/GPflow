
import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.models.cvi import CVI
from gpflow.optimizers import NaturalGradient

plt.style.use("ggplot")

# for reproducibility of this notebook:
rng = np.random.RandomState(123)
tf.random.set_seed(42)

# %%
def func(x):
    return np.sin(x * 3 * 3.14) + 0.3 * np.cos(x * 9 * 3.14) + 0.5 * np.sin(x * 7 * 3.14)

N = 20  # Number of training observations
X = rng.rand(N, 1) * 2 - 1  # X values
Y = func(X) + 0.2 * rng.randn(N, 1)  # Noisy Y values
data = (X, Y)

plt.plot(X, Y, "x", alpha=0.2)
Xt = np.linspace(-1.1, 1.1, 1000)[:, None]
Yt = func(Xt)
_ = plt.plot(Xt, Yt, c="k")

#plt.show()


var_gp = .1
len_gp = .1

lr_natgrad = 1.

m_cvi = CVI(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp),
    gpflow.likelihoods.Gaussian(variance=var_gp))

m_vgp = gpflow.models.VGP(data,
    gpflow.kernels.SquaredExponential(lengthscales=len_gp),
    gpflow.likelihoods.Gaussian(variance=var_gp))


for k in range(10):
    m_cvi.update_variational_parameters(beta=lr_natgrad)
    print('cvi :',  m_cvi.elbo())


natgrad_opt = NaturalGradient(gamma=lr_natgrad)
variational_params = [(m_vgp.q_mu, m_vgp.q_sqrt)]

for k in range(10):
    natgrad_opt.minimize(m_vgp.training_loss, var_list=variational_params)
    print('svgp :', m_vgp.elbo())

