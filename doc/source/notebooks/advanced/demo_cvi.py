
import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.models.cvi import CVI

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


m_cvi = CVI(data,
    gpflow.kernels.SquaredExponential(lengthscales=.1),
    gpflow.likelihoods.Gaussian(variance=.2**2))

m_vgp = gpflow.models.VGP(data,
    gpflow.kernels.SquaredExponential(lengthscales=.1),
    gpflow.likelihoods.Gaussian(variance=.2**2))

print(m_cvi.maximum_log_likelihood_objective(beta=0))

print(m_vgp.elbo())
